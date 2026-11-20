"""Export focal-free Depth Pro 2 graphs to ONNX.

    uv run --extra export python export_onnx.py --size S --out build/DepthPro2-S

Writes, for one static input size:
    onnx/model.onnx, model_fp16.onnx, model_int8.onnx      single frame (1, 3, H, W)
    onnx/model_clip{T}.onnx, model_clip{T}_fp16.onnx        T frames attending to each other
    onnx/model_stream{W}.onnx, model_stream{W}_fp16.onnx    one frame, W-frame cache as state
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import onnx
import torch

from depthpro2_onnx.model import (
    CHECKPOINTS,
    FP32_ATTENTION_BLOCKS,
    ClipDepth,
    StreamDepth,
    check_size,
    load_model,
    to_fp16,
)

OPSET = 18
LICENSE = (
    "Apple Machine Learning Research Model License (apple-amlr): non-commercial research use only. "
    "Apple Machine Learning Research Model is licensed under the Apple Machine Learning Research "
    "Model License Agreement."
)


def git_commit() -> str:
    result = subprocess.run(
        ["git", "describe", "--always", "--dirty", "--exclude=*"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
    )
    return result.stdout.strip() or "unknown"


def export(module: torch.nn.Module, args: tuple, path: Path, inputs, outputs, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with torch.no_grad():
        torch.onnx.export(
            module,
            args,
            str(path),
            dynamo=True,
            opset_version=OPSET,
            input_names=inputs,
            output_names=outputs,
            external_data=False,
            optimize=True,
            verbose=False,
        )
    simplify(path)
    model = onnx.load(str(path))
    onnx.helper.set_model_props(
        model,
        {"depthpro2_onnx": json.dumps(meta, sort_keys=True), "license": LICENSE},
    )
    model.producer_name = "depthpro2-onnx"
    model.doc_string = (
        "Depth Pro 2 without its focal-length network. pixel_values: RGB in [0, 1], NCHW. "
        "f_px: horizontal focal length in pixels of pixel_values. depth_meters: z-depth in meters."
    )
    onnx.save(model, str(path))
    onnx.checker.check_model(str(path))
    size_mb = path.stat().st_size / 1e6
    print(f"wrote {path} ({size_mb:.1f} MB, {time.time() - start:.0f} s)", flush=True)


def simplify(path: Path) -> None:
    import onnxslim

    model = onnxslim.slim(onnx.load(str(path)))
    onnx.save(model, str(path))


def quantize_int8(src: Path, dst: Path) -> None:
    """Dynamic INT8: MatMul weights per channel, activations per tensor at run time."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        src,
        dst,
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul"],
        extra_options={"MatMulConstBOnly": True},
    )
    model = onnx.load(str(dst))
    meta = json.loads({p.key: p.value for p in model.metadata_props}["depthpro2_onnx"])
    onnx.helper.set_model_props(
        model,
        {
            "depthpro2_onnx": json.dumps(meta | {"precision": "int8"}, sort_keys=True),
            "license": LICENSE,
        },
    )
    onnx.save(model, str(dst))
    print(f"wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=sorted(CHECKPOINTS), required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--clip", type=int, nargs="*", default=[4], help="clip lengths T")
    parser.add_argument("--stream", type=int, nargs="*", default=[4], help="stream windows W")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", nargs="*", default=None, help="export only these file stems")
    args = parser.parse_args()
    check_size(args.height, args.width)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out / "onnx"
    repo, revision = CHECKPOINTS[args.size]
    base_meta = {
        "base_model": repo,
        "base_revision": revision,
        "height": args.height,
        "width": args.width,
        "focal_estimator_removed": True,
        "exporter_commit": git_commit(),
        "opset": OPSET,
    }
    pixels = lambda t: torch.rand(t, 3, args.height, args.width, device=device)  # noqa: E731
    f_px = torch.tensor([500.0], device=device)

    jobs = [("model", "single", 1)]
    jobs += [(f"model_clip{t}", "clip", t) for t in args.clip]
    jobs += [(f"model_stream{w}", "stream", w) for w in args.stream]
    for stem, mode, n in jobs:
        for precision in ("fp32", "fp16"):
            name = stem if precision == "fp32" else f"{stem}_fp16"
            if args.only is not None and name not in args.only:
                continue
            model = load_model(args.size, device)
            if precision == "fp16":
                model = to_fp16(model)
            meta = base_meta | {"mode": mode, "precision": precision}
            if precision == "fp16":
                meta["fp32_attention_blocks"] = list(FP32_ATTENTION_BLOCKS)
            if mode == "stream":
                module = StreamDepth(model, n, args.height, args.width).to(device).eval()
                state = module.initial_state(device)
                names = module.state_names
                meta |= {"window": n, "state": names}
                export(
                    module,
                    (pixels(1), f_px, *state),
                    out / f"{name}.onnx",
                    ["pixel_values", "f_px"] + [f"past_{s}" for s in names],
                    ["depth_meters"] + [f"present_{s}" for s in names],
                    meta,
                )
            else:
                module = ClipDepth(model).to(device).eval()
                meta |= {"frames": n}
                export(
                    module,
                    (pixels(n), f_px),
                    out / f"{name}.onnx",
                    ["pixel_values", "f_px"],
                    ["depth_meters"],
                    meta,
                )
            del model, module
            torch.cuda.empty_cache()

    if args.only is None or "model_int8" in args.only:
        quantize_int8(out / "model.onnx", out / "model_int8.onnx")


if __name__ == "__main__":
    main()
