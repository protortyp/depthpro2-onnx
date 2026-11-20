"""Check an exported graph against the PyTorch reference depth of the fixture.

    uv run python verify_parity.py build/DepthPro2-S/onnx/model_fp16.onnx --provider cuda

The gate is fixed: AbsRel < 0.01 and delta1 > 0.995 against PyTorch float32, on reference pixels
up to 200 m. It applies to float32 and float16 graphs; INT8 results are reported, not gated.
Exit status 1 means the graph failed the gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from depthpro2_onnx.metrics import compare, merge
from depthpro2_onnx.runtime import load, to_pixel_values

ROOT = Path(__file__).resolve().parent
GATE_ABS_REL = 0.01
GATE_DELTA1 = 0.995
GATED_PRECISIONS = ("fp32", "fp16")


def run(graph, fixture, refs, ref_precision: str) -> dict:
    f_px = float(fixture["f_px"])
    clip = to_pixel_values(fixture["clip"])
    photo = to_pixel_values(fixture["photo"])
    mode = graph.config["mode"]
    if mode == "single":
        results = [
            compare(
                graph.infer(frames[i : i + 1], f_px),
                refs[f"{ref_precision}/single/{key}"][i : i + 1],
            )
            for key, frames in (("clip", clip), ("photo", photo))
            for i in range(len(frames))
        ]
        return merge(results)
    if mode == "clip":
        t = graph.config["frames"]
        return compare(graph.infer(clip[:t], f_px), refs[f"{ref_precision}/clip{t}/clip"])
    if mode == "stream":
        graph.reset()
        w = graph.config["window"]
        return compare(graph.infer(clip, f_px), refs[f"{ref_precision}/stream{w}/clip"])
    raise ValueError(f"unknown mode {mode!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", type=Path)
    parser.add_argument("--provider", default="cuda", choices=["cpu", "cuda", "tensorrt"])
    parser.add_argument("--json", type=Path, help="write the results here")
    parser.add_argument("--size", choices=["S", "B"], help="model size, for TensorRT engines")
    parser.add_argument("--input", default="480x640", help="HxW, for TensorRT engines")
    args = parser.parse_args()

    graph = load(args.onnx, args.provider)
    config = graph.config
    # Engines carry no ONNX metadata; their size and input come from the command line.
    size = config["base_model"].rsplit("-", 1)[-1] if "base_model" in config else args.size
    tag = f"{config['height']}x{config['width']}" if "height" in config else args.input
    fixture = np.load(ROOT / "fixtures" / f"test_clip_{tag}.npz")
    refs = np.load(ROOT / "fixtures" / "ref_outputs" / f"DepthPro2-{size}_{tag}.npz")

    report = {
        "graph": args.onnx.name,
        "provider": graph.provider,
        "config": config,
        "vs_fp32": run(graph, fixture, refs, "fp32"),
        "vs_bf16": run(graph, fixture, refs, "bf16"),
    }
    vs = report["vs_fp32"]
    gated = config["precision"] in GATED_PRECISIONS
    passed = vs["abs_rel"] < GATE_ABS_REL and vs["delta1"] > GATE_DELTA1
    report["gate"] = {
        "abs_rel_lt": GATE_ABS_REL,
        "delta1_gt": GATE_DELTA1,
        "applies": gated,
        "passed": passed,
    }
    print(json.dumps(report, indent=1))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1) + "\n")
    return 1 if gated and not passed else 0


if __name__ == "__main__":
    sys.exit(main())
