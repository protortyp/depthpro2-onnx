"""Assemble the Hugging Face model repositories in release/ from build/ and results/.

    uv run python scripts/stage_release.py

The model cards come from release_templates/README_<size>.md with {{placeholders}} filled from
the parity and benchmark results, so every number in a card traces back to a results file.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from huggingface_hub import get_safetensors_metadata, hf_hub_download

from depthpro2_onnx.model import CHECKPOINTS

ROOT = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/protortyp/depthpro2-onnx"
HF_USER = "christianengel"
DEFAULT_SIZE = "480x640"
EXTRA_SIZES = ["384x672"]
DESCRIPTIONS = {
    "model": ("single", "fp32", "1 frame"),
    "model_fp16": ("single", "fp16", "1 frame"),
    "model_int8": ("single", "int8", "1 frame"),
    "model_clip4": ("clip", "fp32", "4 frames attending to each other"),
    "model_clip4_fp16": ("clip", "fp16", "4 frames attending to each other"),
    "model_stream4": ("stream", "fp32", "1 frame + 3-frame cache as state"),
    "model_stream4_fp16": ("stream", "fp16", "1 frame + 3-frame cache as state"),
}


def parity_for(size: str, build_dir: str, stem: str) -> dict:
    """Parity on CUDA (the CPU result for INT8, its main target, if present)."""
    folder = ROOT / "results" / "parity" / build_dir
    for provider in ["cpu", "cuda"] if stem.endswith("int8") else ["cuda"]:
        path = folder / f"{stem}_{provider}.json"
        if path.exists():
            return json.loads(path.read_text())
    raise FileNotFoundError(f"no parity result for {build_dir}/{stem}; run scripts/verify_all.sh")


def stage(size: str) -> Path:
    repo, revision = CHECKPOINTS[size]
    name = f"DepthPro2-{size}-ONNX"
    out = ROOT / "release" / name
    if out.exists():
        shutil.rmtree(out)
    (out / "onnx").mkdir(parents=True)

    variants = {}
    builds = [(DEFAULT_SIZE, f"DepthPro2-{size}", "onnx")]
    builds += [(res, f"DepthPro2-{size}-{res}", f"onnx/{res}") for res in EXTRA_SIZES]
    for res, build_dir, target in builds:
        src = ROOT / "build" / build_dir / "onnx"
        (out / target).mkdir(parents=True, exist_ok=True)
        for stem, (mode, precision, frames) in DESCRIPTIONS.items():
            graph = src / f"{stem}.onnx"
            if not graph.exists():
                continue
            shutil.copy2(graph, out / target / graph.name)
            report = parity_for(size, build_dir, stem)
            height, width = (int(v) for v in res.split("x"))
            variants[f"{target}/{graph.name}"] = {
                "mode": mode,
                "precision": precision,
                "frames": frames,
                "height": height,
                "width": width,
                "size_mb": round(graph.stat().st_size / 1e6, 1),
                "parity_vs_pytorch_fp32": {
                    k: round(report["vs_fp32"][k], 6)
                    for k in ("abs_rel", "delta1", "max_abs_near_m", "cosine")
                },
            }

    shutil.copy2(hf_hub_download(repo, "LICENSE", revision=revision), out / "LICENSE")
    shutil.copy2(ROOT / "jetson" / "build_tensorrt.sh", out / "build_tensorrt.sh")
    params = get_safetensors_metadata(repo, revision=revision).parameter_count
    config = {
        "architecture": "DepthPro2 (DINOv3 ViT encoder with cross-frame attention, DPT decoder)",
        "model_type": "depthpro2",
        "base_model": repo,
        "base_revision": revision,
        "parameters": sum(params.values()),
        "focal_estimator_removed": True,
        "removed_network": "apple/DepthPro2-focal (23.5M parameters)",
        "input": {
            "height": int(DEFAULT_SIZE.split("x")[0]),
            "width": int(DEFAULT_SIZE.split("x")[1]),
            "other_sizes": EXTRA_SIZES,
            "pixel_values": "float32 (T, 3, H, W), RGB in [0, 1], not normalized",
            "f_px": "float32 (1,), horizontal focal length in pixels of pixel_values",
            "multiple_of": 32,
        },
        "output": {
            "depth_meters": "float32 (T, H, W), z-depth in meters; inf where infinitely far"
        },
        "stream_state": "past_* inputs and present_* outputs; zeros for the first frame",
        "precisions": sorted({v["precision"] for v in variants.values()}),
        "variants": variants,
        "opset": 18,
        "exporter": GITHUB,
        "license": "apple-amlr",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "example.py").write_text(EXAMPLE.replace("{repo}", f"{HF_USER}/{name}"))
    return out


EXAMPLE = '''"""Metric depth from one photo with a known focal length."""

import json

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

repo = "{repo}"
config = json.load(open(hf_hub_download(repo, "config.json")))
height, width = config["input"]["height"], config["input"]["width"]
session = ort.InferenceSession(
    hf_hub_download(repo, "onnx/model_fp16.onnx"),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

image = Image.open("photo.jpg").convert("RGB")
fx = 1400.0  # horizontal focal length of photo.jpg in pixels, e.g. CameraInfo.k[0]

# Scale to cover the input size and center-crop; the focal length scales with the image.
scale = max(width / image.width, height / image.height)
size = (max(width, round(image.width * scale)), max(height, round(image.height * scale)))
left, top = (size[0] - width) // 2, (size[1] - height) // 2
image = image.resize(size, Image.Resampling.LANCZOS).crop((left, top, left + width, top + height))

pixels = np.asarray(image, np.float32).transpose(2, 0, 1)[None] / 255
f_px = np.array([fx * scale], np.float32)
depth = session.run(None, {"pixel_values": pixels, "f_px": f_px})[0][0]  # (H, W) meters
print(depth.shape, float(np.median(depth[np.isfinite(depth)])))
'''


def fill(template: str, values: dict) -> str:
    def sub(match: re.Match) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"template placeholder {key!r} has no value")
        return str(values[key])

    return re.sub(r"\{\{(\w+)\}\}", sub, template)


def main() -> None:
    for size in CHECKPOINTS:
        out = stage(size)
        template = ROOT / "release_templates" / "MODEL_CARD.md"
        if template.exists():
            values = card_values(size, out)
            (out / "README.md").write_text(fill(template.read_text(), values))
        total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
        print(f"staged {out} ({total:.0f} MB)")


def card_values(size: str, out: Path) -> dict:
    config = json.loads((out / "config.json").read_text())
    rows = []
    for path, v in config["variants"].items():
        p = v["parity_vs_pytorch_fp32"]
        rows.append(
            f"| `{path}` | {v['height']}x{v['width']} | {v['frames']} | {v['precision'].upper()} "
            f"| {v['size_mb']:.0f} | {p['abs_rel']:.4f} | {p['delta1']:.4f} |"
        )
    bench = ROOT / "results" / "benchmark.md"
    bench_rows = []
    if bench.exists():
        lines = bench.read_text().splitlines()
        bench_rows = lines[:2] + [line for line in lines[2:] if f"DepthPro2-{size} " in line]
    env = json.loads((ROOT / "results" / "benchmark.json").read_text())["environment"]
    return {
        "size": size,
        "repo": f"{HF_USER}/DepthPro2-{size}-ONNX",
        "base_model": config["base_model"],
        "base_revision": config["base_revision"][:7],
        "params_m": f"{config['parameters'] / 1e6:.1f}",
        "variants_table": "\n".join(rows),
        "benchmark_table": "\n".join(bench_rows),
        "gpu": env["gpu"],
        "cpu": env["cpu"],
        "ort": env["onnxruntime"],
        "github": GITHUB,
    }


if __name__ == "__main__":
    main()
