"""Export the focal-length network, and auto-depth graphs that chain it with S or B.

    uv run --extra export python export_focal.py --out build/DepthPro2-focal

Writes, per input size (480x640 at onnx/, others at onnx/<HxW>/):
    focal.onnx, focal_fp16.onnx          pixel_values -> f_px (1,), fov_deg (1, 2)
    auto_s_fp16.onnx, auto_b_fp16.onnx   pixel_values -> depth_meters (1, H, W), f_px (1,)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from depthpro2_onnx.model import (
    CHECKPOINTS,
    FOCAL_CHECKPOINT,
    FP32_ATTENTION_BLOCKS,
    AutoDepth,
    FocalLength,
    check_focal_size,
    check_size,
    focal_to_fp16,
    load_focal,
    load_model,
    to_fp16,
)
from export_onnx import OPSET, export, git_commit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="*", default=["480x640", "384x672"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    focal_repo, focal_revision = FOCAL_CHECKPOINT

    for tag in args.sizes:
        height, width = (int(v) for v in tag.split("x"))
        check_size(height, width)
        check_focal_size(height, width)
        out = f"{args.out}/onnx" + ("" if tag == "480x640" else f"/{tag}")
        pixels = torch.rand(1, 3, height, width, device=device)
        base = {
            "focal_model": focal_repo,
            "focal_revision": focal_revision,
            "height": height,
            "width": width,
            "exporter_commit": git_commit(),
            "opset": OPSET,
        }
        fp16_meta = {"fp32_attention_blocks": list(FP32_ATTENTION_BLOCKS)}

        for precision in ("fp32", "fp16"):
            fov = load_focal(device)
            if precision == "fp16":
                fov = focal_to_fp16(fov)
            meta = base | {"mode": "focal", "precision": precision}
            meta |= fp16_meta if precision == "fp16" else {}
            name = "focal" if precision == "fp32" else "focal_fp16"
            export(
                FocalLength(fov).to(device).eval(),
                (pixels,),
                Path(out) / f"{name}.onnx",
                ["pixel_values"],
                ["f_px", "fov_deg"],
                meta,
            )

        for size in CHECKPOINTS:
            repo, revision = CHECKPOINTS[size]
            module = AutoDepth(to_fp16(load_model(size, device)), focal_to_fp16(load_focal(device)))
            meta = (
                base
                | fp16_meta
                | {
                    "mode": "auto",
                    "precision": "fp16",
                    "base_model": repo,
                    "base_revision": revision,
                    "frames": 1,
                }
            )
            export(
                module.to(device).eval(),
                (pixels,),
                Path(out) / f"auto_{size.lower()}_fp16.onnx",
                ["pixel_values"],
                ["depth_meters", "f_px"],
                meta,
            )
            del module
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
