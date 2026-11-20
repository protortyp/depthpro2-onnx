"""PyTorch FP32 references for the focal-length and auto-depth graphs.

Per frame of the fixture: Apple's focal estimate, and S and B depth computed with that estimate
(what the auto graphs return). Run after make_fixture.py.

    uv run --extra export python fixtures/make_focal_refs.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from depthpro2_onnx.model import CHECKPOINTS, ClipDepth, FocalLength, load_focal, load_model

ROOT = Path(__file__).resolve().parent
SIZES = ["480x640", "384x672"]


@torch.no_grad()
def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    focal = FocalLength(load_focal(device)).to(device)
    for tag in SIZES:
        fixture = np.load(ROOT / f"test_clip_{tag}.npz")
        out = {}
        frames = {}
        for key in ("clip", "photo"):
            rgb = np.ascontiguousarray(fixture[key])
            frames[key] = torch.from_numpy(rgb).to(device).permute(0, 3, 1, 2).float() / 255
            estimates = [focal(f[None]) for f in frames[key]]
            out[f"f_px/{key}"] = torch.cat([f for f, _ in estimates])
            out[f"fov_deg/{key}"] = torch.cat([d for _, d in estimates])
        for size in CHECKPOINTS:
            depth = ClipDepth(load_model(size, device)).to(device)
            for key, f_px in (("clip", out["f_px/clip"]), ("photo", out["f_px/photo"])):
                out[f"{size}/{key}"] = torch.cat(
                    [depth(f[None], fp[None]) for f, fp in zip(frames[key], f_px, strict=True)]
                )
        arrays = {k: v.float().cpu().numpy() for k, v in out.items()}
        np.savez_compressed(ROOT / "ref_outputs" / f"focal_{tag}.npz", **arrays)
        print(tag, "f_px", np.round(arrays["f_px/clip"][:4], 1), "photo", arrays["f_px/photo"])


if __name__ == "__main__":
    main()
