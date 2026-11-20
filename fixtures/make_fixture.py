"""Build the parity fixture and the PyTorch reference depth it is checked against.

The frames come from Apple's example video and photo at a pinned commit, verified by hash and
not redistributed here. The fixture's focal length is estimated once with Apple's focal-length
network, so the reference depth is in plausible meters; the exported graphs never run that network.

    uv run --extra export python fixtures/make_fixture.py
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
from pathlib import Path

import av
import numpy as np
import torch
from depth_pro_2.network.fov import FOVEstimator
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file

from depthpro2_onnx.model import CHECKPOINTS, FOCAL_CHECKPOINT, ClipDepth, load_model

ROOT = Path(__file__).resolve().parent
COMMIT = "aa389cb"
ASSETS = {
    "example.mp4": "2b354c7f6d0ed5621e6566602241be94b4631bb6a5559e704b786c4b9ff03ef1",
    "example.jpg": "58f0558ee6672af6f108f216482f5dc41ad28b0c16fea77d8a35d35384146255",
}
URL = "https://raw.githubusercontent.com/apple-aiml-research/ml-depth-pro-2/{commit}/assets/{name}"
NUM_FRAMES = 16


def fetch(name: str) -> bytes:
    cache = ROOT / "cache" / name
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL.format(commit=COMMIT, name=name)) as response:
            cache.write_bytes(response.read())
    data = cache.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != ASSETS[name]:
        raise RuntimeError(f"{name}: sha256 {digest} != {ASSETS[name]}")
    return data


def cover(image: Image.Image, height: int, width: int) -> np.ndarray:
    """Scale ``image`` to cover ``height`` x ``width`` and center-crop it."""
    scale = max(width / image.width, height / image.height)
    size = (round(image.width * scale), round(image.height * scale))
    image = image.resize(size, Image.Resampling.LANCZOS)
    left, top = (size[0] - width) // 2, (size[1] - height) // 2
    return np.array(image.crop((left, top, left + width, top + height)).convert("RGB"))


def load_frames(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    container = av.open(io.BytesIO(fetch("example.mp4")))
    clip = [
        cover(frame.to_image(), height, width)
        for _, frame in zip(range(NUM_FRAMES), container.decode(video=0), strict=False)
    ]
    photo = cover(Image.open(io.BytesIO(fetch("example.jpg"))), height, width)
    return np.stack(clip), photo[None]


def to_tensor(frames: np.ndarray, device: str) -> torch.Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(frames))
        .to(device)
        .permute(0, 3, 1, 2)
        .float()
        .div(255)
    )


@torch.no_grad()
def estimate_focal(frames: torch.Tensor) -> float:
    repo, revision = FOCAL_CHECKPOINT
    fov = FOVEstimator().to(frames.device).eval()
    fov.load_state_dict(load_file(hf_hub_download(repo, "model.safetensors", revision=revision)))
    mean = torch.tensor([0.485, 0.456, 0.406], device=frames.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=frames.device).view(1, 3, 1, 1)
    return float(fov.focal_length_px((frames - mean) / std))


@torch.no_grad()
def reference(
    size: str,
    inputs: dict[str, torch.Tensor],
    f_px: torch.Tensor,
    clips: list[int],
    windows: list[int],
):
    """Per-frame, clip and unbuffered-stream depth in float32 and under bfloat16 autocast.

    Streams run through Apple's ``infer_video``, the reference for the stream graphs.
    """
    out = {}
    for name, precision in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        device = inputs["clip"].device
        model = ClipDepth(load_model(size, device, precision)).to(device).eval()
        for key, frames in inputs.items():
            out[f"{name}/single/{key}"] = torch.cat([model(f[None], f_px) for f in frames])
        for t in clips:
            out[f"{name}/clip{t}/clip"] = model(inputs["clip"][:t], f_px)
        frames = model.normalize(inputs["clip"]).float()
        for w in windows:
            stream = model.model.infer_video(iter(frames), window=w, f_px=f_px, mode="unbuffered")
            out[f"{name}/stream{w}/clip"] = torch.stack([p["depth"] for p in stream])
    return {k: v.float().cpu().numpy() for k, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--clips", type=int, nargs="*", default=[4, 16])
    parser.add_argument("--windows", type=int, nargs="*", default=[4, 8])
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip, photo = load_frames(args.height, args.width)
    tag = f"{args.height}x{args.width}"
    inputs = {"clip": to_tensor(clip, device), "photo": to_tensor(photo, device)}
    f_px = estimate_focal(inputs["clip"])
    np.savez_compressed(ROOT / f"test_clip_{tag}.npz", clip=clip, photo=photo, f_px=f_px)

    refs = ROOT / "ref_outputs"
    refs.mkdir(exist_ok=True)
    for size in CHECKPOINTS:
        arrays = reference(
            size, inputs, torch.tensor([f_px], device=device), args.clips, args.windows
        )
        np.savez_compressed(refs / f"DepthPro2-{size}_{tag}.npz", **arrays)
        summary = {k: float(np.median(v[np.isfinite(v)])) for k, v in arrays.items()}
        print(size, json.dumps(summary, indent=1))
    print(f"f_px {f_px:.2f} at {tag}")


if __name__ == "__main__":
    main()
