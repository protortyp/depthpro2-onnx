"""Gradio demo: metric depth from a photo and a known focal length, on CPU with ONNX Runtime."""

from __future__ import annotations

import json
import math
import tempfile
import time
import urllib.request
from functools import lru_cache
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

REPOS = {"S (24M)": "christianengel/DepthPro2-S-ONNX", "B (97M)": "christianengel/DepthPro2-B-ONNX"}
VARIANTS = {"INT8 weights (smallest)": "onnx/model_int8.onnx", "FP32": "onnx/model.onnx"}
EXAMPLE_URL = "https://raw.githubusercontent.com/apple-aiml-research/ml-depth-pro-2/aa389cb/assets/example.jpg"


@lru_cache(maxsize=4)
def session(repo: str, filename: str) -> tuple[ort.InferenceSession, dict]:
    config = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
    path = hf_hub_download(repo, filename)
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"]), config


def cover_crop(image: Image.Image, height: int, width: int) -> tuple[np.ndarray, float, int, int]:
    scale = max(width / image.width, height / image.height)
    size = (max(width, round(image.width * scale)), max(height, round(image.height * scale)))
    left, top = (size[0] - width) // 2, (size[1] - height) // 2
    resized = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    return np.asarray(resized.crop((left, top, left + width, top + height))), scale, left, top


def colorize(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    inverse = 1.0 / np.clip(depth, near, far)
    t = (inverse - 1.0 / far) / (1.0 / near - 1.0 / far)
    return (matplotlib.colormaps["turbo"](t)[..., :3] * 255).astype(np.uint8)


def predict(image: Image.Image, focal_mode: str, focal: float, model: str, variant: str):
    if image is None:
        raise gr.Error("Upload a photo.")
    sess, config = session(REPOS[model], VARIANTS[variant])
    height, width = config["input"]["height"], config["input"]["width"]
    rgb, scale, _, _ = cover_crop(image, height, width)
    if focal_mode == "fx in pixels of the uploaded photo":
        f_px = focal * scale
    else:
        f_px = 0.5 * width / math.tan(math.radians(focal) / 2)
    pixels = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], np.float32) / 255
    start = time.perf_counter()
    depth = sess.run(None, {"pixel_values": pixels, "f_px": np.array([f_px], np.float32)})[0][0]
    elapsed = (time.perf_counter() - start) * 1e3
    finite = depth[np.isfinite(depth)]
    near, far = np.percentile(finite, 2), np.percentile(finite, 98)
    out = Path(tempfile.mkdtemp()) / "depth_meters.npy"
    np.save(out, depth.astype(np.float32))
    summary = (
        f"{width}x{height} input, f_px {f_px:.1f}, {elapsed:.0f} ms on this Space's CPU. "
        f"Depth 2nd-98th percentile: {near:.2f}-{far:.2f} m, median {np.median(finite):.2f} m."
    )
    return Image.fromarray(rgb), Image.fromarray(colorize(depth, near, far)), summary, str(out)


def example_image() -> str | None:
    path = Path(tempfile.gettempdir()) / "depthpro2_example.jpg"
    if not path.exists():
        try:
            urllib.request.urlretrieve(EXAMPLE_URL, path)
        except OSError:
            return None
    return str(path)


with gr.Blocks(title="Depth Pro 2 ONNX (unofficial)") as demo:
    gr.Markdown(
        "# Depth Pro 2 ONNX, focal-free (unofficial)\n"
        "Metric depth from one photo and a focal length you already know. "
        "The focal-length network is removed. "
        "[S](https://huggingface.co/christianengel/DepthPro2-S-ONNX) · "
        "[B](https://huggingface.co/christianengel/DepthPro2-B-ONNX) · "
        "[Code](https://github.com/protortyp/depthpro2-onnx)\n\n"
        "Weights: Apple Machine Learning Research Model License, **non-commercial research use "
        "only**. Not affiliated with or endorsed by Apple."
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Photo")
            focal_mode = gr.Radio(
                ["horizontal field of view in degrees", "fx in pixels of the uploaded photo"],
                value="horizontal field of view in degrees",
                label="Focal length given as",
            )
            focal = gr.Number(value=60.0, label="Value")
            model = gr.Radio(list(REPOS), value="S (24M)", label="Model")
            variant = gr.Radio(list(VARIANTS), value="INT8 weights (smallest)", label="Graph")
            run = gr.Button("Estimate depth", variant="primary")
        with gr.Column():
            cropped = gr.Image(label="Model input (scaled and center-cropped)")
            depth = gr.Image(label="Depth (red near, blue far)")
            summary = gr.Markdown()
            download = gr.File(label="Depth in meters (.npy, float32)")
    run.click(
        predict, [image, focal_mode, focal, model, variant], [cropped, depth, summary, download]
    )
    example = example_image()
    if example:
        gr.Examples(
            [
                [
                    example,
                    "horizontal field of view in degrees",
                    65.6,
                    "S (24M)",
                    "INT8 weights (smallest)",
                ]
            ],
            [image, focal_mode, focal, model, variant],
            label="Example (Apple's sample photo, fetched from the ml-depth-pro-2 repository)",
        )

if __name__ == "__main__":
    demo.launch()
