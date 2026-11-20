---
license: apple-amlr
license_link: LICENSE
library_name: onnx
pipeline_tag: depth-estimation
base_model: {{base_model}}
base_model_relation: quantized
tags:
  - depth-estimation
  - metric-depth
  - depth-pro-2
  - onnx
  - tensorrt
  - ros2
  - robotics
  - jetson
---

# Depth Pro 2 {{size}} ONNX, focal-free

Unofficial ONNX export of [{{base_model}}](https://huggingface.co/{{base_model}}) for calibrated cameras. Pass the focal length `f_px`; the 23.5M-parameter focal-length network is gone. Output is metric depth in meters. Every graph passes a parity gate against PyTorch. Not affiliated with or endorsed by Apple.

> **Research use only.** These weights are a Model Derivative under the [Apple Machine Learning Research Model License](LICENSE): non-commercial research and academic use only. Apple Machine Learning Research Model is licensed under the Apple Machine Learning Research Model License Agreement.

## Quick start

```sh
pip install onnxruntime-gpu huggingface_hub pillow numpy   # onnxruntime for CPU
```

```python
import json
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

repo = "{{repo}}"
config = json.load(open(hf_hub_download(repo, "config.json")))
session = ort.InferenceSession(
    hf_hub_download(repo, "onnx/model_fp16.onnx"),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

# pixel_values: RGB in [0, 1], shape (1, 3, 480, 640). f_px: fx of that 640x480 image in pixels.
depth = session.run(None, {"pixel_values": pixels, "f_px": np.array([fx], np.float32)})[0]
```

[`example.py`](example.py) resizes and crops any photo and scales `f_px` with it. From ROS 2, `f_px` is `CameraInfo.k[0]` of the rectified image.

## Files

Static shapes, opset 18. AbsRel and delta1 are against PyTorch FP32 on a 17-frame fixture.

| File | Input | Frames | Precision | MB | AbsRel | delta1 |
| --- | --- | --- | --- | ---: | ---: | ---: |
{{variants_table}}

- **Single frame** (`model*.onnx`): each frame on its own.
- **Stream** (`model_stream4*.onnx`): one frame per call, attending to the previous three; Depth Pro 2's unbuffered video mode. The cache goes in as `past_*` and comes out as `present_*`; zeros for the first frame.
- **Clip** (`model_clip4*.onnx`): four frames jointly, as `DepthPro2.infer` on a `(4, 3, H, W)` clip.

I/O:

- `pixel_values`: float32 RGB in [0, 1], NCHW, not normalized.
- `f_px`: float32 `(1,)`, horizontal focal length in pixels of `pixel_values`.
- `depth_meters`: float32 `(T, H, W)`, z-depth in meters; `inf` where infinitely far (sky).

## Speed

{{gpu}}, ONNX Runtime {{ort}}, 480x640, one frame at a time, host array to host array. Peak VRAM is the whole process (NVML).

{{benchmark_table}}

## TensorRT and Jetson

On the device: `./build_tensorrt.sh onnx/model_fp16.onnx` ([script](build_tensorrt.sh), `trtexec --stronglyTyped`). Do not use `--fp16` or ONNX Runtime's TensorRT provider: both may run block 0's attention (logits up to 2e6) or B's LayerNorms in FP16 and return NaN.

## ROS 2

[Node]({{github}}/tree/main/ros2): subscribes to `Image` and `CameraInfo`, takes `f_px` from `k[0]`, publishes `32FC1` depth in meters, matching `CameraInfo`, optional `PointCloud2`.

## Changes from the original

- Focal-length network ([apple/DepthPro2-focal](https://huggingface.co/apple/DepthPro2-focal)) removed; `f_px` is an input.
- ImageNet normalization and `depth = f_px / (W * inverse_depth)` are in the graph.
- No padding: sides must be multiples of 32.
- FP16: FP16 weights; block 0 attention, RoPE tables, depth head and final division stay FP32.
- INT8: weight-only 8-bit MatMul weights (symmetric, blocks of 32). Dynamic activation INT8 collapses this backbone (delta1 < 0.5).
- Stream graphs follow the unbuffered step of Apple's Core ML exporter, with the cache as explicit tensors.

Source: [{{base_model}}@{{base_revision}}](https://huggingface.co/{{base_model}}), {{params_m}}M parameters. Export code: [{{github}}]({{github}}) (Apache-2.0).

## Citation

```bibtex
@article{DepthPro2:2026:arxiv,
  title   = {Depth Pro 2: Sharp and Temporally Consistent Metric Monocular Depth},
  author  = {Peiyun Hu and Marko Mihajlovic and Lyndon Duong and Lahav Lipson and Wei Dong and Ama\"{e}l Delaunoy and Stephan R. Richter and Vladlen Koltun},
  journal = {arXiv preprint arXiv:2611.26603},
  year    = {2026},
}
```
