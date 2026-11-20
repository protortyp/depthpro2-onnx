---
license: apple-amlr
license_link: LICENSE
library_name: onnx
pipeline_tag: depth-estimation
base_model:
  - apple/DepthPro2-focal
  - apple/DepthPro2-S
  - apple/DepthPro2-B
base_model_relation: quantized
tags:
  - depth-estimation
  - metric-depth
  - focal-length
  - camera-calibration
  - depth-pro-2
  - onnx
  - tensorrt
---

# Depth Pro 2 focal length + auto depth, ONNX

Unofficial ONNX export of [apple/DepthPro2-focal](https://huggingface.co/apple/DepthPro2-focal), the network that guesses a photo's focal length. The `auto_*` graphs chain it with Depth Pro 2 S or B: photo in, depth in meters and `f_px` out, no calibration. [Try it in the browser](https://huggingface.co/spaces/christianengel/DepthPro2-ONNX). Not affiliated with or endorsed by Apple.

> **Research use only.** These weights are a Model Derivative under the [Apple Machine Learning Research Model License](LICENSE): non-commercial research and academic use only. Apple Machine Learning Research Model is licensed under the Apple Machine Learning Research Model License Agreement.

Calibrated camera? Use [DepthPro2-S-ONNX](https://huggingface.co/christianengel/DepthPro2-S-ONNX) and pass your `fx`; it's faster and a measured focal beats a guessed one.

## Quick start

```python
import json
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

repo = "{{repo}}"
config = json.load(open(hf_hub_download(repo, "config.json")))
session = ort.InferenceSession(hf_hub_download(repo, "onnx/auto_s_fp16.onnx"),
                               providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

# pixel_values: RGB in [0, 1], shape (1, 3, 480, 640), e.g. a photo scaled to cover and cropped.
depth, f_px = session.run(None, {"pixel_values": pixels})  # (1, 480, 640) meters, (1,) pixels
```

[`example.py`](example.py) does the resize and crop and converts `f_px` back to the original photo.

## Files

Static shapes, opset 18. Errors against PyTorch FP32 on a 17-frame fixture.

| File | Input | Output | Precision | MB | f_px max error | Depth AbsRel |
| --- | --- | --- | --- | ---: | ---: | ---: |
{{variants_table}}

- `pixel_values`: float32 RGB in [0, 1], NCHW, not normalized. 480x640 and 384x672 are the network's own input sizes for 4:3 and 16:9, so nothing is resized inside.
- `f_px`: float32 `(1,)`, focal length in pixels of `pixel_values`. `fov_deg`: horizontal and vertical field of view.
- `depth_meters`: float32 `(1, H, W)`, z-depth in meters; `inf` where infinitely far.

Video: estimate `f_px` on the first frames, take the median, keep it fixed (Apple uses the first 16), and run the depth-only graphs.

## How good is the guess

Parity above is against PyTorch, not ground truth. On two calibrated robot cameras, undistorted:

| Camera | Calibrated fx | Estimated | Error |
| --- | ---: | ---: | ---: |
| Overhead, room-scale view, 1920x1080 | 445.8 | 430.7 | -3.4% |
| Wrist, close-up of gripper and table, 640x480 | 557.2 | 460.6 | -17.3% |

Depth scales with the focal length, so a 17% focal error is a 17% depth error. Ordinary scenes are close; close-ups and odd lenses are not.

## Speed

{{gpu}}, 480x640, one frame per call, host array to host array.

{{benchmark_table}}

## Changes from the original

- Single frame, no resize or crop: inputs are already the network's input size and at most 16:9.
- ImageNet normalization and the FoV-to-`f_px` conversion are in the graph; `1/tan` is written as `cos/sin` (ONNX Runtime has no CUDA `Tan`).
- FP16: FP16 trunk; block 0 attention, RoPE tables, focal head and depth head stay FP32.
- Auto graphs: the focal graph feeds the [DepthPro2-S](https://huggingface.co/christianengel/DepthPro2-S-ONNX) / [B](https://huggingface.co/christianengel/DepthPro2-B-ONNX) FP16 depth graphs.

Source: [apple/DepthPro2-focal@{{focal_revision}}](https://huggingface.co/apple/DepthPro2-focal), 23.5M parameters. Export code: [{{github}}]({{github}}) (Apache-2.0).

## Citation

```bibtex
@article{DepthPro2:2026:arxiv,
  title   = {Depth Pro 2: Sharp and Temporally Consistent Metric Monocular Depth},
  author  = {Peiyun Hu and Marko Mihajlovic and Lyndon Duong and Lahav Lipson and Wei Dong and Ama\"{e}l Delaunoy and Stephan R. Richter and Vladlen Koltun},
  journal = {arXiv preprint arXiv:2611.26603},
  year    = {2026},
}
```
