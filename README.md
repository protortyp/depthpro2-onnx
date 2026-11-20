# Depth Pro 2 ONNX

ONNX, TensorRT and ROS 2 for Apple's [Depth Pro 2](https://github.com/apple-aiml-research/ml-depth-pro-2) S and B, in two versions. Unofficial.

| | Focal-free | Focal |
| --- | --- | --- |
| Input | image + `f_px` | image |
| Output | depth (m) | depth (m) + estimated `f_px` |
| For | calibrated cameras, robots | any photo |
| Hugging Face | [S](https://huggingface.co/christianengel/DepthPro2-S-ONNX) · [B](https://huggingface.co/christianengel/DepthPro2-B-ONNX) | [focal + auto S/B](https://huggingface.co/christianengel/DepthPro2-focal-ONNX) |

The focal version adds Apple's 23.5M-parameter focal network. Its guess: 3.4% off on a room-scale robot camera, 17% on a gripper close-up. Pass `fx` when you have it. [Browser demo](https://huggingface.co/spaces/christianengel/DepthPro2-ONNX).

Focal-free speed:

| Model | Runtime | ms/frame | FPS | VRAM MB | Disk MB | AbsRel |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| S | PyTorch BF16 (Apple default) | 10.3 | 97 | 622 | - | 0.0030 |
| S | ORT CUDA FP16 | 8.8 | 114 | 548 | 51 | 0.0008 |
| S | TensorRT FP16 engine | 3.6 | 278 | 480 | 56 | 0.0014 |
| S | ORT CUDA FP16 stream, window 4 | 11.5 | 87 | 1378 | 51 | 0.0005 |
| S | ORT CPU FP32 (16 cores) | 259.4 | 4 | - | 99 | 0.0000 |
| B | PyTorch BF16 (Apple default) | 11.5 | 87 | 1198 | - | 0.0022 |
| B | ORT CUDA FP16 | 14.9 | 67 | 1322 | 199 | 0.0007 |
| B | TensorRT FP16 engine | 7.1 | 140 | 700 | 204 | 0.0009 |

RTX 3090 (Ryzen 9 5950X for CPU), 480x640, one frame per call, host array to host array. AbsRel against PyTorch FP32 on a 17-frame fixture; every FP32/FP16 graph passes AbsRel < 0.01 and delta1 > 0.995. [All results](results/)

## Use

```python
from huggingface_hub import hf_hub_download
from depthpro2_onnx.runtime import load, to_pixel_values

# Focal-free: you know fx (CameraInfo.k[0]).
graph = load(hf_hub_download("christianengel/DepthPro2-S-ONNX", "onnx/model_stream4_fp16.onnx"))
depth = graph.infer(to_pixel_values(rgb), f_px=fx)  # rgb: uint8 480x640x3; meters, state kept

# Focal: you don't.
import onnxruntime as ort
auto = ort.InferenceSession(
    hf_hub_download("christianengel/DepthPro2-focal-ONNX", "onnx/auto_s_fp16.onnx"),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
depth, f_px = auto.run(None, {"pixel_values": to_pixel_values(rgb)})
```

[ROS 2 node](ros2/) (focal-free) · [TensorRT on Jetson](jetson/build_tensorrt.sh) (both)

## Findings

- **Block 0 attention overflows FP16.** Logits reach 2e6 (S), 2e5 (B). FP16 graphs keep that block in FP32 and land 3-4x closer to FP32 than Apple's BF16.
- **TensorRT `--fp16` returns all-NaN depth.** ONNX Runtime's TensorRT provider fails too, on S or B depending on the build. Use `trtexec --stronglyTyped` engines.
- **Dynamic INT8 breaks DINOv3.** Residual outliers of 1e3-1e4 drop delta1 to 0.08. Weight-only INT8 keeps AbsRel < 0.005 at 38% of FP32 size, but is not faster.

## Reproduce

```sh
uv sync --extra export --group bench
uv run python fixtures/make_fixture.py                      # fixture + PyTorch references
uv run python export_onnx.py --size S --out build/DepthPro2-S
scripts/verify_all.sh                                       # parity gate, every graph
uv run python fixtures/make_focal_refs.py && uv run python export_focal.py --out build/DepthPro2-focal
scripts/verify_focal_all.sh                                 # focal within 1%, auto depth gate
uv run python benchmark.py
ros2/test/run_in_docker.sh build/DepthPro2-S/onnx S         # ROS 2 end-to-end
```

## License

Code: [Apache-2.0](LICENSE). Weights: [Apple Machine Learning Research Model License](https://github.com/apple-aiml-research/ml-depth-pro-2/blob/main/LICENSE_MODEL), non-commercial research use only. The stream graph follows Apple's Core ML exporter (Apple sample code license).
