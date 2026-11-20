# Depth Pro 2 ONNX

Focal-free ONNX, TensorRT and ROS 2 packaging for Apple's [Depth Pro 2](https://github.com/apple-aiml-research/ml-depth-pro-2) S and B. Pass `f_px` from `CameraInfo`; the 23.5M-parameter focal-length network never runs. Unofficial; not affiliated with Apple.

{{benchmark_table}}

{{gpu}}, 480x640, one frame per call, host array to host array. AbsRel against PyTorch FP32 on a 17-frame fixture; every FP32/FP16 graph passes AbsRel < 0.01 and delta1 > 0.995. [All results](results/)

## Use

[DepthPro2-S-ONNX](https://huggingface.co/christianengel/DepthPro2-S-ONNX) · [DepthPro2-B-ONNX](https://huggingface.co/christianengel/DepthPro2-B-ONNX) · [Space](https://huggingface.co/spaces/christianengel/DepthPro2-ONNX)

```python
from huggingface_hub import hf_hub_download
from depthpro2_onnx.runtime import DepthPro2ONNX, to_pixel_values

graph = DepthPro2ONNX(
    hf_hub_download("christianengel/DepthPro2-S-ONNX", "onnx/model_stream4_fp16.onnx")
)
depth = graph.infer(
    to_pixel_values(rgb_480x640), f_px=fx
)  # (1, 480, 640) meters; stream state kept
```

[ROS 2 node](ros2/) · [TensorRT on Jetson](jetson/build_tensorrt.sh)

## Findings

- **Block 0 attention overflows FP16.** Logits reach 2e6 (S) and 2e5 (B). The FP16 graphs keep that block in FP32 and land 5x closer to FP32 than Apple's default BF16.
- **TensorRT `--fp16` gives all-NaN depth.** So does ONNX Runtime's `trt_fp16_enable`. Build the FP16 graph with `trtexec --stronglyTyped`.
- **Dynamic INT8 breaks DINOv3.** Residual outliers of 1e3-1e4 collapse delta1 to 0.08. Weight-only INT8 holds AbsRel < 0.005 at 38% of the FP32 size.

## Reproduce

```sh
uv sync --extra export --group bench
uv run python fixtures/make_fixture.py                      # fixture + PyTorch references
uv run python export_onnx.py --size S --out build/DepthPro2-S
scripts/verify_all.sh                                       # parity gate, every graph
uv run python benchmark.py
ros2/test/run_in_docker.sh build/DepthPro2-S/onnx S         # ROS 2 end-to-end
```

## License

Code: [Apache-2.0](LICENSE). Weights: [Apple Machine Learning Research Model License](https://github.com/apple-aiml-research/ml-depth-pro-2/blob/main/LICENSE_MODEL), non-commercial research use only. The stream graph follows Apple's Core ML exporter (Apple sample code license).
