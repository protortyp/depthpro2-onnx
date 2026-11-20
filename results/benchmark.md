| Model | Runtime | Latency (ms/frame) | FPS | Peak VRAM (MB) | Disk (MB) | AbsRel vs PyTorch FP32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DepthPro2-S | PyTorch BF16 (Apple default) | 10.3 | 96.9 | 622 | - | 0.0030 |
| DepthPro2-S | PyTorch FP32 | 10.5 | 95.1 | 588 | - | 0.0000 |
| DepthPro2-S | ORT CUDA FP32 | 11.6 | 86.2 | 798 | 99 | 0.0004 |
| DepthPro2-S | ORT CUDA FP16 | 8.8 | 114.1 | 548 | 51 | 0.0008 |
| DepthPro2-S | ORT CUDA INT8 | 11.8 | 84.8 | 796 | 38 | 0.0049 |
| DepthPro2-S | TensorRT FP16 engine | 3.6 | 277.8 | 480 | 56 | 0.0014 |
| DepthPro2-S | TensorRT FP16 engine, stream window 4 | 15.1 | 66.2 | 546 | 57 | 0.0011 |
| DepthPro2-S | ORT CUDA FP16 stream, window 4 | 11.5 | 87.0 | 1378 | 51 | 0.0005 |
| DepthPro2-S | ORT CUDA FP16 clip of 4 | 8.8 | 113.2 | 2352 | 53 | 0.0005 |
| DepthPro2-S | ORT CPU FP32 | 259.4 | 3.9 | - | 99 | 0.0000 |
| DepthPro2-S | ORT CPU INT8 | 303.1 | 3.3 | - | 38 | 0.0049 |
| DepthPro2-B | PyTorch BF16 (Apple default) | 11.5 | 86.6 | 1198 | - | 0.0022 |
| DepthPro2-B | PyTorch FP32 | 24.9 | 40.1 | 1112 | - | 0.0000 |
| DepthPro2-B | ORT CUDA FP32 | 22.4 | 44.6 | 1314 | 387 | 0.0002 |
| DepthPro2-B | ORT CUDA FP16 | 14.9 | 67.1 | 1322 | 199 | 0.0007 |
| DepthPro2-B | ORT CUDA INT8 | 23.1 | 43.2 | 1308 | 143 | 0.0030 |
| DepthPro2-B | TensorRT FP16 engine | 7.1 | 140.5 | 700 | 204 | 0.0009 |
| DepthPro2-B | TensorRT FP16 engine, stream window 4 | 38.9 | 25.7 | 830 | 204 | 0.0009 |
| DepthPro2-B | ORT CUDA FP16 stream, window 4 | 21.7 | 46.0 | 2472 | 199 | 0.0005 |
| DepthPro2-B | ORT CUDA FP16 clip of 4 | 19.2 | 52.1 | 5430 | 201 | 0.0006 |
