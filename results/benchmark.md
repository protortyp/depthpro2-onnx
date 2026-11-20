| Model | Runtime | Latency (ms/frame) | FPS | Peak VRAM (MB) | Disk (MB) | AbsRel vs PyTorch FP32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DepthPro2-S | PyTorch BF16 (Apple default) | 10.0 | 99.8 | 622 | - | 0.0030 |
| DepthPro2-S | PyTorch FP32 | 10.5 | 95.1 | 588 | - | 0.0000 |
| DepthPro2-S | ORT CUDA FP32 | 11.5 | 87.0 | 798 | 99 | 0.0004 |
| DepthPro2-S | ORT CUDA FP16 | 8.3 | 120.4 | 548 | 51 | 0.0008 |
| DepthPro2-S | ORT CUDA INT8 | 11.7 | 85.1 | 796 | 38 | 0.0049 |
| DepthPro2-S | ORT TensorRT FP16 | 3.4 | 296.9 | 914 | 51 | 0.0015 |
| DepthPro2-S | ORT CUDA FP16 stream, window 4 | 11.2 | 89.6 | 1378 | 51 | 0.0005 |
| DepthPro2-S | ORT CUDA FP16 clip of 4 | 8.7 | 114.6 | 2352 | 53 | 0.0005 |
| DepthPro2-S | ORT CPU FP32 | 239.2 | 4.2 | - | 99 | 0.0000 |
| DepthPro2-S | ORT CPU INT8 | 253.7 | 3.9 | - | 38 | 0.0049 |
| DepthPro2-B | PyTorch BF16 (Apple default) | 11.3 | 88.6 | 1198 | - | 0.0022 |
| DepthPro2-B | PyTorch FP32 | 24.3 | 41.1 | 1112 | - | 0.0000 |
| DepthPro2-B | ORT CUDA FP32 | 21.8 | 45.9 | 1308 | 387 | 0.0002 |
| DepthPro2-B | ORT CUDA FP16 | 14.5 | 68.8 | 1318 | 199 | 0.0007 |
| DepthPro2-B | ORT CUDA INT8 | 22.7 | 44.0 | 1308 | 143 | 0.0030 |
| DepthPro2-B | ORT TensorRT FP16 | 3.4 | 295.9 | 514 | 199 | inf |
| DepthPro2-B | ORT CUDA FP16 stream, window 4 | 19.9 | 50.3 | 2480 | 199 | 0.0005 |
| DepthPro2-B | ORT CUDA FP16 clip of 4 | 17.1 | 58.6 | 5426 | 201 | 0.0006 |
