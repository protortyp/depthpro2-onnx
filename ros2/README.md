# depthpro2_ros

Metric depth from a calibrated camera. `f_px` comes from `CameraInfo.k[0]`, so the focal-length network never runs.

| Topic | Type | |
| --- | --- | --- |
| `image_topic` (`/camera/image_raw`) | `sensor_msgs/Image` in | `rgb8`, `bgr8`, `rgba8`, `bgra8`, `mono8`; rectified |
| `camera_info_topic` (`/camera/camera_info`) | `sensor_msgs/CameraInfo` in | |
| `~/depth` | `sensor_msgs/Image` out | `32FC1`, meters, same stamp and frame |
| `~/camera_info` | `sensor_msgs/CameraInfo` out | intrinsics of `~/depth` |
| `~/points` | `sensor_msgs/PointCloud2` out | `publish_pointcloud:=true` |

Frames are scaled to cover the graph's input (480x640 or 384x672) and center-cropped. `~/camera_info` has the matching intrinsics.

## Install

Tested on ROS 2 Jazzy (Ubuntu 24.04, Python 3.12). The package symlinks `../depthpro2_onnx`, so clone the whole repository into the workspace.

```sh
cd ~/ros2_ws/src && git clone https://github.com/protortyp/depthpro2-onnx
pip install --break-system-packages "onnxruntime-gpu[cuda,cudnn]" huggingface_hub   # x86 + NVIDIA
# Jetson: onnxruntime-gpu built for your JetPack (Jetson AI Lab pip index)
cd ~/ros2_ws && colcon build --packages-select depthpro2_ros && source install/setup.bash
```

## Run

```sh
ros2 launch depthpro2_ros depthpro2.launch.py \
  image_topic:=/camera/color/image_rect camera_info_topic:=/camera/color/camera_info \
  model_repo:=christianengel/DepthPro2-S-ONNX model_file:=onnx/model_stream4_fp16.onnx \
  publish_pointcloud:=true
```

| Parameter | Default | |
| --- | --- | --- |
| `model_repo`, `model_file`, `model_revision` | `christianengel/DepthPro2-S-ONNX`, `onnx/model_fp16.onnx`, `main` | downloaded once |
| `model_path` | | local `.onnx`; overrides the above |
| `provider` | `cuda` | `cuda` or `cpu` (ONNX Runtime), `tensorrt` (engine); never falls back |
| `publish_pointcloud`, `pointcloud_stride`, `max_depth` | `false`, `2`, `20.0` | |
| `log_timing_every` | `300` | frames between timing logs; `0` off |

Stream graphs reset when timestamps go backwards or the camera resolution changes.

TensorRT (Jetson): build an engine on the device with [`jetson/build_tensorrt.sh`](../jetson/build_tensorrt.sh), then `provider:=tensorrt model_path:=/path/model_fp16.engine`. Needs JetPack's `python3-libnvinfer` and `pip install polygraphy`.

## Test

`test/run_in_docker.sh build/DepthPro2-S/onnx S` builds the package in `ros:jazzy-ros-base`. It publishes the parity fixture at 1280x960 `bgr8` and compares the depth with PyTorch: per frame for single graphs, against Apple's stream for stream graphs.
