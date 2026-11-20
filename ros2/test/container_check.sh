#!/usr/bin/env bash
# Runs inside ros:jazzy-ros-base; see run_in_docker.sh.
set -eo pipefail
apt-get update -qq >/dev/null
apt-get install -y -qq python3-pip python3-opencv python3-colcon-common-extensions >/dev/null 2>&1
pip install -q --break-system-packages "onnxruntime-gpu[cuda,cudnn]==1.23.2" >/dev/null 2>&1
mkdir -p /ws/src && cp -rL /repo/ros2 /ws/src/depthpro2_ros
cd /ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select depthpro2_ros >/dev/null
source install/setup.bash
node=/ws/install/depthpro2_ros/lib/depthpro2_ros/depthpro2_onnx_node
status=0
for cfg in ${CONFIGS:-"cpu model_int8 0" "cuda model_fp16 0" "cuda model_stream4_fp16 4"}; do
  read -r provider graph window <<< "$cfg"
  echo "== $provider $graph"
  "$node" --ros-args -p model_path:="/models/$graph.onnx" -p provider:="$provider" \
    -p publish_pointcloud:=true -p log_timing_every:=16 > /tmp/node.log 2>&1 &
  pid=$!
  python3 /repo/ros2/test/fake_camera_check.py /repo/fixtures/test_clip_480x640.npz \
    "/repo/fixtures/ref_outputs/DepthPro2-${SIZE}_480x640.npz" --window "$window" || status=1
  kill -INT "$pid"
  wait "$pid" 2>/dev/null || true
  grep -E "ms/frame|ERROR" /tmp/node.log || true
done
exit $status
