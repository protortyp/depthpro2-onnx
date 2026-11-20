#!/usr/bin/env bash
# Build the package in ros:jazzy and run the end-to-end check for several graphs.
#   ros2/test/run_in_docker.sh build/DepthPro2-S/onnx S
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
models=$(cd "${1:?models dir}" && pwd)
docker run --rm --device nvidia.com/gpu=all -v "$repo":/repo:ro -v "$models":/models:ro \
  -e SIZE="${2:-S}" ros:jazzy-ros-base bash /repo/ros2/test/container_check.sh
