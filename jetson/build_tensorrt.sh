#!/usr/bin/env bash
# Build a TensorRT engine from a Depth Pro 2 ONNX graph on the device that will run it.
#
#   jetson/build_tensorrt.sh onnx/model_fp16.onnx                 # writes onnx/model_fp16.engine
#   jetson/build_tensorrt.sh onnx/model_fp16.onnx out.engine
#
# Engines are specific to the GPU, TensorRT version and JetPack, so build on the target
# (Jetson Orin Nano / NX / AGX, Thor; JetPack 6.x or 7.x) and do not copy engines between them.
#
# The FP16 graph keeps the first transformer block's attention in FP32: its logits exceed 2e5
# and overflow FP16. --stronglyTyped makes TensorRT follow the graph's precisions. Building the
# FP32 graph with --fp16 instead lets TensorRT pick FP16 for that block. The graphs have static
# shapes, so no optimization profile is needed.
set -euo pipefail

onnx=${1:?usage: build_tensorrt.sh MODEL.onnx [OUT.engine]}
engine=${2:-${onnx%.onnx}.engine}
trtexec=${TRTEXEC:-$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)}
workspace_mib=${WORKSPACE_MIB:-2048}

if [[ ! -x $trtexec ]]; then
  echo "trtexec not found; set TRTEXEC=/path/to/trtexec (JetPack: /usr/src/tensorrt/bin)" >&2
  exit 1
fi

case "$onnx" in
  *_fp16.onnx) precision=(--stronglyTyped) ;;
  *_int8.onnx)
    echo "INT8 graphs use ONNX Runtime's MatMulNBits, which TensorRT cannot parse;" \
      "build the FP16 graph instead." >&2
    exit 1
    ;;
  *) precision=() ;;
esac

timing_cache=${engine%.engine}.timing
"$trtexec" \
  --onnx="$onnx" \
  --saveEngine="$engine" \
  "${precision[@]}" \
  --memPoolSize=workspace:"${workspace_mib}"M \
  --timingCacheFile="$timing_cache" \
  --builderOptimizationLevel=3 \
  --skipInference
echo "wrote $engine"
