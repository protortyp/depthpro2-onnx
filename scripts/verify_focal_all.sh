#!/usr/bin/env bash
# Parity of every focal and auto graph; results in results/parity/DepthPro2-focal/.
set -u
cd "$(dirname "$0")/.."
status=0
for graph in $(find build/DepthPro2-focal/onnx -name "*.onnx" | sort); do
  rel=${graph#build/DepthPro2-focal/onnx/}
  stem=$(echo "${rel%.onnx}" | tr '/' '_')
  providers="cuda"
  case "$rel" in *focal.onnx) providers="cuda cpu" ;; esac
  for provider in $providers; do
    out="results/parity/DepthPro2-focal/${stem}_${provider}.json"
    if uv run --no-sync python verify_focal.py "$graph" --provider "$provider" --json "$out" >/dev/null 2>&1; then
      verdict=ok
    else
      verdict=FAIL
      status=1
    fi
    python3 - "$out" "$verdict" "$rel" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); dv = d.get("depth_vs_fp32")
depth = f"depth AbsRel {dv['abs_rel']:.5f} delta1 {dv['delta1']:.5f}" if dv else ""
print(f"{sys.argv[2]:4} {sys.argv[3]:<26} {d['provider']:<22} f_px max rel {d['focal_max_rel']:.5f} {depth}")
PY
  done
done
exit $status
