#!/usr/bin/env bash
# Parity of every exported graph; results in results/parity/<model>/<graph>_<provider>.json.
set -u
cd "$(dirname "$0")/.."
status=0
for dir in build/DepthPro2-*/onnx; do
  model=$(basename "$(dirname "$dir")")
  for graph in "$dir"/*.onnx; do
    stem=$(basename "$graph" .onnx)
    providers="cuda"
    case "$stem" in model|model_int8) providers="cuda cpu" ;; esac
    for provider in $providers; do
      out="results/parity/$model/${stem}_${provider}.json"
      if uv run --no-sync python verify_parity.py "$graph" --provider "$provider" --json "$out" > /dev/null 2>&1; then
        verdict=ok
      else
        verdict=FAIL
        status=1
      fi
      python3 - "$out" "$verdict" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); v = d["vs_fp32"]
print(f"{sys.argv[2]:4} {d['graph']:<26} {d['provider']:<22} AbsRel {v['abs_rel']:.5f} "
      f"delta1 {v['delta1']:.5f} near-max {v['max_abs_near_m']:.3f} m")
PY
    done
  done
done
exit $status
