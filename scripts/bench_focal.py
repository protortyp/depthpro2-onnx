"""Latency and memory of the focal and auto-depth graphs; writes results/focal_benchmark.json.

uv run --group bench python scripts/bench_focal.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark import PeakGpuMemory, time_runs  # noqa: E402

CONFIGS = [
    ("focal", "ORT CUDA FP16", "cuda", "build/DepthPro2-focal/onnx/focal_fp16.onnx"),
    ("focal", "ORT CPU FP32", "cpu", "build/DepthPro2-focal/onnx/focal.onnx"),
    ("focal", "TensorRT FP16 engine", "tensorrt", "build/trt/F_focal_fp16.engine"),
    ("auto S", "ORT CUDA FP16", "cuda", "build/DepthPro2-focal/onnx/auto_s_fp16.onnx"),
    ("auto S", "TensorRT FP16 engine", "tensorrt", "build/trt/F_auto_s_fp16.engine"),
    ("auto B", "ORT CUDA FP16", "cuda", "build/DepthPro2-focal/onnx/auto_b_fp16.onnx"),
    ("auto B", "TensorRT FP16 engine", "tensorrt", "build/trt/F_auto_b_fp16.engine"),
]


def run_one(provider: str, path: str) -> dict:
    fixture = np.load(ROOT / "fixtures" / "test_clip_480x640.npz")
    pixels = np.ascontiguousarray(fixture["photo"].transpose(0, 3, 1, 2), np.float32) / 255
    memory = PeakGpuMemory() if provider != "cpu" else None
    if provider == "tensorrt":
        from depthpro2_onnx.runtime import load_pip_tensorrt

        load_pip_tensorrt()
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner

        runner = TrtRunner(EngineFromBytes((ROOT / path).read_bytes()))
        runner.activate()
        step = lambda: runner.infer({"pixel_values": pixels})  # noqa: E731
    else:
        import onnxruntime as ort

        from depthpro2_onnx.runtime import providers_for

        session = ort.InferenceSession(str(ROOT / path), providers=providers_for(provider))
        step = lambda: session.run(None, {"pixel_values": pixels})  # noqa: E731
    iters = 10 if provider == "cpu" else 100
    times = time_runs(step, 3 if provider == "cpu" else 20, iters)
    return {"ms": float(np.median(times)), "peak_vram_mb": memory.stop() if memory else None}


def main() -> None:
    if len(sys.argv) == 3:
        print("RESULT " + json.dumps(run_one(sys.argv[1], sys.argv[2])))
        return
    rows = []
    for graph, runtime, provider, path in CONFIGS:
        proc = subprocess.run(
            [sys.executable, __file__, provider, path], capture_output=True, text=True
        )
        line = next((x for x in proc.stdout.splitlines() if x.startswith("RESULT ")), None)
        if line is None:
            print(f"FAILED {graph} {runtime}\n{proc.stderr[-2000:]}")
            continue
        row = {"graph": graph, "runtime": runtime, "disk_mb": (ROOT / path).stat().st_size / 1e6}
        rows.append(row | json.loads(line[7:]))
        vram = rows[-1]["peak_vram_mb"] or 0
        print(f"{graph:<7} {runtime:<22} {rows[-1]['ms']:7.2f} ms  vram {vram:5.0f} MB")
    (ROOT / "results" / "focal_benchmark.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
