"""Run a single-frame TensorRT engine on the parity fixture and score it against PyTorch FP32.

    python3 jetson/check_engine.py onnx/model_fp16.engine --size S

Needs tensorrt, polygraphy and numpy, plus the fixture and references that
fixtures/make_fixture.py writes. Reports the same metrics as verify_parity.py, and latency.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from polygraphy.backend.trt import EngineFromBytes, TrtRunner

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from depthpro2_onnx.metrics import compare, merge  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    parser.add_argument("--size", required=True, choices=["S", "B"])
    parser.add_argument("--fixtures", type=Path, default=ROOT / "fixtures")
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    fixture = np.load(args.fixtures / "test_clip_480x640.npz")
    refs = np.load(args.fixtures / "ref_outputs" / f"DepthPro2-{args.size}_480x640.npz")
    f_px = np.array([fixture["f_px"]], dtype=np.float32)
    frames = {"clip": fixture["clip"], "photo": fixture["photo"]}

    with TrtRunner(EngineFromBytes(args.engine.read_bytes())) as runner:
        results = []
        for key, rgb in frames.items():
            for i, frame in enumerate(rgb):
                pixels = np.ascontiguousarray(frame.transpose(2, 0, 1)[None], np.float32) / 255
                depth = runner.infer({"pixel_values": pixels, "f_px": f_px})["depth_meters"]
                results.append(compare(np.asarray(depth), refs[f"fp32/single/{key}"][i : i + 1]))
        for _ in range(10):
            runner.infer({"pixel_values": pixels, "f_px": f_px})
        times = []
        for _ in range(args.iters):
            start = time.perf_counter()
            runner.infer({"pixel_values": pixels, "f_px": f_px})
            times.append((time.perf_counter() - start) * 1e3)
    report = merge(results) | {
        "engine": args.engine.name,
        "median_ms": float(np.median(times)),
        "p90_ms": float(np.percentile(times, 90)),
    }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
