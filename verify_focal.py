"""Check focal-length and auto-depth graphs against PyTorch FP32 on the fixture.

    uv run python verify_focal.py build/DepthPro2-focal/onnx/auto_s_fp16.onnx --provider cuda

Gates, fixed: every frame's f_px within 1% of PyTorch (a 1% focal error is a 1% depth error);
auto graphs must also pass the depth gate of verify_parity.py (AbsRel < 0.01, delta1 > 0.995).
Exit status 1 means the graph failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from depthpro2_onnx.metrics import compare, merge
from depthpro2_onnx.runtime import providers_for, to_pixel_values
from verify_parity import GATE_ABS_REL, GATE_DELTA1

ROOT = Path(__file__).resolve().parent
GATE_FOCAL_REL = 0.01


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", type=Path)
    parser.add_argument("--provider", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    session = ort.InferenceSession(str(args.onnx), providers=providers_for(args.provider))
    config = json.loads(session.get_modelmeta().custom_metadata_map["depthpro2_onnx"])
    tag = f"{config['height']}x{config['width']}"
    fixture = np.load(ROOT / "fixtures" / f"test_clip_{tag}.npz")
    refs = np.load(ROOT / "fixtures" / "ref_outputs" / f"focal_{tag}.npz")
    size = config["base_model"].rsplit("-", 1)[-1] if config["mode"] == "auto" else None

    focal_rel, depth_results = [], []
    for key in ("clip", "photo"):
        frames = to_pixel_values(fixture[key])
        for i, frame in enumerate(frames):
            outputs = dict(
                zip(
                    [o.name for o in session.get_outputs()],
                    session.run(None, {"pixel_values": frame[None]}),
                    strict=True,
                )
            )
            ref_f = refs[f"f_px/{key}"][i]
            focal_rel.append(abs(float(outputs["f_px"][0]) / float(ref_f) - 1))
            if size:
                depth_results.append(
                    compare(outputs["depth_meters"], refs[f"{size}/{key}"][i : i + 1])
                )

    report = {
        "graph": args.onnx.name,
        "provider": session.get_providers()[0],
        "config": config,
        "focal_max_rel": float(max(focal_rel)),
        "focal_mean_rel": float(np.mean(focal_rel)),
    }
    passed = report["focal_max_rel"] < GATE_FOCAL_REL
    if size:
        report["depth_vs_fp32"] = merge(depth_results)
        passed &= report["depth_vs_fp32"]["abs_rel"] < GATE_ABS_REL
        passed &= report["depth_vs_fp32"]["delta1"] > GATE_DELTA1
    report["gate"] = {
        "focal_rel_lt": GATE_FOCAL_REL,
        "abs_rel_lt": GATE_ABS_REL,
        "delta1_gt": GATE_DELTA1,
        "passed": bool(passed),
    }
    print(json.dumps(report, indent=1))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
