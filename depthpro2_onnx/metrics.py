"""Depth agreement between a prediction and a reference, both in meters."""

from __future__ import annotations

import numpy as np

# Sky and other far points come out at kilometers or infinity and swing by large ratios between
# precisions; robots act within this range, so metrics count reference pixels up to it.
MAX_DEPTH_M = 200.0
# Manipulation and obstacle avoidance mostly happen within this range.
NEAR_DEPTH_M = 10.0


def compare(pred: np.ndarray, ref: np.ndarray, max_depth: float = MAX_DEPTH_M) -> dict[str, float]:
    pred = pred.astype(np.float64)
    ref = ref.astype(np.float64)
    if pred.shape != ref.shape:
        raise ValueError(f"shape {pred.shape} != reference {ref.shape}")
    valid = np.isfinite(ref) & (ref > 0) & (ref <= max_depth)
    p, r = pred[valid], ref[valid]
    finite = np.isfinite(p) & (p > 0)
    # A non-finite or non-positive prediction where the reference is in range counts as a miss.
    ratio = np.where(finite, np.maximum(p / r, r / np.where(finite, p, 1.0)), np.inf)
    rel = np.where(finite, np.abs(p - r) / r, np.inf)
    pf = np.where(finite, p, 0.0)
    cosine = float(pf @ r / (np.linalg.norm(pf) * np.linalg.norm(r)))
    near = r <= NEAR_DEPTH_M
    return {
        "max_abs_m": float(np.max(np.abs(pf - r))),
        "max_abs_near_m": float(np.max(np.abs(pf - r)[near], initial=0.0)),
        "abs_rel": float(np.mean(rel)),
        "delta1": float(np.mean(ratio < 1.25)),
        "cosine": cosine,
        "valid_px": int(valid.sum()),
        "nonfinite_px": int((~finite).sum()),
    }


def merge(results: list[dict[str, float]]) -> dict[str, float]:
    """Pixel-weighted mean of AbsRel and delta1, worst case of the rest."""
    weights = np.array([r["valid_px"] for r in results], dtype=np.float64)
    out = {
        "abs_rel": float(np.average([r["abs_rel"] for r in results], weights=weights)),
        "delta1": float(np.average([r["delta1"] for r in results], weights=weights)),
        "max_abs_m": max(r["max_abs_m"] for r in results),
        "max_abs_near_m": max(r["max_abs_near_m"] for r in results),
        "cosine": min(r["cosine"] for r in results),
        "valid_px": int(weights.sum()),
        "nonfinite_px": sum(r["nonfinite_px"] for r in results),
    }
    return out
