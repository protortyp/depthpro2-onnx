"""Latency, throughput, memory and parity of the graphs and the PyTorch reference.

    uv run --extra export --group bench python benchmark.py --sizes S B

Each configuration runs in its own process so that its peak GPU memory, read from NVML for that
process, includes everything it allocates (CUDA context, workspace, weights, activations).
Latency is wall time from a host RGB array to host depth, one frame at a time, after warmup.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


class PeakGpuMemory:
    """Polls NVML for this process's GPU memory until stopped; ``peak_mb`` is the maximum."""

    def __init__(self, interval: float = 0.005):
        import pynvml

        pynvml.nvmlInit()
        self.nvml = pynvml
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.pid = os.getpid()
        self.peak = 0
        self.interval = interval
        self.running = True
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def _poll(self) -> None:
        while self.running:
            for proc in self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle):
                if proc.pid == self.pid and proc.usedGpuMemory:
                    self.peak = max(self.peak, proc.usedGpuMemory)
            time.sleep(self.interval)

    def stop(self) -> float:
        self.running = False
        self.thread.join()
        return self.peak / 2**20


def time_runs(step, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        step()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        step()
        times.append((time.perf_counter() - start) * 1e3)
    return times


def bench_onnx(config: dict, frame: np.ndarray, f_px: float) -> dict:
    from depthpro2_onnx.runtime import load, to_pixel_values

    memory = PeakGpuMemory() if config["provider"] != "cpu" else None
    graph = load(
        config["path"],
        config["provider"],
        **({"threads": config["threads"]} if config.get("threads") else {}),
    )
    providers = [graph.provider]
    frames = graph.frames
    pixels = to_pixel_values(np.repeat(frame[None], frames, axis=0))
    times = time_runs(lambda: graph.infer(pixels, f_px), config["warmup"], config["iters"])
    return {
        "times_ms": times,
        "frames_per_call": frames,
        "peak_vram_mb": memory.stop() if memory else None,
        "providers": providers,
    }


def bench_torch(config: dict, frame: np.ndarray, f_px: float) -> dict:
    import torch

    from depthpro2_onnx.model import load_model

    memory = PeakGpuMemory()
    precision = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    model = load_model(config["size"], "cuda", precision[config["precision"]])
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(3, 1, 1)

    def step():
        x = torch.from_numpy(frame).cuda().permute(2, 0, 1).float().div(255)
        depth = model.infer((x - mean) / std, f_px=f_px)["depth"]
        return depth.cpu().numpy()

    times = time_runs(step, config["warmup"], config["iters"])
    return {
        "times_ms": times,
        "frames_per_call": 1,
        "peak_vram_mb": memory.stop(),
        "torch_peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
        "providers": [f"torch {torch.__version__} eager"],
    }


def run_one(config: dict) -> dict:
    fixture = np.load(ROOT / "fixtures" / "test_clip_480x640.npz")
    frame, f_px = fixture["photo"][0], float(fixture["f_px"])
    if config["runtime"] == "torch":
        return bench_torch(config, frame, f_px)
    return bench_onnx(config, frame, f_px)


def parity(path: Path, provider: str) -> dict | None:
    out = RESULTS / "parity" / path.parent.parent.name / f"{path.stem}_{provider}.json"
    if not out.exists():
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "verify_parity.py"),
                str(path),
                "--provider",
                provider,
                "--json",
                str(out),
            ],
            check=False,
            capture_output=True,
        )
    return json.loads(out.read_text()) if out.exists() else None


def configs(sizes: list[str], build: Path, warmup: int, iters: int, cpu_iters: int) -> list[dict]:
    out = []
    for size in sizes:
        onnx = build / f"DepthPro2-{size}" / "onnx"
        base = {"size": size, "warmup": warmup, "iters": iters}
        out.append(
            base | {"name": "PyTorch BF16 (Apple default)", "runtime": "torch", "precision": "bf16"}
        )
        out.append(base | {"name": "PyTorch FP32", "runtime": "torch", "precision": "fp32"})
        for provider, label in (("cuda", "ORT CUDA"), ("tensorrt", "ORT TensorRT")):
            for stem, precision in (
                ("model", "FP32"),
                ("model_fp16", "FP16"),
                ("model_int8", "INT8"),
            ):
                if provider == "tensorrt" and stem != "model_fp16":
                    continue
                out.append(
                    base
                    | {
                        "name": f"{label} {precision}",
                        "runtime": "onnx",
                        "provider": provider,
                        "path": str(onnx / f"{stem}.onnx"),
                    }
                )
        for stem, label in (
            ("model_stream4_fp16", "ORT CUDA FP16 stream, window 4"),
            ("model_clip4_fp16", "ORT CUDA FP16 clip of 4"),
        ):
            out.append(
                base
                | {
                    "name": label,
                    "runtime": "onnx",
                    "provider": "cuda",
                    "path": str(onnx / f"{stem}.onnx"),
                }
            )
        if size == "S":
            for stem, precision in (("model", "FP32"), ("model_int8", "INT8")):
                out.append(
                    base
                    | {
                        "name": f"ORT CPU {precision}",
                        "runtime": "onnx",
                        "provider": "cpu",
                        "path": str(onnx / f"{stem}.onnx"),
                        "iters": cpu_iters,
                        "warmup": 2,
                    }
                )
    return out


def environment() -> dict:
    import onnxruntime
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    others = [
        p.usedGpuMemory / 2**20
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        if p.pid != os.getpid() and p.usedGpuMemory
    ]
    cpu = next(
        (line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if "model name" in line),
        platform.processor(),
    )
    return {
        "gpu": pynvml.nvmlDeviceGetName(handle),
        "driver": pynvml.nvmlSystemGetDriverVersion(),
        "cpu": cpu,
        "cpu_threads": os.cpu_count(),
        "onnxruntime": onnxruntime.__version__,
        "other_gpu_processes_mb": others,
        "resolution": "480x640",
    }


def table(rows: list[dict]) -> str:
    lines = [
        "| Model | Runtime | Latency (ms/frame) | FPS | Peak VRAM (MB) | Disk (MB) "
        "| AbsRel vs PyTorch FP32 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        vram = f"{r['peak_vram_mb']:.0f}" if r.get("peak_vram_mb") else "-"
        disk = f"{r['disk_mb']:.0f}" if r.get("disk_mb") else "-"
        rel = f"{r['abs_rel']:.4f}" if r.get("abs_rel") is not None else "-"
        lines.append(
            f"| DepthPro2-{r['size']} | {r['name']} | {r['ms_per_frame']:.1f} | {r['fps']:.1f} "
            f"| {vram} | {disk} | {rel} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="*", default=["S", "B"])
    parser.add_argument("--build", type=Path, default=ROOT / "build")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--cpu-iters", type=int, default=10)
    parser.add_argument("--only", nargs="*", help="run configurations whose name contains these")
    parser.add_argument("--one", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.one:
        print("RESULT " + json.dumps(run_one(json.loads(args.one))))
        return

    env = environment()
    rows = []
    for config in configs(args.sizes, args.build, args.warmup, args.iters, args.cpu_iters):
        if args.only and not any(o in config["name"] for o in args.only):
            continue
        if config["runtime"] == "onnx" and not Path(config["path"]).exists():
            print(f"skip {config['name']}: {config['path']} missing", flush=True)
            continue
        proc = subprocess.run(
            [sys.executable, __file__, "--one", json.dumps(config)],
            capture_output=True,
            text=True,
        )
        line = next((x for x in proc.stdout.splitlines() if x.startswith("RESULT ")), None)
        if line is None:
            print(f"FAILED {config['name']}:\n{proc.stderr[-3000:]}", flush=True)
            continue
        result = json.loads(line[len("RESULT ") :])
        times = np.array(result.pop("times_ms"))
        per_frame = float(np.median(times)) / result["frames_per_call"]
        row = (
            config
            | result
            | {
                "ms_per_frame": per_frame,
                "p90_ms_per_call": float(np.percentile(times, 90)),
                "fps": 1e3 / per_frame,
            }
        )
        if config["runtime"] == "onnx":
            path = Path(config["path"])
            row["disk_mb"] = path.stat().st_size / 1e6
            report = parity(path, config["provider"], config["size"])
            row["abs_rel"] = report["vs_fp32"]["abs_rel"] if report else None
        else:
            ref = np.load(
                ROOT / "fixtures" / "ref_outputs" / f"DepthPro2-{config['size']}_480x640.npz"
            )
            if config["precision"] == "bf16":
                from depthpro2_onnx.metrics import compare

                row["abs_rel"] = compare(ref["bf16/single/photo"], ref["fp32/single/photo"])[
                    "abs_rel"
                ]
            else:
                row["abs_rel"] = 0.0
        rows.append(row)
        print(
            f"{row['name']:<40} {config['size']} {per_frame:7.2f} ms/frame "
            f"{row['fps']:7.1f} fps  vram {row.get('peak_vram_mb') or 0:6.0f} MB",
            flush=True,
        )

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "benchmark.json").write_text(
        json.dumps({"environment": env, "rows": rows}, indent=1)
    )
    markdown = table(rows)
    (RESULTS / "benchmark.md").write_text(markdown + "\n")
    print(json.dumps(env, indent=1))
    print(markdown)


if __name__ == "__main__":
    main()
