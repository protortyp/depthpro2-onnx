"""Runners for the exported graphs: ONNX Runtime (numpy + onnxruntime) or a TensorRT engine."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

PROVIDERS = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
}


def load_pip_tensorrt() -> None:
    """Load the libraries of the ``tensorrt-cu12`` wheel if it is installed.

    JetPack's system TensorRT is on the default library path and needs none of this.
    """
    try:
        import tensorrt_libs
    except ImportError:
        return
    folder = Path(tensorrt_libs.__file__).parent
    for name in ("libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"):
        if (folder / name).exists():
            ctypes.CDLL(str(folder / name), mode=ctypes.RTLD_GLOBAL)


def providers_for(name: str) -> list:
    if name not in PROVIDERS:
        raise ValueError(f"provider must be one of {', '.join(PROVIDERS)}, got {name!r}")
    if name == "cuda" and hasattr(ort, "preload_dlls"):
        # Finds the CUDA and cuDNN libraries of pip's nvidia-* wheels when torch, which would
        # otherwise load them, is not imported.
        ort.preload_dlls()
    return list(PROVIDERS[name])


def load(path: str | Path, provider: str = "cuda", **kwargs):
    """A ``TensorRTEngine`` for ``.engine`` files, else a ``DepthPro2ONNX`` on ``provider``."""
    if str(path).endswith(".engine") or provider == "tensorrt":
        if not str(path).endswith(".engine"):
            raise ValueError(
                "provider tensorrt takes an engine built by jetson/build_tensorrt.sh, not "
                f"{path}; ONNX Runtime's TensorRT provider gives NaN depth on these graphs"
            )
        return TensorRTEngine(path)
    return DepthPro2ONNX(path, provider, **kwargs)


def to_pixel_values(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 ``(H, W, 3)`` or ``(T, H, W, 3)`` to float32 ``(T, 3, H, W)`` in [0, 1]."""
    if rgb.dtype != np.uint8 or rgb.shape[-1] != 3:
        raise TypeError(f"expected RGB uint8 (..., H, W, 3), got {rgb.dtype} {rgb.shape}")
    frames = rgb[None] if rgb.ndim == 3 else rgb
    return np.ascontiguousarray(frames.transpose(0, 3, 1, 2), dtype=np.float32) / 255.0


class DepthPro2ONNX:
    """Runs a single-frame, clip or stream graph.

    ``infer`` takes frames ``(T, 3, H, W)`` (``T`` fixed by the graph) and returns depth
    ``(T, H, W)`` in meters; on a stream graph it steps through the frames in order, carrying the
    state across calls until ``reset``.
    """

    def __init__(
        self,
        path: str | Path,
        provider: str = "cuda",
        threads: int | None = None,
        allow_fallback: bool = False,
    ):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), options, providers=providers_for(provider))
        active, wanted = self.session.get_providers()[0], PROVIDERS[provider][0]
        if active != wanted and not allow_fallback:
            # ONNX Runtime falls back to the next provider with only a log line; a robot
            # silently running on the CPU at a tenth of the frame rate is worse than an error.
            raise RuntimeError(
                f"{wanted} is unavailable (session runs on {active}); install its libraries "
                "or pass allow_fallback=True"
            )
        self.provider = active
        meta = self.session.get_modelmeta().custom_metadata_map
        self.config = json.loads(meta.get("depthpro2_onnx", "{}"))
        inputs = {i.name: i for i in self.session.get_inputs()}
        self.frames, _, self.height, self.width = inputs["pixel_values"].shape
        self.state_inputs = [n for n in inputs if n.startswith("past_")]
        self.state_shapes = {n: inputs[n].shape for n in self.state_inputs}
        self.state_dtype = (
            np.float16
            if self.state_inputs and inputs[self.state_inputs[0]].type == "tensor(float16)"
            else np.float32
        )
        self.on_gpu = self.session.get_providers()[0] != "CPUExecutionProvider"
        self.reset()

    @property
    def is_stream(self) -> bool:
        return bool(self.state_inputs)

    def reset(self) -> None:
        """Forget the frames a stream has seen; start of a new video or camera."""
        self.state = {
            name: self._ortvalue(np.zeros(shape, dtype=self.state_dtype))
            for name, shape in self.state_shapes.items()
        }

    def _ortvalue(self, array: np.ndarray) -> ort.OrtValue:
        device = "cuda" if self.on_gpu else "cpu"
        return ort.OrtValue.ortvalue_from_numpy(array, device, 0)

    def _run(self, pixel_values: np.ndarray, f_px: float) -> np.ndarray:
        binding = self.session.io_binding()
        binding.bind_ortvalue_input("pixel_values", self._ortvalue(pixel_values))
        binding.bind_cpu_input("f_px", np.array([f_px], dtype=np.float32))
        for name, value in self.state.items():
            binding.bind_ortvalue_input(name, value)
        binding.bind_output("depth_meters", "cpu")
        presents = [name.replace("past_", "present_", 1) for name in self.state]
        for name in presents:
            binding.bind_output(name, "cuda" if self.on_gpu else "cpu")
        self.session.run_with_iobinding(binding)
        outputs = binding.get_outputs()
        self.state = dict(zip(self.state, outputs[1:], strict=True))
        return outputs[0].numpy()

    def infer(self, pixel_values: np.ndarray, f_px: float) -> np.ndarray:
        pixel_values = np.ascontiguousarray(pixel_values, dtype=np.float32)
        if pixel_values.shape[1:] != (3, self.height, self.width):
            raise ValueError(
                f"graph takes (T, 3, {self.height}, {self.width}), got {pixel_values.shape}"
            )
        if self.is_stream:
            return np.concatenate([self._run(f[None], f_px) for f in pixel_values])
        if pixel_values.shape[0] != self.frames:
            raise ValueError(f"graph takes {self.frames} frames, got {pixel_values.shape[0]}")
        return self._run(pixel_values, f_px)


class TensorRTEngine:
    """Runs an engine from ``jetson/build_tensorrt.sh`` with the interface of ``DepthPro2ONNX``.

    Needs ``tensorrt`` and ``polygraphy``. Stream state makes a round trip through host memory
    each frame; on Jetson, whose GPU shares the host's memory, that copy is cheap.
    """

    def __init__(self, path: str | Path):
        load_pip_tensorrt()
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner

        self.runner = TrtRunner(EngineFromBytes(Path(path).read_bytes()))
        self.runner.activate()
        inputs = {
            name: (np.dtype(dtype), tuple(shape))
            for name, (dtype, shape) in self.runner.get_input_metadata(
                use_numpy_dtypes=True
            ).items()
        }
        self.frames, _, self.height, self.width = inputs["pixel_values"][1]
        self.state_inputs = [n for n in inputs if n.startswith("past_")]
        self.state_specs = {n: inputs[n] for n in self.state_inputs}
        precision = "fp16" if "fp16" in Path(path).stem else "fp32"
        mode = "stream" if self.state_inputs else ("single" if self.frames == 1 else "clip")
        self.config = {"mode": mode, "precision": precision, "frames": self.frames}
        if self.state_inputs:
            self.config["window"] = self.state_specs["past_filled"][1][0]
        self.provider = "TensorRT engine"
        self.reset()

    @property
    def is_stream(self) -> bool:
        return bool(self.state_inputs)

    def reset(self) -> None:
        self.state = {n: np.zeros(shape, dtype) for n, (dtype, shape) in self.state_specs.items()}

    def _run(self, pixel_values: np.ndarray, f_px: float) -> np.ndarray:
        feed = {"pixel_values": pixel_values, "f_px": np.array([f_px], np.float32)} | self.state
        out = self.runner.infer(feed)
        self.state = {n: np.array(out[n.replace("past_", "present_", 1)]) for n in self.state}
        return np.array(out["depth_meters"])

    def infer(self, pixel_values: np.ndarray, f_px: float) -> np.ndarray:
        pixel_values = np.ascontiguousarray(pixel_values, dtype=np.float32)
        if self.is_stream:
            return np.concatenate([self._run(f[None], f_px) for f in pixel_values])
        return self._run(pixel_values, f_px)
