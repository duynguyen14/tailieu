from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from app.core.config import DevicePreference, ModelBackend, Settings


@dataclass(frozen=True, slots=True)
class RuntimeSelection:
    requested_device: str
    effective_device: str
    backend: str
    gpu_available: bool
    reason: str


def _onnx_cuda_available() -> bool:
    if importlib.util.find_spec("onnxruntime") is None:
        return False
    import onnxruntime as ort

    return "CUDAExecutionProvider" in ort.get_available_providers()


def _paddle_cuda_available() -> bool:
    if importlib.util.find_spec("paddle") is None:
        return False
    import paddle

    return bool(paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0)


def _tensorrt_available() -> bool:
    return importlib.util.find_spec("tensorrt") is not None


def _backend_gpu_available(backend: ModelBackend) -> bool:
    if backend == ModelBackend.ONNX:
        return _onnx_cuda_available()
    if backend == ModelBackend.PADDLE:
        return _paddle_cuda_available()
    if backend == ModelBackend.TENSORRT:
        return _tensorrt_available()
    return False


def resolve_runtime(settings: Settings) -> RuntimeSelection:
    backend = settings.model_backend
    requested = settings.device

    if backend == ModelBackend.MOCK:
        return RuntimeSelection(
            requested_device=requested.value,
            effective_device="cpu",
            backend=backend.value,
            gpu_available=False,
            reason="Mock backend performs no model computation.",
        )

    gpu_available = _backend_gpu_available(backend)
    if requested == DevicePreference.CPU:
        return RuntimeSelection(
            requested_device=requested.value,
            effective_device="cpu",
            backend=backend.value,
            gpu_available=gpu_available,
            reason="CPU explicitly selected.",
        )

    if gpu_available:
        return RuntimeSelection(
            requested_device=requested.value,
            effective_device=f"cuda:{settings.gpu_device_id}",
            backend=backend.value,
            gpu_available=True,
            reason="Compatible GPU provider is available.",
        )

    if settings.allow_cpu_fallback:
        return RuntimeSelection(
            requested_device=requested.value,
            effective_device="cpu",
            backend=backend.value,
            gpu_available=False,
            reason="GPU provider unavailable; CPU fallback enabled.",
        )

    raise RuntimeError(
        f"DEVICE={requested.value} requested for backend={backend.value}, "
        "but no compatible GPU provider is available and CPU fallback is disabled."
    )
