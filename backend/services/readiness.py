"""System readiness checks."""

from __future__ import annotations

from pathlib import Path

from tracks.instructor.core.config import (
    INSTRUCTOR_ONNX_CONFIG,
    INSTRUCTOR_ONNX_DENSE,
    INSTRUCTOR_ONNX_ENCODER,
    INSTRUCTOR_ONNX_TOKENIZER,
)
from tracks.shared.paths import CROSS_ENCODER_DIR, ROOT_DIR

from backend.models.health import ReadyResponse


def check_readiness() -> ReadyResponse:
    checks: dict[str, bool] = {}
    messages: dict[str, str] = {}

    instructor_files = [
        ("encoder", INSTRUCTOR_ONNX_ENCODER),
        ("tokenizer", INSTRUCTOR_ONNX_TOKENIZER),
        ("dense", INSTRUCTOR_ONNX_DENSE),
        ("config", INSTRUCTOR_ONNX_CONFIG),
    ]
    instructor_ok = True
    for name, path in instructor_files:
        ok = path.exists() and (path.is_file() or path.is_dir())
        if not ok:
            instructor_ok = False
            messages[f"instructor_{name}"] = f"Missing: {path}"
    checks["instructor_onnx"] = instructor_ok

    ce_model = CROSS_ENCODER_DIR / "model.onnx"
    ce_tokenizer = CROSS_ENCODER_DIR / "tokenizer"
    ce_ok = ce_model.exists() and ce_tokenizer.is_dir()
    checks["cross_encoder_onnx"] = ce_ok
    if not ce_ok:
        messages["cross_encoder"] = (
            f"Missing cross-encoder artifacts under {CROSS_ENCODER_DIR}. "
            "Run: python tracks/instructor/stage0/run_cross_encoder.py"
        )

    config_ok = (ROOT_DIR / "config.yaml").exists()
    checks["config_yaml"] = config_ok
    if not config_ok:
        messages["config"] = "Missing config.yaml at repo root"

    cuda_ok = _check_cuda_available()
    checks["cuda_available"] = cuda_ok
    if not cuda_ok:
        messages["cuda"] = "CUDA provider not available (required for pool indexing)"

    ready = instructor_ok and ce_ok and config_ok
    return ReadyResponse(ready=ready, checks=checks, messages=messages)


def _check_cuda_available() -> bool:
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        return "CUDAExecutionProvider" in providers
    except Exception:
        return False
