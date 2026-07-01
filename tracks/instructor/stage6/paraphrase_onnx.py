"""CPU ONNX paraphraser — thread-local sessions, one decode per call."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput

from tracks.instructor.stage6.config import Stage6Config


@dataclass(frozen=True)
class ParaphraserPaths:
    encoder_onnx: Path
    tokenizer_dir: Path
    pytorch_dir: Path
    manifest: dict


def _load_paths(paraphraser_dir: Path) -> ParaphraserPaths:
    manifest_path = paraphraser_dir / "manifest.json"
    encoder_onnx = paraphraser_dir / "encoder.onnx"
    tokenizer_dir = paraphraser_dir / "tokenizer"
    pytorch_dir = paraphraser_dir / "pytorch"

    missing = [
        p for p in (manifest_path, encoder_onnx, tokenizer_dir, pytorch_dir) if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Paraphraser artifacts missing. Run:\n"
            "  python tracks/instructor/stage0/run_paraphraser_export.py\n"
            f"Missing: {[str(p) for p in missing]}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return ParaphraserPaths(
        encoder_onnx=encoder_onnx,
        tokenizer_dir=tokenizer_dir,
        pytorch_dir=pytorch_dir,
        manifest=manifest,
    )


class ParaphraseSession:
    """One ORT encoder session + CPU PyTorch decoder per worker thread."""

    def __init__(self, config: Stage6Config, paths: ParaphraserPaths) -> None:
        self._config = config
        self._manifest = paths.manifest
        self._input_prefix = config.input_prefix

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = config.ort_intra_op_threads
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._encoder = ort.InferenceSession(
            str(paths.encoder_onnx),
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = T5Tokenizer.from_pretrained(str(paths.tokenizer_dir))
        self._model = _load_decoder_model(paths.pytorch_dir)
        torch.set_num_threads(max(1, config.ort_intra_op_threads))

    def paraphrase_once(self, text: str, temperature: float) -> str:
        prompt = f"{self._input_prefix}{text}"
        encoded = self._tokenizer(
            prompt,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=self._config.max_length,
        )
        feeds = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
        hidden = self._encoder.run(None, feeds)[0]
        attention_mask = torch.from_numpy(encoded["attention_mask"].astype(np.int64))
        encoder_outputs = BaseModelOutput(
            last_hidden_state=torch.from_numpy(hidden).float(),
        )
        with torch.inference_mode():
            output_ids = self._model.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                max_new_tokens=self._config.max_length,
                do_sample=True,
                temperature=temperature,
                top_p=self._config.top_p,
                repetition_penalty=self._config.repetition_penalty,
            )
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)


_thread_local = threading.local()
_session_config: Stage6Config | None = None
_session_paths: ParaphraserPaths | None = None
_model_load_lock = threading.Lock()


def _load_decoder_model(pytorch_dir: Path) -> T5ForConditionalGeneration:
    """Load T5 decoder weights once per thread; serialized to avoid safetensors races."""
    with _model_load_lock:
        model = T5ForConditionalGeneration.from_pretrained(
            str(pytorch_dir),
            low_cpu_mem_usage=False,
        )
    model.eval()
    model.tie_weights()
    return model


def init_paraphraser(config: Stage6Config) -> None:
    global _session_config, _session_paths
    _session_config = config
    _session_paths = _load_paths(config.paraphraser_dir)
    # Validate load on main thread before workers start.
    _get_session()


def _get_session() -> ParaphraseSession:
    if _session_config is None or _session_paths is None:
        raise RuntimeError("Call init_paraphraser() before paraphrasing.")
    session = getattr(_thread_local, "paraphrase_session", None)
    if session is None:
        session = ParaphraseSession(_session_config, _session_paths)
        _thread_local.paraphrase_session = session
    return session


def paraphrase_once(text: str, temperature: float) -> str:
    return _get_session().paraphrase_once(text, temperature)


def make_paraphrase_fn() -> Callable[[str, float], str]:
    return paraphrase_once
