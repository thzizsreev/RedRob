"""CUDA ONNX Runtime embedder for INSTRUCTOR-large (precompute only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from tracks.instructor.config import (
    CUDA_PROVIDER,
    INSTRUCTOR_ONNX_CONFIG,
    INSTRUCTOR_ONNX_DENSE,
    INSTRUCTOR_ONNX_ENCODER,
    INSTRUCTOR_ONNX_MAX_SEQ_LENGTH,
    INSTRUCTOR_ONNX_TOKENIZER,
)

CUDA_OPTIONS = {
    "device_id": 0,
    "arena_extend_strategy": "kSameAsRequested",
    "gpu_mem_limit": 4 * 1024**3,
    "cudnn_conv_algo_search": "DEFAULT",
}


def _parse_max_seq_length(config_path: Path) -> int:
    if not config_path.exists():
        return INSTRUCTOR_ONNX_MAX_SEQ_LENGTH
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("max_seq_length="):
            return int(line.split("=", 1)[1].strip())
    return INSTRUCTOR_ONNX_MAX_SEQ_LENGTH


def _preflight_artifacts(
    encoder: Path,
    tokenizer: Path,
    dense: Path,
    config: Path,
) -> None:
    required = [
        ("encoder", encoder),
        ("tokenizer", tokenizer),
        ("dense weights", dense),
        ("config", config),
    ]
    missing = [f"  {name}: {path}" for name, path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "INSTRUCTOR ONNX artifacts missing. Run: cd onnx && python export_to_onnx.py\n"
            + "\n".join(missing)
        )
    if not tokenizer.is_dir():
        raise FileNotFoundError(f"Tokenizer directory missing: {tokenizer}")


def _create_session(onnx_path: Path) -> ort.InferenceSession:
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.enable_mem_pattern = True
    sess_options.enable_cpu_mem_arena = False

    providers: list = [
        (CUDA_PROVIDER, CUDA_OPTIONS),
        "CPUExecutionProvider",
    ]
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options,
        providers=providers,
    )
    active = session.get_providers()
    if not active or active[0] != CUDA_PROVIDER:
        raise RuntimeError(
            f"{CUDA_PROVIDER} is not active (providers={active}). "
            "Install onnxruntime-gpu and uninstall CPU onnxruntime."
        )
    print(f"ORT providers: {active}")
    return session


class InstructorONNX:
    def __init__(
        self,
        onnx_path: Path = INSTRUCTOR_ONNX_ENCODER,
        tokenizer_path: Path = INSTRUCTOR_ONNX_TOKENIZER,
        dense_weight_path: Path = INSTRUCTOR_ONNX_DENSE,
        config_path: Path = INSTRUCTOR_ONNX_CONFIG,
        max_seq_length: int | None = None,
    ):
        _preflight_artifacts(onnx_path, tokenizer_path, dense_weight_path, config_path)
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        self.max_seq_length = max_seq_length or _parse_max_seq_length(config_path)
        self.dense_weight = np.load(str(dense_weight_path))
        self.session = _create_session(onnx_path)

    def _tokenize(self, texts: list[str], padding: str = "longest"):
        return self.tokenizer(
            texts,
            padding=padding,
            truncation="longest_first",
            max_length=self.max_seq_length,
            return_tensors="np",
        )

    def encode(
        self,
        instruction_text_pairs: list[list[str]],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(instruction_text_pairs), batch_size):
            batch = instruction_text_pairs[start : start + batch_size]
            instructions = [pair[0].strip() for pair in batch]
            texts = [pair[1].strip() for pair in batch]
            combined = [instr + text for instr, text in zip(instructions, texts)]

            combined_enc = self._tokenize(combined)
            instr_enc = self._tokenize(instructions)

            input_ids = combined_enc["input_ids"].astype(np.int64)
            attention_mask = combined_enc["attention_mask"].astype(np.int64)

            instr_lengths = instr_enc["attention_mask"].sum(axis=1) - 1
            instr_lengths = np.clip(instr_lengths, 0, None)

            seq_len = input_ids.shape[1]
            positions = np.arange(seq_len)[None, :]
            is_instruction = (positions < instr_lengths[:, None]).astype(np.int64)
            pooling_mask = attention_mask * (1 - is_instruction)

            outputs = self.session.run(
                ["last_hidden_state"],
                {"input_ids": input_ids, "attention_mask": attention_mask},
            )
            last_hidden_state = outputs[0]

            mask_expanded = pooling_mask[:, :, None].astype(np.float32)
            summed = (last_hidden_state * mask_expanded).sum(axis=1)
            counts = np.clip(mask_expanded.sum(axis=1), 1e-9, None)
            embeddings = summed / counts
            embeddings = embeddings @ self.dense_weight.T

            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.clip(norms, 1e-9, None)

            all_embeddings.append(embeddings.astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)


def load_embedder() -> InstructorONNX:
    print("Loading INSTRUCTOR-large ONNX embedder...")
    return InstructorONNX()


def unload_embedder(model: InstructorONNX | None) -> None:
    if model is not None:
        del model
