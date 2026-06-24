"""Batched ONNX cross-encoder inference on CPU."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tracks.instructor.stage4.config import Stage4Config


@dataclass
class CrossEncoderSession:
    session: ort.InferenceSession
    tokenizer: PreTrainedTokenizerBase
    output_name: str


def _preflight_model(config: Stage4Config) -> None:
    missing = []
    if not config.onnx_model_path.exists():
        missing.append(f"  onnx model: {config.onnx_model_path}")
    if not config.tokenizer_path.exists():
        missing.append(f"  tokenizer: {config.tokenizer_path}")
    if missing:
        raise FileNotFoundError(
            "Cross-encoder ONNX artifacts missing. Run Phase A first:\n"
            "  pip install -r models/requirements.txt\n"
            "  python models/export_cross_encoder.py\n"
            + "\n".join(missing)
        )


def load_cross_encoder(config: Stage4Config) -> CrossEncoderSession:
    _preflight_model(config)

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = config.num_threads
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(config.onnx_model_path),
        sess_options,
        providers=["CPUExecutionProvider"],
    )
    tokenizer = AutoTokenizer.from_pretrained(str(config.tokenizer_path))
    output_name = session.get_outputs()[0].name
    print(f"Cross-encoder ONNX loaded: {config.onnx_model_path.name}")
    print(f"ORT providers: {session.get_providers()}")
    return CrossEncoderSession(session=session, tokenizer=tokenizer, output_name=output_name)


def _extract_scores(logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits)
    if arr.ndim == 1:
        return arr.astype(np.float32)
    if arr.shape[-1] == 1:
        return arr.reshape(-1).astype(np.float32)
    return arr[:, -1].astype(np.float32)


def score_pairs(
    encoder: CrossEncoderSession,
    pairs: list[tuple[str, str, str]],
    config: Stage4Config,
) -> dict[str, float]:
    """Score (query, passage) pairs; empty passage gets empty_score."""
    scores: dict[str, float] = {}
    pending_ids: list[str] = []
    pending_queries: list[str] = []
    pending_passages: list[str] = []

    def flush_batch() -> None:
        if not pending_ids:
            return
        encoded = encoder.tokenizer(
            pending_queries,
            pending_passages,
            padding=True,
            truncation="only_second",
            max_length=config.max_pair_tokens,
            return_tensors="np",
        )
        feeds = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in encoded:
            feeds["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)
        else:
            feeds["token_type_ids"] = np.zeros_like(feeds["input_ids"])

        logits = encoder.session.run([encoder.output_name], feeds)[0]
        batch_scores = _extract_scores(logits)
        for cid, score in zip(pending_ids, batch_scores):
            scores[cid] = float(score)
        pending_ids.clear()
        pending_queries.clear()
        pending_passages.clear()

    for cid, query, passage in pairs:
        if not passage.strip():
            print(f"WARNING: empty candidate text for {cid}; assigning empty_score")
            scores[cid] = config.empty_score
            continue
        pending_ids.append(cid)
        pending_queries.append(query)
        pending_passages.append(passage)
        if len(pending_ids) >= config.batch_size:
            flush_batch()

    flush_batch()
    return scores
