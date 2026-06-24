#!/usr/bin/env python3
"""
Phase A — Export MS MARCO cross-encoder to ONNX + tokenizer (offline, one-time).

    pip install -r models/requirements.txt
    python models/export_cross_encoder.py

Outputs under models/cross_encoder/:
  model.onnx, config.json, tokenizer/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OUTPUT_DIR = ROOT / "cross_encoder"
ONNX_PATH = OUTPUT_DIR / "model.onnx"
TOKENIZER_DIR = OUTPUT_DIR / "tokenizer"
OPSET = 14

SMOKE_JD = (
    "Senior AI Engineer with production embeddings-based retrieval and vector database "
    "experience, evaluation frameworks, and strong Python in production ML."
)
SMOKE_CANDIDATE = (
    "Staff ML Engineer at Acme: built hybrid FAISS + BM25 retrieval serving 2M queries/day, "
    "designed NDCG offline eval and online A/B tests for ranking quality."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export cross-encoder to ONNX for Stage 4.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def export_onnx(model_id: str, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"
    tokenizer_dir = output_dir / "tokenizer"

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    model.cpu()

    tokenizer.save_pretrained(str(tokenizer_dir))
    print(f"Saved tokenizer to {tokenizer_dir.resolve()}")

    dummy = tokenizer(
        ["query text", "passage text"],
        padding=True,
        truncation=True,
        max_length=16,
        return_tensors="pt",
    )
    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]
    token_type_ids = dummy.get("token_type_ids")
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(input_ids)

    print(f"Exporting to {onnx_path.resolve()}...")
    torch.onnx.export(
        model,
        (input_ids, attention_mask, token_type_ids),
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
        verbose=False,
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX graph is valid.")
    print("Inputs:", [i.name for i in onnx_model.graph.input])
    print("Outputs:", [o.name for o in onnx_model.graph.output])

    return onnx_path, tokenizer_dir


def smoke_test(onnx_path: Path, tokenizer_dir: Path) -> float:
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    encoded = tokenizer(
        SMOKE_JD,
        SMOKE_CANDIDATE,
        padding=True,
        truncation=True,
        max_length=512,
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

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    logits = session.run(None, feeds)[0]
    score = float(logits.reshape(-1)[0])
    print(f"Smoke test score: {score:.6f}")
    return score


def print_manifest(output_dir: Path) -> None:
    artifacts = [
        output_dir / "model.onnx",
        output_dir / "tokenizer",
    ]
    print("\n--- Output files ---")
    for path in artifacts:
        status = "OK" if path.exists() else "MISSING"
        print(f"  [{status}] {path.resolve()}")
    print("\nNext: python tracks/instructor/stage4/run.py")


def main() -> None:
    args = parse_args()
    onnx_path, tokenizer_dir = export_onnx(args.model_id, args.output_dir)
    smoke_test(onnx_path, tokenizer_dir)
    print_manifest(args.output_dir)


if __name__ == "__main__":
    main()
