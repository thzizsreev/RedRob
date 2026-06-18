#!/usr/bin/env python3
"""Export hkunlp/instructor-large T5 encoder to ONNX (encoder only)."""

from __future__ import annotations

import os
from pathlib import Path

import onnx
import numpy as np
import torch
import torch.nn as nn
from InstructorEmbedding import INSTRUCTOR

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "hkunlp/instructor-large"
OUTPUT_DIR = ROOT / "models"
OUTPUT_PATH = OUTPUT_DIR / "instructor-large-encoder.onnx"
OPSET = 17


class EncoderWrapper(nn.Module):
    def __init__(self, t5_encoder: nn.Module):
        super().__init__()
        self.t5_encoder = t5_encoder

    def forward(self, input_ids, attention_mask):
        out = self.t5_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return out.last_hidden_state


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_NAME}...")
    model = INSTRUCTOR(MODEL_NAME)
    model.eval()
    model = model.cpu()

    transformer_module = model._first_module()
    t5_encoder = transformer_module.auto_model
    max_seq_length = transformer_module.max_seq_length

    wrapper = EncoderWrapper(t5_encoder)
    wrapper.eval()

    dummy_input_ids = torch.randint(
        0, t5_encoder.config.vocab_size, (2, 16), dtype=torch.long
    )
    dummy_attention_mask = torch.ones(2, 16, dtype=torch.long)

    print(f"Exporting to {OUTPUT_PATH}...")
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask),
        str(OUTPUT_PATH),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=OPSET,
        do_constant_folding=True,
    )
    print(f"Saved encoder to {OUTPUT_PATH}")

    tokenizer_dir = OUTPUT_DIR / "tokenizer"
    transformer_module.tokenizer.save_pretrained(str(tokenizer_dir))
    print(f"Saved tokenizer to {tokenizer_dir}")

    config_path = OUTPUT_DIR / "config.txt"
    config_path.write_text(f"max_seq_length={max_seq_length}\n", encoding="utf-8")
    print(f"max_seq_length={max_seq_length}")

    dense_weight = model[2].linear.weight.detach().cpu().numpy()
    dense_path = OUTPUT_DIR / "dense_weight.npy"
    np.save(dense_path, dense_weight)
    print(f"Saved dense projection {dense_weight.shape} to {dense_path}")

    print("Running onnx.checker.check_model...")
    onnx_model = onnx.load(str(OUTPUT_PATH))
    onnx.checker.check_model(onnx_model)
    print("ONNX graph is valid.")
    print("Inputs:", [i.name for i in onnx_model.graph.input])
    print("Outputs:", [o.name for o in onnx_model.graph.output])


if __name__ == "__main__":
    main()
