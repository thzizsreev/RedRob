"""Stage 0 — T5 paraphraser ONNX encoder export (offline, one-time)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from transformers import T5EncoderModel, T5ForConditionalGeneration, T5Tokenizer

DEFAULT_MODEL_ID = "humarin/chatgpt_paraphraser_on_T5_base"
OPSET = 14
SMOKE_TEXT = (
    "Mira Verma brings 6-plus-year years of production ML experience at Uber and Flipkart, "
    "most recently developing and scaling the ranking pipeline at Uber."
)


def load_model_id_from_config(config_path: Path | None) -> str:
    if config_path is None or not config_path.exists():
        return DEFAULT_MODEL_ID
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    stage6 = raw.get("stage6", {})
    return str(stage6.get("model_id", DEFAULT_MODEL_ID))


def artifacts_exist(output_dir: Path) -> bool:
    return (
        (output_dir / "encoder.onnx").exists()
        and (output_dir / "tokenizer").exists()
        and (output_dir / "pytorch").exists()
        and (output_dir / "manifest.json").exists()
    )


def export_paraphraser_encoder(model_id: str, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = output_dir / "encoder.onnx"
    tokenizer_dir = output_dir / "tokenizer"
    pytorch_dir = output_dir / "pytorch"

    print(f"Loading {model_id}...")
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    full_model = T5ForConditionalGeneration.from_pretrained(model_id)
    full_model.eval()
    full_model.cpu()

    encoder = full_model.encoder
    encoder.eval()

    tokenizer.save_pretrained(str(tokenizer_dir))
    full_model.config.tie_word_embeddings = True
    full_model.tie_weights()
    full_model.save_pretrained(str(pytorch_dir))
    print(f"Saved tokenizer -> {tokenizer_dir.resolve()}")
    print(f"Saved PyTorch weights -> {pytorch_dir.resolve()} (CPU decoder fallback)")

    dummy = tokenizer(
        [f"paraphrase: {SMOKE_TEXT}"],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]

    print(f"Exporting encoder to {encoder_path.resolve()}...")
    torch.onnx.export(
        encoder,
        (input_ids, attention_mask),
        str(encoder_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
        verbose=False,
    )

    onnx_model = onnx.load(str(encoder_path))
    onnx.checker.check_model(onnx_model)
    print("Encoder ONNX graph is valid.")

    manifest = {
        "model_id": model_id,
        "opset": OPSET,
        "encoder_onnx": "encoder.onnx",
        "tokenizer_dir": "tokenizer",
        "pytorch_dir": "pytorch",
        "decoder_mode": "pytorch_cpu",
        "input_prefix": "paraphrase: ",
        "max_length": 128,
        "max_new_tokens": 128,
        "top_p": 0.92,
        "repetition_penalty": 1.3,
        "providers": ["CPUExecutionProvider"],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path.resolve()}")
    return manifest


def smoke_test(output_dir: Path, manifest: dict) -> str:
    tokenizer = T5Tokenizer.from_pretrained(str(output_dir / "tokenizer"))
    encoded = tokenizer(
        f"paraphrase: {SMOKE_TEXT}",
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=128,
    )
    session = ort.InferenceSession(
        str(output_dir / "encoder.onnx"),
        providers=["CPUExecutionProvider"],
    )
    feeds = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    }
    hidden = session.run(None, feeds)[0]
    print(f"Encoder smoke: hidden shape {hidden.shape}")

    from transformers.modeling_outputs import BaseModelOutput

    model = T5ForConditionalGeneration.from_pretrained(str(output_dir / "pytorch"))
    model.eval()
    encoder_outputs = BaseModelOutput(
        last_hidden_state=torch.from_numpy(hidden).float(),
    )
    output_ids = model.generate(
        encoder_outputs=encoder_outputs,
        attention_mask=torch.from_numpy(encoded["attention_mask"]),
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
        top_p=manifest["top_p"],
        repetition_penalty=manifest["repetition_penalty"],
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Paraphrase smoke: {text[:120]}...")
    return text


def run_paraphraser_export(
    *,
    model_id: str,
    output_dir: Path,
    skip_if_exists: bool = True,
    config_path: Path | None = None,
) -> Path:
    if skip_if_exists and artifacts_exist(output_dir):
        print(f"Paraphraser artifacts already exist at {output_dir.resolve()} — skipping export.")
        return output_dir

    manifest = export_paraphraser_encoder(model_id, output_dir)
    smoke_test(output_dir, manifest)
    return output_dir
