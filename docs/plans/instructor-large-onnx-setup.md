# Setting Up ONNX for Faster Instructor-Large Embeddings

This guide walks through exporting `hkunlp/instructor-large` to ONNX and running it with ONNX Runtime for faster inference, including why the "obvious" approaches (`optimum-cli export onnx`, plugging it into `ORTModelForFeatureExtraction`, etc.) fail for this specific model, and a working manual export + inference pipeline.

## Table of contents

1. [Why this model is hard to export](#1-why-this-model-is-hard-to-export)
2. [Prerequisites](#2-prerequisites)
3. [Project setup](#3-project-setup)
4. [Step 1 — Export the T5 encoder to ONNX](#4-step-1--export-the-t5-encoder-to-onnx)
5. [Step 2 — Verify the exported graph](#5-step-2--verify-the-exported-graph)
6. [Step 3 — Build the ONNX inference wrapper](#6-step-3--build-the-onnx-inference-wrapper)
7. [Step 4 — Validate against the original PyTorch model](#7-step-4--validate-against-the-original-pytorch-model)
8. [Step 5 — Benchmark](#8-step-5--benchmark)
9. [Step 6 (optional) — Quantize for extra speed](#9-step-6-optional--quantize-for-extra-speed)
10. [Project structure](#10-project-structure)
11. [Troubleshooting](#11-troubleshooting)
12. [References](#12-references)

---

## 1. Why this model is hard to export

If you've tried `optimum-cli export onnx --model hkunlp/instructor-large ...` or wrapped it in `optimum.onnxruntime.ORTModelForFeatureExtraction`, it almost certainly failed or produced wrong embeddings. There's a specific reason for that, and it's worth understanding before touching code, because it explains why Cursor's AI agent (or any AI agent) tends to go in circles on this: it keeps reaching for the standard Optimum export path, which doesn't apply here.

`instructor-large` is not a plain Hugging Face `AutoModel`. It's distributed through the `InstructorEmbedding` Python package as a customized `sentence-transformers` model with two stages:

- **`INSTRUCTORTransformer`** — a wrapper around a `T5EncoderModel` backbone.
- **`INSTRUCTORPooling`** — a mean-pooling layer.

The part that breaks automatic exporters is what happens *between* those two stages. INSTRUCTOR encodes the instruction and the text **together** as one sequence (`instruction + text`), so the model can attend across both — but when it pools the token embeddings into a single vector, it **excludes the instruction tokens** from the average. Only the text tokens (and the final end-of-sequence token) contribute to the final embedding. The instruction still shapes the result indirectly, through self-attention, but it isn't averaged in directly.

That instruction-exclusion logic is implemented as plain Python/tensor bookkeeping that runs *before* and *after* the encoder call — it tokenizes the instruction a second time on its own, counts how many tokens it occupies, and uses that to build a mask. None of this is part of the model's learned computation graph, so:

- `optimum-cli` can't auto-detect a task for it (it isn't a registered `AutoModelFor...` class), and even if forced, it would trace the wrong `forward()` method.
- A naive `torch.onnx.export(model, ...)` on the whole `INSTRUCTOR` object will either error out or silently bake in pooling behavior that ignores instructions entirely (plain mean pooling over everything), which quietly degrades embedding quality.

**The fix:** export only the part that's actually a static, traceable neural network — the T5 encoder — and reimplement the instruction-masking + pooling step in plain NumPy at inference time. That logic is deterministic arithmetic (counting tokens, building a mask, averaging), so it's trivial and fast outside the ONNX graph, and it gives you results numerically identical to the original PyTorch model.

---

## 2. Prerequisites

- Python 3.10 or 3.11 (recommended — prebuilt wheels exist for every package below, so you won't need a C++ compiler toolchain)
- ~4 GB of free RAM and ~3 GB of free disk space (instructor-large is ~1.2 GB in fp32; you'll briefly have a PyTorch copy and an ONNX copy on disk at once)
- Cursor with its Python interpreter pointed at the virtual environment you create below (see [Troubleshooting](#11-troubleshooting) if Cursor keeps using the wrong interpreter — this is the single most common cause of "it's not working")

---

## 3. Project setup

```bash
mkdir instructor-onnx && cd instructor-onnx
python -m venv .venv

# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Open this folder in Cursor, then explicitly select `.venv` as the interpreter: `Cmd/Ctrl+Shift+P` → `Python: Select Interpreter` → choose the one inside `.venv`. Do this **before** installing packages or running any script from Cursor's terminal/run button.

Install dependencies:

```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # drop the --index-url if you have a CUDA GPU and want GPU export
pip install transformers sentence-transformers InstructorEmbedding
pip install onnx onnxruntime
```

If you have an NVIDIA GPU and want GPU inference later, install `onnxruntime-gpu` instead of `onnxruntime` (don't install both — they conflict):

```bash
pip uninstall onnxruntime -y
pip install onnxruntime-gpu
```

---

## 4. Step 1 — Export the T5 encoder to ONNX

Create `export_to_onnx.py`:

```python
"""
Export the T5 encoder backbone used by hkunlp/instructor-large to ONNX.

We export *only* the encoder. The instruction-aware pooling that makes
INSTRUCTOR different from a plain sentence-transformer is bookkeeping
done in Python around the encoder call, not part of the learned graph,
so it's reimplemented separately in instructor_onnx.py instead of being
traced here.
"""

import os
import torch
import torch.nn as nn
from InstructorEmbedding import INSTRUCTOR

MODEL_NAME = "hkunlp/instructor-large"
OUTPUT_DIR = "onnx"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "instructor-large-encoder.onnx")
OPSET = 17


class EncoderWrapper(nn.Module):
    """Restricts the traced graph's inputs/outputs to input_ids and attention_mask."""

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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading {MODEL_NAME} (this downloads ~1.2 GB the first time)...")
    model = INSTRUCTOR(MODEL_NAME)
    model.eval()

    # model._first_module() is the INSTRUCTORTransformer wrapper;
    # .auto_model is the underlying transformers T5EncoderModel.
    transformer_module = model._first_module()
    t5_encoder = transformer_module.auto_model
    max_seq_length = transformer_module.max_seq_length

    wrapper = EncoderWrapper(t5_encoder)
    wrapper.eval()

    dummy_input_ids = torch.randint(0, t5_encoder.config.vocab_size, (2, 16), dtype=torch.long)
    dummy_attention_mask = torch.ones(2, 16, dtype=torch.long)

    print("Exporting to ONNX...")
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask),
        OUTPUT_PATH,
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

    transformer_module.tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "tokenizer"))
    print(f"Saved tokenizer to {OUTPUT_DIR}/tokenizer")

    with open(os.path.join(OUTPUT_DIR, "config.txt"), "w") as f:
        f.write(f"max_seq_length={max_seq_length}\n")
    print(f"max_seq_length recorded as {max_seq_length}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
python export_to_onnx.py
```

This will take a few minutes (downloading the model, tracing the graph, writing ~1.2 GB of ONNX weights). When it finishes you should have:

```
onnx/
├── instructor-large-encoder.onnx
├── tokenizer/
└── config.txt
```

---

## 5. Step 2 — Verify the exported graph

Create `verify_onnx.py`:

```python
import onnx

model = onnx.load("onnx/instructor-large-encoder.onnx")
onnx.checker.check_model(model)
print("ONNX graph is structurally valid.")
print("Inputs:", [i.name for i in model.graph.input])
print("Outputs:", [o.name for o in model.graph.output])
```

```bash
python verify_onnx.py
```

If `onnx.checker.check_model` raises an error here, do not move on — fix the export step first (see [Troubleshooting](#11-troubleshooting)).

---

## 6. Step 3 — Build the ONNX inference wrapper

This is the piece that reproduces INSTRUCTOR's instruction-aware pooling. Create `instructor_onnx.py`:

```python
"""
Drop-in, faster replacement for InstructorEmbedding.INSTRUCTOR.encode(),
backed by ONNX Runtime instead of PyTorch.

Reproduces INSTRUCTOR's instruction-aware mean pooling:
  1. Tokenize "instruction + text" together   -> input_ids / attention_mask
     fed into the encoder.
  2. Tokenize the instruction alone           -> used only to count how
     many leading tokens of the combined sequence belong to the
     instruction (its own end-of-sequence token doesn't count).
  3. Zero out those leading instruction-token positions in the
     attention mask before pooling, so the final embedding is a mean
     over the text tokens only. The instruction still influenced those
     text-token representations through self-attention inside the
     encoder; it's just excluded from the final average.
"""

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class InstructorONNX:
    def __init__(
        self,
        onnx_path: str = "onnx/instructor-large-encoder.onnx",
        tokenizer_path: str = "onnx/tokenizer",
        max_seq_length: int = 512,
        providers=None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_seq_length = max_seq_length
        providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)

    def _tokenize(self, texts, padding="longest"):
        return self.tokenizer(
            texts,
            padding=padding,
            truncation="longest_first",
            max_length=self.max_seq_length,
            return_tensors="np",
        )

    def encode(self, instruction_text_pairs, batch_size: int = 32, normalize: bool = True):
        """
        instruction_text_pairs: list of [instruction, text] pairs, e.g.

            [["Represent the Science title: ", "3D ActionSLAM: wearable person tracking"]]

        Returns an (N, hidden_dim) float32 numpy array.
        """
        all_embeddings = []

        for start in range(0, len(instruction_text_pairs), batch_size):
            batch = instruction_text_pairs[start : start + batch_size]
            instructions = [pair[0].strip() for pair in batch]
            texts = [pair[1].strip() for pair in batch]
            combined = [instr + text for instr, text in zip(instructions, texts)]

            combined_enc = self._tokenize(combined)
            instr_enc = self._tokenize(instructions)

            input_ids = combined_enc["input_ids"].astype(np.int64)
            attention_mask = combined_enc["attention_mask"].astype(np.int64)

            # Number of instruction tokens excluding its own end-of-sequence token.
            instr_lengths = instr_enc["attention_mask"].sum(axis=1) - 1
            instr_lengths = np.clip(instr_lengths, 0, None)

            seq_len = input_ids.shape[1]
            positions = np.arange(seq_len)[None, :]                       # (1, seq_len)
            is_instruction = (positions < instr_lengths[:, None]).astype(np.int64)
            pooling_mask = attention_mask * (1 - is_instruction)          # exclude instruction tokens

            outputs = self.session.run(
                ["last_hidden_state"],
                {"input_ids": input_ids, "attention_mask": attention_mask},
            )
            last_hidden_state = outputs[0]

            mask_expanded = pooling_mask[:, :, None].astype(np.float32)
            summed = (last_hidden_state * mask_expanded).sum(axis=1)
            counts = np.clip(mask_expanded.sum(axis=1), 1e-9, None)
            embeddings = summed / counts

            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.clip(norms, 1e-9, None)

            all_embeddings.append(embeddings.astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)


if __name__ == "__main__":
    model = InstructorONNX()
    pairs = [
        ["Represent the Science title: ", "3D ActionSLAM: wearable person tracking in multi-floor environments"],
    ]
    embeddings = model.encode(pairs)
    print(embeddings.shape, embeddings[0][:8])
```

A few notes on the implementation:

- `padding="longest"` (rather than `"max_length"`) is the main speed win over the original library, which always pads every batch out to the full 512-token `max_seq_length`. Padding to only the longest sequence in the current batch means short texts run through far fewer tokens — the math is identical since padded positions are always excluded from pooling, only the wasted compute disappears.
- `normalize=True` matches calling `model.encode(..., normalize_embeddings=True)` on the original model. Set it to `False` if you were using cosine similarity manually and don't want pre-normalized vectors.

---

## 7. Step 4 — Validate against the original PyTorch model

Don't skip this — it's the only way to be sure the instruction-masking logic above was reimplemented correctly for your environment. Create `compare_outputs.py`:

```python
import numpy as np
from InstructorEmbedding import INSTRUCTOR
from instructor_onnx import InstructorONNX

pairs = [
    ["Represent the Science title: ", "3D ActionSLAM: wearable person tracking in multi-floor environments"],
    ["Represent the Wikipedia document for retrieval: ", "Artificial intelligence was founded as an academic discipline in 1956."],
    ["Represent the financial statement for retrieval: ", "Net income rose 12% year over year, driven by strong cloud demand."],
]

print("Running original PyTorch model...")
pt_model = INSTRUCTOR("hkunlp/instructor-large")
pt_embeddings = pt_model.encode(pairs, normalize_embeddings=True)

print("Running ONNX model...")
onnx_model = InstructorONNX()
onnx_embeddings = onnx_model.encode(pairs, normalize=True)

cosine_sim = (pt_embeddings * onnx_embeddings).sum(axis=1)  # both are unit-normalized
max_abs_diff = np.abs(pt_embeddings - onnx_embeddings).max()

print("Per-pair cosine similarity (PyTorch vs ONNX):", cosine_sim)
print("Max absolute element-wise difference:", max_abs_diff)
print("PASS" if cosine_sim.min() > 0.999 else "FAIL — investigate before relying on the ONNX output")
```

```bash
python compare_outputs.py
```

Expect cosine similarities at or extremely close to `1.0` (typically `> 0.9999`) and a max absolute difference on the order of `1e-4` or smaller — that residual comes from floating-point operation ordering differences between PyTorch and ONNX Runtime, not from a logic error. If similarity is meaningfully lower than that, recheck the export step before trusting embeddings produced by this pipeline.

---

## 8. Step 5 — Benchmark

Create `benchmark.py`:

```python
import time
from InstructorEmbedding import INSTRUCTOR
from instructor_onnx import InstructorONNX

pairs = [["Represent the document for retrieval: ", f"This is sample document number {i} about software engineering."] for i in range(64)]

pt_model = INSTRUCTOR("hkunlp/instructor-large")
onnx_model = InstructorONNX()

# Warm-up run (loading lazy kernels, caches, etc.)
pt_model.encode(pairs[:4], normalize_embeddings=True)
onnx_model.encode(pairs[:4], normalize=True)

start = time.perf_counter()
pt_model.encode(pairs, normalize_embeddings=True, batch_size=16)
pt_time = time.perf_counter() - start

start = time.perf_counter()
onnx_model.encode(pairs, normalize=True, batch_size=16)
onnx_time = time.perf_counter() - start

print(f"PyTorch: {pt_time:.2f}s")
print(f"ONNX Runtime: {onnx_time:.2f}s")
print(f"Speedup: {pt_time / onnx_time:.2f}x")
```

```bash
python benchmark.py
```

Typical CPU speedups for this kind of export are commonly in the 1.5x–3x range, mostly coming from avoiding the fixed 512-token padding and from ONNX Runtime's graph optimizations (operator fusion, etc.). GPU speedups vary more by hardware.

---

## 9. Step 6 (optional) — Quantize for extra speed

Dynamic INT8 quantization can give a further CPU speedup (often 1.5x–2x on top of the fp32 ONNX model), at a small accuracy cost. Test it with `compare_outputs.py`-style validation before relying on it for anything similarity-sensitive.

```bash
pip install onnxruntime-tools onnx
```

```python
# quantize.py
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="onnx/instructor-large-encoder.onnx",
    model_output="onnx/instructor-large-encoder-int8.onnx",
    weight_type=QuantType.QInt8,
)
print("Wrote onnx/instructor-large-encoder-int8.onnx")
```

```bash
python quantize.py
```

Then point `InstructorONNX(onnx_path="onnx/instructor-large-encoder-int8.onnx")` at the quantized file and re-run `compare_outputs.py` to confirm cosine similarity is still acceptable for your use case (retrieval/ranking tasks are usually fairly tolerant of small quantization noise; exact-match or clustering tasks may be more sensitive).

---

## 10. Project structure

After completing all steps:

```
instructor-onnx/
├── .venv/
├── export_to_onnx.py
├── verify_onnx.py
├── instructor_onnx.py
├── compare_outputs.py
├── benchmark.py
├── quantize.py
└── onnx/
    ├── instructor-large-encoder.onnx
    ├── instructor-large-encoder-int8.onnx   # if you quantized
    ├── tokenizer/
    └── config.txt
```

To use the model going forward, you only need `instructor_onnx.py` plus the `onnx/` folder — none of the heavier PyTorch/`InstructorEmbedding` dependencies are required at inference time, only `onnxruntime`, `transformers` (for the tokenizer), and `numpy`.

---

## 11. Troubleshooting

**Cursor keeps "fixing" this by suggesting `optimum-cli export onnx`, and it keeps failing.**
That command works for standard `AutoModel`-based Hugging Face models, not for the custom `INSTRUCTOR` class — see [Section 1](#1-why-this-model-is-hard-to-export). Point Cursor's agent at this guide's manual export script instead of letting it retry the CLI path with different flags.

**Cursor runs a different Python/package set than what you installed.**
This is the most common root cause of "very very difficult to setup." Confirm the interpreter Cursor is using matches your `.venv`: `Cmd/Ctrl+Shift+P` → `Python: Select Interpreter`. If you installed packages from a regular terminal but Cursor's integrated terminal or run button uses a different interpreter, you'll get `ModuleNotFoundError` even though `pip list` looks correct.

**`pip install InstructorEmbedding` / `sentencepiece` tries to compile from source and fails (Windows especially).**
Use Python 3.10 or 3.11 — both have prebuilt wheels for `sentencepiece`, `tokenizers`, and `torch` on Windows/macOS/Linux, avoiding the need for a C++ build toolchain. If you're stuck on a newer/older Python version, installing the "Desktop development with C++" workload via Visual Studio Build Tools is the fallback on Windows.

**`TypeError: Descriptors cannot be created directly` or other `protobuf` errors.**
`onnx` and `transformers`/`torch` can disagree on `protobuf` versions. Pin a compatible version:
```bash
pip install "protobuf>=3.20,<5"
```
then reinstall `onnx`/`onnxruntime` if the error persists.

**`RuntimeError: Exporting the operator ... to ONNX opset version N is not supported.`**
Bump `OPSET` in `export_to_onnx.py` (17 or 18 is safe with reasonably current `torch`), or upgrade `torch`/`onnx` — older combinations sometimes lack support for ops T5 relies on (relative position bias, certain reshape patterns).

**The export succeeds but `compare_outputs.py` shows low cosine similarity.**
Almost always means the instruction-masking logic in `instructor_onnx.py` and the original model's tokenizer settings have drifted apart — double check `max_seq_length` in `onnx/config.txt` matches what you pass to `InstructorONNX(max_seq_length=...)`, and confirm you didn't accidentally lowercase or otherwise alter instruction/text strings differently between the two code paths.

**Both `onnxruntime` and `onnxruntime-gpu` are installed and things behave oddly.**
They conflict. Uninstall both, then install only the one you need:
```bash
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime          # CPU
# or
pip install onnxruntime-gpu      # NVIDIA GPU
```

**Running on Apple Silicon (M1/M2/M3).**
Standard `pip install torch onnxruntime` works fine on macOS arm64 with current versions — no special wheels needed. GPU acceleration via `CoreMLExecutionProvider` is possible but optional; CPU performance is already reasonable for this model size.

**Out-of-memory during export.**
The encoder is loaded fully in fp32 (~1.2 GB) plus the ONNX writer briefly holds a serialized copy. Close other heavy applications, or export on a machine/container with at least 4 GB free RAM.

---

## 12. References

- INSTRUCTOR model and library: `hkunlp/instructor-large` on Hugging Face, and the `InstructorEmbedding` PyPI package / `xlang-ai/instructor-embedding` GitHub repository
- ONNX Runtime documentation: https://onnxruntime.ai/docs/
- Hugging Face Optimum ONNX export docs (background on the standard export path, useful for understanding why it doesn't apply here): https://huggingface.co/docs/optimum
