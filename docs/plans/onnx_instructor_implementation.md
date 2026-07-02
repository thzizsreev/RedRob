# ONNX Implementation Guide: instructor-large
### Redrob Hackathon — Track A Speedup

---

## Why This Document Exists

Current precompute runtime on RTX 4050: **~4-5 hours**  
Target after ONNX: **~60-90 minutes**  
Expected speedup: **3-4×** from graph optimization + kernel fusion + GPU execution provider

This document is a step-by-step implementation guide. Follow the sections in order.

---

## Critical Finding: Two Paths, One Correct One

There are two ways to use INSTRUCTOR with ONNX. Only one works cleanly.

### ❌ Path A — InstructorEmbedding library + manual ONNX export
The `InstructorEmbedding` library wraps sentence-transformers with custom pooling logic. Exporting it to ONNX via `torch.onnx.export` or `optimum-cli` is fragile — the custom pooling and instruction-token masking are not standard ONNX ops and break during export.

### ✅ Path B — sentence-transformers native INSTRUCTOR support + backend="onnx"
Since sentence-transformers v3.2, `hkunlp/instructor-large` is **natively supported** as a `SentenceTransformer` model. The library handles `include_prompt=False` pooling automatically. The `backend="onnx"` parameter is a first-class feature of `SentenceTransformer()`. These two features compose cleanly.

**Use Path B. Everything in this document uses Path B.**

---

## Section 1: Dependencies

### Uninstall Conflicts First

```bash
# onnxruntime and onnxruntime-gpu conflict — remove CPU version before installing GPU
pip uninstall onnxruntime -y
```

### Install Required Packages

```bash
pip install --upgrade sentence-transformers>=3.2.0
pip install --upgrade "optimum[onnxruntime-gpu]"
pip install onnx
pip install onnxruntime-gpu  # GPU execution provider for RTX 4050
```

### Verify Installation

```python
import onnxruntime as ort
print(ort.get_available_providers())
# Must include: ['CUDAExecutionProvider', 'CPUExecutionProvider']
# If CUDA is missing, ONNX will fall back to CPU silently — check CUDA/cuDNN setup
```

### Updated requirements.txt for precompute.py

```text
# Track A — ONNX path
sentence-transformers>=3.2.0
optimum[onnxruntime-gpu]
onnxruntime-gpu
onnx

# Existing (unchanged)
polars>=0.20.0
faiss-cpu>=1.7.4
lightgbm>=4.1.0
numpy>=1.24.0
rank_bm25>=0.2.2

# NOTE: InstructorEmbedding package is NO LONGER NEEDED
# NOTE: onnxruntime (CPU) must NOT be installed alongside onnxruntime-gpu
```

---

## Section 2: Step 1 — Export the Model to ONNX (Run Once)

This step converts instructor-large from PyTorch weights to an ONNX graph. Run once, save to disk, never run again.

### Script: `export_onnx.py`

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.backend import export_optimized_onnx_model
import os

ONNX_DIR = "./instructor_large_onnx"
os.makedirs(ONNX_DIR, exist_ok=True)

print("Step 1: Loading instructor-large with ONNX backend for export...")
# backend="onnx" + export=True triggers automatic ONNX export
# sentence-transformers handles instructor pooling (include_prompt=False) natively
model = SentenceTransformer(
    "hkunlp/instructor-large",
    backend="onnx",
    model_kwargs={"export": True},
)

print("Step 2: Applying O4 optimization (FP16 + graph fusion for GPU)...")
# O4 = FP16 mixed precision + all graph optimizations
# Best choice for RTX 4050 CUDA — uses Tensor cores
export_optimized_onnx_model(
    model=model,
    optimization_config="O4",        # FP16 + full graph optimization
    model_name_or_path=ONNX_DIR,
)

print(f"Export complete. Optimized model saved to: {ONNX_DIR}/")
print("Files created:")
for f in os.listdir(ONNX_DIR):
    size_mb = os.path.getsize(os.path.join(ONNX_DIR, f)) / (1024*1024)
    print(f"  {f}: {size_mb:.1f} MB")
```

### What O4 Does Internally
- Converts all FP32 weights to FP16 (halves memory, 2× faster on Tensor cores)
- Fuses multi-head attention into single optimized kernel
- Fuses LayerNorm + activation functions
- Removes redundant transpose operations
- Constant-folds static subgraphs

### Expected Output After Export
```
instructor_large_onnx/
├── model.onnx              # Base export (~1.3GB)
├── model_O4.onnx           # Optimized FP16 model (~650MB)
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── config.json
```

### Optimization Level Reference

| Level | Precision | What it does | Best for |
|---|---|---|---|
| O1 | FP32 | Basic optimizations only | Debugging |
| O2 | FP32 | + operator fusion | CPU inference |
| O3 | FP32 | + GELU approximation | CPU inference |
| **O4** | **FP16** | **+ mixed precision, Tensor cores** | **GPU (RTX 4050) ✅** |

---

## Section 3: Step 2 — Load the Optimized Model for Inference

### How to Load in precompute.py

```python
from sentence_transformers import SentenceTransformer

ONNX_DIR = "./instructor_large_onnx"

print("Loading optimized ONNX model on GPU...")
model = SentenceTransformer(
    ONNX_DIR,
    backend="onnx",
    model_kwargs={
        "file_name": "model_O4.onnx",           # Use optimized file, not base model.onnx
        "provider": "CUDAExecutionProvider",      # RTX 4050 GPU
    },
)

print(f"Model loaded. Providers: {model._modules['0'].session.get_providers()}")
# Should print: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

### Verifying GPU is Actually Being Used

```python
import onnxruntime as ort

# After loading the model, inspect the session
session = model._modules['0'].session
active_providers = session.get_providers()
print(f"Active providers: {active_providers}")

# Check that CUDA is first (highest priority)
assert active_providers[0] == "CUDAExecutionProvider", (
    "WARNING: CUDA not active. Check onnxruntime-gpu installation and CUDA version."
)
```

---

## Section 4: Step 3 — The Encoding Pipeline with ONNX

### Key API Change: prompt= instead of [[instruction, text]]

The original `InstructorEmbedding` API used paired lists:
```python
# OLD — InstructorEmbedding API (do not use)
model.encode([[instruction, text], [instruction, text]])
```

With native sentence-transformers ONNX support, use the `prompt=` parameter:
```python
# NEW — sentence-transformers native API
model.encode(texts, prompt="your instruction here:")
```

The behavior is identical — instruction tokens are included in the forward pass but excluded from mean pooling. The API is cleaner and compatible with the ONNX backend.

### Full Encoding Pipeline: `precompute.py`

```python
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Constants ─────────────────────────────────────────────────────────────────

ONNX_DIR         = "./instructor_large_onnx"
FAISS_INDEX_PATH = "./track_a_hnsw.index"
CANDIDATE_IDS_PATH = "./candidate_ids.npy"

RETRIEVAL_PROMPT = (
    "Represent the AI engineering career history for retrieving candidates "
    "with production experience in semantic search, embeddings-based retrieval, "
    "hybrid search systems, and ranking pipelines:"
)

INFRA_PROMPT = (
    "Represent the AI engineering career history for retrieving candidates "
    "with production experience deploying and scaling ML systems, vector databases, "
    "inference optimization, and MLOps infrastructure:"
)

EVAL_PROMPT = (
    "Represent the AI engineering career history for retrieving candidates "
    "who have designed evaluation frameworks for ranking systems, run A/B tests, "
    "measured NDCG MRR MAP, and built offline-to-online feedback loops:"
)

BATCH_SIZE = 64   # Safe for O4 FP16 on RTX 4050 6GB
                  # FP16 halves memory vs FP32, so batch 64 ≈ old batch 32

# ── Load Model ────────────────────────────────────────────────────────────────

model = SentenceTransformer(
    ONNX_DIR,
    backend="onnx",
    model_kwargs={
        "file_name": "model_O4.onnx",
        "provider": "CUDAExecutionProvider",
    },
)

# ── Load and Prepare Candidates ───────────────────────────────────────────────

import gzip, json

candidates = []
with gzip.open("candidates.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        candidates.append(json.loads(line))

def build_career_text(candidate: dict) -> str:
    """Concatenate career history descriptions, most recent first."""
    history = sorted(
        candidate.get("career_history", []),
        key=lambda x: x.get("start_date", ""),
        reverse=True,
    )
    descriptions = [
        job["description"].strip()
        for job in history
        if job.get("description", "").strip()
    ]
    return " ".join(descriptions)

candidate_ids   = [c["candidate_id"] for c in candidates]
career_texts    = [build_career_text(c) for c in candidates]

print(f"Loaded {len(career_texts)} candidates.")

# ── Three Batched Encode Passes ───────────────────────────────────────────────

print("Pass 1/3: Retrieval subspace...")
retrieval_vecs = model.encode(
    career_texts,
    prompt=RETRIEVAL_PROMPT,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=False,   # We apply per-block normalization manually
    convert_to_numpy=True,
)

print("Pass 2/3: Infrastructure subspace...")
infra_vecs = model.encode(
    career_texts,
    prompt=INFRA_PROMPT,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=False,
    convert_to_numpy=True,
)

print("Pass 3/3: Evaluation subspace...")
eval_vecs = model.encode(
    career_texts,
    prompt=EVAL_PROMPT,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=False,
    convert_to_numpy=True,
)

# ── Per-Block L2 Normalization ────────────────────────────────────────────────
# Normalize each 768-d block independently.
# This preserves intra-block magnitude relationships while making blocks
# unit-normalized. Do NOT normalize the full 2304-d vector globally —
# that would lift weak blocks and suppress strong ones.

def normalize_block(vecs: np.ndarray) -> np.ndarray:
    """L2 normalize each row of a [N x D] matrix."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)   # avoid division by zero
    return vecs / norms

retrieval_vecs_norm = normalize_block(retrieval_vecs)
infra_vecs_norm     = normalize_block(infra_vecs)
eval_vecs_norm      = normalize_block(eval_vecs)

# ── Concatenate Into 2304-d Final Vectors ────────────────────────────────────

candidate_vecs = np.concatenate(
    [retrieval_vecs_norm, infra_vecs_norm, eval_vecs_norm],
    axis=1,
).astype(np.float32)   # FAISS requires float32

print(f"Final vector matrix shape: {candidate_vecs.shape}")   # (100000, 2304)

# ── Build FAISS HNSW Index ───────────────────────────────────────────────────

DIMENSION = 2304

print("Building FAISS HNSW index...")
index = faiss.IndexHNSWFlat(DIMENSION, 32, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 200

index.add(candidate_vecs)

print(f"Index built. Total vectors: {index.ntotal}")

faiss.write_index(index, FAISS_INDEX_PATH)
np.save(CANDIDATE_IDS_PATH, np.array(candidate_ids))

print(f"Saved: {FAISS_INDEX_PATH}")
print(f"Saved: {CANDIDATE_IDS_PATH}")
```

---

## Section 5: Step 4 — JD Query Vector (rank.py)

The JD query vector uses the **same model loaded from the same ONNX file**. This is the only place INSTRUCTOR is loaded in rank.py — 4 encode calls total, negligible time.

```python
# rank.py — JD query vector construction
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

ONNX_DIR = "./instructor_large_onnx"

# Load model for JD encoding only
# This is the only model load in rank.py — happens once at startup
model = SentenceTransformer(
    ONNX_DIR,
    backend="onnx",
    model_kwargs={
        "file_name": "model_O4.onnx",
        "provider": "CUDAExecutionProvider",
    },
)

# JD text per subspace — extracted from the job description
JD_RETRIEVAL_TEXT = """
Production experience with embeddings-based retrieval systems deployed to real users.
Handling embedding drift, index refresh, retrieval-quality regression in production.
Production experience with vector databases or hybrid search infrastructure.
Shipped at least one end-to-end ranking, search, or recommendation system to real users at scale.
"""

JD_INFRA_TEXT = """
Vector database scaling, FAISS, latency optimization, throughput, production deployment.
Background in distributed systems or large-scale inference optimization.
Deploy and maintain machine learning systems on cloud infrastructure at scale.
"""

JD_EVAL_TEXT = """
Hands-on experience designing evaluation frameworks for ranking systems.
NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation.
Evaluation infrastructure, offline benchmarks, online A/B testing, recruiter feedback loops.
"""

# Encode JD — 3 calls, same prompts as precompute
jd_ret  = model.encode([JD_RETRIEVAL_TEXT], prompt=RETRIEVAL_PROMPT, normalize_embeddings=False)
jd_inf  = model.encode([JD_INFRA_TEXT],     prompt=INFRA_PROMPT,     normalize_embeddings=False)
jd_eval = model.encode([JD_EVAL_TEXT],      prompt=EVAL_PROMPT,      normalize_embeddings=False)

# Normalize each block
def normalize_block(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms

# Apply priority weights BEFORE concatenation
# Retrieval is the dominant JD requirement
RETRIEVAL_WEIGHT = 0.50
INFRA_WEIGHT     = 0.30
EVAL_WEIGHT      = 0.20

jd_query_vec = np.concatenate([
    normalize_block(jd_ret)  * RETRIEVAL_WEIGHT,
    normalize_block(jd_inf)  * INFRA_WEIGHT,
    normalize_block(jd_eval) * EVAL_WEIGHT,
], axis=1).astype(np.float32)   # shape: (1, 2304)

# Load index and search
index = faiss.read_index("./track_a_hnsw.index")
index.hnsw.efSearch = 64

scores, indices = index.search(jd_query_vec, k=300)
# indices[0] → array of 300 positions in the FAISS index
# scores[0]  → corresponding inner product scores
```

---

## Section 6: Token Truncation Fix

The coding agent flagged `702 > 512` warnings — passages exceeding the model's token limit. Fix this in `build_career_text`:

```python
from transformers import AutoTokenizer

# Load the same tokenizer the model uses
tokenizer = AutoTokenizer.from_pretrained("hkunlp/instructor-large")

MAX_PASSAGE_TOKENS = 480   # Leave 32 tokens for instruction prefix

def build_career_text(candidate: dict, tokenizer, max_tokens: int = MAX_PASSAGE_TOKENS) -> str:
    history = sorted(
        candidate.get("career_history", []),
        key=lambda x: x.get("start_date", ""),
        reverse=True,
    )
    descriptions = [
        job["description"].strip()
        for job in history
        if job.get("description", "").strip()
    ]
    full_text = " ".join(descriptions)

    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return full_text

    # Truncate from the END — preserves most recent (most relevant) experience
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens, skip_special_tokens=True)
```

---

## Section 7: Fallback — If O4 FP16 Causes Issues

O4 uses FP16. If you see NaN values in embeddings or cosine similarity anomalies, fall back to O3 (FP32 with full graph optimization):

```python
# export_onnx.py — O3 fallback
export_optimized_onnx_model(
    model=model,
    optimization_config="O3",        # FP32, still ~2× faster than baseline PyTorch
    model_name_or_path=ONNX_DIR,
)

# Loading — point to O3 file
model = SentenceTransformer(
    ONNX_DIR,
    backend="onnx",
    model_kwargs={
        "file_name": "model_O3.onnx",
        "provider": "CUDAExecutionProvider",
    },
)
```

O3 gives ~2× speedup vs baseline. O4 gives ~3-4×. Try O4 first.

---

## Section 8: Troubleshooting

### Issue: CUDAExecutionProvider not available
```
Available providers: ['CPUExecutionProvider']
```
**Fix:**
```bash
# Check CUDA version
nvidia-smi   # Should show CUDA version

# Reinstall onnxruntime-gpu matching your CUDA version
# CUDA 11.x:
pip install onnxruntime-gpu==1.16.3

# CUDA 12.x:
pip install onnxruntime-gpu==1.18.0
```

### Issue: _target_device warning from InstructorEmbedding
```
_target_device has been removed
```
**Fix:** This warning only appears if `InstructorEmbedding` is still imported somewhere. With the ONNX path using native sentence-transformers, `InstructorEmbedding` is not imported at all. Remove any `from InstructorEmbedding import INSTRUCTOR` lines.

### Issue: model._modules['0'].session AttributeError
The ONNX session is accessed differently depending on sentence-transformers version:
```python
# Try this to find the session
for name, module in model.named_modules():
    if hasattr(module, 'session'):
        print(f"Session found at: {name}")
        print(module.session.get_providers())
```

### Issue: Export fails with pooling errors
```
TypeError: Pooling.__init__() got unexpected keyword argument
```
**Fix:** Upgrade sentence-transformers:
```bash
pip install --upgrade sentence-transformers>=3.2.0
```
Native INSTRUCTOR support (including correct pooling) was added in v3.2.

### Issue: Embeddings are all-zero or NaN after O4 export
FP16 can cause numerical instability on some layer configurations.
```bash
# Step down to O3
# In export_onnx.py, change:
optimization_config="O4"
# to:
optimization_config="O3"
```

---

## Section 9: Time Estimates After ONNX

Based on your observed baseline of ~1.7s per batch (batch_size=32, FP32, PyTorch):

| Configuration | Batch Size | Precision | Est. Time (3 passes, 100K) |
|---|---|---|---|
| Baseline (your current) | 32 | FP32 PyTorch | ~4-5 hours |
| ONNX O3, CPU provider | 32 | FP32 | ~2-2.5 hours |
| ONNX O3, CUDA provider | 64 | FP32 | ~1-1.5 hours |
| **ONNX O4, CUDA provider** | **64** | **FP16** | **~60-90 min** |
| ONNX O4, CUDA provider | 128 | FP16 | ~40-60 min (test VRAM) |

Batch size 128 with O4 FP16 might fit in 6GB VRAM — test with a small sample first:

```python
# Test VRAM headroom with batch 128
test_texts = career_texts[:128]
test_vecs = model.encode(test_texts, prompt=RETRIEVAL_PROMPT, batch_size=128)
# If this runs without OOM, batch 128 is safe
```

---

## Section 10: File Checklist

### Files Produced by export_onnx.py (run once)
```
instructor_large_onnx/
├── model.onnx          # Base export — not used directly
├── model_O4.onnx       # ← This is what precompute.py loads
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── config.json
```

### Files Produced by precompute.py
```
track_a_hnsw.index      # FAISS HNSW index (2304-d, 100K vectors)
candidate_ids.npy        # numpy array mapping FAISS position → candidate_id
```

### Files Loaded by rank.py
```
instructor_large_onnx/model_O4.onnx   # For JD query encoding only
track_a_hnsw.index                     # For candidate retrieval
candidate_ids.npy                      # For ID lookup
```

---

## Section 11: Summary of All Changes from Previous Architecture

| Aspect | Before (InstructorEmbedding + PyTorch) | After (sentence-transformers + ONNX) |
|---|---|---|
| Library | `InstructorEmbedding` | `sentence-transformers>=3.2` |
| Model load | `INSTRUCTOR('hkunlp/instructor-large')` | `SentenceTransformer(..., backend="onnx")` |
| Instruction API | `[[instruction, text], ...]` | `model.encode(texts, prompt=instruction)` |
| Execution backend | PyTorch CUDA | ONNX Runtime CUDAExecutionProvider |
| Precision | FP32 | FP16 (O4) |
| Batch size | 32 | 64-128 |
| Export step | None | `export_optimized_onnx_model(..., "O4")` — run once |
| Estimated 100K time | 4-5 hours | 60-90 minutes |
| torch in rank.py | Not imported | Not imported (unchanged) |
| Pooling behavior | Custom InstructorEmbedding pooling | Native sbert `include_prompt=False` pooling |
| Pooling correctness | Same | Same — instruction tokens excluded from mean pool |
