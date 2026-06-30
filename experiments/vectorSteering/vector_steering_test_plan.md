# Vector Steering Feasibility Test — Full Plan

This document is a self-contained implementation plan for a coding agent. It tests whether a numeric score can be deterministically converted into a displacement in sentence-embedding space, and decoded back into coherent, calibrated natural language — before any resume-reasoning pipeline is built on top of this mechanism.

---

## 1. Why This Test Exists

The downstream reasoning-generation pipeline depends entirely on one unproven assumption: that a numeric score can be converted into a displacement in embedding space, and that displacement can be deterministically decoded back into coherent, calibrated natural language. Before any resume-reasoning pipeline is built, this assumption needs to be validated in isolation, on tiny inputs, with full visibility into every intermediate vector.

This test does not touch resumes, the job description, or the ranking pipeline. It only tests whether the encode → steer → decode mechanism works at all.

### Two independent unknowns

**Unknown 1 — Inversion fidelity at baseline.** Can the encoder-decoder round trip reproduce a stable, coherent sentence from an unmodified embedding, with zero vector manipulation? This isolates whether the decoder itself is reliable before any steering math is introduced.

**Unknown 2 — Steering fidelity.** Given that Unknown 1 holds, does displacing a vector by a score-scaled direction vector and renormalizing produce text that a human reads as proportionally more positive or negative as the score increases?

These must be tested in this order. If Unknown 1 fails, Unknown 2 cannot be meaningfully measured, because there would be no way to tell whether bad output comes from broken steering math or from an unreliable decoder.

---

## 2. Tool Selection — Constraints and Rationale

Selection criteria: the encoder and decoder must be well-established, actively maintained, documented to work without a GPU, and runnable on Windows 10/11 64-bit without compiled-from-source dependencies.

### Encoder: `sentence-transformers/gtr-t5-base`

The standard GTR-base encoder referenced in the source architecture document and in the published Vec2Text research. Distributed through the `sentence-transformers` library — pure pip-installable wheels, runs on CPU without any platform-specific build step. Produces 768-dimensional embeddings, matching the dimensionality assumed throughout the architecture.

### Decoder: `vec2text` (Morris et al., jxmorris12/vec2text)

The same inversion library cited in the architecture document. Pip-installable, PyTorch-based. Ships pretrained inversion models specifically trained for GTR-base embeddings, so no custom training is required — this is the exact encoder/decoder pairing the published research validated. PyTorch has stable Windows CPU wheels, so the full stack runs on Windows 64-bit without CUDA or a GPU.

### Why not alternatives

SONAR (Meta) is a heavier dependency (fairseq2-based), has less mature Windows support, and is tuned for translation rather than English-only sentiment-direction steering. BERT+GPT2 round-tripping was tried in prior work and is known to hallucinate because GPT2 decoding is autoregressive and stochastic rather than an optimization-based inversion process. Vec2Text's corrector loop is deterministic-leaning (iterative distance minimization), which directly addresses the determinism requirement.

---

## 3. Environment Setup — Windows 64-bit

### Requirements

- Windows 10 or 11, 64-bit
- Python 3.10 or 3.11 (avoid 3.12+ for now — some PyTorch/sentence-transformers wheel combinations lag behind newest Python releases; 3.10/3.11 has the broadest pretrained-model compatibility)
- No GPU required. Both GTR-base and vec2text's inversion models run on CPU, just slower than on GPU.
- Roughly 5-10 GB free disk space for model weights and the Python environment.

### Step 1 — Create an isolated virtual environment

```
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

Confirm the venv is active — the prompt should show `(.venv)` at the start of the line.

### Step 2 — Install PyTorch (CPU build)

Install the CPU-only PyTorch build explicitly, to avoid pulling a CUDA build that won't run without an NVIDIA GPU and matching drivers:

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Step 3 — Install sentence-transformers

```
pip install sentence-transformers
```

Pure pip package, no compiled-from-source requirement on Windows.

### Step 4 — Install vec2text

```
pip install vec2text
```

Pulls in `transformers` and other dependencies automatically. All have Windows-compatible wheels.

### Step 5 — Verify the installation

```python
from sentence_transformers import SentenceTransformer
import vec2text

encoder = SentenceTransformer("sentence-transformers/gtr-t5-base")
test_vec = encoder.encode(["hello world"])
print("Encoder output shape:", test_vec.shape)

corrector = vec2text.load_pretrained_corrector("gtr-base")
print("Vec2text corrector loaded successfully.")
```

Expected output: an encoder shape of `(1, 768)` and a confirmation that the corrector loaded without error. The first run downloads model weights (a few hundred MB to ~1GB) and requires an internet connection once; subsequent runs use the local cache.

### Known Windows-specific notes

- The first `SentenceTransformer` and `vec2text` calls cache models under `%USERPROFILE%\.cache\huggingface`. Ensure free disk space there.
- If `pip install vec2text` fails on a dependency resolution conflict, install inside a fresh venv rather than troubleshooting an existing one.
- No Visual C++ Build Tools should be required, since both libraries ship precompiled wheels for Windows. If a build error mentions a missing compiler, confirm the Python version is 3.10/3.11 and the venv is truly clean.

---

## 4. Test Design — Procedure

### Phase A: Baseline Inversion Stability (No Steering)

Goal: confirm the encode-decode round trip is stable and deterministic before any vector math is introduced.

**Test sentences** (reasoning-style register, not generic web text):

1. "Seven years of production experience building FAISS and Qdrant based hybrid search systems at a fintech product company."
2. "This candidate has no production deployment experience and has only worked in academic research labs."
3. "Strong recruiter engagement with a ninety one percent response rate and active platform use this week."
4. "The candidate's notice period of one hundred twenty days is a significant logistical concern for this role."
5. "Five years at a consulting firm with no exposure to embeddings, retrieval, or ranking systems."

**Procedure**, for each sentence:

1. Encode the sentence with GTR-base to get vector `v`.
2. Decode `v` immediately with vec2text, with zero modification.
3. Record the decoded text.
4. Repeat steps 2-3 four more times on the same unmodified `v` (five decode runs total per sentence).
5. Record whether the five decode outputs are identical, near-identical, or divergent.

**What is being measured:**

- **Fidelity** — does decoded text preserve the meaning and most of the wording of the original sentence?
- **Determinism** — across five repeated decodes of the identical vector, is the output stable, or does it vary run to run?

### Phase B: Steering Direction Sanity Check

Only proceed to Phase B if Phase A passes (see Section 5 for the pass threshold).

**Single axis setup** — overall candidate quality, expressed through one sentence template.

- Anchor (bad): "This candidate's technical experience is completely unsuitable for the role."
- Anchor (good): "This candidate's technical experience is exceptionally well suited for the role."
- Base (neutral, same topic, not an anchor): "This candidate has technical experience in machine learning."

**Procedure:**

1. Encode the bad anchor → `v_bad`. Encode the good anchor → `v_good`. Encode the base → `v_base`.
2. Compute the steering direction: `v_steer = v_good - v_bad`.
3. Sweep across these `S` values in order: `-0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25`.
4. For each `S`:
   - Compute `v_target = v_base + (S * v_steer)`.
   - L2-normalize `v_target` to project it back onto the unit hypersphere.
   - Decode the normalized vector with vec2text.
   - Record the decoded text.
5. Repeat the full sweep three times to check run-to-run determinism at each `S` value.

**What is being measured:**

- **Monotonicity** — reading the seven decoded sentences in `S` order, does a human judge perceive a consistent, increasing trend from negative to positive? This is the single most important result.
- **Coherence** — is every decoded sentence at every `S` value grammatically valid and meaningful, including at intermediate values that don't correspond to either anchor?
- **Determinism** — across the three repeated sweeps, is the decoded text at each `S` value stable?
- **Overshoot behavior** — at `S = -0.25` and `S = 1.25` (past the anchors), does output degrade gracefully (more extreme, still coherent) or break down entirely?

### Recording results

For both phases, log results in a simple structured table: input description, `S` value (Phase B only), run number, decoded text, and a human judgment column (coherent / incoherent, and for Phase B, perceived sentiment direction). Record everything first, then evaluate against Section 5 — do not pre-judge results while running the test.

---

## 5. Evaluation Criteria — Pass/Fail Thresholds

### Phase A: Baseline Inversion Stability

| Criterion | Pass condition | Fail condition |
|---|---|---|
| Fidelity | Decoded text preserves the core meaning of at least 4 of 5 test sentences, with most key terms (numbers, named technologies, role-relevant nouns) intact | Decoded text loses or distorts meaning in 2 or more of 5 sentences |
| Determinism | For at least 4 of 5 sentences, all five repeated decodes of the same unmodified vector are identical or near-identical (trivial wording variation only, no meaning drift) | Repeated decodes of the same vector produce materially different sentences |

**Phase A passes only if both criteria pass.** If Phase A fails, do not proceed to Phase B — escalate per the decision tree in Section 6.

### Phase B: Steering Direction Sanity Check

| Criterion | Pass condition | Fail condition |
|---|---|---|
| Monotonicity | A human reading the 7 decoded sentences in `S` order perceives a consistent negative-to-positive trend, with no more than one out-of-order reversal | Sentiment order is scrambled, reversed, or shows no discernible trend across `S` |
| Coherence | At least 6 of 7 `S` values produce a grammatically valid, meaningful sentence | 2 or more `S` values produce garbled or nonsensical output |
| Determinism | At least 5 of 7 `S` values are stable (identical or near-identical) across the three repeated sweeps | Output varies substantially across repeated sweeps at the same `S` |
| Overshoot | At `S = -0.25` and `S = 1.25`, output remains coherent even if more extreme in tone | Overshoot values produce broken or nonsensical text |

**Phase B passes only if Monotonicity and Coherence both pass.** Determinism and Overshoot are secondary signals that inform the next step but do not alone block proceeding.

---

## 6. Decision Tree

**Phase A fails** → The inverter is not reliable on this sentence register, independent of any steering math. Do not attempt to fix this by adjusting steering logic. Next step: try a different decoder (e.g. a different vec2text checkpoint, or reconsider SONAR despite its heavier Windows footprint), or test whether fidelity improves with shorter/simpler sentences before concluding the entire vector-steering philosophy is not viable with current tooling.

**Phase A passes, Phase B fails monotonicity** → Round-trip meaning preservation works, but algebraic displacement in GTR-base's embedding space does not correspond to controllable semantic displacement for this axis. Next step: do not abandon vector steering outright — first test whether the axis definition itself is the problem (try a more extreme anchor pair, or a single-word axis like "unsuitable" vs "exceptional" instead of full sentences) before concluding the geometry itself is unusable.

**Phase A passes, Phase B passes Monotonicity and Coherence but fails Determinism or Overshoot** → The core mechanism works. Determinism issues may be addressable by checking vec2text's correction-loop configuration (number of correction steps, beam width) rather than the steering math itself. Overshoot failures are lower priority since production scores will be bounded between 0 and 1 and won't require extrapolation past the anchors.

**Both phases pass on all four criteria** → Proceed to defining the JD-relevant axes (technical fit, behavioral availability, JD-disqualifier gating) and the gate/scalar/texture tiering established in prior analysis. This test result becomes the empirical justification for building the actual pipeline on this mechanism.

---

## 7. Explicit Scope Boundary for the Coding Agent

Build only what is described above: environment setup, Phase A script, Phase B script, and a results log in the structured format described in Section 4. Do not build the resume-reasoning pipeline, the JD-matching logic, or any ranking-related code as part of this task — this is strictly an isolated feasibility test of the encode-steer-decode mechanism.
