# Plan 3 — SONAR Vector Steering Experiment
## Encode → Steer → Decode in SONAR's Native 1024d Space

---

## Why SONAR

BERT + GPT2 failed because:
- GPT2 hallucinated wildly when given free continuation
- LangVAE's latent space is 128d requiring a random untrained projector
- Neither model was designed for encode → transform → decode as a closed loop

SONAR is designed exactly as a closed loop:
- One encoder: text → 1024d vector (native space)
- One decoder: 1024d vector → text (same native space)
- No projector needed. No dimension mismatch. No random weights.
- The encoder and decoder share the same 1024d space by design.
- Transformations applied in this space are decoded faithfully.

The key difference from BERT + GPT2:
  In SONAR, encode(text_A) and encode(text_B) produce vectors whose
  interpolation decodes to text semantically between A and B.
  This is exactly what our anchor interpolation steering requires.

---

## Critical Dimension Change

SONAR vectors are 1024d, not 768d.
All vectors in this plan are shape (1024,).
All numpy arrays are shape (1024,).
All torch tensors passed to decode are shape (1, 1024).

---

## Environment and Dependencies

### Python version
```
Python 3.10
```

### Required libraries and exact install commands

```bash
# SONAR requires fairseq2 which has specific torch compatibility
# Install in this exact order

pip install torch==2.0.1 torchaudio==2.0.2
pip install fairseq2==0.2.1 --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.0.1/cpu/
pip install sonar-space==0.2.1
pip install numpy==1.24.0
```

### If fairseq2 index fails, use conda alternative
```bash
conda install -c pytorch pytorch=2.0.1 torchaudio
pip install fairseq2 --pre --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/nightly/cpu/
pip install sonar-space numpy==1.24.0
```

### Models — downloaded automatically on first run
```
Encoder : text_sonar_basic_encoder
          ~1.2GB, cached in ~/.cache/fairseq2/ after first download

Decoder : text_sonar_basic_decoder
          ~1.2GB, cached in ~/.cache/fairseq2/ after first download

Both models are NLLB-1B based transformers.
24-layer encoder, 24-layer decoder.
Both operate in the same 1024d vector space.
CPU inference: ~2-3 seconds per sentence encode or decode.
```

### Imports used in all scripts
```python
import torch
import numpy as np
import os
import json
from sonar.inference_pipelines.text import (
    TextToEmbeddingModelPipeline,
    EmbeddingToTextModelPipeline
)
```

### Directory structure
```
experiment/
├── precompute.py       ← encode all anchors and templates, save .npy files
├── run_experiment.py   ← vector math + steer + decode + output
├── vectors/            ← all precomputed .npy files saved here
└── results/
    └── output.json     ← final output saved here
```

---

## All Hardcoded Input Values

### Candidate resume sections (hardcoded in run_experiment.py)

```python
resume_tech_text = (
    "Built two-stage FAISS retrieval pipeline at Meesho for 50M product catalog. "
    "P99 latency reduced from 340ms to 22ms. End-to-end ownership of the system."
)

resume_career_text = (
    "Promoted to Senior MLE in 18 months. Led a team of 4 engineers. "
    "Full ownership of ML infrastructure decisions."
)

resume_behav_text = (
    "47 days since last login. Response rate 0.18. Notice period 30 days."
)
```

### Scores (hardcoded in run_experiment.py)

```python
s_tech   = 0.88    # high technical fit
s_career = 0.81    # high career depth
s_behav  = 0.19    # low behavioral engagement
```

### Fixed hyperparameters (hardcoded in run_experiment.py)

```python
GAMMA      = 0.55   # template weight in blend. candidate weight = 1 - GAMMA = 0.45
                    # slightly lower than before because SONAR encodes meaning
                    # more densely — less template dominance needed for tone

DELTA      = 0.30   # steer weight applied to blended base. base weight = 0.70
                    # slightly higher than before because SONAR interpolation
                    # is more faithful — larger steer still decodes cleanly

MAX_SEQ_LEN = 64    # max tokens in decoded output per clause
BEAM_SIZE   = 5     # beam search width — default SONAR setting, do not change
TARGET_LANG = "eng_Latn"   # decode target language — English
SOURCE_LANG = "eng_Latn"   # encode source language — English
```

---

## Template Library (hardcoded in precompute.py)

These are reviewer-tone sentences. SONAR encodes them into 1024d vectors.
The coding agent does NOT change these strings.

### Technical Templates

```python
template_tech = {
    "high": (
        "Strong production-grade technical alignment with the retrieval and "
        "ranking requirements of the role, backed by hands-on engineering "
        "ownership at meaningful scale."
    ),
    "mid": (
        "Partial technical overlap with the role's retrieval and ranking "
        "focus, with some relevant exposure but gaps in depth or breadth "
        "of production experience."
    ),
    "low": (
        "Limited direct technical alignment with the role's core retrieval "
        "and embedding systems requirements based on the available profile."
    )
}
```

### Career Templates

```python
template_career = {
    "high": (
        "Senior engineering trajectory with demonstrated end-to-end "
        "ownership, measurable impact, and a clear product-company "
        "background matching the seniority target."
    ),
    "mid": (
        "Developing seniority with some ownership signals but not yet "
        "at the full independent decision-making depth the role requires."
    ),
    "low": (
        "Early career profile or primarily supporting role history "
        "without the seniority and ownership depth the position demands."
    )
}
```

### Behavioral Templates

```python
template_behav = {
    "high": (
        "Active on platform with strong engagement signals and low "
        "friction expected for recruiter outreach and response."
    ),
    "mid": (
        "Moderate platform engagement with some responsiveness signals, "
        "though availability may need confirmation."
    ),
    "low": (
        "Minimal recent platform activity and low responsiveness signals "
        "suggest availability is uncertain and outreach may face friction."
    )
}
```

---

## Anchor Pairs (hardcoded in precompute.py)

These define the two poles of each steering axis.
The coding agent does NOT change these strings.

### Technical Anchors

```python
anchor_tech_hi = (
    "hands-on production engineer who has built and owned retrieval systems, "
    "vector databases, and embedding pipelines at real scale with measurable "
    "latency and quality outcomes"
)

anchor_tech_lo = (
    "candidate with surface-level or theoretical exposure to retrieval concepts "
    "without direct ownership of production retrieval or ranking systems"
)
```

### Career Anchors

```python
anchor_career_hi = (
    "senior individual contributor with full ownership of complex ML systems, "
    "clear promotion trajectory, and cross-functional impact at product companies"
)

anchor_career_lo = (
    "junior or mid-level engineer in a supporting capacity with limited "
    "independent scope and no clear ownership of significant systems"
)
```

### Behavioral Anchors

```python
anchor_behav_hi = (
    "candidate who is actively engaged, logged in recently, responds quickly "
    "to recruiter messages, and has low notice period"
)

anchor_behav_lo = (
    "candidate who has not logged in for an extended period, rarely responds "
    "to recruiter outreach, and shows no recent job-seeking activity"
)
```

---

## Score Bucketing (in run_experiment.py)

```python
def bucket(score):
    if score >= 0.65:
        return "high"
    elif score >= 0.35:
        return "mid"
    else:
        return "low"

bucket_tech   = bucket(s_tech)    # → "high"   (s_tech=0.88)
bucket_career = bucket(s_career)  # → "high"   (s_career=0.81)
bucket_behav  = bucket(s_behav)   # → "low"    (s_behav=0.19)
```

---

## precompute.py — Full Script

```python
import torch
import numpy as np
import os
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

# ── paste all template and anchor strings here exactly as declared above ──

# Load SONAR encoder once
encoder = TextToEmbeddingModelPipeline(
    encoder="text_sonar_basic_encoder",
    tokenizer="text_sonar_basic_encoder"
)

def sonar_encode(text):
    """
    Encode one string into a 1024d SONAR vector.
    Returns numpy array of shape (1024,).
    """
    embedding = encoder.predict([text], source_lang="eng_Latn")
    # embedding shape: torch.Size([1, 1024])
    return embedding[0].numpy()   # shape: (1024,)

os.makedirs("vectors", exist_ok=True)

# Encode and save all 9 templates
np.save("vectors/v_tmpl_tech_high.npy",   sonar_encode(template_tech["high"]))
np.save("vectors/v_tmpl_tech_mid.npy",    sonar_encode(template_tech["mid"]))
np.save("vectors/v_tmpl_tech_low.npy",    sonar_encode(template_tech["low"]))
np.save("vectors/v_tmpl_career_high.npy", sonar_encode(template_career["high"]))
np.save("vectors/v_tmpl_career_mid.npy",  sonar_encode(template_career["mid"]))
np.save("vectors/v_tmpl_career_low.npy",  sonar_encode(template_career["low"]))
np.save("vectors/v_tmpl_behav_high.npy",  sonar_encode(template_behav["high"]))
np.save("vectors/v_tmpl_behav_mid.npy",   sonar_encode(template_behav["mid"]))
np.save("vectors/v_tmpl_behav_low.npy",   sonar_encode(template_behav["low"]))

# Encode and save all 6 anchors
np.save("vectors/v_anch_tech_hi.npy",     sonar_encode(anchor_tech_hi))
np.save("vectors/v_anch_tech_lo.npy",     sonar_encode(anchor_tech_lo))
np.save("vectors/v_anch_career_hi.npy",   sonar_encode(anchor_career_hi))
np.save("vectors/v_anch_career_lo.npy",   sonar_encode(anchor_career_lo))
np.save("vectors/v_anch_behav_hi.npy",    sonar_encode(anchor_behav_hi))
np.save("vectors/v_anch_behav_lo.npy",    sonar_encode(anchor_behav_lo))

print("Precomputation complete. 15 vectors saved to /vectors/")
print("All vectors shape: (1024,)")
```

---

## run_experiment.py — Full Pipeline

### Step 1 — Load models and precomputed vectors

```python
# Load SONAR encoder and decoder
encoder = TextToEmbeddingModelPipeline(
    encoder="text_sonar_basic_encoder",
    tokenizer="text_sonar_basic_encoder"
)

decoder = EmbeddingToTextModelPipeline(
    decoder="text_sonar_basic_decoder",
    tokenizer="text_sonar_basic_encoder"
)

def sonar_encode(text):
    embedding = encoder.predict([text], source_lang=SOURCE_LANG)
    return embedding[0].numpy()    # shape: (1024,)

def sonar_decode(vector_1024d):
    """
    vector_1024d : numpy array of shape (1024,)
    Returns      : decoded string
    """
    t = torch.tensor(vector_1024d, dtype=torch.float32).unsqueeze(0)
    # shape: (1, 1024) — batch size 1
    result = decoder.predict(
        t,
        target_lang=TARGET_LANG,
        max_seq_len=MAX_SEQ_LEN
    )
    return result[0]    # returns list of strings, take first

# Load precomputed vectors
v_tmpl_tech   = np.load(f"vectors/v_tmpl_tech_{bucket_tech}.npy")
v_tmpl_career = np.load(f"vectors/v_tmpl_career_{bucket_career}.npy")
v_tmpl_behav  = np.load(f"vectors/v_tmpl_behav_{bucket_behav}.npy")

v_anch_tech_hi   = np.load("vectors/v_anch_tech_hi.npy")
v_anch_tech_lo   = np.load("vectors/v_anch_tech_lo.npy")
v_anch_career_hi = np.load("vectors/v_anch_career_hi.npy")
v_anch_career_lo = np.load("vectors/v_anch_career_lo.npy")
v_anch_behav_hi  = np.load("vectors/v_anch_behav_hi.npy")
v_anch_behav_lo  = np.load("vectors/v_anch_behav_lo.npy")
```

### Step 2 — Encode candidate resume sections

```python
v_cand_tech   = sonar_encode(resume_tech_text)
v_cand_career = sonar_encode(resume_career_text)
v_cand_behav  = sonar_encode(resume_behav_text)

# All shapes: (1024,)
# Three independent vectors. Never combined with each other.
```

### Step 3 — Blend template with candidate per dimension

```python
# GAMMA = 0.55 (template weight), 1 - GAMMA = 0.45 (candidate weight)

v_base_tech   = GAMMA * v_tmpl_tech   + (1 - GAMMA) * v_cand_tech
v_base_career = GAMMA * v_tmpl_career + (1 - GAMMA) * v_cand_career
v_base_behav  = GAMMA * v_tmpl_behav  + (1 - GAMMA) * v_cand_behav

# Result: 55% reviewer tone from template + 45% candidate specifics
# All shapes: (1024,)
```

### Step 4 — Anchor interpolation per dimension

```python
# Score determines position between the two anchor poles.
# s=1.0 → pure high anchor. s=0.0 → pure low anchor. s=0.5 → midpoint.
# This is steering, not amplification.

v_steer_tech   = (1 - s_tech)   * v_anch_tech_lo   + s_tech   * v_anch_tech_hi
v_steer_career = (1 - s_career) * v_anch_career_lo + s_career * v_anch_career_hi
v_steer_behav  = (1 - s_behav)  * v_anch_behav_lo  + s_behav  * v_anch_behav_hi

# For this test case:
#   v_steer_tech   = 0.12 * v_anch_tech_lo   + 0.88 * v_anch_tech_hi
#   v_steer_career = 0.19 * v_anch_career_lo + 0.81 * v_anch_career_hi
#   v_steer_behav  = 0.81 * v_anch_behav_lo  + 0.19 * v_anch_behav_hi

# Three vectors. Fully independent. NEVER added together.
# All shapes: (1024,)
```

### Step 5 — Apply steer to base per dimension

```python
# DELTA = 0.30 (steer influence). 1 - DELTA = 0.70 (base weight).

v_final_tech   = (1 - DELTA) * v_base_tech   + DELTA * v_steer_tech
v_final_career = (1 - DELTA) * v_base_career + DELTA * v_steer_career
v_final_behav  = (1 - DELTA) * v_base_behav  + DELTA * v_steer_behav

# Composition of each final vector:
#   0.70 × 0.55 = 0.385  ← template tone
#   0.70 × 0.45 = 0.315  ← candidate content
#   0.30        = 0.300  ← anchor steering
# All shapes: (1024,)
```

### Step 6 — Three independent SONAR decodes

```python
clause_tech   = sonar_decode(v_final_tech)
clause_career = sonar_decode(v_final_career)
clause_behav  = sonar_decode(v_final_behav)

# Each decode is one call to EmbeddingToTextModelPipeline.predict()
# Beam search with beam_size=5 (SONAR default)
# No sampling. No temperature. No top_p.
# SONAR decode is deterministic beam search — no hallucination risk.
```

### Step 7 — Concatenate and save

```python
reasoning = (
    clause_tech.strip().rstrip(".")   + ". " +
    clause_career.strip().rstrip(".") + ". " +
    clause_behav.strip().rstrip(".")  + "."
)

print("=== CLAUSE TECH ===")
print(clause_tech)
print()
print("=== CLAUSE CAREER ===")
print(clause_career)
print()
print("=== CLAUSE BEHAV ===")
print(clause_behav)
print()
print("=== FINAL REASONING ===")
print(reasoning)

os.makedirs("results", exist_ok=True)
output = {
    "scores":  {"s_tech": s_tech, "s_career": s_career, "s_behav": s_behav},
    "buckets": {
        "bucket_tech":   bucket_tech,
        "bucket_career": bucket_career,
        "bucket_behav":  bucket_behav
    },
    "clauses": {
        "clause_tech":   clause_tech,
        "clause_career": clause_career,
        "clause_behav":  clause_behav
    },
    "reasoning": reasoning
}
with open("results/output.json", "w") as f:
    json.dump(output, f, indent=2)

print("Saved to results/output.json")
```

---

## Full Computation Trace for This Test Case

```
INPUTS:
  s_tech   = 0.88  →  bucket_tech   = "high"
  s_career = 0.81  →  bucket_career = "high"
  s_behav  = 0.19  →  bucket_behav  = "low"

TEMPLATES LOADED:
  v_tmpl_tech   = vectors/v_tmpl_tech_high.npy    shape (1024,)
  v_tmpl_career = vectors/v_tmpl_career_high.npy  shape (1024,)
  v_tmpl_behav  = vectors/v_tmpl_behav_low.npy    shape (1024,)

CANDIDATE ENCODED (SONAR encoder):
  v_cand_tech   = sonar_encode(resume_tech_text)    shape (1024,)
  v_cand_career = sonar_encode(resume_career_text)  shape (1024,)
  v_cand_behav  = sonar_encode(resume_behav_text)   shape (1024,)

BLEND (GAMMA=0.55):
  v_base_tech   = 0.55 * v_tmpl_tech_high   + 0.45 * v_cand_tech
  v_base_career = 0.55 * v_tmpl_career_high + 0.45 * v_cand_career
  v_base_behav  = 0.55 * v_tmpl_behav_low   + 0.45 * v_cand_behav

ANCHOR INTERPOLATION:
  v_steer_tech   = 0.12 * v_anch_tech_lo   + 0.88 * v_anch_tech_hi
  v_steer_career = 0.19 * v_anch_career_lo + 0.81 * v_anch_career_hi
  v_steer_behav  = 0.81 * v_anch_behav_lo  + 0.19 * v_anch_behav_hi

FINAL VECTORS (DELTA=0.30):
  v_final_tech   = 0.70 * v_base_tech   + 0.30 * v_steer_tech
  v_final_career = 0.70 * v_base_career + 0.30 * v_steer_career
  v_final_behav  = 0.70 * v_base_behav  + 0.30 * v_steer_behav

DECODES (SONAR decoder, beam_size=5):
  clause_tech   = sonar_decode(v_final_tech)
  clause_career = sonar_decode(v_final_career)
  clause_behav  = sonar_decode(v_final_behav)
```

---

## Why SONAR Decode Will Not Hallucinate

BERT + GPT2 hallucinated because GPT2 uses autoregressive sampling —
at each step it samples from a probability distribution and can drift
far from the input vector.

SONAR uses beam search with beam_size=5. Beam search:
- Maintains the 5 most probable sequences at every step
- Always stays grounded to the input vector via cross-attention
- Never drifts into free association

The SONAR decoder cross-attends to the input 1024d vector at every
generation step — not as a temperature hint, but as the actual source
of truth that the decoder is translating. This is how it was trained.
The decode is a translation from vector space to English, not free generation.

---

## Why SONAR Interpolation Works

SONAR's training objective includes:
- Autoencoding loss: encode(text) then decode should recover text
- Cross-lingual translation loss: encode(english) → decode(french)
- MSE similarity loss: similar sentences cluster in the space

This means the space is smooth and continuous. The interpolation:

  v_interpolated = (1 - s) * v_anchor_lo + s * v_anchor_hi

produces a vector that lives on the line between two real text points.
Both endpoints are real text. The decoder was trained on vectors from
this space. The interpolated point decodes to semantically intermediate
text — not to noise, not to hallucination.

This is the fundamental property that BERT + GPT2 did not have.

---

## What the Coding Agent Must Build

```
precompute.py:
  1. Paste all template and anchor strings exactly as declared above
  2. Load SONAR encoder (TextToEmbeddingModelPipeline)
  3. Implement sonar_encode() exactly as written above
  4. Encode and save all 15 vectors to vectors/
  5. Print confirmation with shapes

run_experiment.py:
  1. Paste all hardcoded inputs, scores, hyperparameters exactly as above
  2. Paste all template strings (needed for bucket lookup display only)
  3. Implement bucket() exactly as written above
  4. Load SONAR encoder and decoder
  5. Implement sonar_encode() and sonar_decode() exactly as written above
  6. Load 15 precomputed vectors
  7. Execute Steps 2 through 7 exactly as written above
  8. Print and save output

BEFORE RUNNING:
  1. pip install torch==2.0.1 torchaudio==2.0.2
  2. pip install fairseq2==0.2.1 --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.0.1/cpu/
  3. pip install sonar-space==0.2.1 numpy==1.24.0
  4. python precompute.py    (downloads models on first run, ~2.4GB total)
  5. python run_experiment.py
```

---

## Evaluation After the Run

```
1. TONE CHECK
   Does output sound like a human reviewer, not a resume bullet?
   Expected: yes. Templates (reviewer-written) contribute 38.5%.

2. SPECIFICITY CHECK
   Does output mention FAISS, Meesho, latency, Senior MLE, or 18 months?
   Expected: partially. Candidate content contributes 31.5%.
   SONAR's decode may paraphrase rather than reproduce exact terms.
   That is acceptable and actually desirable — paraphrase = human touch.

3. DIRECTION CHECK
   Does clause_tech sound positive? (s_tech=0.88 → 88% toward hi anchor)
   Does clause_behav sound like a concern? (s_behav=0.19 → 81% toward lo anchor)
   Expected: yes. This is the core thing SONAR interpolation should achieve.

4. INDEPENDENCE CHECK
   Do three clauses cover three different aspects without repeating?
   Expected: yes. Three separate encode → steer → decode pipelines.

5. FLUENCY CHECK (new — specific to SONAR)
   Is the output grammatically correct with no repeated or broken words?
   Expected: yes. Beam search is deterministic and stable.
   If output has repeated phrases, increase MAX_SEQ_LEN.
```

---

## Known Limitation

SONAR was trained as a translation and autoencoding system, not as a
candidate reasoning generator. Its decoder vocabulary and style leans
toward clean factual sentences rather than recruiting-specific language.

The output will be fluent and semantically accurate but may not sound
exactly like a recruiter wrote it. If the direction check passes —
meaning high scores produce stronger-sounding text and low scores
produce concern-sounding text — the theory is validated regardless
of whether the surface style is perfect.

Surface style refinement is a separate problem solved after the
theory is confirmed.
