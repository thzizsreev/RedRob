# Plan 2 — Reasoning Generation via Template Base with Dimensional Steering
## Final Experiment Plan — Option B Decoder (LangVAE Latent Injection)

---

## What Changed From Previous Version

The only change from the previous plan is Step 4 — the decode step.

Everything else is identical:
- All template strings — unchanged
- All anchor strings — unchanged
- All hyperparameters — unchanged
- All hardcoded inputs and scores — unchanged
- Steps 3A through 3E (vector math) — unchanged

Previous decode approach (BROKEN):
  GPT2 received the template text as a prompt and continued it freely.
  The vector was not actually conditioning the generation.
  Output drifted into Wikipedia-style hallucination.

New decode approach (Option B — LangVAE):
  LangVAE couples a BERT encoder with a GPT2 decoder via a VAE latent space.
  Memory injection and embedding injection pass the latent into GPT2 at every step.
  Pretrained checkpoints are available on HuggingFace right now.
  Installable via pip. No manual download required.

---

## Environment and Dependencies

### Python version
```
Python 3.9 or 3.10
```

### Required libraries
```
torch==2.0.1
transformers==4.35.0
numpy==1.24.0
langvae==0.3.0
pythae==0.1.1
saf_datasets
```

### Install command
```bash
pip install torch==2.0.1 transformers==4.35.0 numpy==1.24.0
pip install langvae==0.3.0 pythae==0.1.1
pip install git+https://github.com/neuro-symbolic-ai/saf_datasets.git
```

### Models required

#### Encoder — same as before
```
bert-base-uncased
from transformers import BertTokenizer, BertModel
Downloaded automatically from HuggingFace on first run.
Used to encode all text → 768d vectors.
```

#### Decoder — LangVAE
```
LangVAE couples a BERT encoder with a GPT2 decoder via a VAE latent space.

Two injection channels:

  Memory injection:
    The latent vector is projected to GPT2's hidden size and appended
    to the key-value memory at every transformer layer.
    GPT2 attends to this vector at every generation step.

  Embedding injection:
    The latent vector is projected and added to the token embedding
    at each decoding step before the token enters the transformer.

Pretrained checkpoint used:
  neuro-symbolic-ai/eb-langvae-bert-base-cased-gpt2-l128

  Available on HuggingFace. Downloads automatically on first run.
  Trained on EntailmentBank — explanation and reasoning sentences.
  This training domain is closer to candidate reasoning than Wikipedia.

  Encoder: BERT-base-cased
  Decoder: GPT2-base
  Latent dimension: 128

IMPORTANT — DIMENSION MISMATCH:
  Our vectors are 768d (from BERT mean pooling).
  LangVAE's latent space is 128d.
  A linear projection layer is required before decode.
  nn.Linear(768, 128) with random initialisation.
  This is intentional for the experiment — we are testing whether
  the decode pipeline runs and produces fluent output.
  A trained projector is the next step after this experiment.
```

### LangVAE GitHub repository
```
https://github.com/neuro-symbolic-ai/LangVAE

No cloning required. LangVAE is installed via pip.
The model loads directly from HuggingFace Hub.
```

### Directory structure
```
experiment/
├── precompute.py
├── run_experiment.py
├── vectors/          ← precomputed .npy files saved here
└── results/          ← output.json saved here

NOTE: LangVAE downloads automatically from HuggingFace on first run
      and caches locally in the HuggingFace cache (~/.cache/huggingface/).
```

---

## All Hardcoded Input Values
## (UNCHANGED from previous plan)

### Candidate resume sections

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

### Scores

```python
s_tech   = 0.88
s_career = 0.81
s_behav  = 0.19
```

### Hyperparameters

```python
GAMMA       = 0.60
DELTA       = 0.25
MAX_LENGTH  = 30
TEMPERATURE = 0.85
TOP_P       = 0.92
DO_SAMPLE   = True
```

---

## Template Library
## (UNCHANGED from previous plan)

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

## Anchor Pairs
## (UNCHANGED from previous plan)

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

anchor_career_hi = (
    "senior individual contributor with full ownership of complex ML systems, "
    "clear promotion trajectory, and cross-functional impact at product companies"
)
anchor_career_lo = (
    "junior or mid-level engineer in a supporting capacity with limited "
    "independent scope and no clear ownership of significant systems"
)

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

## Step 1 — Score Bucketing
## (UNCHANGED)

```python
def bucket(score):
    if score >= 0.65:
        return "high"
    elif score >= 0.35:
        return "mid"
    else:
        return "low"

bucket_tech   = bucket(s_tech)    # → "high"
bucket_career = bucket(s_career)  # → "high"
bucket_behav  = bucket(s_behav)   # → "low"
```

---

## Step 2 — Offline Precomputation
## (UNCHANGED — precompute.py)

```python
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import os

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model     = BertModel.from_pretrained("bert-base-uncased")
model.eval()

def bert_encode(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return vector    # shape: (768,)

os.makedirs("vectors", exist_ok=True)

np.save("vectors/v_tmpl_tech_high.npy",   bert_encode(template_tech["high"]))
np.save("vectors/v_tmpl_tech_mid.npy",    bert_encode(template_tech["mid"]))
np.save("vectors/v_tmpl_tech_low.npy",    bert_encode(template_tech["low"]))
np.save("vectors/v_tmpl_career_high.npy", bert_encode(template_career["high"]))
np.save("vectors/v_tmpl_career_mid.npy",  bert_encode(template_career["mid"]))
np.save("vectors/v_tmpl_career_low.npy",  bert_encode(template_career["low"]))
np.save("vectors/v_tmpl_behav_high.npy",  bert_encode(template_behav["high"]))
np.save("vectors/v_tmpl_behav_mid.npy",   bert_encode(template_behav["mid"]))
np.save("vectors/v_tmpl_behav_low.npy",   bert_encode(template_behav["low"]))
np.save("vectors/v_anch_tech_hi.npy",     bert_encode(anchor_tech_hi))
np.save("vectors/v_anch_tech_lo.npy",     bert_encode(anchor_tech_lo))
np.save("vectors/v_anch_career_hi.npy",   bert_encode(anchor_career_hi))
np.save("vectors/v_anch_career_lo.npy",   bert_encode(anchor_career_lo))
np.save("vectors/v_anch_behav_hi.npy",    bert_encode(anchor_behav_hi))
np.save("vectors/v_anch_behav_lo.npy",    bert_encode(anchor_behav_lo))

print("Precomputation complete. 15 vectors saved to /vectors/")
```

---

## Steps 3A through 3E — Vector Math
## (UNCHANGED)

```python
# 3A — encode candidate sections
v_cand_tech   = bert_encode(resume_tech_text)
v_cand_career = bert_encode(resume_career_text)
v_cand_behav  = bert_encode(resume_behav_text)

# 3B — load template vectors by bucket
v_tmpl_tech   = np.load(f"vectors/v_tmpl_tech_{bucket_tech}.npy")
v_tmpl_career = np.load(f"vectors/v_tmpl_career_{bucket_career}.npy")
v_tmpl_behav  = np.load(f"vectors/v_tmpl_behav_{bucket_behav}.npy")

# load anchors
v_anch_tech_hi   = np.load("vectors/v_anch_tech_hi.npy")
v_anch_tech_lo   = np.load("vectors/v_anch_tech_lo.npy")
v_anch_career_hi = np.load("vectors/v_anch_career_hi.npy")
v_anch_career_lo = np.load("vectors/v_anch_career_lo.npy")
v_anch_behav_hi  = np.load("vectors/v_anch_behav_hi.npy")
v_anch_behav_lo  = np.load("vectors/v_anch_behav_lo.npy")

# 3C — blend template with candidate
v_base_tech   = GAMMA * v_tmpl_tech   + (1 - GAMMA) * v_cand_tech
v_base_career = GAMMA * v_tmpl_career + (1 - GAMMA) * v_cand_career
v_base_behav  = GAMMA * v_tmpl_behav  + (1 - GAMMA) * v_cand_behav

# 3D — anchor interpolation
v_steer_tech   = (1 - s_tech)   * v_anch_tech_lo   + s_tech   * v_anch_tech_hi
v_steer_career = (1 - s_career) * v_anch_career_lo + s_career * v_anch_career_hi
v_steer_behav  = (1 - s_behav)  * v_anch_behav_lo  + s_behav  * v_anch_behav_hi

# 3E — apply steer to base
v_final_tech   = (1 - DELTA) * v_base_tech   + DELTA * v_steer_tech
v_final_career = (1 - DELTA) * v_base_career + DELTA * v_steer_career
v_final_behav  = (1 - DELTA) * v_base_behav  + DELTA * v_steer_behav
```

---

## Step 4 — LangVAE Decoder

### What the coding agent must import

```python
import torch
import torch.nn as nn
import numpy as np
from langvae import LangVAE
```

### How to load LangVAE

```python
langvae_model = LangVAE.load_from_hf_hub(
    "neuro-symbolic-ai/eb-langvae-bert-base-cased-gpt2-l128"
)
langvae_model.eval()
langvae_model.decoder.init_pretrained_model()
```

This downloads and loads:
- BERT-base-cased encoder (inside LangVAE — not used for our encoding)
- GPT2-base decoder with VAE injection (used for our decoding)
- Latent dimension: 128

### The projection layer

Our vectors are 768d. LangVAE's latent space is 128d.
A linear projection maps our vectors into LangVAE's space.

```python
# Random initialisation — intentional for this experiment
# Testing pipeline functionality and output fluency first
# A trained projector is the next step after this experiment succeeds
torch.manual_seed(42)    # fixed seed for reproducibility
projector = nn.Linear(768, 128, bias=False)
projector.eval()
```

### The decode function

```python
def langvae_decode(vector_768d):
    """
    vector_768d : numpy array of shape (768,)
                  this is v_final_tech, v_final_career, or v_final_behav

    Returns     : decoded string of max MAX_LENGTH tokens
    """
    # Convert to torch tensor — shape (1, 768)
    z_768 = torch.tensor(vector_768d, dtype=torch.float32).unsqueeze(0)

    # Project to LangVAE's 128d latent space — shape (1, 128)
    with torch.no_grad():
        z_128 = projector(z_768)

    # Decode using LangVAE's GPT2 decoder
    # The decoder's generate method accepts the latent vector directly
    # and injects it into GPT2 via memory and embedding at every step
    with torch.no_grad():
        output_ids = langvae_model.decoder.generate(
            z          = z_128,
            max_length = MAX_LENGTH,
            temperature= TEMPERATURE,
            top_p      = TOP_P,
            do_sample  = DO_SAMPLE
        )

    decoded = langvae_model.decoder.tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )
    return decoded.strip()
```

### Run the three decodes

```python
clause_tech   = langvae_decode(v_final_tech)
clause_career = langvae_decode(v_final_career)
clause_behav  = langvae_decode(v_final_behav)
```

Each decode is fully independent. No shared state between the three calls.

---

## Step 5 — Concatenation and Output
## (UNCHANGED)

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

import json, os
os.makedirs("results", exist_ok=True)

output = {
    "scores":  {"s_tech": s_tech, "s_career": s_career, "s_behav": s_behav},
    "buckets": {"bucket_tech": bucket_tech, "bucket_career": bucket_career,
                "bucket_behav": bucket_behav},
    "clauses": {"clause_tech": clause_tech, "clause_career": clause_career,
                "clause_behav": clause_behav},
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

VECTOR MATH (unchanged):
  v_base_tech   = 0.60 * v_tmpl_tech_high   + 0.40 * v_cand_tech
  v_base_career = 0.60 * v_tmpl_career_high + 0.40 * v_cand_career
  v_base_behav  = 0.60 * v_tmpl_behav_low   + 0.40 * v_cand_behav

  v_steer_tech   = 0.12 * v_anch_tech_lo   + 0.88 * v_anch_tech_hi
  v_steer_career = 0.19 * v_anch_career_lo + 0.81 * v_anch_career_hi
  v_steer_behav  = 0.81 * v_anch_behav_lo  + 0.19 * v_anch_behav_hi

  v_final_tech   = 0.75 * v_base_tech   + 0.25 * v_steer_tech
  v_final_career = 0.75 * v_base_career + 0.25 * v_steer_career
  v_final_behav  = 0.75 * v_base_behav  + 0.25 * v_steer_behav

DECODES (LangVAE):
  clause_tech   = langvae_decode(v_final_tech)
  clause_career = langvae_decode(v_final_career)
  clause_behav  = langvae_decode(v_final_behav)

  Each decode:
    1. Projects v_final (768d) → z_128 (128d) via nn.Linear(768, 128)
    2. Passes z_128 to LangVAE GPT2 decoder
    3. GPT2 injects z_128 into memory and embedding at every token step
    4. Returns decoded string
```

---

## What the Coding Agent Must Build

```
precompute.py
  — identical to previous plan
  — no changes

run_experiment.py
  — keep all hardcoded inputs, scores, hyperparameters identical
  — keep Steps 3A through 3E vector math identical
  — REMOVE: all GPT2 direct continuation code from previous plan
  — REMOVE: decode_vector_to_text() function
  — REMOVE: any legacy decoder imports or sys.path hacks
  — ADD: from langvae import LangVAE
  — ADD: load langvae_model from HuggingFace hub as written above
  — ADD: projector = nn.Linear(768, 128, bias=False) with seed 42
  — ADD: langvae_decode() function exactly as written above
  — REPLACE the three decode calls with langvae_decode() calls
  — keep Step 5 concatenation and output identical

BEFORE RUNNING:
  1. pip install langvae pythae
     pip install git+https://github.com/neuro-symbolic-ai/saf_datasets.git
  2. Run precompute.py first
  3. Run run_experiment.py
  4. LangVAE checkpoint downloads automatically on first run
     (~800MB, cached in ~/.cache/huggingface/)

LangVAE downloads from HuggingFace on first run. No manual checkpoint management.
```

---

## Known Risk

```
PROJECTION IS RANDOM:
  nn.Linear(768, 128) is randomly initialised with seed 42.
  This means our 768d vectors are projected into LangVAE's 128d space
  via a random rotation. The projection is not semantically meaningful.
  The decoder will produce fluent English but the content will not
  reflect our anchor steering in a precise way.

  This is intentional and acceptable for this experiment.
  We are testing two things:
    1. Does the pipeline run end to end without errors?
    2. Does LangVAE produce fluent English from arbitrary 128d vectors?

  If both pass, the next step is training the projector on
  (our 768d vectors, target reasoning text) pairs so the projection
  becomes semantically meaningful.

TRAINING DOMAIN GAP:
  LangVAE was trained on EntailmentBank — explanation sentences.
  Our output domain is candidate reasoning sentences.
  Output may sound like explanations rather than recruiter reasoning.
  This is a known limitation at this stage.
```

---

## Evaluation After the Run

```
1. TONE CHECK
   Does the output sound like a human reviewer, not a resume bullet?

2. SPECIFICITY CHECK
   Does the output reference anything from the candidate's resume
   (FAISS, Meesho, latency, Senior MLE, 18 months)?

3. DIRECTION CHECK
   Does clause_tech sound positive? (s_tech=0.88 → high anchor)
   Does clause_behav sound like a concern? (s_behav=0.19 → low anchor)

4. INDEPENDENCE CHECK
   Do the three clauses cover three different aspects without repeating?
```
