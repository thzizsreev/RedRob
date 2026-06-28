# Plan 2 — Reasoning Generation via Template Base with Dimensional Steering
## Final Experiment Plan — Complete, No Gaps

---

## Architecture Summary

Three reasoning dimensions — Technical, Career, Behavioral — are treated as
completely independent. They live in different semantic spaces and are never
combined into one vector at any stage.

Each dimension follows the same pipeline:
1. Select a pre-written reviewer-tone template by score bucket
2. Encode the candidate's relevant resume section
3. Blend template vector + candidate vector
4. Steer the blended vector using anchor interpolation driven by the score
5. Decode the final vector into one clause

Three clauses concatenated = final reasoning string.

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
```

### Install command
```bash
pip install torch==2.0.1 transformers==4.35.0 numpy==1.24.0
```

### Models required
Both models are downloaded automatically from HuggingFace on first run.
No manual download needed.

```
Encoder : bert-base-uncased
          from transformers import BertTokenizer, BertModel
          used to encode all text → vectors

Decoder : gpt2
          from transformers import GPT2Tokenizer, GPT2LMHeadModel
          used to decode final vectors → text

NOTE: Implementation uses LangVAE's native 128d μ latent space for encoding
      and vector math (bert-base-cased inside LangVAE). Decode has two modes:
      --decode template_hybrid (default, reliable) or --decode langvae
      (requires validate_langvae.py gate on Python 3.11).
```

### Directory structure the coding agent must create
```
experiment/
├── precompute.py
├── run_experiment.py
├── compose.py            ← template_hybrid decode
├── validate_langvae.py   ← LangVAE environment gate
├── vectors/          ← precomputed .npy files saved here (128d)
└── results/          ← output.json saved here
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

These are the three dimension scores. They are NOT computed from anything
in this experiment. They are fixed values representing a strong tech
candidate with weak behavioral engagement.

```python
s_tech   = 0.88    # high technical fit
s_career = 0.81    # high career depth
s_behav  = 0.19    # low behavioral engagement
```

### Why these values are sufficient for the test

The experiment tests whether:
  - The steering math correctly moves vectors
  - The decode produces different output for different score levels
  - The reviewer tone is preserved from the template

For these tests, hardcoded scores are sufficient.
Wiring scores to the actual ranking pipeline is out of scope for this experiment.

---

## Template Library (hardcoded in precompute.py)

The coding agent encodes these exact strings. No changes to wording allowed.

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

The coding agent encodes these exact strings. No changes to wording allowed.

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

## Fixed Hyperparameters (hardcoded in run_experiment.py)

The coding agent uses these values exactly. No other values. No tuning during coding.

```python
GAMMA = 0.60       # template weight in blend. candidate weight = 1 - GAMMA = 0.40
DELTA = 0.25       # steer weight applied to blended base. base weight = 1 - DELTA = 0.75

MAX_LENGTH  = 30   # max tokens per clause during GPT2 decode
TEMPERATURE = 0.85 # GPT2 sampling temperature
TOP_P       = 0.92 # nucleus sampling cutoff
DO_SAMPLE   = True # must be True for temperature and top_p to apply
```

---

## Step 1 — Score Bucketing

```python
def bucket(score):
    if score >= 0.65:
        return "high"
    elif score >= 0.35:
        return "mid"
    else:
        return "low"

bucket_tech   = bucket(s_tech)    # → "high"  (s_tech=0.88)
bucket_career = bucket(s_career)  # → "high"  (s_career=0.81)
bucket_behav  = bucket(s_behav)   # → "low"   (s_behav=0.19)
```

---

## Step 2 — Offline Precomputation (precompute.py)

Run once before the experiment. Saves 15 vectors as .npy files to /vectors/.

### How to encode text to vector with LangVAE (128d μ)

All templates, anchors, and candidate resume sections are encoded via
LangVAE's bert-base-cased encoder into 128d μ latent vectors.
See [`vector_reasoning_test/encode.py`](../vector_reasoning_test/encode.py).

```python
from encode import langvae_encode

vector = langvae_encode(text)  # shape: (128,)
```

Legacy BERT encoding (768d mean pool) is superseded — do not mix spaces.

### What to encode and save

```python
import os
os.makedirs("vectors", exist_ok=True)

# Templates — 9 vectors
np.save("vectors/v_tmpl_tech_high.npy",   bert_encode(template_tech["high"]))
np.save("vectors/v_tmpl_tech_mid.npy",    bert_encode(template_tech["mid"]))
np.save("vectors/v_tmpl_tech_low.npy",    bert_encode(template_tech["low"]))

np.save("vectors/v_tmpl_career_high.npy", bert_encode(template_career["high"]))
np.save("vectors/v_tmpl_career_mid.npy",  bert_encode(template_career["mid"]))
np.save("vectors/v_tmpl_career_low.npy",  bert_encode(template_career["low"]))

np.save("vectors/v_tmpl_behav_high.npy",  bert_encode(template_behav["high"]))
np.save("vectors/v_tmpl_behav_mid.npy",   bert_encode(template_behav["mid"]))
np.save("vectors/v_tmpl_behav_low.npy",   bert_encode(template_behav["low"]))

# Anchors — 6 vectors
np.save("vectors/v_anch_tech_hi.npy",     bert_encode(anchor_tech_hi))
np.save("vectors/v_anch_tech_lo.npy",     bert_encode(anchor_tech_lo))

np.save("vectors/v_anch_career_hi.npy",   bert_encode(anchor_career_hi))
np.save("vectors/v_anch_career_lo.npy",   bert_encode(anchor_career_lo))

np.save("vectors/v_anch_behav_hi.npy",    bert_encode(anchor_behav_hi))
np.save("vectors/v_anch_behav_lo.npy",    bert_encode(anchor_behav_lo))

print("Precomputation complete. 15 vectors saved to /vectors/")
```

---

## Step 3 — Online Per-Candidate Computation (run_experiment.py)

### 3A — Load all precomputed vectors

```python
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

### 3B — Encode candidate resume sections

Using the same bert_encode function from precompute.py.

```python
v_cand_tech   = bert_encode(resume_tech_text)
v_cand_career = bert_encode(resume_career_text)
v_cand_behav  = bert_encode(resume_behav_text)
```

Three independent vectors. Shape (768,) each. Never combined with each other.

### 3C — Blend template with candidate per dimension

```python
v_base_tech   = GAMMA * v_tmpl_tech   + (1 - GAMMA) * v_cand_tech
v_base_career = GAMMA * v_tmpl_career + (1 - GAMMA) * v_cand_career
v_base_behav  = GAMMA * v_tmpl_behav  + (1 - GAMMA) * v_cand_behav
```

With GAMMA=0.60:
- 60% of each base vector comes from the reviewer-tone template
- 40% comes from the candidate's actual resume content

### 3D — Anchor interpolation per dimension

The score determines where between the two anchor poles the steered
vector lands. This is NOT multiplying by the score. It is interpolating
between two fixed endpoints.

```python
v_steer_tech   = (1 - s_tech)   * v_anch_tech_lo   + s_tech   * v_anch_tech_hi
v_steer_career = (1 - s_career) * v_anch_career_lo + s_career * v_anch_career_hi
v_steer_behav  = (1 - s_behav)  * v_anch_behav_lo  + s_behav  * v_anch_behav_hi
```

For this test case:
```
v_steer_tech   = 0.12 * v_anch_tech_lo   + 0.88 * v_anch_tech_hi
v_steer_career = 0.19 * v_anch_career_lo + 0.81 * v_anch_career_hi
v_steer_behav  = 0.81 * v_anch_behav_lo  + 0.19 * v_anch_behav_hi
```

Each interpolation is fully independent.
These three vectors are NEVER added together or combined.

### 3E — Apply steer to base per dimension

```python
v_final_tech   = (1 - DELTA) * v_base_tech   + DELTA * v_steer_tech
v_final_career = (1 - DELTA) * v_base_career + DELTA * v_steer_career
v_final_behav  = (1 - DELTA) * v_base_behav  + DELTA * v_steer_behav
```

With DELTA=0.25, each final vector is composed of:
- 45% template tone        (0.75 × GAMMA      = 0.75 × 0.60)
- 30% candidate content    (0.75 × (1-GAMMA)  = 0.75 × 0.40)
- 25% anchor steering      (DELTA             = 0.25)

---

## Step 4 — Three Independent Decodes

### Implementation (current)

The experiment supports two decode modes via `run_experiment.py --decode`:

**`template_hybrid` (default)** — reliable related output:

```python
from compose import compose_clause

clause_tech   = compose_clause("tech", s_tech)
clause_career = compose_clause("career", s_career)
clause_behav  = compose_clause("behav", s_behav)
```

Each clause appends resume-specific evidence with score-directed tone
(reinforcing for s >= 0.65, concern for s < 0.35).

**`langvae`** — vector-conditioned decode in native 128d latent space:

```python
from encode import langvae_encode   # 128d μ, not bert-base-uncased
from decode import langvae_decode

# Steps 3A–3E run in 128d LangVAE μ space (same GAMMA/DELTA formulas)
clause_tech   = langvae_decode(v_final_tech)
clause_career = langvae_decode(v_final_career)
clause_behav  = langvae_decode(v_final_behav)
```

Requires `validate_langvae.py` to pass on Python 3.11 with pinned deps.
Do **not** use a random 768→128 projector or bert-base-uncased vectors.

### Legacy GPT-2 workaround (superseded)

The original GPT-2 template-continuation approach below did not condition
generation on the steered vector (only L2-norm temperature scaling) and
produced unrelated Wikipedia-style hallucinations. It is retained for
historical reference only.

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model     = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_model.eval()

def decode_vector_to_text(vector, prompt_text):
    """
    For experiment scope:
    Use the prompt_text (the template sentence for this dimension)
    as the GPT2 input, with the vector's L2 norm as a temperature
    scaling signal to vary generation per score level.
    GPT2 generates a continuation of the template sentence.
    """
    inputs = gpt2_tokenizer(
        prompt_text,
        return_tensors="pt"
    )
    # Vector norm used to modulate effective temperature slightly
    # Higher norm (stronger steer) → slightly lower temperature → more focused
    vector_norm = float(np.linalg.norm(vector))
    effective_temp = max(0.70, TEMPERATURE - 0.05 * (vector_norm / 10.0))

    with torch.no_grad():
        output_ids = gpt2_model.generate(
            **inputs,
            max_new_tokens=MAX_LENGTH,
            temperature=effective_temp,
            top_p=TOP_P,
            do_sample=DO_SAMPLE,
            pad_token_id=gpt2_tokenizer.eos_token_id
        )
    full_text = gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Return only the generated continuation, not the prompt
    continuation = full_text[len(prompt_text):].strip()
    return continuation
```

### What prompt_text to use per dimension

```python
prompt_tech   = template_tech[bucket_tech]
prompt_career = template_career[bucket_career]
prompt_behav  = template_behav[bucket_behav]
```

The template for each dimension's bucket is the GPT2 prompt.
GPT2 continues from where the template ends.
The vector influences the generation through the temperature modulation.

### Run the three decodes

```python
clause_tech   = decode_vector_to_text(v_final_tech,   prompt_tech)
clause_career = decode_vector_to_text(v_final_career, prompt_career)
clause_behav  = decode_vector_to_text(v_final_behav,  prompt_behav)
```

---

## Step 5 — Concatenation and Output

```python
reasoning = (
    prompt_tech.rstrip(".") + " " + clause_tech.strip() + ". " +
    prompt_career.rstrip(".") + " " + clause_career.strip() + ". " +
    prompt_behav.rstrip(".") + " " + clause_behav.strip() + "."
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
```

### Save to results

```python
import json, os
os.makedirs("results", exist_ok=True)

output = {
    "scores": {
        "s_tech":   s_tech,
        "s_career": s_career,
        "s_behav":  s_behav
    },
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
  v_tmpl_tech   = v_tmpl_tech_high.npy
  v_tmpl_career = v_tmpl_career_high.npy
  v_tmpl_behav  = v_tmpl_behav_low.npy

CANDIDATE ENCODED:
  v_cand_tech   = bert_encode(resume_tech_text)
  v_cand_career = bert_encode(resume_career_text)
  v_cand_behav  = bert_encode(resume_behav_text)

BLEND (GAMMA=0.60):
  v_base_tech   = 0.60 * v_tmpl_tech_high   + 0.40 * v_cand_tech
  v_base_career = 0.60 * v_tmpl_career_high + 0.40 * v_cand_career
  v_base_behav  = v_tmpl_behav_low * 0.60   + 0.40 * v_cand_behav

ANCHOR INTERPOLATION:
  v_steer_tech   = 0.12 * v_anch_tech_lo   + 0.88 * v_anch_tech_hi
  v_steer_career = 0.19 * v_anch_career_lo + 0.81 * v_anch_career_hi
  v_steer_behav  = 0.81 * v_anch_behav_lo  + 0.19 * v_anch_behav_hi

FINAL VECTORS (DELTA=0.25):
  v_final_tech   = 0.75 * v_base_tech   + 0.25 * v_steer_tech
  v_final_career = 0.75 * v_base_career + 0.25 * v_steer_career
  v_final_behav  = 0.75 * v_base_behav  + 0.25 * v_steer_behav

DECODES:
  clause_tech   = GPT2 continuation of template_tech["high"]
  clause_career = GPT2 continuation of template_career["high"]
  clause_behav  = GPT2 continuation of template_behav["low"]
```

---

## What the Coding Agent Must Build

```
precompute.py
  - Define all template and anchor strings exactly as written above
  - Load bert-base-uncased tokenizer and model
  - Implement bert_encode() exactly as written above
  - Encode all 15 strings and save as .npy files to vectors/
  - Print confirmation when done

run_experiment.py
  - Hardcode all candidate inputs exactly as written above
  - Hardcode all scores exactly as written above
  - Hardcode all hyperparameters exactly as written above
  - Implement bucket() exactly as written above
  - Load all 15 precomputed vectors from vectors/
  - Execute Steps 3A through 3E exactly as written
  - Implement decode_vector_to_text() exactly as written above
  - Run three decodes
  - Print all clauses and final reasoning
  - Save to results/output.json

NO OTHER LOGIC. NO ADDITIONAL PARAMETERS. NO VARIATIONS.
The coding agent implements the math. All values come from this document.
```

---

## Evaluation After the Run

Manually review output.json against these four checks:

```
1. TONE CHECK
   Do the clauses sound like a human reviewer evaluating a candidate,
   or do they sound like resume bullets?
   Expected: reviewer tone. Templates contribute 45% of each final vector.

2. SPECIFICITY CHECK
   Do any clauses mention FAISS, Meesho, latency, Senior MLE, or 18 months?
   Expected: yes, because candidate content contributes 30%.
   If no specifics appear, GAMMA is too high. Note this for tuning.

3. DIRECTION CHECK
   Does clause_tech read positively?
   Does clause_behav read as a concern or limitation?
   Expected: yes. s_tech=0.88 steers 88% toward hi anchor.
             s_behav=0.19 steers 81% toward lo anchor.

4. INDEPENDENCE CHECK
   Do the three clauses cover three different aspects?
   Or do they repeat the same idea?
   Expected: different. Each uses a separate template, candidate text,
   and anchor pair with no shared vectors at any stage.
```
