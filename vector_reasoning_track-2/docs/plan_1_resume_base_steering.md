# Plan 1 — Resume Base with Multi-Dimensional Steering

## Core Idea

The candidate's own resume text is the starting point. Three independent reasoning dimensions (technical relevance, career depth, behavioral availability) each have their own anchor pair. The corresponding score steers between those anchors. Three separate decodes produce three clauses. The clauses concatenate into the final reasoning.

No template. No reviewer tone injected upfront. The output tone emerges from how far each steered vector lands from the candidate's resume base.

---

## What Goes In

**From the candidate:**

```
candidate_id
relevant_resume_section_text   (selected by FAISS similarity to JD query vector)
```

**From the scoring pipeline (stages 2, 3, 4):**

```
s_tech   ∈ [0, 1]    technical fit score (from FAISS dot product with Q2 vector)
s_career ∈ [0, 1]    career depth score (from your stage 2/3 model)
s_behav  ∈ [0, 1]    behavioral signal score (computed from 23 Redrob signals)
```

---

## What Comes Out

A single reasoning string composed of three clauses:

```
reasoning = clause_tech + " " + clause_career + " " + clause_behav
```

Each clause is one GPT2 decode driven by one steered vector.

---

## The Anchor Library

Built once offline. Encoded with BERT. Stored as numpy arrays. Six anchor vectors total.

### Dimension 1 — Technical Relevance

```
anchor_tech_high:
"extensive hands-on production experience in retrieval systems, 
vector search, embedding pipelines, and ranking infrastructure 
directly matching the technical requirements of the role"

anchor_tech_low:
"background primarily in adjacent areas with limited direct 
exposure to retrieval, ranking, or vector database systems 
in production environments"
```

### Dimension 2 — Career Depth and Ownership

```
anchor_career_high:
"clear trajectory of senior engineering ownership, end-to-end 
system design responsibility, and demonstrated measurable impact 
across multiple years at product companies"

anchor_career_low:
"early career profile or primarily supporting role without 
demonstrated independent system ownership or technical decision 
making authority"
```

### Dimension 3 — Behavioral Availability

```
anchor_behav_high:
"actively engaged on the platform with consistent recent activity, 
responsive to recruiter outreach, clearly available and currently 
in the job market"

anchor_behav_low:
"minimal platform activity with low responsiveness rate, 
availability uncertain, likely not actively pursuing new 
opportunities right now"
```

---

## The Formulas

### Step 1 — Encode candidate resume section

```
v_candidate = BERT_encode(resume_section_text)         shape: (768,)
```

### Step 2 — Encode anchors (one time, offline)

```
v_tech_hi   = BERT_encode(anchor_tech_high)
v_tech_lo   = BERT_encode(anchor_tech_low)
v_career_hi = BERT_encode(anchor_career_high)
v_career_lo = BERT_encode(anchor_career_low)
v_behav_hi  = BERT_encode(anchor_behav_high)
v_behav_lo  = BERT_encode(anchor_behav_low)
```

### Step 3 — Anchor interpolation per dimension

This is steering, not amplification. The score determines **where between the two anchors** the steered vector lands.

```
v_steer_tech   = (1 - s_tech)   × v_tech_lo   + s_tech   × v_tech_hi
v_steer_career = (1 - s_career) × v_career_lo + s_career × v_career_hi
v_steer_behav  = (1 - s_behav)  × v_behav_lo  + s_behav  × v_behav_hi
```

At score 1.0 the steered vector is exactly the positive anchor.
At score 0.0 the steered vector is exactly the negative anchor.
At score 0.5 it sits at the midpoint.

### Step 4 — Blend candidate base with each steered vector

The candidate's resume vector provides specificity. The steered vector provides the directional tone. Blend them per dimension.

```
β = 0.55    (candidate weight; 0.55 means slightly more candidate than steer)

v_final_tech   = β × v_candidate + (1 - β) × v_steer_tech
v_final_career = β × v_candidate + (1 - β) × v_steer_career
v_final_behav  = β × v_candidate + (1 - β) × v_steer_behav
```

Three final vectors. Each is a slight pull on the candidate's base in a different direction.

### Step 5 — Three decodes, three clauses

```
clause_tech   = GPT2_decode(v_final_tech,   max_length=20, temperature=0.85, top_p=0.92)
clause_career = GPT2_decode(v_final_career, max_length=20, temperature=0.85, top_p=0.92)
clause_behav  = GPT2_decode(v_final_behav,  max_length=20, temperature=0.85, top_p=0.92)
```

### Step 6 — Concatenate

```
reasoning = clause_tech.strip() + ". " + clause_career.strip() + ". " + clause_behav.strip() + "."
```

---

## Worked Example

**Candidate resume section:**

```
"Built two-stage FAISS retrieval pipeline at Meesho for 50M product 
catalog. P99 latency reduced from 340ms to 22ms. End-to-end ownership 
of ML infrastructure. Promoted to Senior MLE in 18 months."
```

**Scores:**

```
s_tech   = 0.88
s_career = 0.81
s_behav  = 0.19
```

**After interpolation:**

- `v_steer_tech` sits at 88% of the way from low to high tech anchor — strongly near "extensive production experience"
- `v_steer_career` sits at 81% toward high career anchor — strongly near "senior ownership trajectory"
- `v_steer_behav` sits at 19% toward high behavioral anchor — strongly near "minimal platform activity"

**After blending with v_candidate at β=0.55:**

- v_final_tech retains 55% of the Meesho/FAISS specifics, pulled 45% toward strong-tech language
- v_final_career retains 55% of the Senior MLE specifics, pulled 45% toward senior-ownership language
- v_final_behav retains 55% of the candidate base, pulled 45% toward availability-concern language

**Expected decoder output (directional):**

```
clause_tech:    "Production retrieval engineering at scale with measurable 
                 latency outcomes in a real product environment"

clause_career:  "Senior ownership of ML infrastructure with promotion 
                 trajectory and demonstrated end-to-end responsibility"

clause_behav:   "Platform engagement is limited with low recent activity 
                 and uncertain availability for outreach"

final reasoning:
"Production retrieval engineering at scale with measurable latency 
outcomes in a real product environment. Senior ownership of ML 
infrastructure with promotion trajectory and demonstrated end-to-end 
responsibility. Platform engagement is limited with low recent 
activity and uncertain availability for outreach."
```

---

## Hyperparameters to Tune

| Parameter | Default | Range | What it controls |
|---|---|---|---|
| β (candidate blend weight) | 0.55 | 0.40 to 0.70 | Higher β = more candidate specifics, less anchor influence |
| temperature | 0.85 | 0.70 to 1.00 | Higher = more variation, less coherence |
| top_p | 0.92 | 0.85 to 0.95 | Nucleus sampling cutoff |
| max_length per clause | 20 | 15 to 30 | Length of each clause |

---

## Experiment Matrix

Use one synthetic candidate. Vary scores across three profiles. Three β values. Compare 9 outputs total.

| Test | s_tech | s_career | s_behav | β | Expected direction |
|---|---|---|---|---|---|
| 1A | 0.88 | 0.81 | 0.65 | 0.55 | All three clauses positive |
| 1B | 0.88 | 0.81 | 0.19 | 0.55 | Two positive, one concern |
| 1C | 0.30 | 0.40 | 0.65 | 0.55 | Two concerns, one positive |
| 2A | 0.88 | 0.81 | 0.65 | 0.40 | More anchor language, less candidate detail |
| 2B | 0.88 | 0.81 | 0.65 | 0.70 | More candidate detail, weaker steering |
| 3A | 0.50 | 0.50 | 0.50 | 0.55 | All clauses neutral midpoint |

---

## Failure Modes to Watch For

**Resume tone bleeding through**

If clauses sound like resume bullets ("Built X. Led Y. Did Z.") rather than reasoning, the candidate vector is dominating. Reduce β.

**Anchor language drowning content**

If clauses sound generic ("strong candidate with relevant background") without mentioning anything specific to this candidate, the anchor is dominating. Increase β.

**Three clauses contradicting each other**

If clause_tech praises and clause_behav damns and they don't read as coherent reasoning together, the concatenation isn't enough. May need a transition word library or post-processing connector logic.

**Clauses repeating each other**

If all three clauses say roughly the same thing, the anchor directions aren't sufficiently distinct in BERT's space. The anchors need to be rewritten to use more distinct vocabulary.

---

## What This Plan Tests

1. Does anchor interpolation produce smooth, score-aligned tone shifts?
2. Does the candidate base vector survive at β=0.55, or does it get washed out?
3. Do three independent decodes produce three meaningfully different clauses?
4. Does the concatenated output read as natural reasoning, or does it read as three disconnected sentences?
