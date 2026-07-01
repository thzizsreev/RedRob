# Tech Section — Final Plan
## Scoring → Classification → Evidence → Sentence Construction → Paraphrase → Reconstruction

---

## Design Principle

More coverage with stable reasoning over deeper reasoning with information loss.

Every candidate gets one sentence. That sentence covers:
- Who they are and how long they have been building these systems
- Where they built them (full career scope, not just latest role)
- What they built most recently and what outcome it produced
- What verified or named technical depth they bring

Nothing is erased. No second sentence needed. The paraphraser handles variation.

---

## Diagnostic Validation

Signals confirmed by diagnostics. Nothing outside this list is used.

```
USED:
  cross_encoder_score    std=0.924  range=3.88  PRIMARY classification signal
  skill_assessment_scores            evidence for verified technical depth

EXPLICITLY EXCLUDED (confirmed flat):
  skill_score            std=0.000  range=0.000  LOW_VARIANCE — constant
  q3_neg_sim             std=0.008  range=0.046  LOW_VARIANCE — useless
  fused_score            std=0.008  range=0.038  LOW_VARIANCE — useless
  q2_score               range=0.029 in top 100  too compressed to differentiate
  q1_score               range=0.037 in top 100  too compressed to differentiate
```

---

## Input Values Required

```
FROM pipeline:
  cross_encoder_score              float     classification signal
  skill_assessment_scores          dict      {skill_name: float 0-100}
                                             19 candidates have empty dict — handled

FROM candidate JSON:
  candidate_id                     string    seed for deterministic selection
  profile.anonymized_name          string    candidate name
  profile.current_title            string    current job title
  profile.current_company          string    current employer
  career_history[]                 list      ALL roles — not just [0] and [1]
                                             always present, minimum 1 role
  skills[]                         list      [{name, proficiency, endorsements}]
                                             always present per data schema
  pipeline.gates_and_career
    .total_years_exp               float     total years of experience
```

---

## Step 1 — Classification

Single input. Single output label. Nothing else.

Thresholds calibrated to the top 100 pre-filtered pool.
(On the full 300-candidate pool p50=1.85, p75=2.55, p95=3.86.
Top 100 score higher because they are pre-selected.)

```
cross_encoder_score >= 3.5  →  DEEP        30 candidates  (~30%)
cross_encoder_score >= 2.5  →  STRONG      27 candidates  (~27%)
cross_encoder_score >= 1.5  →  MODERATE    28 candidates  (~28%)
cross_encoder_score <  1.5  →  SURFACE     15 candidates  (~15%)
```

The label controls: skeleton shape and strength of language.
All cats now produce exactly one sentence. Two-sentence structure removed.

---

## Step 2 — Evidence Extraction

All extractors run for every candidate. Return values or None.
Never raises on missing data.

---

### 2A — System Type

Source: career_history[0].description only.
Describes what they currently own. Not their oldest role.

Match first pattern that fires.

```
PATTERN                                          LABEL
──────────────────────────────────────────────────────────────────────────────
\branking\b.*\b(layer|pipeline|model|system)\b   "ranking pipeline"
\brecommendation\b.*\bsystem\b                   "recommendation system"
\bsemantic\s+search\b                            "semantic search system"
\bdiscovery\s+feed\b|\branking\s+model\b         "discovery ranking system"
\bML\s+pipeline\b|\bMLflow\b|\bKubeflow\b        "ML infrastructure pipeline"
\bRAG\b|\bretrieval.augmented\b                  "RAG-based retrieval pipeline"
\bwhat\s+users\s+are\s+looking\b                 "relevance matching system"
\bsearch\s+and\s+discovery\b                     "search and discovery system"
\bembedding.based\s+search\b                     "embedding-based search system"
\bmigration\b.*\bsearch\b                        "search migration"

No match:                                        "production ML system"
```

---

### 2B — Primary Metric

Source: career_history[0].description only.
Most recent, most credible outcome. Take first match.

```
Priority 1: percentage with verb context
  pattern:  (improved|increased|reduced|boosted|lifted)\s+[\w\s-]{2,25}\s+by\s+(\d+%)
  example:  "improved revenue-per-search by 12%"
  output:   "improved revenue-per-search by 12%"
  convert:  → participle for sentence injection:
              "improved" → "improving revenue-per-search by 12%"
              "increased" → "increasing X by Y%"
              "reduced" → "reducing X by Y%"
              "boosted" → "boosting X by Y%"
              "lifted" → "lifting X by Y%"

Priority 2: plain percentage in context
  pattern:  \d+%
  output:   30-char surrounding context + percentage
  convert:  → noun phrase: "delivering a {context}{%} improvement"

Priority 3: scale number with unit
  pattern:  \d+(?:\.\d+)?[KMB]\+?\s*(users|documents|records|queries|candidates)
  output:   "serving {N} users/documents/etc"
  convert:  → already a gerund phrase, use as-is

Priority 4: latency improvement
  pattern:  (\d+ms)\s+to\s+(\d+ms)
  output:   "reducing latency from {X}ms to {Y}ms"
  convert:  → already a gerund phrase, use as-is

Priority 5: multiplier
  pattern:  \d+x\s+(improvement|reduction|speedup)
  output:   "delivering a {X}x improvement"
  convert:  → noun phrase with "delivering"

No match → metric_converted = None → metric clause omitted
```

CRITICAL: The converted form (participle or noun phrase) is what enters
the sentence skeleton. The raw extracted string never enters the skeleton.

---

### 2C — Company Scope

Source: ALL career_history roles.

Builds a human-readable string describing where the candidate has worked.
Used in the sentence to give full career coverage.

```
Step 1: Extract company names from all career_history entries in order
Step 2: Deduplicate while preserving order (some candidates repeat company names)
Step 3: Build company_scope string:

  1 unique company:
    company_scope = "at {company_0}"

  2 unique companies:
    company_scope = "at {company_0} and {company_1}"

  3 unique companies:
    company_scope = "across {company_0}, {company_1}, and {company_2}"

  4+ unique companies:
    company_scope = "across {company_0}, {company_1}, and prior companies"

Output: company_scope string
```

---

### 2D — Years Label

Source: pipeline.gates_and_career.total_years_exp

Converts the float to a clean readable string.

```
>= 8.0  →  "{int(years)}-year"      e.g. "9-year"
>= 7.0  →  "nearly {ceil}-year"     e.g. "nearly 8-year"
>= 6.0  →  "{floor:.0f}-plus-year"  e.g. "6-plus-year"
>= 5.0  →  "5-plus-year"
< 5.0   →  "{floor:.0f}-year"       e.g. "4-year"

Output: years_label string — e.g. "6-plus-year", "nearly 8-year", "9-year"
```

---

### 2E — Named Technology

Source: ALL career_history descriptions combined.
Tech used in any production role is valid evidence.

Three-pass extraction. SEARCH category keywords restricted to avoid
over-triggering on the generic word "search".

```
SKILL CATEGORY SETS:
  VECTOR_DB:  {weaviate, qdrant, milvus, pinecone, faiss, chroma, pgvector}
  SEARCH:     {elasticsearch, opensearch, solr, algolia, typesense}
  RANKING_ML: {scikit-learn, xgboost, lightgbm, learning to rank, catboost}
  LLM:        {langchain, llamaindex, fine-tuning llms, haystack, peft, lora}

DESCRIPTION KEYWORD SIGNALS:
  "embedding" OR "vector search" OR "ann" OR "hnsw" OR "faiss"          → VECTOR_DB
  "inverted index" OR "bm25" OR "full-text search" OR
  "opensearch" OR "elasticsearch"                                         → SEARCH
  "ranking" OR "gradient-boost" OR "learning-to-rank"                    → RANKING_ML
  "rag" OR "fine-tun" OR "langchain" OR "llm" OR "prompt"                → LLM

NOTE: bare "search" and bare "retrieval" do NOT trigger SEARCH category.
      They appear in nearly every description and cause false positives.

PASS 1 — Exact match:
  combined = all career_history descriptions joined as one string (lowercased)
  Sort candidate skills by endorsement count (highest first)
  Return first skill whose name appears literally in combined

PASS 2 — Category-keyword match:
  Detect triggered categories from combined text
  For each triggered category (order: VECTOR_DB, SEARCH, RANKING_ML, LLM):
    Walk skills sorted by endorsement
    Return first skill in that category set

PASS 3 — Endorsement fallback:
  Return highest-endorsed skill from full skills list
  (always succeeds — skills list is never empty)

Output: one skill name string
```

---

### 2F — Verified Skill

Source: pipeline.skill_assessment_scores

Relevance filter applied first. Score number NOT used in sentence
(score numbers cause paraphraser hallucination — confirmed from output analysis).

```
IRRELEVANT_SKILL_SET = {
    "image classification", "computer vision", "object detection",
    "speech recognition", "deep learning", "reinforcement learning",
    "time series", "robotics", "nlp",
    "recommendation systems",
    "machine learning",
    "data science",
    "feature engineering",
    "statistical modeling",
    "information retrieval"
}

Step 1: Filter skill_assessment_scores excluding IRRELEVANT_SKILL_SET (case-insensitive)
Step 2: If filtered dict is non-empty:
          verified_skill = key with highest score in filtered dict
        Elif original dict is non-empty:
          verified_skill = key with highest score in original dict
        Else (19 candidates with empty dict):
          verified_skill = None → falls through to named_tech in sentence

Output: verified_skill string or None
        NOTE: score value is extracted but NOT injected into the sentence.
              "verified Elasticsearch depth" is stable.
              "verified Elasticsearch expertise with score 90" causes hallucination.
```

---

## Step 3 — Verb Selection

Deterministic. Reproducible. Same candidate always gets same verb.

```
VERB_POOL = [
    "built and owned",
    "designed and deployed",
    "developed and scaled",
    "architected and shipped",
    "constructed and maintained"
]

import hashlib
seed = int(hashlib.md5((candidate_id + ":verb").encode()).hexdigest(), 16) % 5
verb = VERB_POOL[seed]

NOTE: use hashlib.md5, NOT Python's hash().
      hash() is randomised by PYTHONHASHSEED and changes every run.
      hashlib.md5 is stable across all runs.
```

Gerund form of verb used in sentence (for "most recently X-ing"):
```
"built and owned"          → "building and owning"
"designed and deployed"    → "designing and deploying"
"developed and scaled"     → "developing and scaling"
"architected and shipped"  → "architecting and shipping"
"constructed and maintained" → "constructing and maintaining"
```

---

## Step 4 — Sentence Skeletons

ONE sentence per candidate. No P1/P2 split. No two-phrase structure.

Every skeleton contains three fixed anchors:
  ANCHOR 1 — career identity: name + years + company scope
  ANCHOR 2 — current action: verb + system_type + metric
  ANCHOR 3 — technical depth: skill evidence

Subject rules still apply:
  DEEP/STRONG/MODERATE: candidate name is always the first word
  SURFACE: profile as possessive noun is the first element

Metric injection rules:
  Priority 1 extracted metric → use converted participle form directly
  Priority 2-5 extracted metric → use "delivering {converted_form}"
  No metric → omit metric_clause and its preceding comma entirely

---

### DEEP Skeleton

```
TEMPLATE:
"{name} brings {years_label} years of production ML experience
{company_scope}, most recently {verb_gerund} the {system_type}
at {company_0}{metric_clause}, with verified {skill_evidence}
depth directly relevant to the role's retrieval and
ranking requirements"

SLOTS:
  {name}           → profile.anonymized_name
                     Rule: must be first word of sentence
  {years_label}    → from Step 2D
                     e.g. "6-plus-year"
  {company_scope}  → from Step 2C
                     e.g. "at Uber and Flipkart"
  {verb_gerund}    → gerund form of verb from Step 3
                     e.g. "building and owning"
  {system_type}    → from Step 2A (current role only)
                     e.g. "ranking pipeline"
  {company_0}      → career_history[0].company (current employer)
                     e.g. "Uber"
  {metric_clause}  → ", {converted_metric}" if metric_1 exists
                     "" if metric_1 is None
  {skill_evidence} → verified_skill if not None, else named_tech
                     e.g. "Elasticsearch"
                     NOTE: no score number included

METRIC CLAUSE RULE:
  Priority 1 metric "improved X by Y%"
    → ", improving X by Y%"
  Priority 2+ metric
    → ", delivering {converted_form}"
  None
    → omit entire ", {metric_clause}" including the comma
```

**Fully resolved example — Mira Verma:**
```
"Mira Verma brings 6-plus-year years of production ML experience
at Uber and Flipkart, most recently building and owning the
ranking pipeline at Uber, improving revenue-per-search by 12%,
with verified Elasticsearch depth directly relevant to the role's
retrieval and ranking requirements"
```

**Fully resolved example — no metric:**
```
"Mira Verma brings 6-plus-year years of production ML experience
at Uber and Flipkart, most recently building and owning the
ranking pipeline at Uber, with verified Elasticsearch depth
directly relevant to the role's retrieval and ranking requirements"
```

**Fully resolved example — no verified skill (uses named_tech):**
```
"Kavya Naidu brings 5-plus-year years of production ML experience
at Razorpay and InMobi, most recently architecting and shipping the
ranking pipeline at Razorpay, improving revenue-per-search by 12%,
with Elasticsearch depth directly relevant to the role's
retrieval and ranking requirements"
```

---

### STRONG Skeleton

```
TEMPLATE:
"{name} brings {years_label} years of production ML experience
{company_scope}, most recently {verb_gerund} the {system_type}
at {company_0}{metric_clause}, with {named_tech} experience
aligning with the role's retrieval and ranking requirements"

SLOTS: same as DEEP except:
  {named_tech}     → named_tech always (not verified_skill)
                     STRONG does not claim "verified" — cross_encoder is
                     strong but not at DEEP level

METRIC CLAUSE RULE: same as DEEP
```

**Fully resolved example — Tanya Chopra:**
```
"Tanya Chopra brings 6-plus-year years of production ML experience
across Zoho, Observe.AI, and Paytm, most recently designing and
deploying the semantic search system at Zoho, delivering a 35%
search-relevance improvement, with Qdrant experience aligning
with the role's retrieval and ranking requirements"
```

---

### MODERATE Skeleton

Softer language. Does not claim "verified" or "directly relevant".
Uses "relevant" and "partial alignment" to be honest about the score.

```
TEMPLATE:
"{name} brings {years_label} years of ML experience
{company_scope}, currently working on the {system_type}
at {company_0}{metric_clause}, with {named_tech} exposure
showing partial alignment to the role's retrieval and
ranking focus"

SLOTS: same naming as above

METRIC CLAUSE RULE: same as DEEP
  But metric_clause attaches with ", with {converted_metric} achieved"
  not as a participle clause.
  Example: ", with a 12% revenue-per-search improvement achieved"
```

**Fully resolved example — Ayaan Tiwari:**
```
"Ayaan Tiwari brings 5-plus-year years of ML experience across
Wysa, Glance, and Meta, currently working on the ranking pipeline
at Wysa, with a 12% revenue-per-search improvement achieved, with
Elasticsearch exposure showing partial alignment to the role's
retrieval and ranking focus"
```

**Fully resolved example — no metric:**
```
"Ayaan Tiwari brings 5-plus-year years of ML experience across
Wysa, Glance, and Meta, currently working on the ranking pipeline
at Wysa, with Elasticsearch exposure showing partial alignment
to the role's retrieval and ranking focus"
```

---

### SURFACE Skeleton

Honest about limited semantic depth. Profile-as-possessive subject.
Company scope still included for full coverage.
No metric injection — overstating a surface candidate's evidence
would misrepresent the cross_encoder signal.

```
TEMPLATE:
"{name}'s profile, spanning {company_scope}, shows
keyword-level alignment to retrieval systems{tech_clause},
but limited semantic depth against the role's ranking
and embedding requirements"

SLOTS:
  {name}           → profile.anonymized_name
                     Rule: appears as possessive, not active subject
  {company_scope}  → from Step 2C
                     e.g. "Saarthi.ai, Unacademy, and Swiggy"
  {tech_clause}    → " including {named_tech} exposure" if named_tech exists
                     "" if named_tech is None

NOTE: no metric_clause for SURFACE.
      No years_label for SURFACE — length would overstate the signal.
```

**Fully resolved example — Pranav Trivedi:**
```
"Pranav Trivedi's profile, spanning Saarthi.ai, Unacademy, and
Swiggy, shows keyword-level alignment to retrieval systems
including Weaviate exposure, but limited semantic depth against
the role's ranking and embedding requirements"
```

**Fully resolved example — no named tech:**
```
"Pranav Trivedi's profile, spanning Saarthi.ai, Unacademy, and
Swiggy, shows keyword-level alignment to retrieval systems,
but limited semantic depth against the role's ranking and
embedding requirements"
```

---

## Step 5 — Injection Rules Summary

```
SLOT              SCOPE            RULE
──────────────────────────────────────────────────────────────────────────────
{name}            profile          First element. Possessive for SURFACE.
{years_label}     all_roles        Converted float to readable label.
{company_scope}   all_roles        1/2/3+ companies, deduplicated.
{verb_gerund}     seed             Gerund form of hashlib-selected verb.
{system_type}     role[0] only     Pattern-matched from current description.
{company_0}       role[0] only     Current employer name.
{metric_clause}   role[0] only     Converted to participle or noun phrase.
                                   Omitted entirely if None (no trailing comma).
{skill_evidence}  assessment dict  verified_skill (no score) or named_tech.
{named_tech}      all_roles        Three-pass extraction from combined descriptions.
{tech_clause}     all_roles        " including {named_tech} exposure" or "".

WEIGHT RULE:
  {name} {verb_gerund} {system_type} — these three are the sentence spine
  Everything else is modifying clause
  No single modifier should exceed 8 words
  metric_clause max 8 words — truncate if longer
```

---

## Step 6 — Paraphrase

One phrase per candidate. One paraphrase call per candidate.

```
MODEL:  humarin/chatgpt_paraphraser_on_T5_base

INPUT:  "paraphrase: {filled_skeleton}"

FIXED PARAMETERS (revised from output analysis):
  num_beams             = 5
  num_beam_groups       = 5
  num_return_sequences  = 5
  diversity_penalty     = 1.5    REDUCED from 3.0 — prevents hallucination
  repetition_penalty    = 3.0    REDUCED from 10.0 — less aggressive
  no_repeat_ngram_size  = 2
  max_length            = 128
```

Why reduced: diversity_penalty=3.0 caused complete hallucination on
technical phrases with numbers (e.g. P2 Mira Verma output: "total 60",
"linked GB", "interviewer adds value"). 1.5 produces genuine variation
while staying semantically grounded.

---

## Step 7 — Variation Selection

Deterministic. Same candidate always returns same variation.

```
import hashlib

seed = int(hashlib.md5((candidate_id + ":tech1").encode()).hexdigest(), 16) % 5
selected = variations[seed]
```

Only one seed needed — one phrase, one paraphrase call.

---

## Step 8 — Reconstruction

```
clause_tech = selected.rstrip(".") + "."
```

One string out. Enters reasoning combinator alongside
seniority, availability, and concern clauses.

---

## Applied to All Five Test Candidates

### Mira Verma — DEEP

```
years_exp      = 6.8  → years_label = "6-plus-year"
companies      = [Uber, Flipkart]  → company_scope = "at Uber and Flipkart"
system_type    = "ranking pipeline"  (from Uber description)
metric_1       = "improved revenue-per-search by 12%"
  converted    = "improving revenue-per-search by 12%"
skill_evidence = "Elasticsearch"  (verified, irrelevance-filtered)
verb_gerund    = "building and owning"  (from hashlib seed)

SKELETON FILLED:
"Mira Verma brings 6-plus-year years of production ML experience
at Uber and Flipkart, most recently building and owning the
ranking pipeline at Uber, improving revenue-per-search by 12%,
with verified Elasticsearch depth directly relevant to the role's
retrieval and ranking requirements"

EXPECTED PARAPHRASE DIRECTION:
"Mira Verma has over six years of production ML experience at Uber
and Flipkart, where she has been building and owning the ranking
pipeline at Uber with a 12% revenue-per-search improvement, backed
by verified Elasticsearch expertise directly relevant to the role's
retrieval requirements"
```

### Tanya Chopra — STRONG

```
years_exp      = 6.9  → years_label = "6-plus-year"
companies      = [Zoho, Observe.AI, Paytm]
  → company_scope = "across Zoho, Observe.AI, and Paytm"
system_type    = "semantic search system"  (from Zoho description)
metric_1       = "arch-relevance improvement of 35%"  (Priority 2 plain %)
  converted    = "delivering a 35% arch-relevance improvement"
named_tech     = "Qdrant"  (verified_skill, not used for STRONG)

SKELETON FILLED:
"Tanya Chopra brings 6-plus-year years of production ML experience
across Zoho, Observe.AI, and Paytm, most recently designing and
deploying the semantic search system at Zoho, delivering a 35%
arch-relevance improvement, with Qdrant experience aligning with
the role's retrieval and ranking requirements"
```

### Ayaan Tiwari — MODERATE

```
years_exp      = 5.2  → years_label = "5-plus-year"
companies      = [Wysa, Glance, Meta]
  → company_scope = "across Wysa, Glance, and Meta"
system_type    = "ranking pipeline"
metric_1       = "improved revenue-per-search by 12%"
  converted (MODERATE) = "a 12% revenue-per-search improvement achieved"
named_tech     = "Elasticsearch"  (pass 1 — appears literally in descriptions)

SKELETON FILLED:
"Ayaan Tiwari brings 5-plus-year years of ML experience across
Wysa, Glance, and Meta, currently working on the ranking pipeline
at Wysa, with a 12% revenue-per-search improvement achieved, with
Elasticsearch exposure showing partial alignment to the role's
retrieval and ranking focus"
```

### Pranav Trivedi — SURFACE

```
companies      = [Saarthi.ai, Unacademy, Swiggy]
  → company_scope = "Saarthi.ai, Unacademy, and Swiggy"
named_tech     = "Weaviate"  (pass 2 — VECTOR_DB triggered by "sentence-transformers")

SKELETON FILLED:
"Pranav Trivedi's profile, spanning Saarthi.ai, Unacademy, and
Swiggy, shows keyword-level alignment to retrieval systems
including Weaviate exposure, but limited semantic depth against
the role's ranking and embedding requirements"
```

### Kavya Naidu — DEEP (no verified skill)

```
years_exp      = 5.9  → years_label = "5-plus-year"
companies      = [Razorpay, InMobi]
  → company_scope = "at Razorpay and InMobi"
system_type    = "ranking pipeline"
metric_1       = "improved revenue-per-search by 12%"
  converted    = "improving revenue-per-search by 12%"
verified_skill = None  (empty skill_assessment_scores)
named_tech     = "Elasticsearch"  (pass 1 — appears in InMobi description)
skill_evidence = "Elasticsearch"  (falls back to named_tech)

SKELETON FILLED:
"Kavya Naidu brings 5-plus-year years of production ML experience
at Razorpay and InMobi, most recently architecting and shipping the
ranking pipeline at Razorpay, improving revenue-per-search by 12%,
with Elasticsearch depth directly relevant to the role's retrieval
and ranking requirements"
```

---

## All Variants Summary

```
CAT       SUBJECT      YEARS  COMPANIES    METRIC     SKILL EVIDENCE
──────────────────────────────────────────────────────────────────────────────
DEEP      name first   YES    all roles    current    verified_skill → named_tech
STRONG    name first   YES    all roles    current    named_tech only
MODERATE  name first   YES    all roles    current    named_tech ("exposure")
SURFACE   possessive   NO     all roles    NO         named_tech ("including X")
```

---

## What Changed From Previous Version

```
REDESIGN 1 — Collapsed two-sentence to one sentence
  Two-phrase structure (P1 + P2) removed entirely.
  Every tech_cat produces one sentence with three anchors.
  Eliminates P2 hallucination problem (Mira Verma "total 60", "linked GB").

REDESIGN 2 — Career scope covers all roles not just current
  company_scope built from ALL career_history companies.
  years_label from total_years_exp.
  A 3-role candidate shows all three companies in the sentence.
  Nothing erased.

REDESIGN 3 — Score number removed from sentence
  "verified Elasticsearch depth" replaces "verified Elasticsearch score of 90".
  Score numbers cause paraphraser hallucination. Confirmed from output analysis.

REDESIGN 4 — Unified metric conversion before injection
  Priority 1 → participle ("improving X by Y%")
  Priority 2-5 → noun phrase ("delivering a X% improvement")
  Raw extracted string never enters skeleton unconverted.

REDESIGN 5 — Paraphraser parameters reduced
  diversity_penalty: 3.0 → 1.5
  repetition_penalty: 10.0 → 3.0
  Prevents hallucination on technical phrases with numbers.

REDESIGN 6 — IRRELEVANT_SKILL_SET expanded
  Added domain labels and generic terms confirmed from output analysis:
  "recommendation systems", "machine learning", "data science",
  "feature engineering", "statistical modeling", "information retrieval"

REDESIGN 7 — SEARCH category keywords restricted
  Removed bare "search" and "retrieval" from SEARCH trigger keywords.
  These appear in nearly every description and caused false positives.

BUG FIX — hash() replaced with hashlib.md5
  Python hash() randomises across runs.
  hashlib.md5 is stable and reproducible.

BUG FIX — All career_history descriptions used for named_tech
  Previous [:2] cap removed. All roles now contribute.

BUG FIX — metric_2 scans all prior roles not just role[1]
  9 candidates lost their secondary metric. Now scanned forward.
  (metric_2 no longer used in sentence — only metric_1 from current role.
   metric_2 preserved for potential use in STRONG fallback if needed.)
```

---

## What the Coding Agent Must Build

```
tech_section.py

CONSTANTS (hardcoded, do not derive):
  VERB_POOL              list of 5 verbs as specified above
  VERB_GERUND_MAP        dict mapping each verb to its gerund form
  IRRELEVANT_SKILL_SET   set as specified above (expanded version)
  SYSTEM_TYPE_PATTERNS   list of 10 (pattern, label) tuples as specified above
  SKILL_CATEGORIES       list of 4 (name, set) tuples as specified above
  CATEGORY_KEYWORDS      list of 4 (name, list) tuples — SEARCH restricted version

FUNCTIONS:

1. classify_tech(cross_encoder_score: float) -> str
   Returns: "DEEP" | "STRONG" | "MODERATE" | "SURFACE"

2. extract_system_type(description: str) -> str
   10 patterns in order. Default "production ML system".

3. extract_primary_metric(description: str) -> str | None
   5-priority regex. Returns raw extracted string or None.

4. convert_metric_for_skeleton(metric: str, cat: str) -> str | None
   Converts raw metric to injection-ready form.
   DEEP/STRONG: participle for Priority 1, "delivering {noun}" for 2-5
   MODERATE: "a {X}% {subject} improvement achieved" noun phrase
   Returns None if metric is None.
   Truncates to 8 words max.

5. build_company_scope(career_history: list) -> str
   Deduplicates companies. Builds 1/2/3/4+ scope string as specified.

6. build_years_label(total_years_exp: float) -> str
   Converts float to readable label as specified.

7. extract_named_tech(skills: list, descriptions: list[str]) -> str
   Three-pass extraction. Combined string check for Pass 1.
   Restricted SEARCH keywords for Pass 2. Always returns a string.

8. extract_verified_skill(skill_assessment_scores: dict) -> str | None
   Expanded IRRELEVANT_SKILL_SET filter. Returns skill name or None.
   Does NOT return score — score is not used in sentence.

9. select_verb(candidate_id: str) -> str
   hashlib.md5 seed. Returns verb string.

10. get_verb_gerund(verb: str) -> str
    Returns gerund form from VERB_GERUND_MAP.

11. build_tech_sentence(
        tech_cat, name, years_label, company_scope,
        verb_gerund, system_type, company_0,
        metric_converted, skill_evidence, named_tech
    ) -> str
    Builds the filled skeleton string.
    Applies all injection rules.
    Handles None metric_converted (omit clause + comma).
    Returns complete sentence before paraphrasing.

12. paraphrase(text: str) -> list[str]
    Returns 5 variations.
    diversity_penalty=1.5, repetition_penalty=3.0

13. select_variation(variations: list[str], candidate_id: str) -> str
    hashlib.md5 with slot ":tech1". Returns one string.

14. build_tech_clause(candidate: dict, paraphrase_fn) -> str
    Orchestrates steps 1-13.
    Accesses cross_encoder_score from:
      candidate["pipeline"]["retrieval_scores"]["cross_encoder_score"]
    Accesses skill_assessment_scores from:
      candidate["pipeline"]["skill_assessment_scores"]
    Accesses total_years_exp from:
      candidate["pipeline"]["gates_and_career"]["total_years_exp"]
    Returns final tech clause string.

TEST:
  Run against all five candidates from the test batch:
    CAND_0018549 Mira Verma   (DEEP)
    CAND_0061265 Tanya Chopra (STRONG)
    CAND_0086151 Ayaan Tiwari (MODERATE)
    CAND_0018722 Pranav Trivedi (SURFACE)
    CAND_0051630 Kavya Naidu  (DEEP, no verified skill)

  For each candidate print:
    tech_cat, years_label, company_scope, system_type,
    metric_1 (raw), metric_converted, skill_evidence,
    named_tech, verb, verb_gerund,
    filled skeleton (before paraphrase),
    all 5 paraphrase variations,
    selected variation,
    final clause_tech

  Verify each filled skeleton against the expected resolved examples
  in this plan. If any filled skeleton does not match, print MISMATCH
  and the actual vs expected strings.

  Save all outputs to tech_section_output.json.
```
