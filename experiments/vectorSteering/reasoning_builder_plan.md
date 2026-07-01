# Reasoning Builder — Complete Pipeline Plan
## Schema, Formulas, Templates, and Implementation Guide

---

## What This Pipeline Does

Takes one candidate dict from `team_results.json` and produces a three-sentence
human-readable reasoning string describing their fit for the role.

The pipeline has seven distinct stages:

```
1. Base Calculations     → pure math from numeric fields → category labels
2. Evidence Extraction   → regex from text fields → structured facts
3. Priority Scoring      → threshold checks → decide what is worth mentioning
4. Sentence Assembly     → fill templates with computed values → raw sentences
5. Paraphrase            → run each raw sentence through paraphraser → 5 variations
6. Variation Selection   → deterministic seed → pick one variation per sentence
7. Reconstruction        → join three selected sentences → final reasoning string
```

No ML inference except the paraphraser at Step 5.
No randomness except seeded by candidate_id (same candidate always gets same output).
No external API calls.

---

## The Three Sentences — Semantic Roles

Every candidate gets exactly three sentences. Each sentence answers one question
a recruiter actually asks when evaluating a candidate.

```
SENTENCE 1 — TECHNICAL IDENTITY
  Question: what have they built, where, for how long, and does it match?
  Sources:  career_history + cross_encoder_score + skill_assessment_scores
  Always contains: name, years, companies, current system, metric, skill evidence

SENTENCE 2 — CAREER QUALITY
  Question: what kind of engineer are they and do they clear the JD's filters?
  Sources:  gates_and_career flags
  Always contains: career type, disqualifier statement
  Conditionally contains: tenure, pre-LLM signal, sweet spot, tech caveat

SENTENCE 3 — AVAILABILITY AND FRICTION
  Question: can we hire them and how hard will outreach be?
  Sources:  behavioral_signals + stage5_scoring
  Always contains: notice period, availability assessment, outreach recommendation
  Conditionally contains: days since active, response rate, github, applications
```

---

## Input Field Map

Every field the pipeline reads from the candidate dict.
All access paths are from the root of the candidate dict.

```
FIELD                                    PATH
──────────────────────────────────────────────────────────────────────────────
candidate_id                             candidate["candidate_id"]
name                                     candidate["profile"]["anonymized_name"]
current_company                          candidate["profile"]["current_company"]
cross_encoder_score                      candidate["pipeline"]["retrieval_scores"]
                                                  ["cross_encoder_score"]
skill_assessment_scores                  candidate["pipeline"]["skill_assessment_scores"]
total_years_exp                          candidate["pipeline"]["gates_and_career"]
                                                  ["total_years_exp"]
pre_llm_production_ml                    candidate["pipeline"]["gates_and_career"]
                                                  ["pre_llm_production_ml"]
product_company_fraction                 candidate["pipeline"]["gates_and_career"]
                                                  ["product_company_fraction"]
consulting_company_count                 candidate["pipeline"]["gates_and_career"]
                                                  ["consulting_company_count"]
avg_tenure_per_employer                  candidate["pipeline"]["gates_and_career"]
                                                  ["avg_tenure_per_employer"]
llm_framework_only                       candidate["pipeline"]["gates_and_career"]
                                                  ["llm_framework_only"]
recent_ai_only                           candidate["pipeline"]["gates_and_career"]
                                                  ["recent_ai_only"]
in_sweet_spot                            candidate["pipeline"]["gates_and_career"]
                                                  ["in_sweet_spot"]
notice_period_days                       candidate["pipeline"]["gates_and_career"]
                                                  ["notice_period_days"]
days_since_active                        candidate["pipeline"]["stage5_scoring"]
                                                  ["days_since_active"]
open_to_work_flag                        candidate["pipeline"]["behavioral_signals"]
                                                  ["open_to_work_flag"]
applications_submitted_30d               candidate["pipeline"]["behavioral_signals"]
                                                  ["applications_submitted_30d"]
recruiter_response_rate                  candidate["pipeline"]["behavioral_signals"]
                                                  ["recruiter_response_rate"]
offer_acceptance_rate                    candidate["pipeline"]["behavioral_signals"]
                                                  ["offer_acceptance_rate"]
github_activity_score                    candidate["pipeline"]["behavioral_signals"]
                                                  ["github_activity_score"]
career_history[]                         candidate["career_history"]
                                         (list of role dicts, all roles used)
skills[]                                 candidate["skills"]
                                         (list of skill dicts, all used)
```

---

## Constants

All constants are hardcoded. The coding agent does not derive or modify these.

### Verb Pool and Gerund Map

Five verbs for sentence 1. Verb is selected deterministically by candidate_id seed.
Gerund form used in the sentence template.

```
VERB_POOL = [
    "built and owned",
    "designed and deployed",
    "developed and scaled",
    "architected and shipped",
    "constructed and maintained",
]

VERB_GERUND_MAP = {
    "built and owned":             "building and owning",
    "designed and deployed":       "designing and deploying",
    "developed and scaled":        "developing and scaling",
    "architected and shipped":     "architecting and shipping",
    "constructed and maintained":  "constructing and maintaining",
}
```

### System Type Patterns

Applied to career_history[0].description in order. First match wins.

```
Pattern                                        Label
──────────────────────────────────────────────────────────────────────
\branking\b.*\b(layer|pipeline|model|system)\b "ranking pipeline"
\brecommendation\b.*\bsystem\b                 "recommendation system"
\bsemantic\s+search\b                          "semantic search system"
\bdiscovery\s+feed\b|\branking\s+model\b       "discovery ranking system"
\bML\s+pipeline\b|\bMLflow\b|\bKubeflow\b      "ML infrastructure pipeline"
\bRAG\b|\bretrieval.augmented\b                "RAG-based retrieval pipeline"
\bwhat\s+users\s+are\s+looking\b               "relevance matching system"
\bsearch\s+and\s+discovery\b                   "search and discovery system"
\bembedding.based\s+search\b                   "embedding-based search system"
\bmigration\b.*\bsearch\b                      "search migration"
No match                                       "production ML system"
```

### Metric Extraction Patterns

Applied to description text in priority order. First match wins.

```
Priority 1 — Verb-context percentage
  Pattern: (improved|increased|reduced|boosted|lifted)\s+([\w\s\-]{2,25})\s+by\s+(\d+(?:\.\d+)?%)
  Example input:  "Final model improved revenue-per-search by 12%"
  Raw output:     "improved revenue-per-search by 12%"
  Participle:     "improving revenue-per-search by 12%"
  Noun phrase:    "a 12% revenue-per-search improvement"

Priority 2 — Scale number with unit
  Pattern: \d+(?:\.\d+)?[KMB]\+?\s*(users|documents|records|queries|candidates)
  Example input:  "system serving 10M+ users"
  Raw output:     "10M+ users"
  Participle:     "serving 10M+ users"
  Noun phrase:    "serving 10M+ users"

Priority 3 — Latency improvement
  Pattern: (\d+ms)\s+to\s+(\d+ms)
  Example input:  "latency from 340ms to 22ms"
  Participle:     "reducing latency from 340ms to 22ms"
  Noun phrase:    "a latency reduction from 340ms to 22ms"

Priority 4 — Plain percentage with context
  Pattern: (\d+(?:\.\d+)?%)
  Takes 25 chars of context before the match
  Participle:     "delivering a {X}% improvement"
  Noun phrase:    "a {X}% improvement"

No match → metric = None → metric clause omitted entirely including comma
```

### Participle Conversion Map

Used to convert extracted verb to present participle for sentence 1.

```
"improved"  → "improving"
"increased" → "increasing"
"reduced"   → "reducing"
"boosted"   → "boosting"
"lifted"    → "lifting"
```

### Description Tech Extraction Pattern

Named tools mentioned in description text. Used in sentence 1 optional clause.
Max 2 items returned, order of first appearance.

```
Pattern: \b(FAISS|Qdrant|Weaviate|Milvus|Pinecone|Elasticsearch|OpenSearch|
            BM25|sentence-transformers|sentence_transformers|bge-base|all-MiniLM|
            XGBoost|LightGBM|scikit-learn|MLflow|Kubeflow|LangChain|LlamaIndex|
            PEFT|LoRA|QLoRA|Hugging\s*Face|HNSW|ANNOY|ScaNN)\b
```

### Irrelevant Skill Set

Skills filtered out before selecting verified_skill.
These are domain labels or generic terms, not technical tools.

```
IRRELEVANT_SKILL_SET = {
    "image classification", "computer vision", "object detection",
    "speech recognition", "deep learning", "reinforcement learning",
    "time series", "robotics", "nlp", "recommendation systems",
    "machine learning", "data science", "feature engineering",
    "statistical modeling", "information retrieval",
}
```

### Skill Category Sets and Keywords

Used in three-pass named_tech extraction.
SEARCH keywords are intentionally restricted — bare "search" and "retrieval"
appear in nearly every description and cause false positives.

```
SKILL_CATEGORIES = [
    ("VECTOR_DB",  {weaviate, qdrant, milvus, pinecone, faiss, chroma, pgvector}),
    ("SEARCH",     {elasticsearch, opensearch, solr, algolia, typesense}),
    ("RANKING_ML", {scikit-learn, xgboost, lightgbm, learning to rank, catboost}),
    ("LLM",        {langchain, llamaindex, fine-tuning llms, haystack, peft, lora}),
]

CATEGORY_KEYWORDS = [
    ("VECTOR_DB",  ["embedding", "vector search", "ann", "hnsw", "faiss"]),
    ("SEARCH",     ["inverted index", "bm25", "full-text search",
                    "opensearch", "elasticsearch"]),
    ("RANKING_ML", ["ranking", "gradient-boost", "learning-to-rank"]),
    ("LLM",        ["rag", "fine-tun", "langchain", "llm", "prompt"]),
]
```

---

## Section 1 — Base Calculations

These run first. Pure math and logic from numeric and boolean fields.
No string building except converting numbers to readable labels.

### 1A — Tech Category

```
INPUT:  cross_encoder_score (float)
OUTPUT: tech_cat (string)

FORMULA:
  >= 3.5  →  "DEEP"        30% of top-100 pool. Strong semantic depth.
  >= 2.5  →  "STRONG"      27%. Solid match.
  >= 1.5  →  "MODERATE"    28%. Partial alignment.
  <  1.5  →  "SURFACE"     15%. Keyword match only.

PURPOSE: Controls sentence 1 template variant, sentence 2 tech caveat,
         sentence 3 outreach recommendation tone.

FUNCTION: calculate_tech_cat(cross_encoder_score: float) -> str
```

### 1B — Years Label

```
INPUT:  total_years_exp (float)
OUTPUT: years_label (string)

FORMULA:
  >= 8.0  →  "{int(years)}-year"          e.g. "9-year"
  >= 7.0  →  "nearly {ceil(years)}-year"  e.g. "nearly 8-year"
  >= 6.0  →  "{int(years)}-plus-year"     e.g. "6-plus-year"
  >= 5.0  →  "5-plus-year"
  <  5.0  →  "{int(years)}-year"          e.g. "4-year"

PURPOSE: Injected into sentence 1 as human-readable experience span.

FUNCTION: calculate_years_label(total_years_exp: float) -> str
```

### 1C — Experience Type

```
INPUT:  pre_llm_production_ml (bool)
OUTPUT: experience_type (string)

FORMULA:
  True  →  "production ML"
  False →  "ML engineering"

PURPOSE: Injected into sentence 1. Distinguishes pre-LLM builders from
         post-2022 practitioners without stating the flag explicitly.

FUNCTION: calculate_experience_type(pre_llm: bool) -> str
```

### 1D — Notice Days Label

```
INPUT:  notice_period_days (int)
OUTPUT: notice_label (string)

FORMULA:
  0     →  "immediate availability"
  1+    →  "{notice_days}-day notice period"
  e.g.  30  →  "30-day notice period"
        60  →  "60-day notice period"
        90  →  "90-day notice period"

PURPOSE: Injected into sentence 3 as readable logistics statement.

FUNCTION: calculate_notice_days_label(notice_days: int) -> str
```

### 1E — Availability Assessment

```
INPUT:  days_since_active (int)
        notice_period_days (int)
        open_to_work_flag (bool)
        applications_submitted_30d (int)
        offer_acceptance_rate (float)

OUTPUT: availability_assessment (string) — one of five labels

FORMULA (evaluated in order, first match wins):

  IMMEDIATE:
    open_to_work AND days_since_active <= 30 AND notice_days <= 30
    →  "immediate availability and low outreach friction"

  SERIOUS_MOVER:
    applications_30d >= 10 AND offer_acceptance >= 0.80
    →  "active job-seeking with strong commitment signals"
    NOTE: checked before ACTIVE because behavioral commitment overrides
          recency signals — a candidate with 10+ applications and 80%+
          offer acceptance is demonstrably serious regardless of login date.

  ACTIVE:
    open_to_work AND days_since_active <= 60 AND notice_days <= 60
    AND (applications_30d >= 5 OR offer_acceptance >= 0.70)
    →  "active availability with manageable timeline"

  MODERATE_FRICTION:
    open_to_work AND (days_since_active > 60 OR notice_days > 60)
    →  "moderate friction — timeline confirmation needed"

  LOW_SIGNAL:
    NOT open_to_work OR days_since_active > 90
    →  "low availability signal — confirm interest before outreach"

FUNCTION: calculate_availability_assessment(
              days_since_active, notice_days, open_to_work,
              applications_30d, offer_acceptance) -> str
```

### 1F — Outreach Recommendation

```
INPUT:  tech_cat (string)
        availability_assessment (string)

OUTPUT: outreach_recommendation (string)

FORMULA:
  is_low_friction = "immediate" OR "active" OR "strong commitment" in assessment
  is_friction     = "moderate friction" in assessment
  is_low_signal   = "low availability" in assessment

  DEEP + is_low_friction  →  "strong outreach candidate"
  DEEP + is_friction      →  "recommended for outreach with timeline negotiation"
  DEEP + is_low_signal    →  "outreach recommended but availability must be confirmed"
  STRONG + is_low_friction →  "solid outreach candidate"
  STRONG + other          →  "worth outreach with timeline confirmation"
  MODERATE + any          →  "worth outreach pending direct technical evaluation"
  SURFACE + any           →  "low outreach priority unless screening confirms depth"

FUNCTION: calculate_outreach_recommendation(tech_cat, availability_assessment) -> str
```

### 1G — Career Characterization

```
INPUT:  product_company_fraction (float)
        consulting_company_count (int)

OUTPUT: career_characterization (string)

FORMULA:
  product_frac == 1.0 AND consulting == 0
    →  "career is entirely at product companies"
  product_frac >= 0.7 AND consulting == 0
    →  "career is predominantly at product companies"
  consulting >= 1 AND product_frac >= 0.5
    →  "career combines product company and consulting firm experience"
  consulting >= 1 AND product_frac < 0.5
    →  "career is primarily at consulting firms"
  else
    →  "career spans mixed company types"

FUNCTION: calculate_career_characterization(product_frac, consulting_count) -> str
```

### 1H — Disqualifier Statement

```
INPUT:  consulting_company_count (int)
        llm_framework_only (bool)
        recent_ai_only (bool)

OUTPUT: disqualifier_statement (string)

FORMULA:
  Collect flags that apply:
    consulting_count >= 1  →  flag: "consulting firm history is an explicit JD disqualifier"
    llm_framework_only     →  flag: "LLM framework-only experience falls below the JD's threshold"
    recent_ai_only         →  flag: "production ML experience is primarily post-2022"

  No flags:
    →  "clearing the JD's explicit disqualifiers"

  One flag:
    →  "though {flag}"

  Two or more flags:
    →  "though {flag1} and {flag2}"

FUNCTION: calculate_disqualifier_statement(
              consulting_count, llm_framework_only, recent_ai_only) -> str
```

---

## Section 2 — Evidence Extraction

Regex-based extraction from text fields and structured list fields.
Returns clean values or None. Never raises on missing data.

### 2A — System Type

```
INPUT:  career_history[0].description (string)
OUTPUT: system_type (string)

METHOD: Apply SYSTEM_TYPE_PATTERNS in order. Return label of first match.
        Default "production ML system" if no match.

SCOPE:  Current role description ONLY (career_history[0]).
        System type reflects what they currently own, not oldest role.

FUNCTION: extract_system_type(description: str) -> str
```

### 2B — Primary Metric

```
INPUT:  career_history[0].description (string)
OUTPUT: dict or None

  dict = {
    "raw":                  original extracted string
    "converted_participle": for DEEP/STRONG sentence 1
    "converted_noun":       for MODERATE sentence 1
  }

METHOD: Apply METRIC EXTRACTION PATTERNS in priority order.
        Return dict with all three forms populated.
        Return None if no pattern matches.

SCOPE:  Current role description ONLY.

SUBJECT WORD LIMIT: When building converted forms, truncate the metric
                    subject to 4 words max to keep metric_clause under
                    8 words total.
                    "revenue-per-search" = 1 hyphenated word — keep as-is.
                    "7-day retention" = 2 words — keep as-is.
                    "overall business application performance metric" = 5 words
                    → truncate to "overall business application performance"

FUNCTION: extract_primary_metric(description: str) -> dict | None
```

### 2C — Description Tech

```
INPUT:  career_history[0].description (string)
OUTPUT: list of strings (max 2 items, order of first appearance)

METHOD: Apply DESCRIPTION_TECH_RE pattern.
        Collect unique matches (case-insensitive dedup by normalized form).
        Return first 2 found.
        Return empty list if none found.

PURPOSE: Captures tools explicitly named in description text that may not
         appear in the skills list. e.g. "sentence-transformers and FAISS"
         from Tanya Chopra's Zoho description.

SCOPE:  Current role description ONLY.

FUNCTION: extract_description_tech(description: str) -> list[str]
```

### 2D — Company Scope

```
INPUT:  career_history[] (all roles)
OUTPUT: (companies_list: list[str], company_scope: str)

METHOD:
  Walk all career_history entries in order.
  Extract company name from each role.
  Deduplicate while preserving order of first appearance.

  Build company_scope string:
    1 unique company  →  "at {c0}"
    2 unique companies →  "at {c0} and {c1}"
    3 unique companies →  "across {c0}, {c1}, and {c2}"
    4+ unique companies → "across {c0}, {c1}, and prior companies"

SCOPE:  ALL career_history roles. This is intentional.
        Career scope should reflect the full career, not just the current role.

FUNCTION: extract_company_scope(career_history: list) -> tuple[list[str], str]
```

### 2E — Named Technology

```
INPUT:  skills[] (all skills)
        all career_history descriptions combined
OUTPUT: named_tech (string) — always returns something (never None)

METHOD: Three-pass extraction.

  Pass 1 — Exact match:
    combined = all descriptions joined and lowercased
    Sort skills by endorsement count descending
    Return first skill whose name appears literally in combined
    Rationale: if the candidate explicitly named the tool in their description,
               that is the most credible evidence

  Pass 2 — Category keyword match:
    Check which CATEGORY_KEYWORDS appear in combined
    For each triggered category in order (VECTOR_DB → SEARCH → RANKING_ML → LLM):
      Walk skills sorted by endorsement
      Return first skill whose name is in that category set
    Rationale: description uses concept words not tool names, but candidate
               has a relevant tool in their skills list

  Pass 3 — Endorsement fallback:
    Return name of highest-endorsed skill
    Always succeeds — skills list is never empty per data schema

NOTE: SEARCH category does NOT trigger on bare "search" or "retrieval".
      These appear in almost every description. Only specific technical
      terms (bm25, inverted index, full-text search, opensearch,
      elasticsearch) trigger the SEARCH category.

FUNCTION: extract_named_tech(skills: list, descriptions: list[str]) -> str
```

### 2F — Verified Skill

```
INPUT:  skill_assessment_scores (dict: {skill_name: float})
OUTPUT: verified_skill (string or None)

METHOD:
  If dict is empty → return None

  Filter out keys in IRRELEVANT_SKILL_SET (case-insensitive comparison).
  If filtered dict is non-empty → return key with highest score.
  If all skills are irrelevant → return key with highest score from original dict.
  (edge case — better an adjacent skill than nothing)

IMPORTANT: Return skill NAME only. Do NOT return the score value.
           Score numbers cause paraphraser hallucination (confirmed from
           output analysis — "total 60", "linked GB", etc.)
           "verified Elasticsearch depth" is stable.
           "verified Elasticsearch expertise with an assessed score of 90" is not.

FUNCTION: extract_verified_skill(skill_assessment_scores: dict) -> str | None
```

### 2G — Best Prior Metric

```
INPUT:  career_history[] (all roles)
OUTPUT: metric dict or None

METHOD:
  Scan career_history[1:] forward (skip current role).
  Call extract_primary_metric on each description.
  Return the first non-None result found.
  Return None if no prior role has an extractable metric.

PURPOSE: Provides secondary evidence when sentence construction needs
         additional evidence from prior roles.
         Currently reserved for optional use — sentence 1 uses only
         the current role metric (metric_1). This is available if
         needed for future expansion.

FUNCTION: extract_best_metric_from_prior_roles(career_history: list) -> dict | None
```

---

## Section 3 — Priority Signal Scoring

These functions check whether a signal is strong enough to be worth
mentioning in the reasoning. Signals that are common, expected, or
in the middle range are silently omitted. Only signals that are
genuinely noteworthy at either extreme are surfaced.

This is the mechanism that makes reasoning specific and non-generic.
If 98% of candidates have the same value for a field, mentioning it
adds nothing. These functions implement that logic per field.

### 3A — Tenure Signal

```
INPUT:  avg_tenure_per_employer (float)
OUTPUT: string or None

THRESHOLD:
  >= 3.0  →  "stable tenure"
              (positive — worth mentioning, demonstrates ownership depth)
  <= 1.5  →  "short average tenure raising ownership depth questions"
              (concern — worth mentioning, relevant to JD's 3+ year preference)
  else    →  None
              (middle range 1.5-3.0 — not noteworthy either way)

FUNCTION: score_tenure_signal(avg_tenure: float) -> str | None
```

### 3B — Pre-LLM Signal

```
INPUT:  pre_llm_production_ml (bool)
OUTPUT: string or None

THRESHOLD:
  True  →  "pre-LLM production ML ownership"
            (positive — explicitly valued by JD, worth mentioning)
  False →  None
            (absence is implied by career section, not explicitly stated)

FUNCTION: score_pre_llm_signal(pre_llm: bool) -> str | None
```

### 3C — Sweet Spot Signal

```
INPUT:  in_sweet_spot (bool)
OUTPUT: string or None

THRESHOLD:
  True  →  "placing them in the role's target experience band"
            (positive — explicit JD requirement met, worth closing sentence 2 with)
  False →  None
            (being outside the band is conveyed by years_label implicitly)

FUNCTION: score_sweet_spot_signal(in_sweet_spot: bool) -> str | None
```

### 3D — Tech Depth Caveat

```
INPUT:  tech_cat (string)
OUTPUT: string or None

THRESHOLD:
  "MODERATE"  →  "cross-encoder alignment is partial — direct technical evaluation recommended"
  "SURFACE"   →  "semantic depth against the role's core requirements is limited"
  "DEEP"      →  None  (no caveat needed — strong alignment)
  "STRONG"    →  None  (no caveat needed — solid alignment)

PURPOSE: Appears at the end of sentence 2. Carries the technical depth
         signal into the career/quality sentence so it contextualizes
         the career quality claims.

FUNCTION: score_tech_depth_caveat(tech_cat: str) -> str | None
```

### 3E — Activity Signal

```
INPUT:  days_since_active (int)
OUTPUT: string or None

THRESHOLD:
  > 45 days  →  "{days} days since last platform login"
                 (worth flagging — above typical active-user threshold)
  <= 45 days →  None
                 (recent activity is baseline expectation, not noteworthy)

FUNCTION: score_activity_signal(days_since_active: int) -> str | None
```

### 3F — Response Rate Signal

```
INPUT:  recruiter_response_rate (float, 0.0-1.0)
OUTPUT: string or None

THRESHOLD:
  >= 0.80  →  "strong recruiter responsiveness"
               (positive — exceptional, worth mentioning as hiring ease signal)
  <  0.50  →  "low response rate — outreach may face delays"
               (concern — below acceptable threshold)
  else     →  None
               (0.50-0.80 is expected range, not noteworthy)

FUNCTION: score_response_signal(recruiter_response_rate: float) -> str | None
```

### 3G — GitHub Signal

```
INPUT:  github_activity_score (float, -1 to 100; -1 = no GitHub linked)
OUTPUT: string or None

THRESHOLD:
  >= 70   →  "active GitHub presence supporting technical claims"
              (positive — above-average external validation signal)
  -1      →  None  (no GitHub linked)
  else    →  None  (below threshold, not worth mentioning)

NOTE: This is one of the few signals that provides external validation
      evidence. Worth mentioning when genuinely high.

FUNCTION: score_github_signal(github_activity_score: float) -> str | None
```

### 3H — Applications Signal

```
INPUT:  applications_submitted_30d (int)
        open_to_work_flag (bool)
OUTPUT: string or None

THRESHOLD:
  >= 10 AND open_to_work     →  "actively applying to roles"
                                 (positive — demonstrates genuine market activity)
  == 0 AND NOT open_to_work  →  "no recent applications"
                                 (concern — passive and not flagged available)
  else                       →  None
                                 (middle range, implied by availability assessment)

FUNCTION: score_applications_signal(applications_30d: int, open_to_work: bool) -> str | None
```

---

## Section 4 — Sentence Assembly

Takes all computed and extracted values. Builds the raw sentence strings.
No paraphrasing happens here. Output is complete English sentences
ready to be paraphrased.

### Sentence 1 Template

```
PURPOSE: Establishes technical identity. Answers "what have they built,
         where, for how long, and does it match the role."

TEMPLATE VARIANTS BY tech_cat:

  DEEP, STRONG, MODERATE:
  ────────────────────────
  "{name} brings {years_label} years of {experience_type} experience
   {company_scope}, most recently {verb_gerund} the {system_type}
   at {company_0}{metric_clause}{desc_tech_clause},
   {skill_clause} {jd_alignment}"

  SURFACE:
  ─────────
  "{name}'s profile, spanning {company_scope}, shows keyword-level
   alignment to retrieval systems{tech_or_named_clause}, but limited
   semantic depth against the role's ranking and embedding requirements"

SLOT DEFINITIONS:

  {name}
    Source:  profile.anonymized_name
    Rule:    First word of sentence for DEEP/STRONG/MODERATE.
             First word as possessive for SURFACE.

  {years_label}
    Source:  calculate_years_label(total_years_exp)
    Example: "6-plus-year"
    Rule:    Not used in SURFACE variant.

  {experience_type}
    Source:  calculate_experience_type(pre_llm)
    Values:  "production ML" or "ML engineering"
    Rule:    Not used in SURFACE variant.

  {company_scope}
    Source:  extract_company_scope(career_history)[1]
    Example: "at Uber and Flipkart"
             "across Zoho, Observe.AI, and Paytm"
    Rule:    ALL career_history companies, deduped.

  {verb_gerund}
    Source:  VERB_GERUND_MAP[select_verb(candidate_id)]
    Example: "building and owning"
    Rule:    Not used in SURFACE variant.

  {system_type}
    Source:  extract_system_type(career_history[0].description)
    Example: "ranking pipeline"
    Rule:    From current role only. Not used in SURFACE variant.

  {company_0}
    Source:  profile.current_company
    Example: "Uber"
    Rule:    Current employer only. Not used in SURFACE variant.

  {metric_clause}
    Source:  extract_primary_metric(career_history[0].description)
    Rule:
      tech_cat DEEP or STRONG:
        metric not None → ", {metric['converted_participle']}"
        metric is None  → ""  (omit including comma)
      tech_cat MODERATE:
        metric not None → ", with {metric['converted_noun']} achieved"
        metric is None  → ""  (omit including comma)
      tech_cat SURFACE:
        always ""  (no metric in SURFACE variant)
    Example (DEEP):    ", improving revenue-per-search by 12%"
    Example (MODERATE): ", with a 12% revenue-per-search improvement achieved"

  {desc_tech_clause}
    Source:  extract_description_tech(career_history[0].description)
    Rule:
      2 items  →  ", using {item[0]} and {item[1]}"
      1 item   →  ", using {item[0]}"
      0 items  →  ""
    Example: ", using sentence-transformers and FAISS"
    Rule:    Omitted in SURFACE variant.

  {skill_clause}
    Source:  extract_verified_skill(skill_assessment_scores)
             extract_named_tech(skills, all_descriptions)
    Rule:
      verified_skill not None → "backed by verified {verified_skill} depth"
      verified_skill is None  → "with {named_tech} experience"
    Example: "backed by verified Elasticsearch depth"
             "with Qdrant experience"

  {jd_alignment}
    Source:  tech_cat
    Values:
      DEEP     → "directly matching the role's retrieval and ranking requirements"
      STRONG   → "aligning with the role's retrieval and ranking requirements"
      MODERATE → "showing partial alignment to the role's retrieval and ranking focus"
    Rule:    Not used in SURFACE variant (SURFACE has its own closing clause).

  {tech_or_named_clause}  [SURFACE only]
    Source:  extract_named_tech(skills, all_descriptions)
    Rule:
      named_tech not empty → " including {named_tech} exposure"
      named_tech empty     → ""

FULLY RESOLVED EXAMPLES:

  Mira Verma — DEEP:
    "Mira Verma brings 6-plus-year years of production ML experience
     at Uber and Flipkart, most recently developing and scaling the
     ranking pipeline at Uber, improving revenue-per-search by 12%,
     backed by verified Elasticsearch depth directly matching the role's
     retrieval and ranking requirements"

  Tanya Chopra — STRONG (with description tech):
    "Tanya Chopra brings 6-plus-year years of production ML experience
     across Zoho, Observe.AI, and Paytm, most recently designing and
     deploying the semantic search system at Zoho, delivering a 35%
     arch-relevance improvement, using sentence-transformers and FAISS,
     with Qdrant experience aligning with the role's retrieval and
     ranking requirements"

  Ayaan Tiwari — MODERATE:
    "Ayaan Tiwari brings 5-plus-year years of ML engineering experience
     across Wysa, Glance, and Meta, most recently constructing and
     maintaining the ranking pipeline at Wysa, with a 12%
     revenue-per-search improvement achieved, with Elasticsearch
     exposure showing partial alignment to the role's retrieval
     and ranking focus"

  Pranav Trivedi — SURFACE:
    "Pranav Trivedi's profile, spanning Saarthi.ai, Unacademy, and
     Swiggy, shows keyword-level alignment to retrieval systems
     including Weaviate exposure, but limited semantic depth against
     the role's ranking and embedding requirements"

  Kavya Naidu — DEEP, no verified skill:
    "Kavya Naidu brings 5-plus-year years of production ML experience
     at Razorpay and InMobi, most recently architecting and shipping
     the ranking pipeline at Razorpay, improving revenue-per-search
     by 12%, with Elasticsearch experience directly matching the role's
     retrieval and ranking requirements"

FUNCTION: assemble_sentence_1(
    name, years_label, experience_type, company_scope,
    verb_gerund, system_type, company_0,
    metric, desc_tech, verified_skill, named_tech, tech_cat
) -> str
```

---

### Sentence 2 Template

```
PURPOSE: Establishes career quality and JD filter status. Answers "what
         kind of engineer are they and do they clear the explicit
         requirements."

TEMPLATE:
  "{career_characterization}{optional_qualifiers}, {disqualifier_statement}
   {optional_sweet_spot}{optional_tech_caveat}"

SLOT DEFINITIONS:

  {career_characterization}
    Source:  calculate_career_characterization(product_frac, consulting_count)
    Example: "career is entirely at product companies"
    Rule:    Always first element.

  {optional_qualifiers}
    Source:  score_tenure_signal(avg_tenure)
             score_pre_llm_signal(pre_llm)
    Rule:
      both signals are None       → ""
      one signal not None         → ", with {signal}"
      both signals not None       → ", with {signal1} and {signal2}"
    Example: ", with stable tenure and pre-LLM production ML ownership"

  {disqualifier_statement}
    Source:  calculate_disqualifier_statement(consulting_count, llm_only, recent_ai)
    Example: "clearing the JD's explicit disqualifiers"
             "though consulting firm history is an explicit JD disqualifier"
    Rule:    Always present, joined with comma after optional_qualifiers.

  {optional_sweet_spot}
    Source:  score_sweet_spot_signal(in_sweet_spot)
    Rule:
      not None → ", {sweet_spot_signal}"
      None     → ""
    Example: ", placing them in the role's target experience band"

  {optional_tech_caveat}
    Source:  score_tech_depth_caveat(tech_cat)
    Rule:
      not None → "; {tech_caveat}"
      None     → ""
    Example: "; cross-encoder alignment is partial — direct technical evaluation recommended"

FULLY RESOLVED EXAMPLES:

  Mira Verma:
    "Career is entirely at product companies, with stable tenure and
     pre-LLM production ML ownership, clearing the JD's explicit
     disqualifiers, placing them in the role's target experience band"

  Ayaan Tiwari (MODERATE):
    "Career is entirely at product companies, with pre-LLM production
     ML ownership, clearing the JD's explicit disqualifiers; cross-encoder
     alignment is partial — direct technical evaluation recommended"

FUNCTION: assemble_sentence_2(
    career_characterization, tenure_signal, pre_llm_signal,
    disqualifier_statement, sweet_spot_signal, tech_depth_caveat
) -> str
```

---

### Sentence 3 Template

```
PURPOSE: Availability and hiring logistics. Answers "can we hire them
         and how hard will outreach be."

TEMPLATE:
  "[{activity_signal}; ]{notice_label} with {availability_assessment}
   [; {response_signal}][; {github_signal}][; {applications_signal}]
   — {outreach_recommendation}"

SLOT DEFINITIONS:

  {activity_signal}
    Source:  score_activity_signal(days_since_active)
    Rule:
      not None → prepend as first element, followed by "; "
      None     → omit entirely
    Example: "63 days since last platform login; "

  {notice_label}
    Source:  calculate_notice_days_label(notice_period_days)
    Example: "60-day notice period"
    Rule:    Always present. Starts the sentence if activity_signal is None,
             second element if activity_signal is not None.

  {availability_assessment}
    Source:  calculate_availability_assessment(...)
    Example: "moderate friction — timeline confirmation needed"
    Rule:    Always present, joined with "with" after notice_label.

  {response_signal}
    Source:  score_response_signal(recruiter_response_rate)
    Rule:    Appended with "; " if not None.

  {github_signal}
    Source:  score_github_signal(github_activity_score)
    Rule:    Appended with "; " if not None.

  {applications_signal}
    Source:  score_applications_signal(applications_30d, open_to_work)
    Rule:    Appended with "; " if not None.

  {outreach_recommendation}
    Source:  calculate_outreach_recommendation(tech_cat, availability_assessment)
    Example: "recommended for outreach with timeline negotiation"
    Rule:    Always last, preceded by " — " (em dash with spaces).

ASSEMBLY RULE:
  Build a list of non-None parts.
  Join all parts with "; ".
  Append " — {outreach_recommendation}" at the end.

FULLY RESOLVED EXAMPLES:

  Mira Verma (DEEP, moderate friction, 63 days):
    "63 days since last platform login; 60-day notice period with
     moderate friction — timeline confirmation needed — recommended
     for outreach with timeline negotiation"

  Tanya Chopra (STRONG, low friction):
    "30-day notice period with active availability with manageable
     timeline — solid outreach candidate"

  Ayaan Tiwari (MODERATE, manageable friction):
    "60-day notice period with moderate friction — timeline
     confirmation needed — worth outreach pending direct technical
     evaluation"

FUNCTION: assemble_sentence_3(
    activity_signal, notice_label, availability_assessment,
    response_signal, github_signal, applications_signal,
    outreach_recommendation
) -> str
```

---

## Section 5 — Paraphraser Configuration

```
MODEL:  humarin/chatgpt_paraphraser_on_T5_base

INPUT FORMAT:  "paraphrase: {raw_sentence}"

FIXED PARAMETERS:
  num_beams             = 5
  num_beam_groups       = 5
  num_return_sequences  = 5
  diversity_penalty     = 1.5   (reduced from 3.0 — prevents hallucination on
                                  technical phrases with numbers)
  repetition_penalty    = 3.0   (reduced from 10.0 — less aggressive)
  no_repeat_ngram_size  = 2
  max_length            = 128

Each sentence is paraphrased independently.
Sentences are NEVER concatenated before paraphrasing.
Returns list of 5 string variations.

FUNCTION: load_paraphraser() -> Callable
  Returns a paraphrase(text: str) -> list[str] function.
  Model loaded once and reused for all sentences and all candidates.
```

---

## Section 6 — Variation Selection

```
MECHANISM: Deterministic seed using hashlib.md5.
           Same candidate_id + slot always produces same index.
           Different slots produce different indices.

WHY hashlib.md5:
  Python's built-in hash() is randomised by PYTHONHASHSEED.
  hash("CAND_0018549:s1") returns a different value every Python run.
  hashlib.md5 is stable across all runs, all platforms.

FORMULA:
  seed = int(hashlib.md5((candidate_id + ":" + slot).encode()).hexdigest(), 16)
         % len(variations)
  selected = variations[seed]

SLOTS USED:
  "verb"   →  verb selection (not a paraphrase slot)
  "s1"     →  sentence 1 variation
  "s2"     →  sentence 2 variation
  "s3"     →  sentence 3 variation

FUNCTION: stable_seed(text: str, modulus: int) -> int
FUNCTION: select_variation(variations: list[str], candidate_id: str, slot: str) -> str
```

---

## Section 7 — Reconstruction

```
Three selected variations joined with period-space.
Each sentence stripped of trailing period before joining to avoid double periods.

FORMULA:
  reasoning = (
      s1.rstrip(".") + ". " +
      s2.rstrip(".") + ". " +
      s3.rstrip(".") + "."
  )

OUTPUT: one string — the final reasoning for this candidate.
```

---

## Section 8 — Orchestrator

```
FUNCTION: build_reasoning(candidate: dict, paraphrase_fn: Callable) -> dict

INPUT:  candidate dict (full team_results.json schema)
        paraphrase_fn (loaded once by load_paraphraser())

EXECUTION ORDER:
  1.  Extract all field values from candidate dict using paths in Input Field Map
  2.  Run all base calculations (Section 1) — 8 functions
  3.  Run all evidence extractions (Section 2) — 7 functions
  4.  Run all priority signal scorers (Section 3) — 8 functions
  5.  Select verb: stable_seed(cid + ":verb", 5) → VERB_POOL[seed]
      Get gerund: VERB_GERUND_MAP[verb]
  6.  Assemble sentence 1 (Section 4)
  7.  Assemble sentence 2 (Section 4)
  8.  Assemble sentence 3 (Section 4)
  9.  Paraphrase s1_raw → 5 variations
  10. Paraphrase s2_raw → 5 variations
  11. Paraphrase s3_raw → 5 variations
  12. Select s1: select_variation(vars, cid, "s1")
  13. Select s2: select_variation(vars, cid, "s2")
  14. Select s3: select_variation(vars, cid, "s3")
  15. Reconstruct: s1 + ". " + s2 + ". " + s3 + "."

OUTPUT DICT:
  {
    "candidate_id":  str
    "tech_cat":      str
    "s1_raw":        str   (sentence 1 before paraphrase)
    "s2_raw":        str   (sentence 2 before paraphrase)
    "s3_raw":        str   (sentence 3 before paraphrase)
    "s1_selected":   str   (selected paraphrase variation for sentence 1)
    "s2_selected":   str   (selected paraphrase variation for sentence 2)
    "s3_selected":   str   (selected paraphrase variation for sentence 3)
    "reasoning":     str   (final reconstructed reasoning string)
  }
```

---

## Section 9 — Implementation Guide for Coding Agent

### File name
```
reasoning_builder.py
```

### Function list — in implementation order

```
CONSTANTS (no functions, just assignments at module level):
  PARAPHRASE_MODEL, VERB_POOL, VERB_GERUND_MAP, SYSTEM_TYPE_PATTERNS,
  METRIC_VERB_RE, METRIC_SCALE_RE, METRIC_LATENCY_RE, METRIC_PLAIN_PCT_RE,
  PARTICIPLE_MAP, DESCRIPTION_TECH_RE, IRRELEVANT_SKILL_SET,
  SKILL_CATEGORIES, CATEGORY_KEYWORDS

BASE CALCULATIONS:
  calculate_tech_cat(cross_encoder_score: float) -> str
  calculate_years_label(total_years_exp: float) -> str
  calculate_experience_type(pre_llm: bool) -> str
  calculate_notice_days_label(notice_days: int) -> str
  calculate_availability_assessment(days_since_active, notice_days,
      open_to_work, applications_30d, offer_acceptance) -> str
  calculate_outreach_recommendation(tech_cat, availability_assessment) -> str
  calculate_career_characterization(product_frac, consulting_count) -> str
  calculate_disqualifier_statement(consulting_count, llm_only, recent_ai) -> str

EVIDENCE EXTRACTION:
  extract_system_type(description: str) -> str
  extract_primary_metric(description: str) -> dict | None
  extract_description_tech(description: str) -> list[str]
  extract_company_scope(career_history: list) -> tuple[list[str], str]
  extract_named_tech(skills: list, descriptions: list[str]) -> str
  extract_verified_skill(skill_assessment_scores: dict) -> str | None
  extract_best_metric_from_prior_roles(career_history: list) -> dict | None

PRIORITY SIGNAL SCORING:
  score_tenure_signal(avg_tenure: float) -> str | None
  score_pre_llm_signal(pre_llm: bool) -> str | None
  score_sweet_spot_signal(in_sweet_spot: bool) -> str | None
  score_tech_depth_caveat(tech_cat: str) -> str | None
  score_activity_signal(days_since_active: int) -> str | None
  score_response_signal(recruiter_response_rate: float) -> str | None
  score_github_signal(github_activity_score: float) -> str | None
  score_applications_signal(applications_30d: int, open_to_work: bool) -> str | None

SENTENCE ASSEMBLY:
  assemble_sentence_1(name, years_label, experience_type, company_scope,
      verb_gerund, system_type, company_0, metric, desc_tech,
      verified_skill, named_tech, tech_cat) -> str
  assemble_sentence_2(career_characterization, tenure_signal, pre_llm_signal,
      disqualifier_statement, sweet_spot_signal, tech_depth_caveat) -> str
  assemble_sentence_3(activity_signal, notice_label, availability_assessment,
      response_signal, github_signal, applications_signal,
      outreach_recommendation) -> str

PARAPHRASE AND SELECTION:
  stable_seed(text: str, modulus: int) -> int
  select_variation(variations: list[str], candidate_id: str, slot: str) -> str
  load_paraphraser() -> Callable[[str], list[str]]

ORCHESTRATOR:
  build_reasoning(candidate: dict, paraphrase_fn: Callable) -> dict
```

### Test runner

Hardcode the five test candidates at the bottom of the file.
Candidate dicts must match the full team_results.json schema exactly
including the nested pipeline structure.

Test candidates (use the full dicts from the batch already provided):
```
CAND_0018549  Mira Verma     DEEP
CAND_0061265  Tanya Chopra   STRONG
CAND_0086151  Ayaan Tiwari   MODERATE
CAND_0018722  Pranav Trivedi SURFACE
CAND_0051630  Kavya Naidu    DEEP (no verified skill)
```

Run the test and for each candidate print:
```
=== {candidate_id} ({tech_cat}) ===
S1 RAW:       {s1_raw}
S2 RAW:       {s2_raw}
S3 RAW:       {s3_raw}
S1 SELECTED:  {s1_selected}
S2 SELECTED:  {s2_selected}
S3 SELECTED:  {s3_selected}
REASONING:    {reasoning}
```

Save all five result dicts to `reasoning_output.json`.

### Raw sentence verification checks

Before running the paraphraser, verify these conditions for CAND_0018549.
Print PASS or FAIL with actual value for each check.

```
S1 starts with "Mira Verma brings"
S1 contains "at Uber and Flipkart"
S1 contains "ranking pipeline"
S1 contains "improving revenue-per-search by 12%"
S1 contains "Elasticsearch"
S1 ends with "retrieval and ranking requirements"

S2 starts with "career is entirely at product companies"  (case-insensitive)
S2 contains "stable tenure"
S2 contains "pre-LLM production ML ownership"
S2 contains "clearing the JD's explicit disqualifiers"

S3 contains "63 days since last platform login"
S3 contains "60-day notice period"
S3 contains "moderate friction"
S3 ends with "timeline negotiation"
```

Do not proceed to paraphrase until all checks PASS.
