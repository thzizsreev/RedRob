# Q1/Q2 Vector Calibration — Test Cases
## 5 Inputs, Expected Outputs, Acceptance Criteria

---

## HOW TO USE THIS

For each config variant you want to test:
1. Encode each test input against Q1 and Q2 query vectors
2. Record the raw dot product score
3. Compare against expected output ranges below
4. A config passes only if ALL 5 cases meet their expected range simultaneously

---

## TEST CASE 1 — The Ideal Keyword-Rich Candidate
*Represents your current top 2-3. Must always score high. If this drops, the config is broken.*

**Input text:**
```
Built and operated a hybrid retrieval system at scale using FAISS and Elasticsearch.
Designed sentence-transformer embeddings pipeline with BGE-large, handled embedding drift
and incremental index refresh. Implemented NDCG, MRR, and MAP evaluation framework with
offline-to-online correlation analysis. Ran A/B tests on ranking changes. Shipped
learning-to-rank model using XGBoost trained on click-through and engagement signals.
Served 10M+ users at sub-50ms p95 latency. 7 years at product companies, Flipkart and Uber.
```

**Why this case:** Pure keyword-rich, tool-explicit, metric-explicit. Baseline for how high
a perfect candidate should score. Every config must score this as the reference ceiling.

**Expected Q1 score:** `≥ 0.90`
**Expected Q2 score:** `≥ 0.88`
**Role:** Ceiling anchor. If any config scores this below 0.88, reject that config immediately.

---

## TEST CASE 2 — Myra: Outcome Language, No Keywords
*The primary failure case. Currently q1_rank ~170. Must improve.*

**Input text:**
```
Rebuilt the candidate-JD matching pipeline from scratch, taking NDCG@10 from 0.72 to 0.91,
operating at single-digit-millisecond p95 retrieval latency. Migrated keyword-only retrieval
to hybrid setup combining sparse and dense vectors with domain fine-tuned models. Reduced
p95 retrieval latency by 60% while improving relevance on held-out eval set by 18%.
Spent substantial time on incremental index refresh, embedding drift monitoring, and
offline-online metric correlation. Fine-tuned LLaMA-2-7B and Mistral-7B for domain-specific
matching using preference pairs from recruiter labels. Built end-to-end ranking pipeline:
embedding generation, vector retrieval, learning-to-rank re-scoring, behavioral signal
integration. 7.8 years across Ola, Zomato, Amazon — all product companies.
```

**Why this case:** This is Myra's actual career description language. Rich in outcomes
and operational detail but avoids tool brand names (no "Qdrant", no "Pinecone", no "RAG").
The current single-query vector doesn't recognise her. A good config must.

**Expected Q1 score:** `≥ 0.87` (currently scoring ~0.82, needs meaningful lift)
**Expected Q2 score:** `≥ 0.86`
**Expected Q1 rank vs Test Case 1:** within 0.05 score gap (she should be close to ceiling,
not 0.08+ below)
**Role:** Primary rescue target. The config lives or dies on this case.

---

## TEST CASE 3 — Strong Candidate, Mixed Language
*Represents your current rank 5-15. Must stay consistently high.*

**Input text:**
```
Senior ML engineer with 7 years at product companies. Shipped semantic search system
for 500K document corpus using sentence-transformers upgraded to bge-base, with FAISS
nearest-neighbor retrieval and query expansion for vocabulary mismatch. Improved search
relevance 35% over prior BM25 setup validated through human relevance judgments.
Owned ranking layer evolution from hand-tuned scoring to learning-to-rank over 9 months.
Designed relevance labeling pipeline using click-through and explicit human judgments.
Improved revenue-per-search by 12%. Qdrant and OpenSearch production experience.
```

**Why this case:** Mix of outcome language ("improved relevance 35%") and explicit tools
("Qdrant", "FAISS", "bge-base"). Represents the majority of your strong candidates.
Should score high on both the keyword facet and outcome facet.

**Expected Q1 score:** `≥ 0.88`
**Expected Q2 score:** `≥ 0.86`
**Role:** Middle-ground validator. Confirms the centroid doesn't break candidates who
already work well.

---

## TEST CASE 4 — Good Skills, Wrong Career Shape
*Someone who knows the tools but has the wrong background. Should score high on Q1 but low on Q2.*

**Input text:**
```
Research engineer at academic lab with 5 years experience. Published papers on dense
retrieval using FAISS and Qdrant. Implemented NDCG and MAP evaluation for information
retrieval benchmarks. Built RAG pipeline using Pinecone and LangChain for research
prototype. Expert in sentence-transformers, BGE, E5 embeddings. Strong Python.
No production deployments to real users. Work is entirely in research context
with no commercial product experience.
```

**Why this case:** Tests that Q1 and Q2 discriminate independently. This candidate
knows the vocabulary and tools (should score reasonably on Q1) but has never shipped
to real users and has no product company background (should score low on Q2).
If Q2 can't distinguish this from Test Case 1, Q2 is broken.

**Expected Q1 score:** `0.80 – 0.88` (knows tools but no production depth)
**Expected Q2 score:** `≤ 0.78` (wrong career shape — research lab, no product company)
**Expected Q1-Q2 gap:** Q2 must be at least 0.06 lower than Q1
**Role:** Q1 vs Q2 discrimination check.

---

## TEST CASE 5 — Keyword Stuffer, Hollow Experience
*The trap the JD explicitly warns about. Must score low.*

**Input text:**
```
AI Engineer with experience in RAG, LangChain, Pinecone, Qdrant, FAISS, Weaviate,
Milvus, OpenSearch, Elasticsearch, sentence-transformers, BGE, E5, NDCG, MRR, MAP,
A/B testing, embeddings, semantic search, vector databases, retrieval systems,
recommendation systems, LLM fine-tuning, LoRA, QLoRA, PEFT, learning-to-rank.
Currently building LangChain demos and tutorial projects. 3 years experience.
No production systems shipped to real users at scale.
```

**Why this case:** Every keyword from the JD is present. No actual work described.
This is the keyword stuffer the plan.md explicitly warns about. A config that scores
this near Test Case 1 is broken — it's just doing keyword matching, not semantic
understanding. INSTRUCTOR should penalise the absence of operational context.

**Expected Q1 score:** `≤ 0.82`
**Expected Q2 score:** `≤ 0.75`
**Expected gap vs Test Case 1:** Q1 must be at least 0.08 lower than Test Case 1
**Role:** Anti-gaming check. If this scores close to Test Case 1, the config fails.

---

## SCORING SUMMARY TABLE

Fill this in after running each config:

```
Config          | TC1 Q1 | TC1 Q2 | TC2 Q1 | TC2 Q2 | TC3 Q1 | TC4 Q1 | TC4 Q2 | TC5 Q1 | TC5 Q2 | PASS?
----------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|------
0 (baseline)    |        |        |        |        |        |        |        |        |        |
1 (equal)       |        |        |        |        |        |        |        |        |        |
2 (outcome+)    |        |        |        |        |        |        |        |        |        |
3 (keyword+)    |        |        |        |        |        |        |        |        |        |
...             |        |        |        |        |        |        |        |        |        |
```

**A config passes if:**
- TC1 Q1 ≥ 0.90
- TC2 Q1 ≥ 0.87 AND TC2 Q2 ≥ 0.86
- TC3 Q1 ≥ 0.88
- TC4 Q2 ≤ 0.78 AND (TC4 Q1 - TC4 Q2) ≥ 0.06
- TC5 Q1 ≤ 0.82 AND (TC1 Q1 - TC5 Q1) ≥ 0.08

Pick the passing config where TC2 Q1 is highest (Myra improved most).
