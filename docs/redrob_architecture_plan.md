# Redrob Hackathon — Candidate Ranking Architecture Plan

**Challenge:** Intelligent Candidate Discovery & Ranking Challenge
**Goal:** Rank top 100 candidates from a 100K pool against the Senior AI Engineer JD
**Compute constraints:** CPU-only, 16GB RAM, 5-minute runtime, no hosted LLM calls during ranking

---

## Why This Architecture

The JD explicitly states their current system is "mostly BM25 + rule-based scoring, working but not great" — that's our floor to beat, not a strawman. The hackathon also explicitly built in traps (keyword stuffers, plain-language experts, behavioral twins, ~80 honeypots) designed to break naive keyword-matching or naive embedding-only approaches. So the architecture below leans on **time-tested, well-understood techniques** — lexical retrieval, dense retrieval, rank fusion, gradient-boosted ranking over engineered features — rather than novel or overengineered components. Each phase is independently testable before the next one builds on it.

---

## Phase 0 — Data Validation & Honeypot/Disqualification Masking

**Goal:** Build a reusable boolean mask over all 100K candidates. Not a filtered copy — a mask, so later phases can use it as a hard exclude or a soft feature.

**Checks to implement (deterministic, schema-grounded):**

| Check | Logic | Source |
|---|---|---|
| Expert skill, ~0 experience | `skills[].proficiency == "expert"` AND `duration_months` near 0 | Documented honeypot example |
| Experience sum mismatch | `sum(career_history[].duration_months)` vs `profile.years_of_experience` — flag large divergence | Generalized from honeypot pattern |
| Education/career timeline conflict | Career history starting before plausible degree completion | Inferred |
| Schema-boundary artifacts | Values sitting exactly at field min/max (often synthetic-data tells) | Inferred |
| Pure research, no production | Career history with no production-deployment signal at all | JD hard disqualifier |
| Consulting-only career | Every employer is TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini, no exceptions | JD hard disqualifier |

**Known limitation:** the "8 years at a company founded 3 years ago" example needs a `company_founded_year` field that **does not exist** in `candidate_schema.json`. This honeypot type is only catchable via text pattern matching against `career_history[].description` / `profile.summary` (e.g. regex for "founded in", "since our founding"), and even then with limited recall. We accept this gap rather than over-engineer a fragile text-parsing solution — we only need top-100 honeypot rate under 10%, not full-pool recall.

**Output:** `candidate_id -> {is_honeypot, is_hard_disqualified, reasons[]}`

---

## Phase 1 — Lexical Retrieval (BM25)

**Goal:** Fast, dependency-light baseline ranking. Directly comparable to the JD's stated current system.

- Concatenate per candidate: headline + summary + career_history descriptions + skills as sentences.
- Build a BM25 index (`rank_bm25` or similar) over all 100K.
- Query with terms drawn from the JD's "things you absolutely need" section.
- **This is our sanity-check floor** — every later phase should be compared against this baseline to confirm we're actually improving, not just adding complexity.

---

## Phase 2 — Dense Retrieval + Rank Fusion

**Goal:** Catch what BM25 misses — specifically the JD's named trap: a "Tier 5 plain-language expert" who built a recommendation system without ever writing "RAG" or "Pinecone."

- One embedding per candidate (single passage vector — not a block-weighted multi-vector scheme; added complexity isn't justified at this scale).
- Small CPU-friendly bi-encoder (e.g. `bge-small` or `MiniLM`), flat or simple FAISS index.
- Query with a JD-derived query string.
- **Fuse BM25 ranking + embedding ranking via Reciprocal Rank Fusion (RRF)** — standard, well-tested, no tuning of fusion weights required.
- BM25 anchors against the inverse trap: keyword stuffers who'd score artificially high on embedding similarity alone.
- **Output:** shortlist of a few hundred to ~2,000 candidates (not the final 100).

---

## Phase 3 — Feature Engineering (Shortlist Only)

Now cheap, since the pool is much smaller. Compute per shortlisted candidate:

- BM25 score, embedding similarity score
- Years-of-experience fit vs JD's 5–9 year band (soft range per JD's own caveat)
- Location/relocation fit (Pune/Noida preference)
- Notice period fit (JD prefers sub-30-day)
- Phase 0 mask outputs (honeypot, hard-disqualified)
- All 23 `redrob_signals` as direct features (recruiter response rate, last active date recency, profile completeness, GitHub activity, etc.)
- JD negative-space patterns: title-chaser trajectory (short tenures + title inflation), framework-enthusiast signal, CV/speech/robotics-without-NLP pattern

---

## Phase 4 — Learned Ranking Over the Shortlist

- Start with a **documented, defensible weighted formula** over Phase 3 features rather than a black-box trained model — no labels exist, and a transparent formula is stronger for the Stage 4 "defend your work" interview since every weight traces back to a stated JD priority.
- If desired, self-label a small sample and train a LightGBM/XGBoost ranker on top — standard second-stage ranking approach, fast to train on CPU.
- Mirrors the JD's own philosophy: "ship a working ranker in a week even if obviously suboptimal."

---

## Phase 5 — Output Assembly & Validation

- Sort by final score, take top 100.
- Generate `reasoning` column — specific, candidate-grounded (cite actual matching evidence), never templated or hallucinated. Stage 4 manual review penalizes generic or contradictory reasoning.
- Enforce monotonic non-increasing scores; break ties deterministically (secondary signal or `candidate_id` ascending).
- Run `validate_submission.py`.
- Confirm honeypot rate in top 100 is comfortably under the 10% disqualification threshold.

---

## Why Phase Order Matters

Each phase has a natural checkpoint to catch errors before they compound:

1. Confirm BM25 alone looks reasonable before adding embeddings.
2. Confirm the fused shortlist excludes obvious keyword stuffers and includes obvious plain-language fits before building features on top of it.
3. Confirm the honeypot mask catches the two documented examples cleanly before trusting it across the full 100K.

---

## Differences From `vector_encoding_plan.md` — and Why

This plan diverges from the earlier `vector_encoding_plan.md` draft in several deliberate ways. Documenting the reasoning so the team can weigh in or push back.

### Vector architecture: three-block weighted vectors → single dense vector + BM25 fusion

`vector_encoding_plan.md` builds a 1152-d vector per candidate as three independent 384-d sub-vectors (retrieval/infra/eval), each requiring its own anchor construction, threshold-tuned sentence classification, and separate encode call — four BGE calls per candidate, 400K model calls total. This plan uses one embedding per candidate and leans on BM25 as the primary lexical signal, fused via Reciprocal Rank Fusion.

**Why:** the block-weighted design solves a precision problem (controlling exactly how much "infra" vs "retrieval" vs "eval" experience contributes to score) that's elegant in theory but expensive and fragile in practice — it requires hand-tuning three thresholds (0.35/0.38/0.32) against manually labeled data, careful anchor sentence selection, and is vulnerable to its own documented failure mode (cross-block text leakage during sentence extraction). A single embedding plus BM25 fusion gets most of the same discriminative power without that tuning surface. At 100K candidates, the added precision of block-diagonal weighting is unlikely to be the deciding factor in score quality; the bigger risk is threshold tuning being subtly wrong and degrading recall in a way that's hard to detect with no live leaderboard to catch it.

### Retrieval method: dense-only → hybrid (lexical + dense)

The original plan is dense-retrieval-only — FAISS HNSW over BGE embeddings, no lexical component. This plan makes BM25 the backbone, with dense retrieval as a complementary signal fused in.

**Why:** the JD states its current system is BM25-based, making BM25 the natural baseline to benchmark against rather than skip entirely. More importantly, dense-only retrieval has a specific failure mode this dataset is designed to punish: it can't distinguish a keyword stuffer's profile (semantically close to the query because it's saturated with the right vocabulary) from a genuinely strong candidate as cleanly as hybrid retrieval can, since BM25's term-frequency/document-length normalization naturally discounts vocabulary-stuffed-but-context-thin profiles in a way pure embedding similarity doesn't. Hybrid retrieval is also the more time-tested industry default — it's what Elasticsearch, Weaviate, and most production search systems converge on rather than dense-only.

### Honeypot/trap handling: implicit ("not by the vector pipeline") → explicit Phase 0 with concrete rules

The original plan mentions honeypots only to say they're out of scope for the vector pipeline and "handled downstream by the tabular mask" — no rules are specified anywhere in that document. This plan makes honeypot/disqualification masking Phase 0, with concrete, schema-grounded checks.

**Why:** the original plan was scoped narrowly to retrieval only (its own header states downstream ranking is "out of scope here"), so this is less a contradiction and more a gap that needed filling once we looked at the full pipeline rather than just the retrieval stage. Doing it first as a mask, rather than late as an afterthought, lets every downstream phase use it as either a hard filter or a feature without recomputing anything.

### Ranking stage: unspecified → LightGBM/weighted-formula with explicit JD-grounded features

The original plan stops at FAISS index construction and says downstream ranking is "out of scope," sketching only a high-level mention of LightGBM in its diagram with no feature list. This plan specifies Phase 3/4 concretely: years-of-experience fit, location/notice-period fit, all 23 `redrob_signals`, and JD negative-space patterns (title-chasers, framework enthusiasts).

**Why:** this mostly fills in what was always going to be needed but wasn't written yet — but the choice to default to a transparent weighted formula over a trained black-box model is a deliberate decision, not just a gap-fill. With no ground truth to train against, a documented formula is more defensible at the Stage 4 "defend your work" interview, since every weight traces back to a specific JD statement rather than to an opaque training process that would need explaining away.

### Net effect

The original plan is a deep, well-engineered solution to one sub-problem (precision-controlled retrieval) with real sophistication in it. This plan is broader and intentionally simpler across the *whole* pipeline, trading some retrieval precision for lower tuning risk, stronger baseline comparability, and full coverage of stages the original document explicitly didn't address.

---

## Open Items / Needs Before Full Build

- [ ] Confirm `company_founded_year` is genuinely absent from the dataset (only in `candidate_schema.json` currently reviewed) — if present elsewhere, Phase 0 honeypot coverage improves.
- [ ] Decide: heuristic weighted formula vs. self-labeled LightGBM for Phase 4.
- [ ] Confirm compute budget split: precompute (BM25 index, embeddings, FAISS index) has no time limit; only the `rank.py` step producing the CSV is bound by 5 min / 16GB / CPU-only.
