# TRACER — Trap-aware Retrieval And Career Evidence Ranker

**Redrob India Runs Track 1 — Senior AI Engineer**

TRACER is a two-phase candidate–JD ranker: offline semantic indexing + online trap-aware structured scoring. Designed for NDCG@10/50, honeypot survival (>10% traps in top-100 = disqualification), and Stage 3–5 reproduction.

---

## Why TRACER (not another embedding ranker)

| Layer | What it does |
|-------|----------------|
| **Hybrid retrieval** | MiniLM dense + TF-IDF sparse over full career text |
| **Multi-query JD facets** | 4 facet queries max-pooled — surfaces retrieval-only or LTR-only specialists |
| **Cross-encoder rerank** | ms-marco-MiniLM on coarse top-5K (~1 min GPU); bi-encoder fallback if download fails |
| **Structured features** | Title + career (44%), trust-weighted skills, YoE, location, assessments |
| **TRM (Trap-Risk Modulator)** | Multiplicative `(1 − trap_risk)` on soft trap signals |
| **Post-rank filter** | Rank top-150 → filter unsafe → emit 100 |
| **Evidence reasoning** | Cites actual career phrases and skill durations — no hallucination |

---

## Architecture

```
OFFLINE (GPU OK)                    ONLINE (CPU, no network)
─────────────────                   ─────────────────────────
candidates.jsonl                    stream 100K
    │                                   │
    ├─ hybrid semantic (MiniLM+TF-IDF)  ├─ O(1) semantic lookup
    ├─ multi-query facet scores         ├─ hard honeypot → 0
    └─ [optional] rerank top-5K         ├─ TRACER score + TRM
                                        ├─ top-150 heap
                                        ├─ post-filter → 100
                                        └─ submission.csv
```

### Score formula

```
semantic = 0.55×hybrid + 0.25×multi_query + 0.20×rerank_boost (0 if absent)

base = 0.22×semantic + 0.22×title + 0.22×career + 0.12×skill
     + 0.07×experience + 0.05×location + 0.07×assessment − penalties

final = base × (1 − trap_risk) × behavioral_multiplier
```

---

## Honeypot defense (4 layers)

1. **Hard reject** — expert-zero≥4, timeline impossible, junior+expert stuffing, negative titles
2. **TRM soft modulation** — desc reuse, keyword stuffer, title/skill mismatch (cap 0.45)
3. **Post-rank filter** — skip trap_risk≥0.35, weak title + low semantic
4. **`submit_guard.py`** — mandatory pre-upload; exit 1 if any hard trap in top-100

---

## Results (verified)

| Metric | Value |
|--------|-------|
| Rank time (100K CPU) | ~32s |
| Honeypots in top-100 | **0** |
| Negative titles | **0** |
| STRONG + GOOD | **94/100** |
| Top candidate | CAND_0018499 — Senior ML Engineer, 7.2 YoE |

---

## Reproduce

```bash
cd final

# Step 1: hybrid embeddings (one-time, or copy artifacts/)
gpy scripts/precompute_embeddings.py --backend hybrid --device cuda --candidates /path/to/candidates.jsonl

# Step 2: multi-query facets (~8 min GPU)
gpy scripts/precompute_mq.py --candidates /path/to/candidates.jsonl --artifacts artifacts --device cuda

# Step 3: cross-encoder rerank top-5K (~1 min GPU with ms-marco)
gpy scripts/precompute_rerank.py --candidates /path/to/candidates.jsonl --artifacts artifacts --device cuda

# Rank (CPU only)
python rank.py --candidates /path/to/candidates.jsonl --out submission.csv --artifacts artifacts

# Validate
python validate_submission.py submission.csv
python scripts/submit_guard.py --submission submission.csv --candidates /path/to/candidates.jsonl
python scripts/verify_top100.py --candidates /path/to/candidates.jsonl
```

---

## Stage 5 talking points

> "We built TRACER — offline hybrid+multi-query retrieval, optional cross-encoder rerank on the head, online structured scoring with a Trap-Risk Modulator. No LLM at inference. We audited 0 honeypots in top-100. The design mirrors how Redrob would ship candidate search: index once, rank fast, hard-reject traps."

---

## Layout

```
final/
├── rank.py
├── ranker/          # TRACER modules
├── scripts/         # precompute, submit_guard, verify
├── artifacts/       # generated locally (not in git)
├── submission.csv
└── ARCHITECTURE.md
```
