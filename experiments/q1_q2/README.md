# Q1/Q2 Vector Test

**One formula. One config. Five test cases.**

## Formula

For each query (Q1 and Q2):

```
v_f = encode_weighted_query(facet_text_f, subspace_weights)   # per facet
Q   = sum(alpha_f * v_f)                                      # alphas sum to 1.0
score = dot(candidate_vector, Q)
```

Same Q1 and Q2 vectors score all five synthetic cases (and the full candidate pool when integrated).

## Config

Everything is in `input/configs.yaml`:

- `facets.q1` / `facets.q2` — facet texts (fixed)
- `subspace.q1` / `subspace.q2` — retrieval/infra/eval block weights (fixed)
- `weights.q1` / `weights.q2` — **alphas that combine facet vectors** (tune these)

## Run

```bash
python experiments/q1_q2/run_test.py
```

## Pass

- TC1 Q1 ≥ 0.90
- TC2 Q1 ≥ 0.87 and Q2 ≥ 0.86
- TC3 Q1 ≥ 0.88
- TC4 Q2 ≤ 0.78 and (Q1 − Q2) ≥ 0.06
- TC5 Q1 ≤ 0.82 and (TC1 Q1 − TC5 Q1) ≥ 0.08

See [docs/plans/q1q2_vector_test_plan.md](../../docs/plans/q1q2_vector_test_plan.md).
