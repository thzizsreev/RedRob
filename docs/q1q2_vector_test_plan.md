# Q1/Q2 Vector Test Plan

## Purpose

Build **one Q1 vector** and **one Q2 vector** for Stage 3 retrieval.  
Same vectors score every candidate. Stage 5 uses the resulting `q1_score` / `q2_score` ranks.

- **Q1** — technical retrieval depth: production ops, outcomes, evaluation, hybrid architecture.
- **Q2** — career depth: product-company trajectory, shipper mindset, senior IC ownership.

## One formula

```
For each facet f:
  v_f = encode_weighted_query(model, facet_text_f, subspace_weights)

Q1 = sum over q1 facets of  alpha_f * v_f     (alpha sums to 1.0)
Q2 = sum over q2 facets of  alpha_f * v_f

score = candidate_vector @ Q
```

- `subspace_weights` — fixed per Q1/Q2 (retrieval / infra / eval block mix). Production values.
- `alpha_f` — **the only tuning knobs**: how much each facet contributes to the final Q1 or Q2 vector.

No per-candidate weighting. No multiple configs. One weight block in `experiments/q1_q2/input/configs.yaml`.

## Facets

**Q1 (4 facets)**  
`A_operational`, `B_outcome_language`, `C_evaluation_depth`, `D_hybrid_architecture`

**Q2 (3 facets)**  
`A_product_trajectory`, `B_shipper_mindset`, `C_senior_ic_ownership`

Texts and weights: `experiments/q1_q2/input/configs.yaml`.

## Pass criteria (five synthetic cases)

| Case | Rule |
|------|------|
| TC1 | Q1 ≥ 0.90 |
| TC2 | Q1 ≥ 0.87 and Q2 ≥ 0.86 |
| TC3 | Q1 ≥ 0.88 |
| TC4 | Q2 ≤ 0.78 and (TC4 Q1 − TC4 Q2) ≥ 0.06 |
| TC5 | Q1 ≤ 0.82 and (TC1 Q1 − TC5 Q1) ≥ 0.08 |

Tune `weights.q1` and `weights.q2` until all pass. Re-run:

```bash
python experiments/q1_q2/run_test.py
```

## After pass

Copy facet texts + alphas + subspace weights into `config.yaml` / Stage 3 query precompute, re-run Stage 3→5.

## Inputs

- Vectors: `artifacts/runtime/stage0/candidate_vectors.npy`
- ONNX: `onnx/models/instructor-large-encoder.onnx`
- Test cases: `experiments/q1_q2/input/synthetic_cases.yaml`
