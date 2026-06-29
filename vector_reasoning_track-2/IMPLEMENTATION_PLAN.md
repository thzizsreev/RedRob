# Plan 3 SONAR — Implementation Summary

Isolated experiment under `vector_reasoning_track-2/`.

**Full spec:** [docs/vector_steering_plan_3_sonar.md](../docs/vector_steering_plan_3_sonar.md)

## Architecture

Three independent dimensions (Technical, Career, Behavioral):

1. Bucket score → `high` / `mid` / `low`
2. Load precomputed SONAR template + anchor vectors (`precompute.py`, 1024d)
3. Encode candidate resume snippet with SONAR encoder
4. Blend: `v_base = GAMMA * v_tmpl + (1-GAMMA) * v_cand` (GAMMA=0.55)
5. Steer: `v_steer = (1-s)*v_lo + s*v_hi`
6. Final: `v_final = (1-DELTA)*v_base + DELTA*v_steer` (DELTA=0.30)
7. Decode: `sonar_decode(v_final)` → one clause per dimension
8. Concatenate three clauses → `results/output.json` (no template text in output)

## Hyperparameters

| Param | Value |
|-------|-------|
| GAMMA | 0.55 |
| DELTA | 0.30 |
| MAX_SEQ_LEN | 64 |
| BEAM_SIZE | 5 (SONAR default) |
| Vector dim | 1024 |

## Folder layout

```
vector_reasoning_track-2/
├── constants.py
├── encode.py
├── decode.py
├── precompute.py
├── run_experiment.py
├── vectors/          # 15 × .npy (gitignored)
└── results/          # output.json (gitignored)
```

## Evaluation (manual)

1. **Tone** — reviewer-like, not raw resume bullets
2. **Specificity** — may paraphrase FAISS, Meesho, Senior MLE (31.5% candidate weight)
3. **Direction** — tech/career positive; behav concern at s=0.19
4. **Independence** — three distinct aspects
5. **Fluency** — no repeated/broken words (beam search)

## Known limitation

SONAR was trained for translation/autoencoding, not recruiter copy. Output should be fluent and directionally correct; surface style may not match a human recruiter exactly.
