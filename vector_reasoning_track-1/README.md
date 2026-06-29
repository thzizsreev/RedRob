# Plan 1 — Resume Base Steering

Self-contained experiment for Plan 1: single resume-base vector steering across three dimensions (technical, career, behavioral), with three independent GPT-2 decodes concatenated into final reasoning.

See [plan_1_resume_base_steering.md](plan_1_resume_base_steering.md) for the design spec.

## Quick start

```bash
cd vector_reasoning_track-1
pip install -r requirements.txt
python precompute.py
python validate_decode.py          # optional gate
python run_experiment.py           # default test 1B → results/output.json
python run_matrix.py               # 9 cases → results/matrix.json
```

First LangVAE run downloads ~800MB checkpoint from HuggingFace.

## Architecture

1. Encode one resume section → `v_candidate`
2. Load 6 precomputed anchor vectors (hi/lo per dimension)
3. Per dimension: interpolate anchors by score, blend with `v_candidate` at β
4. Decode each `v_final` via LangVAE's GPT-2 decoder
5. Concatenate three clauses (no template prefix)

## Vector space note

Plan pseudocode uses `bert-base-uncased` 768d mean-pool. This implementation uses LangVAE's 128d μ latent (`neuro-symbolic-ai/eb-langvae-bert-base-cased-gpt2-l128`) because raw GPT-2 cannot decode unconstrained 768d BERT vectors. Steering formulas are identical; only the latent dimension differs.

## CLI

```bash
python run_experiment.py --s-tech 0.88 --s-career 0.81 --s-behav 0.19 --beta 0.55
python run_experiment.py --resume-json ../data/sample_candidates.json --candidate-id CAND_0000001
```

## LangVAE environment (optional)

For reliable decode quality:

```bash
pip install -r requirements-pinned.txt   # Python 3.11 recommended
python validate_decode.py
```

## Known limitations

- LangVAE was pretrained on entailment-style text; recruiter reasoning output may be weak or generic
- Scores and resume are hardcoded or CLI-provided; no Stage 5 parquet wiring in v1
- Evaluate output using the failure-mode checklist in the plan (resume bleed, anchor drowning, repetition, contradiction)

## Models

| Role | Model |
|------|--------|
| Encode (resume + anchors) | LangVAE encoder → 128d μ |
| Decode | LangVAE GPT-2 decoder |

No imports from `tracks/` or other repo packages.
