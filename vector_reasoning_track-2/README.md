# Vector Reasoning Steering Experiment

Isolated test of Plan 2 — template + candidate blend with dimensional anchor steering and dual decode modes.

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) and [docs/vector_steering_plan_2_final.md](../docs/vector_steering_plan_2_final.md).

## Quick start (recommended)

```bash
cd vector_reasoning_test
pip install -r requirements.txt
python precompute.py
python run_experiment.py
```

Output: `results/output.json` with related recruiter reasoning (default `template_hybrid` decode).

## Decode modes

| Mode | Command | When to use |
|------|---------|-------------|
| `template_hybrid` | `python run_experiment.py` | **Default.** Reliable output with template + resume-specific clauses |
| `langvae` | `python run_experiment.py --decode langvae` | After `validate_langvae.py` passes on Python 3.11 |

## LangVAE environment (optional)

LangVAE decode requires a pinned stack matching the HF checkpoint:

```bash
# Python 3.11 recommended
pip install -r requirements-pinned.txt
python validate_langvae.py
```

If the gate fails, stay on `--decode template_hybrid`.

## Models

| Role | Model |
|------|--------|
| Encode (templates, anchors, resume) | LangVAE encoder → 128d μ latent |
| Decode (`template_hybrid`) | Deterministic [`compose.py`](compose.py) |
| Decode (`langvae`) | LangVAE `neuro-symbolic-ai/eb-langvae-bert-base-cased-gpt2-l128` |

LangVAE checkpoint downloads from HuggingFace on first run (~800MB, cached locally).

## Inputs

All test inputs are hardcoded in [`constants.py`](constants.py):

- `resume_tech_text`, `resume_career_text`, `resume_behav_text`
- `s_tech`, `s_career`, `s_behav`

No imports from `tracks/` or other repo packages.
