# Vector Reasoning Steering — Implementation Plan

Isolated experiment under `vector_reasoning_test/`.

**Source documents:**

- [docs/vector_steering_plan_2_final.md](../docs/vector_steering_plan_2_final.md) — vector math, templates, evaluation criteria
- [docs/vector_steering_reasoning_plan_2_langvae_implementation.md](../docs/vector_steering_reasoning_plan_2_langvae_implementation.md) — LangVAE decode reference

## Root cause of unrelated output (fixed)

The original LangVAE path produced garbage (`"a is a a kind of a"`) due to:

1. **Random 768→128 projector** — untrained `nn.Linear` mapped steered vectors to arbitrary latents
2. **Embedding space mismatch** — pipeline used `bert-base-uncased` 768d; LangVAE expects 128d μ from `bert-base-cased`
3. **Environment mismatch** — checkpoint targets Python 3.11; LangVAE decode fails on 3.13 / unpinned deps
4. **Step 5 dropped templates** — final string lost reviewer-tone prefix

Steps 1–3E (inputs + formulas) were always correct.

## Phased architecture

| Phase | Decode mode | Status |
|-------|-------------|--------|
| 1 | `--decode template_hybrid` (default) | Reliable related output |
| 0 | `validate_langvae.py` gate | Must pass before trusting LangVAE |
| 2 | `--decode langvae` | 128d native latent steering |

## Constraints

- All code and outputs live in this folder only
- No modifications to `tracks/`, `config.yaml`, or other repo packages
- v1 uses hardcoded resume snippets and scores from the steering doc

## Architecture

Three independent dimensions (Technical, Career, Behavioral):

1. Bucket score → `high` / `mid` / `low`
2. Load precomputed template + anchor **128d LangVAE μ vectors** (`precompute.py`)
3. Encode candidate resume snippet with LangVAE encoder
4. Blend: `v_base = GAMMA * v_tmpl + (1-GAMMA) * v_cand` (GAMMA=0.60)
5. Steer: `v_steer = (1-s)*v_lo + s*v_hi`; `v_final = (1-DELTA)*v_base + DELTA*v_steer` (DELTA=0.25)
6. **Decode (mode-dependent):**
   - `template_hybrid`: deterministic resume clause via `compose.py`
   - `langvae`: `decode_sentences(z_final)` via LangVAE GPT2 decoder
7. **Step 5:** `template[bucket] + " " + clause` per dimension → `results/output.json`

## Folder layout

```
vector_reasoning_test/
├── IMPLEMENTATION_PLAN.md
├── README.md
├── requirements.txt
├── requirements-pinned.txt   # Python 3.11 + pinned deps for LangVAE gate
├── constants.py
├── compose.py                # Phase 1 template_hybrid clauses
├── encode.py                 # langvae_encode() → 128d μ
├── decode.py                 # langvae_decode()
├── validate_langvae.py       # encode→decode gate
├── precompute.py
├── run_experiment.py         # --decode template_hybrid|langvae
├── vectors/                  # 15 × .npy (128d, gitignored)
└── results/                  # output.json (gitignored)
```

## Dependencies

**Default (template_hybrid):**

```bash
pip install -r requirements.txt
```

**LangVAE decode (requires Python 3.11):**

```bash
pip install -r requirements-pinned.txt
python validate_langvae.py   # must exit 0 before --decode langvae
```

## Run commands

```bash
cd vector_reasoning_test
pip install -r requirements.txt
python precompute.py
python run_experiment.py                              # template_hybrid (default)
python run_experiment.py --decode template_hybrid     # explicit
python validate_langvae.py                            # gate for LangVAE
python run_experiment.py --decode langvae             # only after gate passes
```

First LangVAE run downloads checkpoint from HuggingFace (~800MB).

## Inputs (hardcoded)

All in [`constants.py`](constants.py):

| Input | Location |
|-------|----------|
| Resume snippets (tech / career / behav) | `resume_*_text` |
| Dimension scores | `s_tech=0.88`, `s_career=0.81`, `s_behav=0.19` |
| Templates + anchors | `template_*`, `anchor_*` |
| Hyperparams | `GAMMA`, `DELTA`, `MAX_LENGTH`, etc. |

## LangVAE decode details

- Checkpoint: `neuro-symbolic-ai/eb-langvae-bert-base-cased-gpt2-l128`
- Latent dim: **128** (all vector math in LangVAE μ space; no random projector)
- API: `model.decode_sentences(z)` — canonical library path in [`decode.py`](decode.py)
- `TEMPERATURE` / `TOP_P` / `DO_SAMPLE` in constants are unused by LangVAE decode

## Expected trace (hardcoded test case)

| Input | Value | Bucket |
|-------|-------|--------|
| s_tech | 0.88 | high |
| s_career | 0.81 | high |
| s_behav | 0.19 | low |

## Post-run evaluation

1. **Tone** — reviewer voice from template prefix + resume clause
2. **Specificity** — FAISS, Meesho, latency, Senior MLE, 18 months, 47 days
3. **Direction** — tech/career positive; behav concern
4. **Independence** — three distinct aspects

## Non-goals (v1)

- Wiring scores from Stage 5 parquet
- Batch mode over top-100 CSV
- Modifying `tracks/instructor/stage5/reasoning.py`

## Optional future extension (Phase 3)

- Train projector or fine-tune LangVAE on recruiter reasoning pairs
- Wire real scores from `artifacts/runtime/stage5/stage5_scored.parquet`

## Relationship to main pipeline

Stage 5 uses deterministic template concatenation ([tracks/instructor/stage5/reasoning.py](../tracks/instructor/stage5/reasoning.py)). This experiment tests vector-steering + decode in isolation.
