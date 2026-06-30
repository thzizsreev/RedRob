# Vector Steering Feasibility Test

Isolated encode-steer-decode test for GTR-base + vec2text. Validates whether numeric scores can be converted to embedding-space displacements and decoded back into coherent text **before** any resume-reasoning pipeline is built.

Full specification: [../vector_steering_test_plan.md](../vector_steering_test_plan.md)

## Environment (Windows 64-bit)

Use Python **3.10 or 3.11** in a **separate venv** (do not reuse the main pipeline `env/` — different model stack).

```powershell
cd experiments\vectorSteering\testing
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip

# CPU-only PyTorch (required on machines without CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

First run downloads model weights (~1 GB) to `%USERPROFILE%\.cache\huggingface`.

## Verify installation

```powershell
python verify_install.py
```

Expected: encoder shape `(1, 768)` and corrector loads without error.

## Run tests

From `experiments/vectorSteering/testing/`, edit paths and settings at the top of `run_test.py`:

```python
INPUT_CONFIG = TEST_DIR / "input" / "config.yaml"
OUTPUT_DIR = TEST_DIR / "output"
RUN_PHASE = "all"       # "a", "b", or "all"
FORCE_PHASE_B = False
```

Then run:

```powershell
python run_test.py
python run_phase_a.py   # Phase A only (uses same input/output paths)
python run_phase_b.py   # Phase B only (respects Phase A gate)
```

Outputs land in `output/`:

- `phase_a_results.csv` — 25 rows
- `phase_b_results.csv` — 21 rows (7 S values x 3 sweeps)
- `summary.json` — automated determinism checks

## Human review workflow

Automated checks cover **determinism only**. Fidelity, monotonicity, coherence, and overshoot require human judgment per the plan (Section 5).

1. Open `output/phase_a_results.csv` and fill `human_fidelity` for each sentence group.
2. Open `output/phase_b_results.csv` and fill `human_coherent` and `human_sentiment`.
3. Compare decoded text at `S=-0.25` and `S=1.25` for overshoot behavior.

Pass thresholds (from plan):

| Phase | Criterion | Pass |
|-------|-----------|------|
| A | Fidelity | 4/5 sentences preserve core meaning |
| A | Determinism | 4/5 sentences stable across 5 decodes (automated) |
| B | Monotonicity | Consistent negative-to-positive trend across S |
| B | Coherence | 6/7 S values produce valid sentences |
| B | Determinism | 5/7 S values stable across 3 sweeps (automated) |

## Decision tree (Section 6)

- **Phase A fails** → try a different decoder/checkpoint or shorter sentences; do not tune steering math.
- **Phase A passes, B monotonicity fails** → try more extreme anchors or a single-word axis before abandoning steering.
- **Monotonicity + coherence pass, determinism fails** → tune `decoder.num_steps` / `sequence_beam_width` in `input/config.yaml`.
- **All pass** → proceed to JD-relevant axes and production pipeline design.

## Embedding note

This harness uses `SentenceTransformer("sentence-transformers/gtr-t5-base")` as specified in the plan. Vec2text's published GTR examples use mean-pooled encoder outputs; if Phase A fidelity is poor, compare against the transformers mean-pool path before concluding the decoder is unsuitable.
