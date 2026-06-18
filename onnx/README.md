# INSTRUCTOR-large ONNX sprint

Minimal test: export the T5 encoder, run ONNX Runtime inference on a few sample texts.

See [implementation.md](implementation.md) for the full process, failure modes, precautions, and main-project integration checklist.

See [../documents/instructor-large-onnx-setup.md](../documents/instructor-large-onnx-setup.md) for why optimum/sentence-transformers export does not work for INSTRUCTOR.

## Setup

```powershell
cd onnx
pip install -r requirements.txt
```

## Run

```powershell
python export_to_onnx.py    # one-time (~5–10 min), writes models/
python run_encode.py        # ONNX vectors on 3 sample pairs
```

Done when `run_encode.py` prints `shape: (3, 768)` with no errors.

## Artifacts

```
models/
  instructor-large-encoder.onnx
  dense_weight.npy
  tokenizer/
  config.txt
```
