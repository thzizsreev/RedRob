import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_NAME = "humarin/chatgpt_paraphraser_on_T5_base"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_BF16 = DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()
DTYPE = torch.bfloat16 if USE_BF16 else torch.float32

if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

print(f"Device: {DEVICE}", flush=True)
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}", flush=True)
    print(f"Dtype: {'bfloat16' if USE_BF16 else 'float32'}", flush=True)

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, dtype=DTYPE)
model.to(DEVICE)
model.eval()


def _to_device(batch):
    return {key: value.to(DEVICE) for key, value in batch.items()}


def encode(text):
    inputs = _to_device(
        tokenizer(
            f"paraphrase: {text}",
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
    )
    with torch.inference_mode():
        encoder_outputs = model.encoder(**inputs)
    return encoder_outputs


def decode(encoder_outputs, temperature):
    with torch.inference_mode():
        output_ids = model.generate(
            encoder_outputs=encoder_outputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=temperature,
            top_p=0.92,
            repetition_penalty=1.3,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


phrase_tech = """Mira Verma, working as Recommendation Systems Engineer at Uber, "
    "owned the end-to-end ranking pipeline that improved revenue-per-search "
    "by 12%, with verified Elasticsearch depth directly matching the role's "
    "retrieval and learning-to-rank requirements"""
phrase_career = """across 6.8 years entirely at product companies including Uber and 
Flipkart, maintained stable tenure with senior ML ownership predating 
the LLM era and no consulting history — clearing all explicit JD 
disqualifiers"""
phrase_behav = """impact claims are well-evidenced internally but external validation 
score is low with no public papers or open-source work to independently 
verify system depth; sixty-day notice adds timeline friction"""

temperatures = [0.01, 0.3, 0.7, 1.0, 1.3]
phrase_v2 = (
    "Mira Verma, working as Recommendation Systems Engineer at Uber, "
    "owned the end-to-end ranking pipeline that improved revenue-per-search "
    "by 12%, with verified Elasticsearch depth directly matching the role's "
    "retrieval and learning-to-rank requirements"
)
phrase_v3 = (
    "Mira Verma built and owned the end-to-end ranking pipeline at Uber "
    "as Recommendation Systems Engineer, achieving a 12% improvement in "
    "revenue-per-search with verified Elasticsearch expertise directly "
    "matching the role's retrieval and learning-to-rank requirements"
)

phrase_v4 = (
    "Mira Verma's internal impact evidence is strong with a documented "
    "12% revenue improvement at Uber and a 6% retention lift at Flipkart, "
    "yet no public papers, open-source projects, or conference talks exist "
    "to allow independent verification of this depth, and her sixty-day "
    "notice period pushes the earliest start two months beyond the role's "
    "sub-thirty-day preference"
)

phrase_concern_1 = (
    "Mira Verma's production impact is well-documented through measurable "
    "outcomes including a 12% revenue-per-search improvement at Uber and "
    "a 6% retention lift at Flipkart, however no public papers, open-source "
    "contributions, or external conference talks exist to independently "
    "validate the technical depth of the systems she describes"
)

phrase_concern_2 = (
    "Mira Verma carries a sixty-day notice period that pushes the earliest "
    "possible start date two full months beyond the role's stated preference "
    "for a sub-thirty-day onboarding timeline, which introduces concrete "
    "scheduling friction for an immediate hiring need"
)

phrase_A = (
    "Mira Verma carries a sixty-day notice period that delays the start "
    "date two months, while the role requires sub-thirty-day availability "
    "for immediate onboarding"
)

# Test B — grammatical subject is the candidate, heavy clause second
phrase_B = (
    "Mira Verma, despite the role requiring sub-thirty-day availability, "
    "carries a sixty-day notice period that delays the earliest start "
    "date by two full months"
)

# Test C — grammatical subject is the constraint, not the candidate
phrase_C = (
    "The role's sub-thirty-day onboarding preference creates friction "
    "with Mira Verma's sixty-day notice period, pushing the earliest "
    "start date two months beyond the hiring timeline"
)

phrases = {
    "phrase_A": phrase_A,
    "phrase_B": phrase_B,
    "phrase_C": phrase_C,
    
    # "phrase_career": phrase_career,
    # "phrase_behav": phrase_behav,
}

results = {}

# Warm up GPU kernels before timed work.
if DEVICE.type == "cuda":
    _warm = decode(encode("warmup"), 0.7)
    del _warm
    torch.cuda.synchronize()

for key, text in phrases.items():
    print(f"\n=== PHRASE: {key} ===")
    print(f'INPUT: "{text}"')
    print()

    encoder_outputs = encode(text)
    phrase_results = {"input": text}

    for temp in temperatures:
        output = decode(encoder_outputs, temp)
        temp_key = str(temp)
        phrase_results[temp_key] = output
        print(f"temp={temp} → {output}", flush=True)

    print()
    results[key] = phrase_results

results_path = Path(__file__).resolve().parent / "results.json"
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
    f.write("\n")

print(f"Saved results -> {results_path}")
