import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_NAME = "humarin/chatgpt_paraphraser_on_T5_base"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
model.eval()


def encode(text):
    inputs = tokenizer(
        f"paraphrase: {text}",
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    with torch.no_grad():
        encoder_outputs = model.encoder(**inputs)
    return encoder_outputs


def decode(encoder_outputs, temperature):
    with torch.no_grad():
        output_ids = model.generate(
            encoder_outputs=encoder_outputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=temperature,
            top_p=0.92,
            repetition_penalty=1.3,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


phrase_tech = "demonstrated strong production expertise in retrieval systems with measurable latency impact"
phrase_career = "senior engineer with clear ownership trajectory and end-to-end system responsibility"
phrase_behav = "actively engaged on platform with strong availability and responsiveness signals"

temperatures = [0.3, 0.7, 1.0, 1.3]

phrases = {
    "phrase_tech": phrase_tech,
    "phrase_career": phrase_career,
    "phrase_behav": phrase_behav,
}

results = {}

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
