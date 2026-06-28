#!/usr/bin/env python3
"""Environment gate: verify LangVAE native encode->decode produces fluent English."""

from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from encode import get_langvae_model, langvae_encode

GARBAGE_PATTERNS = (
    re.compile(r"a is a a kind of a", re.IGNORECASE),
    re.compile(r"\bis is is\b"),
    re.compile(r"0{3,}"),
    re.compile(r"[a-z]0{2,}", re.IGNORECASE),
)

TEST_SENTENCES = [
    "The hypothesis is entailed because mammals require oxygen to survive.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "Strong production-grade technical alignment with retrieval systems at scale.",
]


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]{3,}", text))


def is_fluent_decode(text: str) -> bool:
    if not text or not text.strip():
        return False
    for pattern in GARBAGE_PATTERNS:
        if pattern.search(text):
            return False
    if _word_count(text) < 5:
        return False
    alpha_chars = sum(c.isalpha() for c in text)
    if alpha_chars / max(len(text), 1) < 0.5:
        return False
    return True


def validate_roundtrip() -> tuple[int, int]:
    model = get_langvae_model()
    passed = 0
    for sentence in TEST_SENTENCES:
        import torch

        z = langvae_encode(sentence)
        z_t = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            decoded = model.decode_sentences(z_t)[0].strip()
        if is_fluent_decode(decoded):
            passed += 1
    return passed, len(TEST_SENTENCES)


def main() -> int:
    print("LangVAE environment validation")
    print(f"Python {sys.version.split()[0]}")
    print()

    model = get_langvae_model()
    passed = 0

    for sentence in TEST_SENTENCES:
        import torch

        z = langvae_encode(sentence)
        z_t = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            decoded = model.decode_sentences(z_t)[0].strip()

        ok = is_fluent_decode(decoded)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] IN:  {sentence}")
        print(f"       OUT: {decoded}")
        print()
        if ok:
            passed += 1

    total = len(TEST_SENTENCES)
    if passed == total:
        print("Gate PASSED - LangVAE encode->decode is usable for --decode langvae")
        return 0

    print(
        f"Gate FAILED ({passed}/{total} passed). "
        "Use Python 3.11 and requirements-pinned.txt, then re-run. "
        "Default to --decode template_hybrid until gate passes."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
