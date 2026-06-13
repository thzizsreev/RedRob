"""Stage 2: Sentence splitting from role-contextualized segments."""

from __future__ import annotations

import re


def split_into_sentences(segments: list[str], *, min_length: int = 20) -> list[str]:
    """Split segments into individual sentences, dropping short fragments."""
    all_sentences: list[str] = []

    for segment in segments:
        sentences = re.split(
            r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+",
            segment,
        )

        for sentence in sentences:
            sub_sentences = re.split(r"\n+|\s*;\s*(?=[A-Z0-9])", sentence)
            for s in sub_sentences:
                s = s.strip()
                if len(s) >= min_length:
                    all_sentences.append(s)

    return all_sentences
