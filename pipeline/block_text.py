"""Stage 4: Token-budget-aware block text construction."""

from __future__ import annotations

from pipeline.config import EMPTY_BLOCK_TEXT, MAX_TOKENS_PER_BLOCK


def build_block_text(
    assigned_sentences: list[tuple[str, float]],
    max_tokens: int = MAX_TOKENS_PER_BLOCK,
) -> str:
    """Sort by similarity, fill token budget greedily from the top."""
    if not assigned_sentences:
        return EMPTY_BLOCK_TEXT

    sorted_sentences = sorted(assigned_sentences, key=lambda x: x[1], reverse=True)

    selected: list[str] = []
    total_tokens = 0.0

    for sentence, _score in sorted_sentences:
        estimated_tokens = len(sentence.split()) * 1.3
        if total_tokens + estimated_tokens > max_tokens:
            break
        selected.append(sentence)
        total_tokens += estimated_tokens

    if not selected:
        return EMPTY_BLOCK_TEXT

    return " ".join(selected)
