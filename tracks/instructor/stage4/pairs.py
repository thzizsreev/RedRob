"""Build and tokenize (JD, candidate) pairs for cross-encoder scoring."""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase

from tracks.instructor.stage4.config import Stage4Config


def truncate_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
) -> str:
    if not text.strip():
        return ""
    tokens = tokenizer.encode(
        text,
        add_special_tokens=False,
        max_length=max_tokens,
        truncation=True,
    )
    return tokenizer.decode(tokens, skip_special_tokens=True)


def prepare_jd_text(config: Stage4Config, tokenizer: PreTrainedTokenizerBase) -> str:
    return truncate_text(config.jd_text, tokenizer, config.max_jd_tokens)


def prepare_candidate_texts(
    candidate_ids: list[str],
    text_by_id: dict[str, str],
    config: Stage4Config,
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, str]:
    prepared: dict[str, str] = {}
    for cid in candidate_ids:
        raw = text_by_id.get(cid, "").strip()
        prepared[cid] = truncate_text(raw, tokenizer, config.max_candidate_tokens)
    return prepared


def build_query_passage_pairs(
    candidate_ids: list[str],
    jd_text: str,
    candidate_texts: dict[str, str],
) -> list[tuple[str, str, str]]:
    """Return (candidate_id, query, passage) tuples preserving input order."""
    pairs: list[tuple[str, str, str]] = []
    for cid in candidate_ids:
        pairs.append((cid, jd_text, candidate_texts.get(cid, "")))
    return pairs
