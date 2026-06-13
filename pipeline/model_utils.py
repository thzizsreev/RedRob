"""Model dimension helper compatible with sentence-transformers API changes."""

from __future__ import annotations


def embedding_dim(model) -> int:
    if hasattr(model, "get_embedding_dimension"):
        return model.get_embedding_dimension()
    return model.get_sentence_embedding_dimension()
