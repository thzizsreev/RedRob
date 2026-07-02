"""Dense + sparse hybrid semantic similarity for v3 ranking."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Literal

import numpy as np

from ranker import jd_config as jd

Backend = Literal["tfidf", "minilm", "bge", "hybrid"]


def candidate_to_text(candidate: dict) -> str:
    """Build searchable profile text prioritizing career evidence over skill lists."""
    p = candidate.get("profile", {})
    chunks = [
        p.get("headline", ""),
        p.get("current_title", ""),
        p.get("summary", ""),
    ]
    for h in candidate.get("career_history", []):
        chunks.append(
            f"{h.get('title', '')} at {h.get('company', '')} in {h.get('industry', '')}: "
            f"{h.get('description', '')}"
        )
    for s in candidate.get("skills", [])[:25]:
        chunks.append(
            f"Skill {s.get('name', '')} proficiency {s.get('proficiency', '')} "
            f"used {s.get('duration_months', 0)} months"
        )
    for edu in candidate.get("education", [])[:2]:
        chunks.append(
            f"{edu.get('degree', '')} in {edu.get('field_of_study', '')} "
            f"from {edu.get('institution', '')}"
        )
    return " ".join(c for c in chunks if c)[:8000]


def _minmax01(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _ssl_env() -> None:
    import os

    try:
        import certifi

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except ImportError:
        pass


def _get_device(preferred: str = "auto") -> str:
    if preferred == "cpu":
        return "cpu"
    if preferred in ("cuda", "auto"):
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
    return "cpu"


def _device_label(device: str) -> str:
    if device != "cuda":
        return device
    try:
        import torch

        return f"cuda:{torch.cuda.get_device_name(0)}"
    except Exception:
        return "cuda"


def _resolve_local_snapshot(model_name: str) -> str | None:
    """Resolve a fully-local snapshot path from the HF hub cache."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return None

    target = model_name.replace("\\", "/")
    for repo in scan_cache_dir().repos:
        if repo.repo_id != target:
            continue
        snapshots = repo.repo_path / "snapshots"
        if not snapshots.is_dir():
            continue
        for snap in sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if (snap / "modules.json").exists() or (snap / "config.json").exists():
                return str(snap)
    return None


def _load_sentence_model(model_name: str, device: str = "auto"):
    from sentence_transformers import SentenceTransformer

    _ssl_env()
    dev = _get_device(device)
    local_path = _resolve_local_snapshot(model_name)
    if local_path:
        try:
            model = SentenceTransformer(local_path, local_files_only=True, device=dev)
            print(f"[embeddings] loaded {model_name} from cache on {_device_label(dev)}")
            return model
        except Exception as exc:
            print(f"[embeddings] local snapshot load failed for {model_name}: {exc}")
    try:
        model = SentenceTransformer(model_name, local_files_only=True, device=dev)
    except Exception:
        model = SentenceTransformer(model_name, device=dev)
    print(f"[embeddings] loaded {model_name} on {_device_label(dev)}")
    return model


def _encode_dense(model, texts: list[str], batch_size: int = 256) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)


class SemanticIndex:
    """Precomputed TRACER semantic scores: hybrid + multi-query + rerank boost."""

    def __init__(
        self,
        id_to_row: dict[str, int],
        hybrid_scores: np.ndarray,
        backend: str,
        model_name: str,
        mq_scores: np.ndarray | None = None,
        rerank_boost: dict[str, float] | None = None,
    ) -> None:
        self.id_to_row = id_to_row
        self.hybrid_scores = hybrid_scores
        self.semantic_scores = hybrid_scores  # backwards compat
        self.mq_scores = mq_scores
        self.rerank_boost = rerank_boost or {}
        self.backend = backend
        self.model_name = model_name

    @classmethod
    def load(cls, artifacts_dir: Path) -> SemanticIndex:
        artifacts_dir = Path(artifacts_dir)
        meta = json.loads((artifacts_dir / "meta.json").read_text(encoding="utf-8"))
        hybrid_path = artifacts_dir / "hybrid_scores.npy"
        if not hybrid_path.exists():
            hybrid_path = artifacts_dir / "semantic_scores.npy"
        hybrid = np.load(hybrid_path).astype(np.float32)
        ids = json.loads((artifacts_dir / "candidate_ids.json").read_text(encoding="utf-8"))
        id_to_row = {cid: i for i, cid in enumerate(ids)}

        mq_scores = None
        mq_path = artifacts_dir / "mq_scores.npy"
        if mq_path.exists():
            mq_scores = np.load(mq_path).astype(np.float32)

        rerank_boost: dict[str, float] = {}
        rerank_path = artifacts_dir / "rerank_boost.json"
        if rerank_path.exists():
            rerank_boost = json.loads(rerank_path.read_text(encoding="utf-8"))

        return cls(
            id_to_row,
            hybrid,
            meta.get("backend", "tracer"),
            meta.get("model", "tracer"),
            mq_scores=mq_scores,
            rerank_boost=rerank_boost,
        )

    def _fused_semantic(self, row: int, candidate_id: str) -> float:
        hybrid = float(self.hybrid_scores[row])
        mq = float(self.mq_scores[row]) if self.mq_scores is not None else hybrid
        rerank = float(self.rerank_boost.get(candidate_id, 0.0))
        fused = (
            jd.SEMANTIC_HYBRID_W * hybrid
            + jd.SEMANTIC_MQ_W * mq
            + jd.SEMANTIC_RERANK_W * rerank
        )
        return max(0.0, min(1.0, fused))

    def semantic_score(self, candidate_id: str) -> float | None:
        row = self.id_to_row.get(candidate_id)
        if row is None:
            return None
        return self._fused_semantic(row, candidate_id)

    @classmethod
    def build_sparse(cls, texts: list[str], out_dir: Path) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        corpus = [jd.JD_TEXT] + texts
        vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
        )
        matrix = vectorizer.fit_transform(corpus)
        sims = cosine_similarity(matrix[1:], matrix[0]).ravel().astype(np.float32)
        with (out_dir / "tfidf_vectorizer.pkl").open("wb") as f:
            pickle.dump(vectorizer, f)
        return np.clip(sims, 0.0, 1.0)

    @classmethod
    def build_tfidf(cls, candidates_path: Path, out_dir: Path) -> SemanticIndex:
        from ranker.io import iter_candidates

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ids: list[str] = []
        texts: list[str] = []
        for candidate in iter_candidates(candidates_path):
            ids.append(candidate["candidate_id"])
            texts.append(candidate_to_text(candidate))

        sims = cls.build_sparse(texts, out_dir)
        np.save(out_dir / "sparse_scores.npy", sims)
        np.save(out_dir / "semantic_scores.npy", sims)
        (out_dir / "candidate_ids.json").write_text(json.dumps(ids), encoding="utf-8")
        (out_dir / "meta.json").write_text(
            json.dumps({"backend": "tfidf", "model": "sklearn-tfidf-8000", "count": len(ids)}),
            encoding="utf-8",
        )
        return cls({cid: i for i, cid in enumerate(ids)}, sims, "tfidf", "sklearn-tfidf-8000")

    @classmethod
    def build_dense_backend(
        cls,
        candidates_path: Path,
        out_dir: Path,
        model_name: str,
        backend_label: str,
        batch_size: int = 256,
        query_prefix: str = "",
        device: str = "auto",
    ) -> tuple[np.ndarray, list[str]]:
        from ranker.io import iter_candidates

        model = _load_sentence_model(model_name, device=device)
        ids: list[str] = []
        texts: list[str] = []
        for candidate in iter_candidates(candidates_path):
            ids.append(candidate["candidate_id"])
            texts.append(candidate_to_text(candidate))

        job_vec = model.encode([query_prefix + jd.JD_TEXT], normalize_embeddings=True)[0]
        embeddings = _encode_dense(model, texts, batch_size)
        sims = np.clip((embeddings @ job_vec + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)

        np.save(out_dir / "candidate_embeddings.npy", embeddings)
        np.save(out_dir / "job_embedding.npy", job_vec.astype(np.float32))
        np.save(out_dir / "dense_scores.npy", sims)
        return sims, ids

    @classmethod
    def build_minilm(
        cls,
        candidates_path: Path,
        out_dir: Path,
        batch_size: int = 256,
        device: str = "auto",
    ) -> SemanticIndex:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sims, ids = cls.build_dense_backend(
            candidates_path, out_dir, jd.EMBED_MODEL, "minilm", batch_size, device=device
        )
        np.save(out_dir / "semantic_scores.npy", sims)
        (out_dir / "candidate_ids.json").write_text(json.dumps(ids), encoding="utf-8")
        (out_dir / "meta.json").write_text(
            json.dumps(
                {
                    "backend": "minilm",
                    "model": jd.EMBED_MODEL,
                    "count": len(ids),
                    "device": _device_label(_get_device(device)),
                }
            ),
            encoding="utf-8",
        )
        return cls({cid: i for i, cid in enumerate(ids)}, sims, "minilm", jd.EMBED_MODEL)

    @classmethod
    def build_hybrid(
        cls,
        candidates_path: Path,
        out_dir: Path,
        batch_size: int = 256,
        dense_model: str | None = None,
        device: str = "auto",
    ) -> SemanticIndex:
        """Fuse dense (BGE/MiniLM) + sparse TF-IDF with fallback chain."""
        from ranker.io import iter_candidates

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ids: list[str] = []
        texts: list[str] = []
        for candidate in iter_candidates(candidates_path):
            ids.append(candidate["candidate_id"])
            texts.append(candidate_to_text(candidate))

        sparse = cls.build_sparse(texts, out_dir)
        np.save(out_dir / "sparse_scores.npy", sparse)

        dense_models = []
        if dense_model:
            dense_models.append(dense_model)
        dense_models.extend([jd.BGE_MODEL, jd.EMBED_MODEL])

        dense: np.ndarray | None = None
        used_model = "sparse-only"
        used_backend = "hybrid"
        query_prefix = ""

        for model_name in dense_models:
            try:
                prefix = jd.BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""
                dense, _ = cls.build_dense_backend(
                    candidates_path,
                    out_dir,
                    model_name,
                    "hybrid",
                    batch_size,
                    query_prefix=prefix,
                    device=device,
                )
                used_model = model_name
                query_prefix = prefix
                break
            except Exception as exc:
                print(f"[hybrid] dense model {model_name} failed: {exc}")

        if dense is not None:
            hybrid = (
                jd.HYBRID_DENSE_WEIGHT * _minmax01(dense)
                + jd.HYBRID_SPARSE_WEIGHT * _minmax01(sparse)
            )
        else:
            hybrid = _minmax01(sparse)
            used_backend = "hybrid-sparse-fallback"

        hybrid = np.clip(hybrid, 0.0, 1.0).astype(np.float32)
        np.save(out_dir / "semantic_scores.npy", hybrid)
        (out_dir / "candidate_ids.json").write_text(json.dumps(ids), encoding="utf-8")
        (out_dir / "meta.json").write_text(
            json.dumps(
                {
                    "backend": used_backend,
                    "model": used_model,
                    "dense_model": used_model,
                    "sparse_model": "sklearn-tfidf-8000",
                    "fusion": f"{jd.HYBRID_DENSE_WEIGHT}*dense+{jd.HYBRID_SPARSE_WEIGHT}*sparse",
                    "count": len(ids),
                    "query_prefix": query_prefix,
                    "device": _device_label(_get_device(device)),
                }
            ),
            encoding="utf-8",
        )
        return cls({cid: i for i, cid in enumerate(ids)}, hybrid, "hybrid", used_model)

    @classmethod
    def build_from_candidates(
        cls,
        candidates_path: Path,
        out_dir: Path,
        backend: Backend = "hybrid",
        batch_size: int = 256,
        device: str = "auto",
    ) -> SemanticIndex:
        if backend == "hybrid":
            return cls.build_hybrid(candidates_path, out_dir, batch_size, device=device)
        if backend == "minilm":
            return cls.build_minilm(candidates_path, out_dir, batch_size, device=device)
        return cls.build_tfidf(candidates_path, out_dir)


EmbeddingIndex = SemanticIndex
