# Vector Encoding Pipeline — AI Agent Implementation Plan
## Redrob Hackathon: Senior AI Engineer Candidate Ranking

---

## WHAT YOU ARE BUILDING

You are building the **offline precomputation pipeline** that transforms 100,000 raw candidate profiles into a 768-dimensional block-weighted vector index. This index is the foundation of Stage 1 retrieval — the mechanism that narrows 100,000 candidates down to a shortlist for downstream ranking.

This document covers everything up to and including the FAISS index construction. Downstream ranking (LightGBM, behavioral signals, CSV output) is out of scope here.

---

## CONTEXT: WHY THIS ARCHITECTURE EXISTS

### The problem with standard embedding

A standard encoder (MiniLM, BERT, etc.) compresses an entire candidate profile into one opaque vector. The geometry of that vector was decided by the model's training corpus — Wikipedia, GitHub, Stack Overflow. It has no concept of which skills matter more for this specific job.

When you query with "senior AI engineer production retrieval experience", the encoder produces a vector that has high activations spread across retrieval, infra, eval, and generic ML concepts simultaneously. You cannot tell it "production deployment matters 2x more than evaluation experience." The importance of each dimension was baked in during training, not by you.

### The solution: block-diagonal priority weighting

Instead of one opaque 768-d vector, you build a **concatenation of three independent 256-d sub-vectors**, each encoding a different skill domain:

```
candidate_vector = [v_retrieval (0-255) | v_infra (256-511) | v_eval (512-767)]
```

At query time, you apply explicit priority weights to the query sub-vectors before concatenation:

```
query_vector = [0.3 * q_retrieval | 0.6 * q_infra | 0.1 * q_eval]
```

The dot product then decomposes mathematically into three weighted sub-scores:

```
score = q · c
      = 0.3(q_r · c_r) + 0.6(q_i · c_i) + 0.1(q_e · c_e)
```

This means infra skill contributes 2x more to the final score than retrieval skill, and 6x more than eval skill — exactly as the JD intends. You have imposed your own importance structure on top of the encoder's geometry.

### Why inner product, not cosine similarity

FAISS must be configured to use **inner product (dot product)**, NOT cosine similarity or L2 distance.

Cosine similarity normalizes the full 768-d vector globally before computing similarity. Global normalization collapses the block structure — a very strong infra sub-vector influences the normalization of the retrieval and eval sub-vectors, blurring the block boundaries you carefully constructed.

Inner product preserves each block's contribution proportional to its magnitude. The per-block L2 normalization (applied separately to each 256-d sub-vector during encoding) combined with the query-side weight scalars gives you full control over the contribution of each block.

### Why the block structure does NOT require orthogonality

Orthogonality is a property *within* a single vector — it would mean `dot(v_retrieval, v_infra) = 0`. You do not need this.

The block structure works because of **positional isolation** — the index ranges do not overlap. When computing `q · c`, a query's retrieval dimensions (0-255) only multiply against a candidate's retrieval dimensions (0-255). Cross-block terms are zero by construction because the index ranges are disjoint. This is positional isolation, not geometric orthogonality.

The actual failure mode to avoid is not cross-block geometric drift — it is **cross-block text leakage** during sentence extraction. If your infra text extraction accidentally includes retrieval-heavy sentences, the infra sub-vector will encode retrieval signal. Fix this at the text extraction step, not with Gram-Schmidt.

---

## JOB DESCRIPTION ANALYSIS

The JD explicitly defines three tiers of requirement. These directly map to the three block weights.

### Non-negotiable (maps to infra block, weight 0.6)

> "Production experience with vector databases or hybrid search infrastructure — Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS. The operational experience does."

> "Has shipped at least one end-to-end ranking, search, or recommendation system to real users at meaningful scale."

> "If you've spent your career in pure research environments without any production deployment — we will not move forward."

Production deployment is the hardest filter. A candidate who understands retrieval theoretically but has never shipped to real users is explicitly excluded. Infra gets the highest weight (0.6).

### Required (maps to retrieval block, weight 0.3)

> "Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5) deployed to real users. Embedding drift, index refresh, retrieval-quality regression in production."

Strong retrieval understanding is required, but it is subordinate to the deployment requirement. A strong infra candidate who knows retrieval well outranks a retrieval theorist without production experience. Retrieval gets weight 0.3.

### Important but not disqualifying (maps to eval block, weight 0.1)

> "Hands-on experience designing evaluation frameworks for ranking systems — NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation."

> "Weeks 9-12: Set up the evaluation infrastructure."

Eval is real but tertiary — the JD schedules it for weeks 9-12, after shipping (weeks 4-8). Eval gets weight 0.1.

### Data traps the pipeline must handle

**Keyword stuffers:** Candidates with every AI buzzword but titles like "Marketing Manager" or no production context. Their block texts will be populated with skill-list sentences that have no surrounding evidence context. These encode to low-similarity vectors against anchors built from production-evidence sentences.

**Plain-language experts:** Candidates who "built recommendation systems at a product company" without using words like "RAG" or "Pinecone." BGE's semantic space handles these — "recommendation system at scale" is geometrically close to "retrieval pipeline production" in a well-trained encoder. Threshold-based assignment catches them where keyword matching would miss them.

**Honeypots:** ~80 profiles with impossible timelines (8 years experience at a 3-year-old company) or implausible skill combinations. These are handled downstream by the tabular mask, not by the vector pipeline. Your vector pipeline should not special-case them.

---

## DEPENDENCIES

```text
sentence-transformers>=2.6.0    # BGE model loading and encoding
faiss-cpu>=1.7.4                # HNSW index construction and search
numpy>=1.24.0                   # vector math
```

Do NOT use:
- `torch` at query/rank time (RAM constraint — use onnxruntime instead for online inference)
- `pandas` (memory bloat — use polars for tabular data)
- Any external API calls at rank time (air-gapped environment)

The offline precompute pipeline (this document) has no time constraint and may use sentence-transformers directly. The online rank.py script must use onnxruntime to load the BGE model.

---

## THE ENCODER: BGE-SMALL-EN-V1.5

### Why BGE over MiniLM

MiniLM is a symmetric encoder — it assumes query and document are the same kind of text object and compresses both into the same geometric space. This is wrong for your use case.

BGE is an asymmetric bi-encoder trained specifically on retrieval pairs: (short intent-expressing query, long evidence-rich document). It was trained with hard negatives — pairs where the negative document looks similar on the surface but doesn't actually answer the query. This forces the model to learn fine-grained distinctions.

The `query:` and `passage:` prefixes are instruction tokens that shift the internal representations into different subspaces:

- `query:` → vector points toward "what answers this?"
- `passage:` → vector points toward "what does this describe?"

The dot product between a `query:` vector and a `passage:` vector measures **relevance**, not similarity. This is exactly the right operation for matching a job requirement against a candidate's experience.

### Model size and performance

`BAAI/bge-small-en-v1.5` is 33M parameters, ~130MB on disk. It produces 384-d vectors natively. You will use only the first 256 dimensions per block — see the truncation note in Stage 5.

Alternatively, use a model that natively produces 256-d vectors. If using the full 384-d output, you must truncate or project to 256-d consistently for all candidates and the query.

**Simplest approach:** use `bge-small-en-v1.5` and take the full 384-d vector per block, making the final concatenated vector 1152-d instead of 768-d. Adjust FAISS index dimensionality accordingly. The architecture is identical — only the dimensionality changes.

### Loading the model

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
# model output dimension: 384
# context window: 512 tokens
```

---

## STAGE 0: ANCHOR VECTOR CONSTRUCTION

**Runs once. Output is three anchor vectors saved to disk.**

The anchors define the geometric targets for each block. Every candidate sentence is scored by its cosine similarity to these anchors to determine which block it belongs to.

### What an anchor is

An anchor is the mean of multiple BGE-encoded sentences that represent what a strong candidate's evidence should look like for that block. It is built from JD sentences — not synthetic sentences, not generic retrieval/infra/eval text. Using JD sentences as anchors means the anchor literally represents "what this specific job needs," not "what retrieval experience generically looks like."

### Step 0.1 — Define anchor seed sentences

These are extracted directly from the JD. They express requirements, not descriptions.

```python
JD_RETRIEVAL_SENTENCES = [
    "production experience with embeddings-based retrieval systems deployed to real users",
    "handling embedding drift index refresh retrieval quality regression in production",
    "sentence-transformers OpenAI embeddings BGE E5 or similar deployed to real users",
    "hybrid retrieval dense sparse production search architecture at scale",
    "vector databases Pinecone Weaviate Qdrant Milvus FAISS operational production experience"
]

JD_INFRA_SENTENCES = [
    "production experience with vector databases or hybrid search infrastructure operational",
    "deploying machine learning systems to real users at meaningful scale",
    "large-scale inference optimization distributed systems production deployment",
    "shipped end-to-end ranking search recommendation system to real users at scale",
    "production deployment latency quality tradeoffs system reliability"
]

JD_EVAL_SENTENCES = [
    "evaluation frameworks for ranking systems NDCG MRR MAP offline to online correlation",
    "A/B test interpretation recruiter feedback loops rigorous ranking evaluation",
    "offline benchmarks online A/B testing feedback loops iterative improvement",
    "how to evaluate a ranking system rigorously statistical validation",
    "learning to rank models evaluation measurement methodology"
]
```

### Step 0.2 — Encode and compute mean anchor

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def build_anchor(sentences: list[str], model) -> np.ndarray:
    """
    Encode sentences with query: prefix (these express requirements).
    Take the mean. L2 normalize.
    Returns: shape (384,) normalized anchor vector.
    """
    prefixed = [f"query: {s}" for s in sentences]
    vecs = model.encode(prefixed, normalize_embeddings=False)
    anchor = np.mean(vecs, axis=0)
    anchor = anchor / np.linalg.norm(anchor)
    return anchor.astype(np.float32)

anchor_r = build_anchor(JD_RETRIEVAL_SENTENCES, model)
anchor_i = build_anchor(JD_INFRA_SENTENCES, model)
anchor_e = build_anchor(JD_EVAL_SENTENCES, model)

# persist to disk — reused for all 100K candidates
import os
os.makedirs("artifacts/anchors", exist_ok=True)
np.save("artifacts/anchors/anchor_retrieval.npy", anchor_r)
np.save("artifacts/anchors/anchor_infra.npy", anchor_i)
np.save("artifacts/anchors/anchor_eval.npy", anchor_e)

print(f"Anchor shapes: {anchor_r.shape}, {anchor_i.shape}, {anchor_e.shape}")
# expected: (384,) (384,) (384,)
```

### Why mean over single sentence

A single anchor sentence is brittle — it captures one phrasing of the concept. "Deployed FAISS" and "production vector search" and "shipped search infrastructure" all describe the same infra signal but encode to slightly different vectors. The mean of five diverse phrasings builds a centroid equidistant from all valid expressions of that concept. Both phrasings score high against the mean anchor where a single-sentence anchor might geometrically favor one phrasing over another.

---

## STAGE 1: CANDIDATE TEXT EXTRACTION

**Input:** one raw candidate record (dict) from `candidates.jsonl.gz`

**Output:** list of role-contextualized text segments (not yet sentences)

### What the candidate record looks like

```json
{
  "candidate_id": "CAND_0042871",
  "current_title": "Senior ML Engineer",
  "experience": [
    {
      "title": "ML Engineer",
      "company": "Flipkart",
      "duration_months": 36,
      "description": "Built FAISS-based ANN index over 50M product embeddings. Reduced p99 latency from 340ms to 22ms. Deployed Qdrant for semantic product search."
    },
    {
      "title": "Data Scientist",
      "company": "Myntra",
      "duration_months": 18,
      "description": "Built recommendation models. Used collaborative filtering and content-based approaches."
    }
  ],
  "skills": ["Python", "FAISS", "Qdrant", "PyTorch", "NDCG", "Docker", "Kubernetes"],
  "education": [{"degree": "B.Tech", "field": "Computer Science", "institution": "IIT Delhi"}],
  "redrob_signals": { ...23 behavioral signals... }
}
```

### Step 1.1 — Build role-contextualized segments

```python
def build_candidate_segments(record: dict) -> list[str]:
    """
    Extract text segments from candidate record.
    Each segment is prefixed with role context.

    CRITICAL: Do NOT concatenate everything into one blob.
    "Built FAISS index" means different things from a 6-year
    senior role vs a 6-month internship. Role context preserves
    this distinction in the text that gets encoded.
    """
    segments = []

    # experience descriptions — most signal lives here
    for exp in record.get("experience", []):
        title = exp.get("title", "").strip()
        company = exp.get("company", "").strip()
        description = exp.get("description", "").strip()
        duration = exp.get("duration_months", 0)

        if not description:
            continue

        # prepend role context to the description
        # this ensures "built FAISS index" carries seniority signal
        context_prefix = f"{title} at {company}"
        segments.append(f"{context_prefix}: {description}")

    # skills list — encode as a declarative sentence, not a raw list
    # "Python FAISS Qdrant" encodes poorly — no syntactic context
    # "technical skills include Python FAISS Qdrant Docker" encodes better
    skills = record.get("skills", [])
    if skills:
        segments.append(f"technical skills include: {', '.join(skills)}")

    # current title — short but useful signal
    current_title = record.get("current_title", "").strip()
    if current_title:
        segments.append(f"current role: {current_title}")

    return segments
```

---

## STAGE 2: SENTENCE SPLITTING

**Input:** list of role-contextualized segments from Stage 1

**Output:** list of individual sentences (strings, minimum 20 characters)

### Step 2.1 — Split segments into sentences

```python
import re

def split_into_sentences(segments: list[str]) -> list[str]:
    """
    Split each segment into individual sentences.
    Handles:
    - Standard sentence endings (. ! ?)
    - Resume-style bullet points (newlines, semicolons)
    - Drops fragments shorter than 20 characters
    """
    all_sentences = []

    for segment in segments:
        # split on sentence-ending punctuation followed by space
        # negative lookbehind avoids splitting on abbreviations
        # e.g. "p99." or "e.g." or "v1.5"
        sentences = re.split(
            r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+',
            segment
        )

        for sentence in sentences:
            # further split on newlines and semicolons
            # common in resume bullet-point style descriptions
            sub_sentences = re.split(r'\n+|\s*;\s*(?=[A-Z0-9])', sentence)

            for s in sub_sentences:
                s = s.strip()
                # drop fragments — too short to carry semantic signal
                if len(s) >= 20:
                    all_sentences.append(s)

    return all_sentences
```

---

## STAGE 3: SENTENCE-LEVEL BLOCK ASSIGNMENT

This is the core of the pipeline. Every sentence is independently scored against all three anchors. Sentences above the per-block threshold are assigned to that block. A sentence can belong to multiple blocks simultaneously.

### The asymmetric encoding principle

The anchors were built with `query:` prefix — they express requirements.
Each candidate sentence must be encoded with `passage:` prefix — it describes evidence.

The BGE model was trained on (query, passage) pairs. The dot product between a `query:` vector and a `passage:` vector measures **relevance** — does this passage answer this query? This is the correct operation. Never encode a candidate sentence with `query:` prefix.

### Step 3.1 — Batch encode all sentences

```python
def encode_sentences(sentences: list[str], model) -> np.ndarray:
    """
    Encode all sentences for one candidate in a single batch call.
    passage: prefix — these are evidence, not queries.
    Returns: shape (N, 384) where N = number of sentences.

    IMPORTANT: batch encoding is ~10x faster than encoding one at a time.
    For 100K candidates with ~20 sentences each, this difference is
    the boundary between 30 minutes and 5 hours of precompute time.
    """
    prefixed = [f"passage: {s}" for s in sentences]
    vecs = model.encode(
        prefixed,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=False  # we normalize per-block separately
    )
    return vecs.astype(np.float32)  # shape: (N, 384)
```

### Step 3.2 — Threshold-based soft multi-block assignment

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Starting threshold values — tune empirically on labeled sample
# Lower = more sentences included (higher recall, more noise)
# Higher = fewer sentences included (higher precision, risk of empty blocks)
THRESHOLDS = {
    "retrieval": 0.35,
    "infra":     0.38,  # slightly higher — infra vocabulary appears in many
                        # non-engineering contexts ("deployed marketing campaign")
                        # higher threshold reduces false positives
    "eval":      0.32   # slightly lower — eval language is more varied and
                        # domain-specific; false positive risk is lower
}


def assign_sentences_to_blocks(
    sentences: list[str],
    sentence_vecs: np.ndarray,
    anchors: dict[str, np.ndarray],
    thresholds: dict[str, float]
) -> dict[str, list[tuple[str, float]]]:
    """
    Soft multi-block assignment.
    Each sentence is scored against ALL three anchors independently.
    A sentence above threshold for a block goes into that block.
    A sentence can appear in multiple blocks simultaneously.

    This handles mixed sentences like:
    "Deployed Qdrant achieving 8ms p99 latency measured via NDCG benchmarks"
    → contains infra signal (deployed, latency, p99)
    → contains retrieval signal (Qdrant)
    → contains eval signal (NDCG, benchmarks)
    Hard single-block assignment would lose two of these signals.
    Soft assignment preserves all three.

    Returns dict: block_name → list of (sentence, similarity_score)
    """
    blocks = {
        "retrieval": [],
        "infra":     [],
        "eval":      []
    }

    for sentence, vec in zip(sentences, sentence_vecs):
        sim_r = cosine_similarity(vec, anchors["retrieval"])
        sim_i = cosine_similarity(vec, anchors["infra"])
        sim_e = cosine_similarity(vec, anchors["eval"])

        if sim_r >= thresholds["retrieval"]:
            blocks["retrieval"].append((sentence, sim_r))

        if sim_i >= thresholds["infra"]:
            blocks["infra"].append((sentence, sim_i))

        if sim_e >= thresholds["eval"]:
            blocks["eval"].append((sentence, sim_e))

    return blocks
```

### Why threshold over top-k

Top-k forces a fixed number of sentences per block regardless of actual relevance:

- A verbose candidate with 30 sentences where the top-5 score only 0.28-0.32 gets 5 sentences of marginal relevance forced into the block.
- A specialist candidate with 3 sentences all scoring 0.75-0.85 gets only 3 sentences — identical to a weak candidate with 5 mediocre sentences.

Threshold-based selection:
- A keyword stuffer whose sentences score 0.15-0.28 gets an **empty block** — the empty block text encodes to a near-zero relevance vector, which is a strong suppression signal in retrieval.
- A plain-language expert whose sentences score 0.36-0.41 (above threshold, no exact keywords) gets correctly included.
- A strong specialist with 3 high-scoring sentences gets exactly those 3 sentences and nothing else.

The empty block case is especially valuable for detecting keyword stuffers. An empty retrieval block is a stronger signal than a low score — it means the candidate had zero sentences describing actual retrieval work, regardless of how many buzzwords appeared in their skills list.

---

## STAGE 4: BLOCK TEXT CONSTRUCTION WITH TOKEN BUDGET

**Input:** block assignments dict from Stage 3

**Output:** three text strings (one per block), token-budget-aware

### Step 4.1 — Sort by similarity, fill to token budget

```python
MAX_TOKENS_PER_BLOCK = 380
# BGE-small has a 512 token context window
# 380 leaves headroom for "passage: " prefix tokens
# and prevents silent truncation of the most relevant sentences
# (BGE silently truncates overflow — you will not get an error)

EMPTY_BLOCK_TEXT = "no relevant experience"
# Encodes to a specific low-relevance vector.
# Better than empty string (undefined behavior in some encoders)
# or zero vector (undefined in cosine space).


def build_block_text(
    assigned_sentences: list[tuple[str, float]],
    max_tokens: int = MAX_TOKENS_PER_BLOCK
) -> str:
    """
    From the list of (sentence, similarity_score) pairs for one block:
    1. Sort by similarity score descending (most relevant first)
    2. Fill token budget greedily from the top
    3. Return concatenated text

    Why sort before applying budget:
    The most semantically relevant sentences go in first.
    If token budget cuts the list at sentence 7, you have included
    the 6 most relevant sentences. Without sorting, a verbose but
    low-relevance sentence early in the profile consumes budget
    that should belong to a high-relevance sentence later.
    """
    if not assigned_sentences:
        return EMPTY_BLOCK_TEXT

    # sort by similarity descending
    sorted_sentences = sorted(assigned_sentences, key=lambda x: x[1], reverse=True)

    selected = []
    total_tokens = 0

    for sentence, score in sorted_sentences:
        # rough token estimate: word count * 1.3 accounts for subword tokenization
        # BGE tokenizer averages ~1.3 tokens per English word
        estimated_tokens = len(sentence.split()) * 1.3

        if total_tokens + estimated_tokens > max_tokens:
            break

        selected.append(sentence)
        total_tokens += estimated_tokens

    if not selected:
        return EMPTY_BLOCK_TEXT

    return " ".join(selected)
```

---

## STAGE 5: BLOCK ENCODING INTO SUB-VECTORS

**Input:** three block text strings from Stage 4

**Output:** three L2-normalized sub-vectors, each shape (384,)

### Step 5.1 — Encode each block with passage: prefix

```python
def encode_block(block_text: str, model) -> np.ndarray:
    """
    Encode one block's text into a 384-d sub-vector.
    passage: prefix — this is evidence text, not a query.
    L2 normalize the output.

    Why L2 normalize each sub-vector separately (not the full 768-d):
    If you normalize the full concatenated vector, a very strong infra
    block (high magnitude) inflates the normalization denominator and
    shrinks the retrieval and eval sub-vectors proportionally.
    This cross-block interference undermines the block structure.

    Normalizing each 256-d (or 384-d) block independently ensures
    all three subspaces operate on equal geometric scale.
    The priority weighting (0.3/0.6/0.1) is then applied at query
    construction time, not baked into sub-vector magnitudes.
    """
    vec = model.encode(
        f"passage: {block_text}",
        normalize_embeddings=False
    )
    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)
```

---

## STAGE 6: CANDIDATE VECTOR ASSEMBLY

**Input:** three encoded sub-vectors from Stage 5

**Output:** one 768-d (or 1152-d if using full BGE-small output) candidate vector

### Step 6.1 — Concatenate sub-vectors

```python
def assemble_candidate_vector(
    v_retrieval: np.ndarray,
    v_infra: np.ndarray,
    v_eval: np.ndarray
) -> np.ndarray:
    """
    Concatenate three sub-vectors into one block-structured vector.

    CRITICAL: Do NOT apply priority weights here.
    Weights are applied to the QUERY vector at rank time.
    The candidate vector stores raw (normalized) sub-vectors.
    Applying weights to candidate vectors would make them
    incomparable across different queries.

    Do NOT L2 normalize the full concatenated vector.
    See Stage 5 note on why per-block normalization is correct.

    Returns: shape (1152,) float32
    """
    return np.concatenate([v_retrieval, v_infra, v_eval]).astype(np.float32)
```

---

## STAGE 7: FULL PER-CANDIDATE PIPELINE

Combining all stages into one function call per candidate.

```python
def process_one_candidate(
    record: dict,
    model,
    anchors: dict[str, np.ndarray],
    thresholds: dict[str, float]
) -> tuple[str, np.ndarray]:
    """
    Full pipeline for one candidate record.
    Returns: (candidate_id, 1152-d vector)
    """
    candidate_id = record["candidate_id"]
    dim = model.get_sentence_embedding_dimension()  # 384 for bge-small

    # graceful fallback for empty/unparseable profiles
    null_vector = np.zeros(dim * 3, dtype=np.float32)

    # Stage 1: text extraction
    segments = build_candidate_segments(record)
    if not segments:
        return candidate_id, null_vector

    # Stage 2: sentence splitting
    sentences = split_into_sentences(segments)
    if not sentences:
        return candidate_id, null_vector

    # Stage 3: batch encode all sentences (one model call)
    sentence_vecs = encode_sentences(sentences, model)

    # Stage 4: threshold-based soft multi-block assignment
    block_assignments = assign_sentences_to_blocks(
        sentences, sentence_vecs, anchors, thresholds
    )

    # Stage 5: build token-budget-aware block texts
    block_texts = {
        block: build_block_text(assignments)
        for block, assignments in block_assignments.items()
    }

    # Stage 6: encode each block text into sub-vector
    v_r = encode_block(block_texts["retrieval"], model)
    v_i = encode_block(block_texts["infra"], model)
    v_e = encode_block(block_texts["eval"], model)

    # Stage 7: assemble final candidate vector
    candidate_vector = assemble_candidate_vector(v_r, v_i, v_e)

    return candidate_id, candidate_vector
```

---

## STAGE 8: BATCH PROCESSING AND FAISS INDEX CONSTRUCTION

### Step 8.1 — Stream candidates and build index

```python
import faiss
import json
import gzip

def build_vector_index(
    candidates_path: str,   # path to candidates.jsonl.gz
    model,
    anchors: dict[str, np.ndarray],
    thresholds: dict[str, float],
    output_dir: str = "artifacts"
) -> None:
    """
    Process all candidates and build FAISS HNSW index.
    Streams from gzipped JSONL — never loads all 100K records into RAM.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    dim = model.get_sentence_embedding_dimension() * 3  # 384 * 3 = 1152

    # HNSW index with inner product metric
    # METRIC_INNER_PRODUCT — preserves block weight structure
    # M=32: graph connectivity parameter
    #   higher M = better recall at query time, more RAM at build time
    #   32 is standard for this scale and quality requirement
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    # efConstruction=200: higher = better index quality during build
    # no time constraint during offline precompute — use high value

    id_map = {}       # faiss integer id → candidate_id string
    vectors = []
    batch_size = 500  # accumulate before adding to index (faster than one-by-one)

    def flush_batch(vectors_batch, start_idx):
        if not vectors_batch:
            return
        arr = np.array(vectors_batch, dtype=np.float32)
        index.add(arr)

    i = 0
    current_batch = []

    with gzip.open(candidates_path, "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            candidate_id, vec = process_one_candidate(
                record, model, anchors, thresholds
            )

            id_map[i] = candidate_id
            current_batch.append(vec)

            if len(current_batch) >= batch_size:
                flush_batch(current_batch, i - len(current_batch) + 1)
                current_batch = []

            i += 1
            if i % 5000 == 0:
                print(f"  processed {i:,} candidates")

    # flush remaining
    if current_batch:
        flush_batch(current_batch, i - len(current_batch))

    # persist index and id map
    faiss.write_index(index, f"{output_dir}/candidate_index.faiss")
    with open(f"{output_dir}/id_map.json", "w") as f:
        json.dump(id_map, f)

    print(f"Index built: {index.ntotal:,} vectors, dim={dim}")
    print(f"Saved to {output_dir}/candidate_index.faiss")
```

### Step 8.2 — FAISS index configuration notes

| Parameter | Value | Reason |
|---|---|---|
| Index type | `IndexHNSWFlat` | HNSW graph for O(log N) approximate search |
| Metric | `METRIC_INNER_PRODUCT` | Preserves block weight structure (not cosine, not L2) |
| M | 32 | Graph connectivity — standard for recall/RAM tradeoff at 100K scale |
| efConstruction | 200 | Build quality — high value acceptable since precompute has no time limit |
| efSearch | 64 (set at query time) | Query recall — set in rank.py, not here |
| dtype | float32 | FAISS requirement — do not use float64 |

**Do NOT** call `faiss.normalize_L2()` on the full candidate vectors before adding to the index. Per-block normalization was already applied in Stage 5. Global normalization would collapse the block structure.

---

## STAGE 9: QUERY VECTOR CONSTRUCTION (reference — used in rank.py)

This stage runs online at rank time, not during precompute. It is included here for completeness so the relationship between candidate vectors and query vectors is clear.

```python
def build_query_vector(
    model,   # loaded via onnxruntime at rank time
    weights: tuple = (0.3, 0.6, 0.1)
) -> np.ndarray:
    """
    Build the weighted query vector from fixed JD-derived query texts.
    Weights: (retrieval, infra, eval)
    Applied BEFORE concatenation — this is where priority structure lives.

    These query texts are fixed and derived from JD analysis.
    They are NOT generated dynamically.
    """
    q_retrieval = "query: production embeddings retrieval semantic search RAG \
                   vector database experience shipped to real users at scale"

    q_infra = "query: deploying scaling vector database production latency \
               optimization inference throughput operational experience"

    q_eval = "query: NDCG MRR MAP evaluation framework ranking A/B testing \
              offline benchmark online experiment design feedback loop"

    v_r = model.encode(q_retrieval)
    v_i = model.encode(q_infra)
    v_e = model.encode(q_eval)

    # normalize each sub-vector independently
    v_r = v_r / np.linalg.norm(v_r)
    v_i = v_i / np.linalg.norm(v_i)
    v_e = v_e / np.linalg.norm(v_e)

    # apply priority weights before concatenation
    w_r, w_i, w_e = weights

    query_vector = np.concatenate([
        w_r * v_r,
        w_i * v_i,
        w_e * v_e
    ]).astype(np.float32)

    return query_vector

# The dot product at retrieval time then decomposes as:
# score = q · c
#       = (0.3 * q_r) · c_r  +  (0.6 * q_i) · c_i  +  (0.1 * q_e) · c_e
#       = 0.3 * sim_retrieval  +  0.6 * sim_infra  +  0.1 * sim_eval
```

---

## COMPLETE DATA FLOW

```
OFFLINE PRECOMPUTE (precompute.py)
══════════════════════════════════════════════════════════════

[STAGE 0 — runs once]
JD text
  ├── 5 retrieval requirement sentences
  │     → BGE encode with "query:" prefix
  │     → mean → L2 normalize → anchor_r  (384-d)
  │
  ├── 5 infra requirement sentences
  │     → BGE encode with "query:" prefix
  │     → mean → L2 normalize → anchor_i  (384-d)
  │
  └── 5 eval requirement sentences
        → BGE encode with "query:" prefix
        → mean → L2 normalize → anchor_e  (384-d)

[STAGES 1-7 — repeated for each of 100,000 candidates]

raw candidate JSON record
  │
  ├── [Stage 1] build_candidate_segments()
  │     role-prefixed experience segments
  │     skills as declarative sentence
  │     current title
  │
  ├── [Stage 2] split_into_sentences()
  │     10-40 individual sentences per candidate
  │     minimum 20 characters, drop fragments
  │
  ├── [Stage 3] encode_sentences() — ONE batch BGE call
  │     "passage: {sentence}" for each sentence
  │     → (N, 384) matrix
  │     ↓
  │     cosine_sim vs anchor_r → threshold 0.35 → retrieval sentences
  │     cosine_sim vs anchor_i → threshold 0.38 → infra sentences
  │     cosine_sim vs anchor_e → threshold 0.32 → eval sentences
  │     [soft: one sentence can qualify for multiple blocks]
  │
  ├── [Stage 4] build_block_text() × 3
  │     sort sentences by similarity score descending
  │     fill greedily to 380 token budget
  │     empty block → "no relevant experience"
  │
  ├── [Stage 5] encode_block() × 3
  │     "passage: {block_text}" → BGE encode
  │     L2 normalize each sub-vector independently
  │     → v_r (384-d), v_i (384-d), v_e (384-d)
  │
  └── [Stage 6] assemble_candidate_vector()
        concatenate [v_r | v_i | v_e]
        NO global normalization
        NO weight application (weights go on query side)
        → 1152-d float32 candidate vector

[STAGE 8 — index construction]
all 100,000 candidate vectors
  → FAISS IndexHNSWFlat(dim=1152, M=32, metric=INNER_PRODUCT)
  → efConstruction=200
  → index.add(vectors)
  → saved: artifacts/candidate_index.faiss
  → saved: artifacts/id_map.json  {faiss_int_id: "CAND_XXXXXXX"}


ONLINE RANK TIME (rank.py) — reference only
══════════════════════════════════════════════════════════════

[query vector — built once]
JD → three fixed query strings
  → BGE encode with "query:" prefix (via onnxruntime)
  → normalize each sub-vector
  → apply weights: [0.3*q_r | 0.6*q_i | 0.1*q_e]
  → 1152-d query vector

[retrieval]
tabular mask (years_exp BETWEEN 5 AND 9, is_honeypot=False)
  → IDSelector bitmask for FAISS
  → index.search(query_vector, k=300, params={efSearch: 64})
  → top 300 candidate IDs

[downstream — out of scope for this document]
top 300 → LightGBM ranking → top 100 → CSV output
```

---

## TUNING PARAMETERS

Three parameters require empirical validation against labeled profiles. Starting values are provided but must be tested.

### Threshold tuning

```
THRESHOLDS = {
    "retrieval": 0.35,
    "infra":     0.38,
    "eval":      0.32
}
```

To validate: manually label 200 candidate profiles across four categories:
1. Strong fit (infra + retrieval + eval, all production)
2. Weak fit (research background, no production)
3. Keyword stuffer (buzzwords, wrong role context)
4. Plain-language expert (no buzzwords, but real production evidence)

For each category, inspect the block texts that result from your thresholds. Ask:
- Do strong fit candidates get populated blocks with specific, evidence-rich sentences?
- Do keyword stuffers get empty or near-empty blocks?
- Do plain-language experts get correctly included in relevant blocks?

Adjust thresholds until behavior is correct for all four categories. Lowering a threshold includes more sentences (higher recall, more noise). Raising it excludes more (higher precision, risk of empty blocks for valid candidates).

### Token budget

```
MAX_TOKENS_PER_BLOCK = 380
```

If strong candidates are getting their best sentences cut, increase toward 450. If you see BGE truncation warnings in output, decrease toward 320. The BGE-small-v1.5 hard limit is 512 tokens including the `passage:` prefix.

### HNSW efSearch (set in rank.py, not precompute.py)

```
efSearch = 64  # starting value
```

Higher efSearch = better recall, slower query. At 100K candidates on CPU, efSearch=64 typically completes in under 30 seconds. If you have budget, raise to 128 for better recall. Validate by checking whether ground-truth strong candidates appear in the top-300 output.

---

## OUTPUT ARTIFACTS

After running the full precompute pipeline, the following files must exist:

```
artifacts/
├── anchors/
│   ├── anchor_retrieval.npy    # shape (384,) float32
│   ├── anchor_infra.npy        # shape (384,) float32
│   └── anchor_eval.npy         # shape (384,) float32
├── candidate_index.faiss       # HNSW index, 100K vectors, dim=1152
└── id_map.json                 # {int_id: "CAND_XXXXXXX"} for 100K candidates
```

These artifacts are loaded by `rank.py` at query time. They must be present before `rank.py` is executed.

---

## IMPLEMENTATION NOTES FOR AI AGENTS

1. **Run Stage 0 first, before processing any candidates.** The anchors must exist before Stage 3 can run. Do not attempt to build anchors inline per-candidate.

2. **Never load all 100K records into RAM simultaneously.** Stream from the gzipped JSONL line by line. The 16GB RAM constraint is real.

3. **One BGE model call per candidate for sentence encoding (Stage 3).** This is the batch encode call over all sentences. Do not loop and encode one sentence at a time — it is 10-50x slower.

4. **Three additional BGE model calls per candidate for block encoding (Stage 5).** These are unavoidable. One per block, after block text is constructed.

5. **Total BGE calls per candidate: 4.** 1 batch call for sentence encoding + 3 calls for block encoding. At 100K candidates, this is 400K model calls total. With batch_size=64 in sentence encoding, actual forward passes are fewer.

6. **Float32 everywhere.** FAISS requires float32. Ensure all numpy arrays are cast to float32 before adding to the index or comparing against anchors.

7. **The null vector (all zeros) for empty profiles is intentional.** A candidate with no parseable text gets a zero vector. This scores zero against any query under inner product — they will not appear in retrieval results. This is correct behavior.

8. **Do not modify the FAISS index after construction.** `IndexHNSWFlat` does not support deletion or modification. If a candidate needs to be excluded, use the IDSelector bitmask at query time (Stage 4.2 in rank.py), not index modification.
