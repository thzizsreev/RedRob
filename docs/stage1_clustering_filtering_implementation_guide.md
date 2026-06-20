# Implementation Guide: Stage 1 Cluster-Based Filtering

**Audience:** a coding agent implementing this from scratch. This covers three things in order: (1) how to build the clusters, (2) how to rank clusters against the JD anchor, (3) how to use that ranking to filter candidates. Each section has concrete function signatures, parameters, and the reasoning behind each choice so you don't substitute your own defaults.

**Pre-requisite (assumed already built elsewhere, not covered here):** every candidate already has a 768-d (or whatever dimension) block-division embedding vector stored, e.g. in a Polars DataFrame or numpy array keyed by `candidate_id`. The JD anchor vector(s) are also already computed (see `build_anchor` pattern below if not).

---

## Part 1 — Building the Clusters

### 1.1 Why this pipeline shape
Block-division vectors are high-dimensional (768-d). HDBSCAN (and clustering in general) performs poorly directly in high dimensions because distance metrics become less meaningful ("curse of dimensionality"). The standard fix, and what was used to produce the validated clustering result in this project, is: **reduce dimensionality with UMAP first, then cluster the reduced representation with HDBSCAN.** Do not run HDBSCAN directly on the raw 768-d vectors.

### 1.2 Dependencies
```text
umap-learn>=0.5.5
hdbscan>=0.8.33
numpy>=1.24.0
polars>=0.20.0
```

### 1.3 Step-by-step

**Step A — Load all candidate vectors into a single matrix.**
```python
import numpy as np

# candidate_ids: list[str], length N
# vectors: np.ndarray of shape (N, 768), float32, one row per candidate_id, same order
```

**Step B — Reduce dimensionality with UMAP.**
```python
import umap

def reduce_for_clustering(vectors: np.ndarray, n_components: int = 2, random_state: int = 42) -> np.ndarray:
    """
    n_components=2 is used for visualization (matches the validated UMAP plots
    already produced for this project). For the actual clustering used in filtering,
    you may use a higher n_components (e.g. 10-15) to retain more structure than
    2D allows — 2D was for human inspection, not necessarily the best clustering input.
    Decide explicitly: if you want filtering results to match the already-validated
    2D clustering exactly, keep n_components=2. If you want richer structure, use more
    dimensions but note this is a NEW clustering result, not the one already validated.
    """
    reducer = umap.UMAP(
        n_components=n_components,
        random_state=random_state,
        metric="cosine",  # cosine is standard for UMAP on embeddings; note this is
                           # UMAP's internal distance metric for the reduction step only —
                           # it does NOT affect the inner-product similarity scoring used
                           # elsewhere in this project, which stays dot-product.
    )
    return reducer.fit_transform(vectors)
```

**Step C — Cluster the reduced vectors with HDBSCAN.**
```python
import hdbscan

def cluster_candidates(reduced_vectors: np.ndarray, min_cluster_size: int = 15) -> np.ndarray:
    """
    Returns an array of cluster labels, one per candidate, same order as input.
    Label -1 means "noise" / not assigned to any cluster — HDBSCAN does this
    deliberately, unlike k-means which forces every point into a cluster.

    min_cluster_size: the smallest grouping HDBSCAN will treat as a real cluster.
    The validated run on ~1000 candidates produced 4 clusters (sizes 45, 250, 524, 181)
    with this in a reasonable range (10-20). Tune if your candidate pool size differs
    significantly (100K full pool will need a proportionally different value — do not
    blindly reuse the same min_cluster_size used on the 1000-candidate validation sample;
    scale it roughly in proportion to pool size, then sanity-check resulting cluster count
    and sizes before trusting it).
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",  # operates on the UMAP-reduced space, not the original
                              # embedding space — euclidean is standard here
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(reduced_vectors)
```

**Step D — Persist the mapping.**
```python
import polars as pl

def save_cluster_assignments(candidate_ids: list[str], labels: np.ndarray, path: str):
    df = pl.DataFrame({"candidate_id": candidate_ids, "cluster_label": labels})
    df.write_parquet(path)
```

### 1.4 What NOT to do here
- Do not run HDBSCAN on raw 768-d vectors directly — always UMAP-reduce first.
- Do not hardcode any cluster label as "the good cluster." Labels are arbitrary integers assigned by the algorithm; this is enforced in Part 2.
- Do not discard label `-1` (noise) silently — see Part 3.4 for explicit handling.

---

## Part 2 — Ranking Clusters Against the JD Anchor

### 2.1 Why median, not mean
Clusters have observed internal density variation (some clusters are tight, some have long tails of weaker matches). Mean similarity is sensitive to that long tail and can be misleading; median is more robust to it. Use median for cluster-level aggregation.

### 2.2 Implementation
```python
def compute_candidate_similarity(candidate_vec: np.ndarray, anchor_vec: np.ndarray) -> float:
    """Inner product / dot product similarity — NOT cosine. Both vectors are
    expected to already be unit-normalized at construction time (see anchor-building
    pattern from Part 1 of the embedding strategy doc), so dot product here is
    equivalent to cosine in magnitude but preserves the orthogonal subspace
    structure that normalizing post-hoc would destroy."""
    return float(np.dot(candidate_vec, anchor_vec))


def rank_clusters_by_anchor_similarity(
    candidate_ids: list[str],
    vectors: np.ndarray,
    cluster_labels: np.ndarray,
    anchor_vec: np.ndarray,
) -> list[tuple[int, float, int]]:
    """
    Returns a list of (cluster_label, median_similarity, cluster_size) tuples,
    sorted descending by median_similarity. This ranking is recomputed every run —
    never cached or assumed stable across reruns, since HDBSCAN label assignment
    is not guaranteed stable.
    """
    from collections import defaultdict
    import statistics

    cluster_to_sims = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        sim = compute_candidate_similarity(vectors[i], anchor_vec)
        cluster_to_sims[label].append(sim)

    results = []
    for label, sims in cluster_to_sims.items():
        med = statistics.median(sims)
        results.append((label, med, len(sims)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results
```

**Critical:** this function is called fresh every time `rank.py` runs against a new JD or a re-clustered pool. Do not persist or hardcode the output ranking — the whole point is that cluster identity is meaningless until ranked against the current anchor.

---

## Part 3 — Filtering Algorithm

### 3.1 Architecture summary
```
[candidate vectors] --> [UMAP reduce] --> [HDBSCAN cluster] --> [rank clusters by median anchor similarity]
                                                                          |
                                                                          v
                                          [walk ranked clusters, add whole clusters to filtered set,
                                           stop once cumulative count >= floor]
                                                                          |
                                                                          v
                                          [output: unordered set of candidate_ids -- NO scores, NO ranks]
```

### 3.2 Core filtering function
```python
def filter_candidates_by_cluster(
    candidate_ids: list[str],
    cluster_labels: np.ndarray,
    ranked_clusters: list[tuple[int, float, int]],  # output of rank_clusters_by_anchor_similarity
    floor: int = 100,
    include_noise_as_last_resort: bool = False,
) -> set[str]:
    """
    Implements: rank clusters -> add whole clusters in ranked order -> stop at floor.
    No candidate-level scoring or ordering happens here. Output is an unordered set.
    """
    # Build label -> list[candidate_id] lookup once
    from collections import defaultdict
    label_to_ids = defaultdict(list)
    for cid, label in zip(candidate_ids, cluster_labels):
        label_to_ids[label].append(cid)

    filtered_set: set[str] = set()

    for label, median_sim, size in ranked_clusters:
        if label == -1:
            # noise / unclustered points -- see 3.4, handle explicitly, do not
            # silently include or exclude
            if not include_noise_as_last_resort:
                continue
        filtered_set.update(label_to_ids[label])
        if len(filtered_set) >= floor:
            break

    return filtered_set
```

### 3.3 Tie-breaking
If two clusters have identical median similarity (rare with float scores but possible), break ties deterministically by cluster label ascending, so re-running on the same data produces the same filtered set:
```python
results.sort(key=lambda x: (-x[1], x[0]))  # descending similarity, then ascending label
```

### 3.4 Handling noise points (label -1) — explicit decision required
HDBSCAN assigns `-1` to points it can't confidently place in any cluster. This is not automatically "bad" — it could contain genuine outlier-good candidates whose profiles don't closely resemble others, or genuine outlier-bad candidates. Two acceptable approaches, pick one and document the choice:

- **Conservative (default):** treat noise as lowest-priority — never pulled in unless every real cluster has been exhausted and the floor still isn't met (extremely unlikely at 100K scale, but handle it rather than crashing).
- **Inclusive:** rank noise points individually by anchor similarity and only include the noise points above some similarity threshold, added after all real clusters. This reintroduces a small amount of candidate-level filtering logic, so only do this if conservative handling leaves the floor unmet in practice.

Do not default to silently dropping noise or silently including all of it — both are unexamined defaults that should be a conscious call.

### 3.5 What this stage explicitly does NOT do
- Does not apply YOE, honeypot, or consulting-firm hard gates — those run in the ranking phase on the *filtered* set, not here.
- Does not compute or attach a per-candidate score to the output set.
- Does not sort or rank the output set in any way — the output is a Python `set`, not a list, to make this structurally explicit (sets have no order).
- Does not fractionally include part of a cluster — clusters are atomic units of inclusion.

### 3.6 End-to-end call pattern
```python
def run_stage1_filtering(
    candidate_ids: list[str],
    vectors: np.ndarray,
    anchor_vec: np.ndarray,
    floor: int = 100,
) -> set[str]:
    reduced = reduce_for_clustering(vectors)
    labels = cluster_candidates(reduced)
    ranked = rank_clusters_by_anchor_similarity(candidate_ids, vectors, labels, anchor_vec)
    filtered_set = filter_candidates_by_cluster(candidate_ids, labels, ranked, floor=floor)
    return filtered_set
```

---

## Part 4 — Validation Checklist Before Trusting This on the Full 100K Pool

- [ ] Confirm `min_cluster_size` scaled appropriately for 100K (don't reuse the value tuned on the 1000-candidate sample without re-checking resulting cluster count/sizes).
- [ ] Confirm the top-ranked cluster after re-running on 100K is dominated by AI/ML-relevant titles, the same sanity check performed on the 1000-candidate sample (Section 3 of prior clustering analysis) — do not assume it transfers without checking.
- [ ] Confirm `filtered_set` size is comfortably ≥ floor and not pathologically large (e.g. if the top cluster alone is 40,000 candidates, something is wrong with clustering granularity, not the filtering logic).
- [ ] Confirm noise-point handling decision (3.4) was made explicitly, not defaulted silently.
- [ ] Confirm wall-clock time for the full UMAP + HDBSCAN pass on 100K candidates fits within whatever time budget Stage 1 is allotted within the overall 5-minute online constraint, or confirm this entire clustering step runs offline in `precompute.py` rather than online in `rank.py` (strongly recommended — clustering 100K points is not a sub-5-minute-safe operation to run online; this should be precomputed once, with only the cluster-ranking-against-anchor step run online in `rank.py`, since the anchor depends on the JD and isn't known until then).
