# Implementation Guide: K-Means Clustering for Stage 1 Filtering

**Audience:** a coding agent implementing this from scratch. This covers building k-means clusters from candidate vectors, selecting `n_clusters` to hit the target filtered-set granularity (100-200), and wiring the result into the existing cluster-ranking and filtering logic.

**Pre-requisite (assumed already built):** every candidate has a 768×3-d block-division embedding vector (Retrieval + Infra + Eval, concatenated), keyed by `candidate_id`. The JD anchor vector(s) are already computed.

**This document replaces only the *clustering* step. The ranking-clusters-by-anchor-similarity and filter-by-whole-clusters logic from the prior implementation guide are unchanged and reused as-is — see Part 3.**

---

## Part 1 — Why K-Means Here

K-means was chosen over density-based methods (HDBSCAN, OPTICS) specifically because the requirement is **direct control over granularity** — the prior density-based attempt produced one cluster of ~6,000 candidates, far above the 100-200 target range, and tuning density parameters to force smaller clusters risks manufacturing unstable, non-reproducible sub-clusters where no real density boundary exists. K-means sidesteps this entirely: you specify `n_clusters` directly, and every point is assigned to exactly one cluster — no noise label, no ambiguity, no dependency on whether real density gaps exist in the data.

**Trade-off being accepted:** k-means clusters are not "natural" density groupings — they partition space into roughly even, locally-spherical regions around learned centroids. This is acceptable here because clusters are not being used to assert "real groups exist," they're being used as a recall mechanism whose only job is to subdivide the candidate pool so the top-ranked group lands in the target size range. The cluster-ranking-by-anchor-similarity step downstream still does the work of deciding which partition is JD-relevant.

---

## Part 2 — Building the Clusters

### 2.1 Dependencies
```text
scikit-learn>=1.2.2
umap-learn>=0.5.3
numpy>=1.24.0
polars>=0.20.0
```

### 2.2 Dimensionality reduction (unchanged requirement)
K-means, like all distance-based clustering, degrades in high dimensions (curse of dimensionality flattens distance differences). UMAP-reduce before clustering, same as in the prior HDBSCAN pipeline.

```python
import umap
import numpy as np

def reduce_for_clustering(vectors: np.ndarray, n_components: int = 15, random_state: int = 42) -> np.ndarray:
    """
    n_components=15 (not 2) is recommended for the clustering input, as opposed to
    the 2D reduction used for visualization. 2D was for human inspection of plots;
    a higher-dimensional reduction retains more of the original 768x3-d structure
    for the actual clustering decision. K-means in particular benefits from a
    moderate number of dimensions (not too few, which loses structure; not too many,
    which reintroduces high-dimensional distance problems).
    """
    reducer = umap.UMAP(
        n_components=n_components,
        random_state=random_state,
        metric="cosine",
    )
    return reducer.fit_transform(vectors)
```

### 2.3 Selecting `n_clusters` — this is the core new problem to solve

K-means requires `n_clusters` up front. There is no universally correct value — it must be tuned against your actual 100K pool to make the **top-ranked cluster** (not just any cluster) land in the 100-200 range. This requires an iterative search, not a single guess.

**Recommended approach: target-size-driven binary search, not the classic elbow/silhouette method.**

The classic ways to pick `n_clusters` (elbow method on inertia, silhouette score) optimize for "what clustering best explains the data's natural variance" — that is not your actual goal. Your goal is narrower and more specific: *the single top-ranked cluster must be 100-200 candidates*. Optimizing for general cluster quality is the wrong objective function for this task; optimizing directly for top-cluster size is the right one.

```python
from sklearn.cluster import KMeans

def cluster_candidates_kmeans(reduced_vectors: np.ndarray, n_clusters: int, random_state: int = 42) -> np.ndarray:
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return kmeans.fit_predict(reduced_vectors)


def find_n_clusters_for_target_top_size(
    reduced_vectors: np.ndarray,
    candidate_ids: list[str],
    vectors_full: np.ndarray,   # original (pre-UMAP) vectors, needed for anchor similarity
    anchor_vec: np.ndarray,
    target_min: int = 100,
    target_max: int = 200,
    n_clusters_low: int = 20,
    n_clusters_high: int = 500,
    max_iterations: int = 12,
) -> int:
    """
    Binary-searches n_clusters so that the TOP-RANKED cluster (by median anchor
    similarity, using the same ranking function as the filtering pipeline) falls
    within [target_min, target_max]. This directly optimizes the quantity that
    matters, rather than a proxy metric like inertia or silhouette score.

    Increasing n_clusters generally shrinks average cluster size, including the
    top-ranked one, so this relationship is roughly monotonic and binary search
    is a reasonable strategy -- but it is NOT guaranteed perfectly monotonic
    (the top-ranked cluster's *identity* can shift between iterations as n_clusters
    changes, since a different partition can surface a different "best" cluster).
    Treat this as a heuristic search, not an exact solve -- log every iteration's
    result and inspect manually rather than trusting the final value blindly.
    """
    low, high = n_clusters_low, n_clusters_high
    best_n = None
    history = []

    for _ in range(max_iterations):
        mid = (low + high) // 2
        labels = cluster_candidates_kmeans(reduced_vectors, n_clusters=mid)
        ranked = rank_clusters_by_anchor_similarity(candidate_ids, vectors_full, labels, anchor_vec)
        top_label, top_sim, top_size = ranked[0]
        history.append((mid, top_size, top_sim))

        if target_min <= top_size <= target_max:
            best_n = mid
            break
        elif top_size > target_max:
            low = mid + 1   # need more, smaller clusters
        else:
            high = mid - 1  # too fragmented, top cluster too small -- back off

        if low > high:
            break

    if best_n is None:
        # No exact hit -- pick the iteration whose top_size was closest to the
        # target range midpoint, and surface this clearly rather than silently
        # accepting a bad fit.
        target_mid = (target_min + target_max) / 2
        best_n = min(history, key=lambda h: abs(h[1] - target_mid))[0]
        print(f"WARNING: no exact n_clusters hit target range. "
              f"Using n_clusters={best_n} as closest match. Full search history: {history}")

    return best_n
```

**Note:** `rank_clusters_by_anchor_similarity` here is the exact same function from the prior HDBSCAN-based implementation guide (Part 2 of that document) — it is algorithm-agnostic and takes cluster labels as input regardless of which clustering method produced them. Reuse it unchanged.

### 2.4 Practical search range guidance
Given the HDBSCAN result (top cluster ≈ 6,000 out of ~100K, i.e. roughly 6% of the pool), a reasonable starting point for `n_clusters_low`/`n_clusters_high` in the binary search is informed by that ratio: if you want a ~100-200 candidate top cluster from a 100K pool, you're looking for a partition where the top cluster is roughly 0.1-0.2% of the pool, which — if clusters were even in size — would suggest `n_clusters` somewhere in the 500-1000 range. In practice clusters are not even (the top-ranked one will likely be smaller than average, since it's a narrow, specific profile per the JD's own admission that good matches are rare), so expect the actual answer to require fewer clusters than that naive estimate. Start the search range wide (e.g. 50 to 2000) on the first run against the full 100K pool, narrow it once you see where results land.

---

## Part 3 — Ranking and Filtering (Unchanged, Reused As-Is)

These functions are identical to the prior HDBSCAN-based guide and are algorithm-agnostic — they operate on `cluster_labels` regardless of what produced them.

```python
def compute_candidate_similarity(candidate_vec: np.ndarray, anchor_vec: np.ndarray) -> float:
    """Inner product / dot product, not cosine."""
    return float(np.dot(candidate_vec, anchor_vec))


def rank_clusters_by_anchor_similarity(
    candidate_ids: list[str],
    vectors: np.ndarray,
    cluster_labels: np.ndarray,
    anchor_vec: np.ndarray,
) -> list[tuple[int, float, int]]:
    """Returns (cluster_label, median_similarity, cluster_size), sorted descending."""
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

    results.sort(key=lambda x: (-x[1], x[0]))  # descending similarity, tie-break ascending label
    return results


def filter_candidates_by_cluster(
    candidate_ids: list[str],
    cluster_labels: np.ndarray,
    ranked_clusters: list[tuple[int, float, int]],
    floor: int = 100,
) -> set[str]:
    """
    Walk ranked clusters, add whole clusters, stop once cumulative count >= floor.
    No candidate-level scoring. K-means has no noise label (-1), so unlike the
    HDBSCAN version, every cluster is a real candidate for inclusion -- there is
    no separate noise-handling branch needed here.
    """
    from collections import defaultdict
    label_to_ids = defaultdict(list)
    for cid, label in zip(candidate_ids, cluster_labels):
        label_to_ids[label].append(cid)

    filtered_set: set[str] = set()
    for label, median_sim, size in ranked_clusters:
        filtered_set.update(label_to_ids[label])
        if len(filtered_set) >= floor:
            break

    return filtered_set
```

---

## Part 4 — End-to-End Call Pattern

```python
def run_stage1_filtering_kmeans(
    candidate_ids: list[str],
    vectors: np.ndarray,
    anchor_vec: np.ndarray,
    floor: int = 100,
    target_max: int = 200,
) -> set[str]:
    reduced = reduce_for_clustering(vectors)

    n_clusters = find_n_clusters_for_target_top_size(
        reduced_vectors=reduced,
        candidate_ids=candidate_ids,
        vectors_full=vectors,
        anchor_vec=anchor_vec,
        target_min=floor,
        target_max=target_max,
    )

    labels = cluster_candidates_kmeans(reduced, n_clusters=n_clusters)
    ranked = rank_clusters_by_anchor_similarity(candidate_ids, vectors, labels, anchor_vec)
    filtered_set = filter_candidates_by_cluster(candidate_ids, labels, ranked, floor=floor)
    return filtered_set
```

**Important operational note:** `find_n_clusters_for_target_top_size` re-runs k-means up to `max_iterations` times (default 12), each on the full UMAP-reduced 100K-candidate pool. This is a real cost — k-means on 100K points is fast per run, but 12 runs plus the UMAP reduction (done once, reused across iterations) needs to be checked against your time budget. **This entire search-and-cluster process belongs in offline `precompute.py`, never in the online `rank.py` 5-minute window** — same constraint as the HDBSCAN version. Only the final `rank_clusters_by_anchor_similarity` + `filter_candidates_by_cluster` steps (cheap, just dot products and set operations) run online, once the JD anchor is known.

---

## Part 5 — Validation Checklist

- [ ] Run the `n_clusters` search on the actual 100K pool, not a sample — cluster size ratios will not transfer reliably from the 1,000-candidate validation set.
- [ ] After finding `n_clusters`, manually verify the top-ranked cluster's title distribution is still AI/ML-dominant (same sanity check as before) — k-means partitions are more artificial than density-based ones, so this is not guaranteed by construction and must be re-checked.
- [ ] Confirm `find_n_clusters_for_target_top_size` actually converges (`best_n is not None` without hitting the warning branch) — if it consistently falls back to the closest-match warning, the target range itself may need revisiting, or the underlying data may not partition cleanly at any granularity (investigate before assuming the search logic is broken).
- [ ] Confirm total wall-clock time for the offline search-and-cluster process is acceptable for your offline compute budget (no hard limit per the hackathon spec, but should still complete in reasonable time for iteration during development).
- [ ] Re-run the 5 synthetic test profiles (Profile 1-5 from earlier validation) through this k-means pipeline and confirm Profile 1 and Profile 3 still land in the top-ranked cluster, consistent with the HDBSCAN result — this confirms the algorithm switch didn't break the validated positive-extraction behavior.
- [ ] `random_state` is fixed (42) for reproducibility — confirm this is intentional and not something that should be varied/seeded differently across runs for your specific submission process.
