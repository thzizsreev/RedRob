# Vector Semantic Richness Testing Pipeline
### Redrob Hackathon — Track A Embedding Validation (No Ground Truth Available)

---

## 1. Purpose & Scope

This pipeline answers one question: **does the current embedding model produce noticeably differentiated, semantically coherent vectors, or is the 2204-d space a relatively undifferentiated blob?**

This is *not* a ranking evaluation. It does not attempt to measure NDCG, precision, or fit-to-JD quality. It is a structural diagnostic of the embedding space itself — used to decide whether the embedder needs to change before you invest further in the retrieval/ranking layers downstream.

The output of this pipeline is not a score you submit anywhere. It's a feedback loop: run it, look at the chart and the cluster contents, decide if the embedder is differentiating well, adjust the embedder (instruction prompts, base model, extraction logic), re-run, compare.

This document covers: sampling, dimensionality reduction, clustering, quantitative scoring, interactive visualization, manual inspection, and the iteration loop. It assumes the 2204-d vectors already exist (produced by your `precompute.py`/extraction step) — this pipeline only reads them, it does not regenerate them.

---

## 2. Algorithm Decision: One Clustering Method, Not Three

**Decision: use HDBSCAN as the single clustering algorithm.** No separate files/pipelines for K-means, DBSCAN, and Agglomerative.

Reasoning:

- **K-means** assumes spherical, equally-sized clusters and requires you to pre-specify K. You have no prior belief about how many "natural" candidate archetypes exist in a 100K pool — picking K is itself a guess, and a wrong K silently distorts every downstream interpretation (forced merges, forced splits).
- **Agglomerative clustering** is informative but doesn't scale comfortably to a few thousand high-dimensional vectors per run without real care about linkage choice, and still requires you to choose a cut height/number of clusters after the fact — same problem as K-means, one step later.
- **HDBSCAN** (Hierarchical DBSCAN) needs no pre-specified cluster count, naturally handles clusters of different density and size (semantic clusters are rarely uniform — "AI/ML engineers" will be a much denser, larger region than "robotics with NLP crossover"), and — critically for this dataset — it **natively labels low-density points as noise/outliers** instead of forcing them into a cluster. That last property is a free bonus: outlier-labeled points are exactly where you'd expect keyword-stuffers, honeypots, and genuinely ambiguous profiles to land. You get a built-in anomaly signal without building one separately.

Running three algorithms in parallel triples the analysis and interpretation burden for marginal extra insight, since the question you're asking ("is there real structure here?") is best answered by one well-chosen method run carefully, not three run shallowly. If HDBSCAN's results look ambiguous after inspection, that's the trigger to bring in a second algorithm as a cross-check — not before.

---

## 3. Stage 1: Sampling

Don't cluster the full 100K every iteration — too slow for a fast feedback loop, and unnecessary for a structural diagnostic.

- Sample size: 2,000–5,000 candidates per run. Large enough for clusters to form reliably, small enough to re-run in minutes.
- **Stratify the sample**, don't draw purely at random. Use whatever weak metadata you already have (declared title keywords, years_of_experience bucket, skill-tag count bucket) to make sure the sample spans the breadth of the pool rather than over-representing the most common profile type.
- Deliberately oversample your known reference points: the synthetic trap profiles from earlier (keyword-stuffer, tier-5 plain-language, honeypot, control strong-fit) should always be injected into every sample run, even though they're synthetic. They act as fixed landmarks — if their relative position in the cluster structure changes wildly between embedder versions, that's a meaningful signal independent of anything else in the sample.
- Keep the sample ID list fixed across iterations where possible (same 3,000 candidates each time you test a new embedder version). This makes cluster-to-cluster comparisons across versions valid instead of confounded by sampling noise.

---

## 4. Stage 2: Dimensionality Reduction (Two Separate UMAP Runs)

Run UMAP twice, for two different purposes — conflating them is a common mistake:

**Run A — Clustering-space UMAP.** Reduce the 2204-d vectors to roughly 10–15 dimensions. HDBSCAN (like most density-based methods) suffers from the curse of dimensionality directly on 2204-d raw vectors — distances become less meaningful, density becomes harder to estimate. Reducing to a moderate intermediate dimensionality first, then clustering in that space, is the standard practice (this is the same approach used by topic-modeling pipelines like BERTopic: UMAP-to-~10d, then HDBSCAN). Use `n_neighbors` around 15–30 and `min_dist` near 0 for this run, since you want it optimized for cluster separation, not visual aesthetics.

**Run B — Visualization-space UMAP.** Separately, reduce the same 2204-d vectors straight to 2 dimensions, purely for plotting. Use a higher `min_dist` (0.1–0.3) here so the plot is visually legible rather than maximally compressed — clustering-space and visualization-space UMAP runs optimize for different things and should not share output.

Also run a quick **PCA-to-2D** pass alongside Run B, even though you've already decided UMAP is the primary tool. PCA's explained-variance-ratio for the first 2–3 components is a cheap, complementary number: if PCA captures very little variance (under ~10%), it confirms the structure is non-linear and reinforces that UMAP (not PCA) is the right lens — useful supporting evidence when you write up what you found, not a replacement for the UMAP plot.

---

## 5. Stage 3: Clustering with HDBSCAN

Run HDBSCAN on the Stage 2 Run A output (the ~10–15-d clustering space), not on the raw 2204-d vectors and not on the 2D visualization space (clustering on a 2D projection throws away too much structure and will give misleadingly clean-looking clusters).

Key parameters to set deliberately, not leave at defaults:

- `min_cluster_size`: start around 1–2% of your sample size (e.g., ~30–60 for a 3,000-candidate sample). Too small and you get hundreds of meaningless micro-clusters; too large and distinct sub-populations get merged into mush.
- `min_samples`: controls how conservative the algorithm is about calling something noise vs. a real cluster. Higher values produce more noise-labeled points and tighter, more conservative clusters. Start near the default (equal to `min_cluster_size`) and loosen if too much of the sample ends up labeled noise.
- Distance metric: use cosine distance on the embeddings if your downstream FAISS index also uses an angle/inner-product-style similarity; use Euclidean if you've already normalized vectors going in. Be consistent with whatever metric your actual retrieval system uses — testing differentiation under a different metric than production uses defeats the purpose.

Record, per run: number of clusters found, size of each cluster, and percentage of the sample labeled noise (-1).

---

## 6. Stage 4: Quantitative Scoring

Three numbers to compute per run, tracked across embedder iterations:

1. **Silhouette score**, computed on the *original high-dimensional* clustering input (the Stage 2 Run A 10–15-d space), not the 2D visualization. Silhouette scores computed on a 2D projection are flattering and unreliable — UMAP can visually separate things that aren't really separated in the underlying space. Range is -1 to 1; for noisy real-world embedding data, anything consistently above ~0.2–0.3 with semantically coherent clusters is a reasonable working bar — there's no universal "good" threshold, so treat this as a relative number to track across iterations, not an absolute pass/fail line.
2. **Noise ratio**: percentage of the sample HDBSCAN couldn't confidently assign to any cluster. A very high noise ratio (most of the sample is "noise") suggests the embedding space lacks structure altogether. A very low noise ratio (almost nothing is noise) is also a yellow flag — it suggests HDBSCAN is being too permissive and likely merging genuinely different profiles together.
3. **Cluster count and size distribution**: are there a handful of well-formed, differently-sized clusters, or one giant cluster swallowing everything plus a scatter of singletons? The former is what healthy structure looks like; the latter suggests weak differentiation.

These three numbers, tracked side by side across embedder versions, are your primary "is this getting better or worse" signal — cheaper to compute and compare than re-running manual inspection every time.

---

## 7. Stage 5: Interactive Visualization

Use the Stage 2 Run B 2D coordinates as the plot, colored by HDBSCAN cluster label (noise points in a distinct neutral color, e.g. grey).

For the "click a point, see details" requirement (the Neo4j-node-style interaction): build this as a Plotly scatter plot with each point's `customdata` populated with the candidate's metadata (candidate_id, declared title, years of experience, a short skills list, and which subspace dominated their vector if you're also doing the per-subspace breakdown in Stage 7). Plotly's hover tooltip alone gives you this without extra infrastructure; if you want true click-to-pin behavior (so you can click multiple points and keep their info on screen rather than only seeing it on hover), that requires a small amount of JS click-event wiring on top of the same Plotly figure, or a lightweight Dash app — both are reasonable, the Dash route gives you more room to add filters/search later if this becomes a recurring tool rather than a one-off diagnostic.

Plot **both** the raw-cluster-colored version and a version colored by your fixed synthetic landmark profiles (the trap candidates from Stage 1) standing out distinctly (e.g., large marker, black outline) so you can visually track where they land relative to the rest of the cloud on every iteration.

---

## 8. Stage 6: Manual Cluster Inspection Protocol

Numbers alone (silhouette, noise ratio) tell you *whether* there's structure, not *whether the structure is the right kind*. This step is what closes that gap.

For each cluster HDBSCAN finds (skip this for the noise group, handle that separately in Stage 8):

- Pull 5–8 random members from the cluster.
- Read their actual profile text (title, top skills, a sentence or two of experience).
- Ask: do these profiles share an obvious, describable theme (e.g., "all senior retrieval/search engineers," "all CV/robotics people," "all non-technical roles with AI buzzwords")? Write that theme down in one line per cluster.
- Flag any cluster where the members look unrelated to each other — that's evidence of either too-large `min_cluster_size`/`min_samples` settings merging distinct groups, or genuinely weak embedding differentiation in that region of the space.

This produces a one-line label per cluster, which is what makes the colored scatter plot interpretable to anyone (including yourself, two days later) without re-reading raw profiles every time.

---

## 9. Stage 7: Subspace-Level Breakdown (Optional, Run if Stage 3-6 Looks Ambiguous)

If your 2204-d vector is built from concatenated, differently-weighted blocks, repeat Stages 2–6 on each block independently (run UMAP + HDBSCAN on just the retrieval block's slice of the vector, then just the infra block's slice, then just the eval block's slice).

This tells you *which* block is carrying the differentiating signal. If one block produces tight, coherent clusters and another produces an undifferentiated blob, that block's extraction step (the offline categorization/instruction prompt) is the one that needs attention — far more actionable than knowing only that "the whole vector" is weak or strong.

---

## 10. Stage 8: Honeypot / Outlier Cross-Check

HDBSCAN's noise-labeled points are a free signal worth checking deliberately, not just skipping past.

Pull the full set of noise-labeled points from your sample and cross-reference them against any honeypot indicators you can already detect heuristically (e.g., years-at-company exceeding company age, if that's checkable from your synthetic landmark profiles or any other quick rule). If a meaningfully higher proportion of noise points are honeypot-like compared to the overall sample, that's a useful secondary signal — it suggests the embedding space is at least partially picking up on the same "this profile reads as implausible" quality that defines a honeypot, even though it was never trained or instructed to detect honeypots specifically. This is a bonus finding, not something to rely on as your actual honeypot filter — that job still belongs to the explicit tabular checks in Track B.

---

## 11. Iteration Loop Across Embedder Versions

Every time you change something about the embedder (swap the base model, change the offline extraction/instruction prompts, change how the three subspaces are weighted or constructed):

1. Re-run Stages 1–6 on the **same fixed sample** (same candidate IDs, including the same synthetic landmarks).
2. Compare silhouette score, noise ratio, and cluster count/size distribution against the previous version's numbers.
3. Compare the one-line cluster themes from Stage 6 — are they getting more specific and coherent, or staying vague/mixed?
4. Check where the synthetic landmark profiles land — is the keyword-stuffer landmark drifting toward noise/its own isolated cluster (good) or blending into the strong-fit cluster (bad)? Is the tier-5 plain-language landmark landing near genuinely technical clusters (good) or off in its own disconnected region (bad)?

Treat an embedder change as an improvement only if multiple signals agree (quantitative scores improve *and* cluster themes get more coherent *and* landmarks move in the expected direction). A single improved number without the qualitative read backing it up is a weak basis for concluding the embedder actually got better.

---

## 12. Chart Interpretation Checklist (Quick Reference)

When looking at the 2D UMAP plot, scan for:

- **Distinct, separated regions of color** rather than one undifferentiated cloud — evidence of real structure.
- **Cluster size variety** — a mix of larger and smaller clusters is more plausible than perfectly uniform cluster sizes, which can indicate an algorithm artifact rather than real structure.
- **Noise points scattered at the edges/between clusters**, not forming their own dense blob — noise should look like genuine in-between/ambiguous cases, not a hidden cluster the algorithm failed to find.
- **Landmark profile positions** — keyword-stuffer should sit apart from genuinely strong clusters; tier-5 plain-language should sit *within* a strong technical cluster despite lacking jargon; honeypot should sit in or near the noise region.
- **No single giant cluster absorbing the majority of the sample** — if 80%+ of points land in one cluster, the embedding isn't differentiating much within the dominant profile type, even if it successfully separates a few outlier groups.

---

## 13. Output Artifacts

Per run, persist:

- The sampled candidate ID list (for reproducibility across iterations).
- The Stage 2 Run A clustering-space coordinates and Run B visualization-space coordinates.
- The HDBSCAN cluster labels per candidate.
- The silhouette score, noise ratio, and cluster size distribution as a small summary record (one row per embedder version, so you can track them in a simple table over time).
- The Stage 6 one-line cluster theme labels.
- The interactive HTML plot itself, so past runs can be reopened and re-examined without recomputation.

---

## 14. When to Stop Iterating on This Diagnostic

This pipeline is a means, not an end. Stop treating it as the bottleneck once: clusters are visibly coherent on manual inspection, silhouette/noise-ratio numbers have plateaued across two or three embedder versions (no longer meaningfully improving), and the synthetic landmark profiles consistently land where you'd expect. At that point, the marginal value shifts from "does the embedding space have structure" to "does that structure actually predict JD fit" — which is a different, ranking-oriented question this pipeline deliberately doesn't try to answer, and is where the gold-set NDCG check from earlier becomes the right next tool.
