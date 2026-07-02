# REDROB HACKATHON: ULTIMATE AI AGENT CONTEXT & ARCHITECTURE BIBLE

**CRITICAL DIRECTIVE FOR AI AGENTS:** You are building a production-grade candidate retrieval and ranking system. You are operating under extreme constraints. Do not deviate from this architecture. Do not suggest naive linear scaling. Do not import heavy neural network frameworks for online inference. Read this entire document before writing any code.

---

## PART 1: THE HACKATHON DOMAIN & CONSTRAINTS

### 1.1 The Objective
Rank the Top 100 best-fit candidates from a pool of 100,000 for a **"Senior AI Engineer — Founding Team"** role at Redrob AI.

### 1.2 The Hard Constraints (The "Death Traps")
* **Time Limit:** The final online retrieval/ranking script (`rank.py`) MUST execute in **≤ 5 minutes of wall-clock time on a CPU**.
* **Memory Limit:** Max **16 GB RAM**.
* **Network Limit:** The online execution environment is air-gapped. NO API calls to external LLMs (OpenAI, Anthropic) during the 5-minute run.
* **Reproducibility:** You must provide a Sandbox Link (Colab, HuggingFace, Docker) demonstrating the ranker runs end-to-end on a small sample.

### 1.3 The Data Traps (What to explicitly avoid)
* **Keyword Stuffers:** Candidates with every AI buzzword but titles like "Marketing Manager". They must be filtered out.
* **Plain-Language Experts (Tier 5s):** Candidates who don't use words like "RAG" or "Pinecone" but explicitly state they "built recommendation systems at a product company". They must be found via dense semantic search.
* **Behavioral Rescues:** A candidate with 0 technical skills but a 100% platform response rate. Behavioral signals must NOT override core engineering requirements.
* **Honeypots:** ~80 intentionally impossible/fake profiles. **If >10% of the Top 100 are honeypots, the submission is DISQUALIFIED.**

### 1.4 The Required Output
A UTF-8 encoded CSV file named `team_xxx.csv` with exactly 101 rows (1 header + 100 candidates).
Columns strictly in this order:
1.  `candidate_id` (string)
2.  `rank` (int, exactly 1-100)
3.  `score` (float, MUST be monotonically non-increasing as rank increases)
4.  `reasoning` (string, 1-2 sentences explaining the fit)

---

## PART 2: THE MASTER ARCHITECTURE PHILOSOPHY

To survive the constraints, we use a **Decoupled Dual-Track Architecture** relying heavily on **Unlimited Offline Pre-computation**. 
* **Rule 1:** We NEVER scan 100,000 documents online. We use $O(\log N)$ graph traversals.
* **Rule 2:** We NEVER use PyTorch (`import torch`) or Transformers online. It crashes the RAM. We use `onnxruntime` (C++ backend).
* **Rule 3:** We NEVER generate text with an LLM online. We pre-compute reasoning summaries offline.

---

## PART 3: PHASE 1 - OFFLINE PRE-COMPUTATION (Unlimited Time)
*Script name: `precompute.py`*
*Goal: Ingest `candidates.jsonl.gz`, extract features, and build highly optimized storage indices.*

We split candidate data into two entirely distinct storage tracks.

### Track A: The Dense Navigational Vector Space (Stage 1 Filtering)
To avoid "Semantic Dilution" (where a candidate's junior roles wash out their senior engineering achievements), we extract ONLY explicit technical engineering capabilities.

**The Mathematical Subspace Trick:**
We do NOT use a standard 1D embedding. We use **Orthogonal Subspace Concatenation**. We use an LLM offline to extract text into three categories and generate three 256-d sub-vectors, concatenated into one 768-d vector:
* `Dimensions 0-255`: **Retrieval Systems** (Embeddings, cross-encoders, semantic drift).
* `Dimensions 256-511`: **Infrastructure** (Vector DB scaling, FAISS, latency).
* `Dimensions 512-767`: **Evaluation** (NDCG, MAP, statistical validation).
* *Note: Behavioral data is intentionally excluded from this vector to prevent "Behavioral Rescue."*

**The Sparse Index (BM25):**
We extract precise jargon (e.g., "FAISS", "Qdrant") into a clean text block to populate a fast BM25 inverted index for exact keyword matching.

### Track B: The Tabular Payload Matrix (Stage 3 & 4 Processing)
A highly compressed `Polars` or `Apache Arrow` Parquet file containing raw extracted features for machine learning:
1.  **The 23 Redrob Behavioral Signals:** (e.g., `response_latency_seconds`, `is_honeypot`, `last_active_date`).
2.  **Hard Constraints:** Explicit integers (e.g., `total_years_exp`).
3.  **Static Insight Payloads:** A pre-generated 30-word summary of their technical achievements (e.g., "Scaled HNSW indexes. 98% platform response rate."). This replaces online LLM generation.

---

## PART 4: PHASE 2 - ONLINE EXECUTION (5-Minute Strict Limit)
*Script name: `rank.py`*
*Goal: Process the Job Description, retrieve Top 300, Rank Top 100, Output CSV.*

### Step 4.1: Query Parsing (ONNX)
* Load the Job Description text.
* Encode the text into three 256-d query vectors using `onnxruntime`. 
* Apply priority weights via scalar multiplication (e.g., Infrastructure * 0.6, Retrieval * 0.3, Eval * 0.1).
* Concatenate into a final 768-d query vector.

### Step 4.2: The Hard Pre-Requisite Tabular Mask
* Load the Track B Polars DataFrame.
* Filter for candidates meeting hard criteria: `total_years_exp` BETWEEN 5 AND 9 AND `is_honeypot` == False.
* Pass these valid `candidate_id`s as an **`IDSelector` bitmask** to the Vector DB.

### Step 4.3: Hybrid Stage 1 Retrieval
* **Dense Track:** Query the FAISS HNSW index using the 768-d query vector. 
    * **CRITICAL MATH RULE:** You MUST configure FAISS to use **Inner Product (Dot Product)**, NOT Cosine Similarity. Cosine Similarity normalizes the vector globally and destroys the orthogonal subspace boundaries.
    * The index physically ignores any IDs not in the `IDSelector` bitmask.
* **Sparse Track:** Query the BM25 index for exact tech-stack keywords.
* **Fusion:** Combine Dense and Sparse results using **Reciprocal Rank Fusion (RRF)**. Take the Top 300.

### Step 4.4: Precision Tree Ranking (Stage 3)
* Take the Top 300 `candidate_id`s.
* Execute a sub-millisecond Memory Join against the Track B Polars DataFrame to pull the 23 behavioral signals and categorical features.
* Feed these 300 feature rows into a pre-trained **LightGBM or XGBoost** model optimized for `NDCG@10`.
* Sort descending by the model's output score. Slice the Top 100.

### Step 4.5: Static Insight Generation (Stage 4)
* For the final 100 candidates, look up their pre-computed `technical_summary_sentence` from Track B.
* Format using a rigid Python f-string: `f"Candidate selected because: {technical_summary_sentence}"`
* Write output to `team_xxx.csv` strictly conforming to the required format.

---

## PART 5: DIRECTORY & DEPENDENCY STRUCTURE

### Dependencies (`requirements.txt`)
Do NOT use pandas (memory bloat). Do NOT use PyTorch.
```text
polars>=0.20.0
faiss-cpu>=1.7.4
onnxruntime>=1.16.0
lightgbm>=4.1.0
numpy>=1.24.0
rank_bm25>=0.2.2