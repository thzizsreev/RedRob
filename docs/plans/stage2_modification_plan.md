# STAGE 2 — MODIFICATION PLAN
## For AI Coding Agent

---

## CONTEXT: WHAT CURRENTLY EXISTS

Stage 2 is a working, tested, single-pass tabular gate. Before touching anything, the coding agent must understand the exact current state so nothing already working is broken.

### What the current code does

The main loop in `gate.py` processes each candidate in this strict order:
1. Honeypot check (timeline rules R1–R5 in `honeypot_rules.py` + skill rules H3–H5 in `checks/skills.py`) → hard remove on any hit
2. Experience band check (`checks/experience.py`) → hard remove if outside [4, 10] years
3. Title / keyword stuffer check (`checks/title.py`) → hard remove if non-engineering title
4. Availability flags (`checks/availability.py`) → soft flags only, never removes

### What the current output contains

`stage2_gated.parquet` currently has exactly these columns per survivor:
`candidate_id`, `cluster_id`, `cluster_rank`, `dist_to_centroid`, `total_years_exp`, `exp_band`, `in_sweet_spot`, `title_family`, `skill_kw_density`, `title_ambiguous`, `stale_profile`, `low_responder`, `not_open`, `honeypot_anomaly_score`

### What the current code explicitly does NOT do (from section 14 of the report)

- Pure research-only or consulting-only career patterns → deferred
- LangChain-only / framework-enthusiast profiles → deferred
- No production code in 18 months → deferred
- Location and notice period → deferred
- Optional nice-to-haves as positive signals → deferred
- Title-chasing (short tenure hops) → deferred
- Product vs services company fraction → deferred

**The purpose of this modification plan is to implement exactly these deferred items** — either as new hard removes (where the JD treats them as absolute disqualifiers) or as new computed feature columns (where the JD treats them as graded signals for downstream scoring in Stage 5).

---

## MODIFICATION PHILOSOPHY — READ BEFORE CODING

**Do not change any existing check.** Checks A (experience), B (title), C (honeypot), D (availability) are working correctly. The only changes to existing files are:
- `gate.py` — add new check calls into the existing loop
- `config.py` — add new config fields to the dataclass
- `config.yaml` — add new keys under `stage2:`
- `gate.py` row dict — add new output columns

**New checks go in new files.** Each new check gets its own file in `checks/`. This mirrors the existing pattern (`checks/experience.py`, `checks/title.py`, etc.) and keeps the logic isolated and testable independently.

**New checks run AFTER existing checks, BEFORE availability.** The evaluation order becomes:
1. Honeypot (existing — unchanged)
2. Experience band (existing — unchanged)
3. Title / keyword stuffer (existing — unchanged)
4. **Check E — Consulting-only career (NEW — hard remove)**
5. **Check F — Research-only career (NEW — hard remove with soft-flag fallback)**
6. **Check G — No production code in 18 months (NEW — soft flag only)**
7. **Check H — LangChain-only shallow AI (NEW — hard remove with escape hatch)**
8. Availability (existing — unchanged, always runs last, never removes)

**Feature columns are computed alongside the checks, not separately.** Every new check also computes feature values that flow to Stage 5. These are not separate passes — they are computed once inside the check function and returned alongside the remove/flag decision.

**When in doubt, flag instead of remove.** The JD uses language like "we will probably not move forward" for most of these — not "we will never." Hard removes are reserved for the cases where the JD uses absolute language or where the signal is unambiguous. Ambiguous signals become soft flags with float or boolean values that Stage 5 weights appropriately.

---

## PART 1 — NEW HARD REMOVE CHECKS

---

### CHECK E — Consulting-Only Career

**File to create:** `checks/consulting.py`

**JD basis:** *"People who have only worked at consulting firms in their entire career — we've had bad fit experiences in both directions. If you're currently at one of these companies but have prior product-company experience, that's fine."*

**Key distinction the JD makes:** it is not about the current employer. It is about whether the entire career history contains only consulting/services companies with no product-company experience at any point. A candidate currently at Infosys who previously worked at a startup is explicitly acceptable.

**What a "consulting/services company" means:** the JD names specific firms: TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini. Beyond named firms, the category includes large IT services companies, staffing firms, and outsourcing companies. The config should hold the named-firm list and allow extension.

**What a "product company" means:** any company that builds and sells its own software product or platform to external customers. Startups, SaaS companies, tech companies, AI companies, marketplaces, consumer apps. Distinguished from companies whose primary business is staffing other companies' engineering work or providing outsourced IT services.

**Logic:**

Iterate over the candidate's full work history (all roles across all employers). For each unique employer, classify it as one of three types:
- `consulting` — matches the named consulting firm list (case-insensitive substring match) OR company name contains signals like "consulting", "solutions", "services", "technologies" combined with patterns that suggest IT outsourcing (no single heuristic is perfect — use the named list as the primary signal and the pattern match as a secondary flag)
- `product` — any employer not classified as consulting that has indicators of being a product company: startup, tech company, known product companies, companies that don't match consulting patterns
- `unknown` — cannot be classified with confidence

**Decision rules:**

- If the candidate has at least one `product` employer anywhere in their history → **KEEP**. Reason: the JD explicitly says prior product experience is sufficient.
- If every employer in history is classified as `consulting` or `unknown`, AND at least one is confidently `consulting` → **HARD REMOVE**, reason code `consulting_only_career`
- If every employer is `unknown` (can't confidently classify any as consulting or product) → **KEEP**, soft flag `career_type_unknown = True`. Do not remove on uncertainty.
- If fewer than 2 employers total → **KEEP**, soft flag `single_employer = True`. Too little history to classify confidently.

**Feature columns to compute and return (always, even if not removing):**

- `product_company_count` (int) — number of distinct employers classified as product
- `consulting_company_count` (int) — number of distinct employers classified as consulting
- `product_company_fraction` (float 0–1) — product_company_count / total employer count. This is the primary Stage 5 feature for career composition.
- `career_type` (string enum: `product_heavy`, `mixed`, `consulting_heavy`, `unknown`) — derived from the fraction: product_heavy if >0.6, mixed if 0.2–0.6, consulting_heavy if <0.2, unknown if < 2 classifiable employers
- `consulting_only_flag` (bool) — True only when the hard remove condition is met. Even if not removed (e.g. soft-flagged), this is False.

**Config keys to add under `stage2:` in `config.yaml`:**

```yaml
stage2:
  consulting:
    named_firms:
      - tcs
      - tata consultancy
      - infosys
      - wipro
      - accenture
      - cognizant
      - capgemini
      - hcl
      - tech mahindra
      - mphasis
      - hexaware
      - mindtree
      - l&t infotech
      - ltimindtree
      - niit technologies
      - patni
      - mastech
      - kforce
      - igate
    consulting_signal_words:
      - consulting
      - consultancy
      - outsourcing
      - staffing
      - it services
    min_employers_to_classify: 2
    product_fraction_threshold: 0.0  # any product employer = keep
```

**Important implementation note:** company name matching must be case-insensitive and must handle partial matches (e.g. "TCS Digital" should match "tcs"). Use substring matching on normalized (lowercase, stripped) company names against the named_firms list. Do not use exact match.

---

### CHECK F — Research-Only / No Production Career

**File to create:** `checks/research.py`

**JD basis:** *"If you've spent your career in pure research environments (academic labs, research-only roles) without any production deployment — we will not move forward. We are explicit about this. We've tried it twice and it didn't work."*

**What "research-only" means structurally:** a career where every role has been in an academic institution or a pure research lab with no evidence of production software deployment to real users. The signal is in role titles and employer types — not in what skills the candidate lists.

**Research role signals (in role titles):**
- PhD, Postdoc, Post-doctoral, Doctoral, Research Intern, Research Fellow, Research Assistant, Research Scientist at academic institution, Research Associate, Principal Investigator, Professor, Lecturer, Teaching Assistant, Graduate Researcher
- Note: "Research Scientist" at a product company (e.g. "Research Scientist at Google") is NOT a research-only role — it is a production-adjacent role. The institution matters as much as the title.

**Academic/pure-research employer signals:**
- University, College, Institute of Technology, IIT, IISc, MIT, Stanford, CMU, lab names ending in "Research Lab" or "Research Institute", national labs, government research bodies

**Production role signals (escape hatch):**
- Any role titled: Engineer, Developer, SDE, MLE, Applied Scientist, Architect, Tech Lead, Staff, Principal (at a non-academic employer)
- Any employer that is a product company (as classified in Check E)

**Logic:**

Classify each role as `research_role` or `production_role` based on title + employer type combination.

- If the candidate has at least one `production_role` → **KEEP**, no flag needed
- If every role is `research_role` AND total employer count is ≥ 2 → **HARD REMOVE**, reason code `research_only_career`
- If every role is `research_role` AND total employer count is 1 (e.g. single PhD program) → **HARD REMOVE**, reason code `research_only_career` — single academic institution is unambiguous
- If the classification is mixed but heavily research (> 80% research roles) → **KEEP**, soft flag `research_heavy = True`
- If classification is uncertain (< 2 classifiable roles) → **KEEP**, no flag

**Feature columns to compute and return:**

- `research_role_count` (int)
- `production_role_count` (int)
- `research_fraction` (float 0–1) — research roles / total roles
- `research_heavy` (bool) — True if research_fraction > 0.8 but not fully research-only (those are hard removed)
- `has_any_production_role` (bool) — the key escape hatch signal; True if at least one production role exists

**Config keys to add:**

```yaml
stage2:
  research:
    research_title_signals:
      - phd
      - postdoc
      - post-doctoral
      - doctoral
      - research intern
      - research fellow
      - research assistant
      - research associate
      - principal investigator
      - professor
      - lecturer
      - teaching assistant
      - graduate researcher
    academic_employer_signals:
      - university
      - college
      - institute of technology
      - iit
      - iisc
      - research lab
      - research institute
      - national lab
    production_title_signals:
      - engineer
      - developer
      - sde
      - mle
      - applied scientist
      - architect
      - tech lead
      - staff
    research_heavy_threshold: 0.8
    min_roles_to_classify: 1
```

---

### CHECK H — LangChain-Only / Shallow Recent AI

**File to create:** `checks/shallow_ai.py`

**JD basis:** *"If your AI experience consists primarily of recent (under 12 months) projects using LangChain to call OpenAI — we will probably not move forward, unless you can demonstrate substantial pre-LLM-era ML production experience."*

**The JD's conditional is critical:** this is NOT a blanket ban on LangChain. It is a ban on candidates whose ENTIRE AI experience started in the last 12 months AND consists only of calling LLM APIs through frameworks — WITHOUT any prior ML production experience. The escape hatch (pre-LLM ML production experience) must be checked before removing.

**What "LangChain-only / shallow AI" looks like structurally:**
- ML/AI work started very recently (< 12 months before reference date as measured by the earliest ML-related role or project date)
- Skills list is dominated by LLM orchestration frameworks: LangChain, LlamaIndex, AutoGPT, CrewAI, OpenAI API, Anthropic API, Azure OpenAI — without presence of foundational ML skills
- No skills or roles indicating pre-LLM experience: scikit-learn, PyTorch, TensorFlow, traditional NLP, classical ML, statistical modeling, anything that pre-dates the LLM era

**What "pre-LLM-era ML production experience" looks like:**
- Any ML role (not just research) that started before 2022 (approximate start of the LLM-mass-adoption era)
- Skills like: scikit-learn, XGBoost, traditional NLP (spaCy, NLTK, word2vec), recommendation systems, classical ranking, PyTorch/TensorFlow for non-LLM tasks, feature engineering, data pipelines

**Logic:**

Step 1: Determine if the candidate is a "recent AI only" profile:
- Find the earliest date of any ML/AI-related role or the earliest date of any ML skill acquisition (use work history start dates of AI/ML-titled roles)
- If earliest ML/AI date is < 12 months before reference date → candidate is `recent_ai_only = True`
- If candidate has no ML/AI work history at all → skip this check entirely (other checks handle non-engineers)

Step 2: If `recent_ai_only = True`, check the escape hatch:
- Does the candidate have any ML role starting before 2022? If yes → `pre_llm_production_ml = True` → **KEEP**
- Does the candidate's skills list contain foundational pre-LLM ML skills (scikit-learn, XGBoost, PyTorch for non-LLM, traditional NLP)? If yes → `pre_llm_production_ml = True` → **KEEP**
- If neither condition is met → **HARD REMOVE**, reason code `shallow_recent_ai_only`

Step 3: If `recent_ai_only = False` (has older ML experience) → **KEEP**, compute features anyway

**Feature columns to compute and return:**

- `pre_llm_production_ml` (bool) — True if any ML production experience predates 2022
- `recent_ai_only` (bool) — True if all ML experience is < 12 months old
- `llm_framework_only` (bool) — True if skills are dominated by LLM orchestration frameworks with no foundational ML skills
- `ml_experience_start_year` (int, nullable) — year of earliest ML/AI role or skill

**Config keys to add:**

```yaml
stage2:
  shallow_ai:
    llm_era_start_year: 2022
    recent_ai_window_months: 12
    llm_framework_signals:
      - langchain
      - llamaindex
      - llama-index
      - autogpt
      - crewai
      - openai api
      - anthropic api
      - azure openai
      - chatgpt api
    pre_llm_skill_signals:
      - scikit-learn
      - sklearn
      - xgboost
      - lightgbm
      - word2vec
      - fasttext
      - spacy
      - nltk
      - traditional nlp
      - recommendation system
      - collaborative filtering
      - classical ml
      - feature engineering
    min_llm_framework_skills: 2  # need at least 2 LLM framework skills to flag as framework-only
```

---

## PART 2 — NEW SOFT FLAG CHECKS

---

### CHECK G — No Production Code in 18 Months

**File to create:** `checks/coding_recency.py`

**JD basis:** *"If you are a senior engineer who hasn't written production code in the last 18 months because you've moved into 'architecture' or 'tech lead' roles — we will probably not move forward. This role writes code."*

**Why this is a soft flag, not a hard remove:** "probably not move forward" is softer language than the other disqualifiers. Also, distinguishing "writes code as an architect" from "pure architecture with no coding" is very hard to do deterministically from a profile. The signal here is imperfect enough that a hard remove risks too many false positives. Flag it, let Stage 5 apply a penalty.

**What to look for:**

The check inspects the most recent 18 months of the candidate's work history (roles where the date range overlaps the last 18 months from `current_date`).

Role titles that signal pure architecture / management with no coding:
- Chief Architect, Enterprise Architect, Solution Architect (without "Software" or "ML" prefix)
- Engineering Manager, VP Engineering, Director of Engineering, Head of Engineering
- CTO, Technical Director
- Principal Architect (without engineering/software qualifier)

Role titles that explicitly signal continued hands-on coding:
- Software Engineer, ML Engineer, Senior Engineer, Staff Engineer, Principal Engineer (these retain IC coding)
- Applied Scientist, Research Engineer (retained coding in research context)

**Logic:**

Look at all roles active in the last 18 months (their date range overlaps the window). Classify each as `management_only`, `hands_on`, or `ambiguous`.

- If ALL roles in the last 18 months are `management_only` → `stale_coding = True`
- If ANY role in the last 18 months is `hands_on` → `stale_coding = False`
- If only `ambiguous` roles in the window → `stale_coding = False` (benefit of the doubt)
- If no roles overlap the last 18 months (candidate is currently between roles) → `stale_coding = False`, set `currently_between_roles = True` instead

This is a **soft flag only**. No candidate is hard-removed by this check.

**Feature columns to compute and return:**

- `stale_coding` (bool) — True if the last 18 months contains only management/architecture roles
- `currently_between_roles` (bool) — True if no role overlaps the last 18 months
- `months_since_last_ic_role` (float, nullable) — months since the most recent hands-on engineering role ended. Null if currently in IC role.

**Config keys to add:**

```yaml
stage2:
  coding_recency:
    stale_coding_window_months: 18
    management_title_signals:
      - chief architect
      - enterprise architect
      - solution architect
      - engineering manager
      - vp engineering
      - director of engineering
      - head of engineering
      - cto
      - technical director
      - principal architect
    hands_on_title_signals:
      - software engineer
      - ml engineer
      - machine learning engineer
      - applied scientist
      - research engineer
      - sde
      - data engineer
      - backend engineer
      - staff engineer
      - principal engineer
```

---

## PART 3 — NEW FEATURE COLUMNS (NO REMOVE DECISION)

These are signals the JD talks about that require computation from the work history or behavioral fields, but which are **never** a reason to remove a candidate at Stage 2. They are computed alongside the checks above and returned as additional columns in the survivor row. Stage 5 uses them as scoring inputs.

---

### FEATURE F1 — Title Chasing (Tenure Per Employer)

**JD basis:** *"If your career trajectory shows you optimizing for Senior → Staff → Principal titles by switching companies every 1.5 years, we're not a fit."*

**Computed in:** `checks/consulting.py` (it already iterates work history for employer classification — compute tenure while there) OR a new `checks/career_shape.py` if the consulting check is not the right home.

**How to compute:**

From the work history, for each role compute its duration in years. Then:
- `avg_tenure_per_employer` (float) — average duration in years across all roles. Short average = potential title-chaser.
- `short_hop_count` (int) — number of roles lasting less than `short_hop_threshold` years (default 1.5 years). The JD names 1.5 years explicitly.
- `title_progression_jumps` (int) — number of times a candidate moved from a lower seniority title to a higher seniority title between consecutive employers (e.g. Senior → Staff between two different companies). This is the specific pattern the JD describes.

**Config keys:**

```yaml
stage2:
  career_shape:
    short_hop_threshold_years: 1.5
```

---

### FEATURE F2 — Location Tier

**JD basis:** *"Pune/Noida preferred. Hyderabad, Pune, Mumbai, Delhi NCR welcome. Outside India: case-by-case, no visa sponsorship."*

**Computed in:** new `checks/logistics.py`

**How to compute:**

Read the candidate's current location from the profile. Normalize to lowercase. Classify into one of four tiers:

- `preferred` — Noida, Pune, Delhi NCR, Delhi, Gurgaon, Gurugram, Faridabad, Noida Extension
- `acceptable` — Hyderabad, Mumbai, Bangalore, Bengaluru, Chennai
- `outside_india` — any location that is clearly not an Indian city
- `unknown` — cannot determine location from available data

Also read `willing_to_relocate` from the 23 behavioral signals. If `willing_to_relocate == True`, upgrade `acceptable` to `preferred` and `unknown` to `acceptable`. Do NOT upgrade `outside_india` — the JD says no visa sponsorship regardless.

**Feature column to output:**

- `location_tier` (string enum: `preferred`, `acceptable`, `outside_india`, `unknown`)

**Config keys:**

```yaml
stage2:
  logistics:
    preferred_locations:
      - noida
      - pune
      - delhi
      - delhi ncr
      - ncr
      - gurgaon
      - gurugram
      - faridabad
      - new delhi
    acceptable_locations:
      - hyderabad
      - mumbai
      - bangalore
      - bengaluru
      - chennai
      - kolkata
    india_signals:
      - india
```

---

### FEATURE F3 — External Validation Score

**JD basis:** *"People whose work has been entirely on closed-source proprietary systems for 5+ years without external validation (papers, talks, open-source) — we will probably not move forward."*

**Computed in:** `checks/availability.py` can be extended, OR new `checks/validation.py`

**How to compute:**

This is a composite of signals already in the 23 behavioral fields plus any profile fields:

- `github_activity_score` from the 23 signals: already available. Values: -1 = no GitHub linked, 0–100 = activity score.
- Profile fields: check if any of these exist and are non-empty: `publications`, `patents`, `open_source_contributions`, `blog_url`, `talks`, `portfolio_url`
- Skills: presence of any open-source library name in skills that implies contribution (e.g. "contributed to HuggingFace", "maintainer of X")

**Composite:**

```
external_validation_score = 0 (base)
+ github_activity_score / 100 * 0.6   (if github linked; -1 counts as 0)
+ 0.2 (if any publication/patent present)
+ 0.2 (if blog/talks/portfolio present)
```

Result is a float 0–1. Higher = more externally visible.

**Feature column to output:**

- `external_validation_score` (float 0–1)
- `has_github` (bool) — `github_activity_score != -1`

---

## PART 4 — CHANGES TO EXISTING FILES

---

### `checks/availability.py` — ADD notice period passthrough

The 23 behavioral signals include `notice_period_days`. Currently `availability.py` only reads `last_active_date`, `recruiter_response_rate`, and `open_to_work_flag`. 

**Add to `evaluate_availability` return value:**

- `notice_period_days` (int, nullable) — read directly from `redrob_signals.notice_period_days`. Do not threshold it here — pass the raw value to Stage 5.

**JD basis:** *"Sub-30-day notice ideal. 30+ = higher bar. Still in scope but bar increases."*

No removal logic. Just pass the value through as a column.

---

### `gate.py` — ADD new check calls and new row columns

The main loop must be modified to:

1. Import the five new check modules: `consulting`, `research`, `coding_recency`, `shallow_ai`, `logistics` (or `career_shape` if that is the home for F1)

2. Call them in the correct order after the existing title check, before availability:

```
[existing: honeypot → experience → title]
→ Check E: consulting.evaluate_consulting(record, config)
   → if result.remove: log and continue
→ Check F: research.evaluate_research(record, config)
   → if result.remove: log and continue
→ Check G: coding_recency.evaluate_coding_recency(record, config)
   → never removes; updates flags
→ Check H: shallow_ai.evaluate_shallow_ai(record, config)
   → if result.remove: log and continue
[existing: availability]
```

3. Add all new columns to the survivor row dict:

```python
row = {
    # --- existing columns (unchanged) ---
    "candidate_id": ...,
    "cluster_id": ...,
    "cluster_rank": ...,
    "dist_to_centroid": ...,
    "total_years_exp": ...,
    "exp_band": ...,
    "in_sweet_spot": ...,
    "title_family": ...,
    "skill_kw_density": ...,
    "title_ambiguous": ...,
    "stale_profile": ...,
    "low_responder": ...,
    "not_open": ...,
    "honeypot_anomaly_score": None,

    # --- NEW columns from Check E ---
    "product_company_count": consulting.product_company_count,
    "consulting_company_count": consulting.consulting_company_count,
    "product_company_fraction": consulting.product_company_fraction,
    "career_type": consulting.career_type,

    # --- NEW columns from Check F ---
    "research_fraction": research.research_fraction,
    "research_heavy": research.research_heavy,
    "has_any_production_role": research.has_any_production_role,

    # --- NEW columns from Check G ---
    "stale_coding": coding_recency.stale_coding,
    "currently_between_roles": coding_recency.currently_between_roles,
    "months_since_last_ic_role": coding_recency.months_since_last_ic_role,

    # --- NEW columns from Check H ---
    "pre_llm_production_ml": shallow_ai.pre_llm_production_ml,
    "recent_ai_only": shallow_ai.recent_ai_only,
    "llm_framework_only": shallow_ai.llm_framework_only,
    "ml_experience_start_year": shallow_ai.ml_experience_start_year,

    # --- NEW columns from Feature F1 (career shape) ---
    "avg_tenure_per_employer": career_shape.avg_tenure_per_employer,
    "short_hop_count": career_shape.short_hop_count,
    "title_progression_jumps": career_shape.title_progression_jumps,

    # --- NEW columns from Feature F2 (logistics) ---
    "location_tier": logistics.location_tier,

    # --- NEW columns from Feature F3 (validation) ---
    "external_validation_score": validation.external_validation_score,
    "has_github": validation.has_github,

    # --- NEW column from availability extension ---
    "notice_period_days": avail.notice_period_days,
}
```

---

### `config.py` — ADD new fields to Stage2Config dataclass

The `Stage2Config` dataclass must be extended to hold all new config blocks. Each new check's config should be its own nested dataclass or a simple dict read from YAML. The exact implementation pattern should follow whatever `config.py` already uses for existing fields (dataclass fields vs raw dict — match the existing style).

New top-level config groups to add:
- `consulting` — for Check E
- `research` — for Check F
- `coding_recency` — for Check G
- `shallow_ai` — for Check H
- `career_shape` — for Feature F1
- `logistics` — for Feature F2

---

### `config.yaml` — ADD all new keys under `stage2:`

All new keys described in each check section above must be added to `config.yaml` under `stage2:`. The full additions are specified in each check's "Config keys to add" subsection. Do not change any existing key.

---

## PART 5 — NEW OUTPUT SCHEMA

The complete `stage2_gated.parquet` output schema after modifications. All existing columns are unchanged. New columns are marked **NEW**.

| Column | Type | Source | Notes |
|--------|------|--------|-------|
| `candidate_id` | string | existing | |
| `cluster_id` | int | existing | |
| `cluster_rank` | int, nullable | existing | |
| `dist_to_centroid` | float | existing | |
| `total_years_exp` | float | existing | |
| `exp_band` | string | existing | `in_band` / `near_band` |
| `in_sweet_spot` | bool | existing | |
| `title_family` | string | existing | |
| `skill_kw_density` | float | existing | |
| `title_ambiguous` | bool | existing | |
| `stale_profile` | bool | existing | |
| `low_responder` | bool | existing | |
| `not_open` | bool | existing | |
| `honeypot_anomaly_score` | null | existing | reserved |
| `product_company_count` | int | **NEW** Check E | |
| `consulting_company_count` | int | **NEW** Check E | |
| `product_company_fraction` | float | **NEW** Check E | primary career-composition feature for Stage 5 |
| `career_type` | string | **NEW** Check E | `product_heavy` / `mixed` / `consulting_heavy` / `unknown` |
| `research_fraction` | float | **NEW** Check F | |
| `research_heavy` | bool | **NEW** Check F | |
| `has_any_production_role` | bool | **NEW** Check F | |
| `stale_coding` | bool | **NEW** Check G | soft flag only |
| `currently_between_roles` | bool | **NEW** Check G | |
| `months_since_last_ic_role` | float, nullable | **NEW** Check G | |
| `pre_llm_production_ml` | bool | **NEW** Check H | escape hatch signal |
| `recent_ai_only` | bool | **NEW** Check H | |
| `llm_framework_only` | bool | **NEW** Check H | |
| `ml_experience_start_year` | int, nullable | **NEW** Check H | |
| `avg_tenure_per_employer` | float | **NEW** F1 | anti title-chasing signal |
| `short_hop_count` | int | **NEW** F1 | |
| `title_progression_jumps` | int | **NEW** F1 | |
| `location_tier` | string | **NEW** F2 | `preferred` / `acceptable` / `outside_india` / `unknown` |
| `external_validation_score` | float | **NEW** F3 | 0–1 |
| `has_github` | bool | **NEW** F3 | |
| `notice_period_days` | int, nullable | **NEW** avail ext | |

---

## PART 6 — NEW REMOVAL REASON CODES

The `stage2_removed_log.csv` currently logs reason codes. The following new reason codes must be added:

| Reason code | Check | Condition |
|-------------|-------|-----------|
| `consulting_only_career` | Check E | Every employer classified as consulting, at least one confidently consulting |
| `research_only_career` | Check F | Every role classified as research/academic, ≥ 1 employer |
| `shallow_recent_ai_only` | Check H | recent_ai_only AND no pre-LLM ML experience AND no foundational ML skills |

---

## PART 7 — EVALUATION ORDER AFTER MODIFICATIONS

```
For each candidate in id_set:

  1. HONEYPOT (existing — unchanged)
     honeypot_rules.py (R1–R5) + checks/skills.py (H3–H5)
     → hard remove on any hit
     → reason logged to honeypot_log + removed_log

  2. EXPERIENCE BAND (existing — unchanged)
     checks/experience.py
     → hard remove if outside [4, 10]
     → reason: exp_out_of_band

  3. TITLE / KEYWORD STUFFER (existing — unchanged)
     checks/title.py
     → hard remove if non_eng title
     → reason: keyword_stuffer | non_eng_title

  4. CHECK E — CONSULTING-ONLY (NEW)
     checks/consulting.py
     → hard remove if every employer is consulting
     → reason: consulting_only_career
     → always computes: product_company_fraction, career_type, counts

  5. CHECK F — RESEARCH-ONLY (NEW)
     checks/research.py
     → hard remove if every role is academic/research
     → reason: research_only_career
     → always computes: research_fraction, has_any_production_role

  6. CHECK G — CODING RECENCY (NEW — SOFT FLAG ONLY)
     checks/coding_recency.py
     → NEVER removes
     → always computes: stale_coding, months_since_last_ic_role

  7. CHECK H — SHALLOW AI (NEW)
     checks/shallow_ai.py
     → hard remove if recent_ai_only AND no escape hatch
     → reason: shallow_recent_ai_only
     → always computes: pre_llm_production_ml, ml_experience_start_year

  8. FEATURE COMPUTATION (NEW — NO REMOVAL)
     career shape (F1): avg_tenure, short_hop_count, title_progression_jumps
     logistics (F2): location_tier
     validation (F3): external_validation_score, has_github

  9. AVAILABILITY (existing — unchanged, extended with notice_period_days)
     checks/availability.py
     → NEVER removes
     → computes: stale_profile, low_responder, not_open, notice_period_days

  → append full row to survivors
```

---

## PART 8 — WHAT NOT TO CHANGE

The following must remain exactly as implemented. Do not refactor, restructure, or move these:

- `honeypot_rules.py` — no changes
- `checks/experience.py` — no changes
- `checks/title.py` — no changes
- `checks/skills.py` — no changes
- The existing `evaluate_availability` return values — only extend with `notice_period_days`
- The `io.py` loading/streaming/writing logic — no changes
- The `run.py` entry point — no changes to paths or the `run()` call signature
- The existing `Stage2Config` field names — only add new fields, never rename
- The existing output columns in `stage2_gated.parquet` — only append new columns, never drop or rename existing ones

---

## PART 9 — VALIDATION AFTER MODIFICATIONS

After running the modified Stage 2, the operator should verify:

1. **Row count:** still in [2,000, 5,000] range. If the new checks reduce output below 2,000, thresholds in E/F/H are too aggressive. Loosen `consulting.min_employers_to_classify` or increase ambiguity handling.

2. **New removal breakdown:** `stage2_removed_log.csv` should show new reason codes. Expect:
   - `consulting_only_career`: a small number (likely < 200 in a 6K sample — not every candidate is consulting-only)
   - `research_only_career`: small number (likely < 100)
   - `shallow_recent_ai_only`: small number (< 50)
   - If any new reason code accounts for > 500 removals, that check is misfiring — audit it

3. **New column distributions:** spot-check the new columns in `stage2_gated.parquet`:
   - `product_company_fraction`: should range across 0–1, not cluster at extremes
   - `location_tier`: check counts of each tier — majority should be `preferred` or `acceptable` for a dataset targeting Indian engineers
   - `avg_tenure_per_employer`: should be normally distributed around 2–4 years for realistic engineers
   - `external_validation_score`: many zeros expected (most candidates don't have public GitHub/papers)

4. **No regressions in existing columns:** verify that `exp_band`, `title_family`, `skill_kw_density`, `stale_profile`, `low_responder`, `not_open` distributions are unchanged from the pre-modification run. These should be identical since the existing checks were not modified.

5. **Honeypot count unchanged:** the honeypot removal count should be identical to the pre-modification run (1,783 `honeypot_skill_years_impossible` + 374 `exp_out_of_band` as per the last observed run). New checks run after honeypots, so they cannot affect honeypot counts.
