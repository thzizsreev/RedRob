# HONEYPOT RESEARCH PLAN — Pure-LLM Honeypot Study

**Status: Research artifact, not production pipeline.**
This is explicitly *not* part of `precompute.py` or `rank.py`. Its output is understanding — what honeypots in this dataset actually look like — not a filter to be wired into the submission. Do not import this script's output directly into `rank.py`'s honeypot exclusion logic without independently re-validating it; that would make your rule-based filter a downstream copy of this study's blind spots rather than an independent check.

---

## 1. Objective

Use an LLM, with no labeled data and no rule-based pre-filtering, to read individual candidate profiles in full and judge whether each is a honeypot — then study the *reasoning*, not just the resulting list, to understand what honeypot patterns actually exist in this dataset.

This is deliberately the opposite design from the rule-based honeypot filter discussed for `precompute.py`. That filter encodes a few hypothesized contradiction types (tenure vs. company founding, proficiency vs. years-used, timeline arithmetic) and only catches what it was told to look for. This study exists to find out whether those hypotheses are right, incomplete, or missing entire honeypot categories — by letting the LLM look at raw profiles with comparatively open instructions and seeing what it actually flags and why.

**Explicit non-goals:**
- Not optimized for runtime, cost, or reproducibility under hackathon compute constraints.
- Not intended to be called from `rank.py` (would violate the no-hosted-LLM-online rule even if it were fast enough).
- Not a substitute for the offline rule-based filter — a complementary, independent check on it.

---

## 1.1 Findings from initial sample records (update this section as more are reviewed)

Two real records (CAND_0038208, CAND_0000002) were reviewed before building the prompt. Neither was a "honeypot" in the impossible-profile sense — no tenure-vs-founding-date or proficiency-vs-zero-years contradictions appeared in either. Instead, both surfaced a different, previously unhypothesized pattern:

**Title/description decoupling.** In `career_history`, each role entry has its own `title` and `description`. In both records, every entry's `description` text describes work that does not match that entry's `title` — e.g., a role titled "DevOps Engineer" with a description entirely about Android/Kotlin mobile development; a role titled "Operations Manager" with a description about mechanical engineering hardware design. The description text is internally coherent on its own — it reads like a real role description — it's just attached to the wrong title. This appeared consistently across both reviewed records, which raises the possibility it's systemic to how this dataset was generated, not a deliberate honeypot trap. This needs to be checked across a larger sample before concluding anything — if it's universal across most candidates, it's a data-generation quirk to design around (e.g., don't trust `title` in isolation, always cross-check against `description`), not a honeypot signal. If it's concentrated in a subset, it may be diagnostic.

**Skills-array red herrings, independent of the keyword-stuffer-in-headline trap.** CAND_0000002's prose (`summary`, all `career_history` descriptions) honestly describes a marketing/ops/hardware/design career with no ML substance — but the `skills` array includes "Kafka" and "Feature Engineering" at intermediate proficiency, unconnected to anything in the career narrative. This is the JD's keyword-stuffer trap, but implemented via the structured `skills` array rather than the `headline`/`summary` text — meaning a prompt or rule that only inspects prose fields for keyword padding would miss it. The `skills` array needs to be checked against the career narrative specifically, not treated as independently trustworthy.

**Neither record was a honeypot at all.** Both are plausible (if messy or off-target) synthetic profiles. This is a useful prior: expect the large majority of any sample to be non-honeypot non-fits or non-honeypot fits, with true honeypots a small minority even within an AI/ML-adjacent stratum. Don't let early honeypot-detection runs over-fire on every internally-imperfect profile — internal messiness (title/description mismatch, scattered skills) appears common across the dataset broadly, while genuine impossibility (the JD's named examples) is likely rarer and more specific.

**Schema notes for prompt-building**, confirmed from real records (supersedes guesses in earlier discussion):
- `career_history` is a list of role objects, each with its own `title`, `description`, `start_date`, `end_date`, `duration_months`, `industry`, `company_size` — titles and descriptions must be compared per-entry, not assumed consistent.
- `skills` is a structured array of objects: `name`, `proficiency` (beginner/intermediate/expert presumably), `endorsements`, `duration_months` — no separate "years used" field; `duration_months` is the closest analog and may be a better field name for the proficiency-mismatch check than originally assumed.
- `skill_assessment_scores` (inside `redrob_signals`) was an empty `{}` in both sampled records — this field may be sparse/often-empty in practice; do not build a rule that depends on it being populated without checking prevalence first.
- `education` includes a `tier` field (e.g., `tier_3`, `tier_4`) alongside institution/degree/grade — not previously accounted for.
- `offer_acceptance_rate: -1` and `github_activity_score: -1` both appeared as sentinel "no data" values in both records, consistent with the signals doc's stated -1 convention — confirms these need explicit -1 handling in any scoring logic, not just clamping.

---

## 2. Why a stratified sample, not the first N or a random N

A sample built carelessly here just confirms whatever bias is already in the sampling. Two failure modes to avoid:

- **Sampling only "obviously AI-relevant" candidates** (e.g., by keyword pre-filter) biases toward honeypots that are *designed* to look good on paper — which is most of them, per the JD's own description — but tells you nothing about whether legitimate strong candidates get false-flagged.
- **Pure random sampling across all 100K** will be mostly irrelevant candidates (marketing managers, unrelated fields) and waste LLM calls on profiles that were never going to be confused for honeypots anyway.

**Sampling strategy:**

1. **Stratum A — AI/ML-adjacent pool.** Candidates with at least some technical/AI signal (broad net, not a tight filter) — this is where honeypots actually live, since they're built to look plausible within this population.
2. **Stratum B — known trap categories from the JD**, sampled deliberately so the study can observe how the LLM handles them specifically:
   - Keyword-stuffers with non-AI titles (e.g., "Marketing Manager" with a skills list full of AI jargon)
   - Tier 5 plain-language candidates (no "RAG"/"Pinecone" vocabulary, but production system-building language)
   - Behavioral-rescue candidates (strong response rate / activity signals, weak technical depth)
   - Consulting-only career candidates (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini)
   - CV/speech/robotics-only candidates without NLP/IR exposure
3. **Stratum C — small random sample from the full 100K**, unfiltered, as a control group to catch anything strata A/B's construction might miss entirely.

Sample size per stratum is left open — start small (low hundreds) to validate the prompt and process work, then scale up if early results look clean and worth the additional LLM spend. There's no fixed target; stop scaling once additional candidates stop surfacing new honeypot *patterns* (saturation), not once a round number is hit.

---

## 3. Prompt design

Two hard requirements, both non-negotiable regardless of sample size:

### 3.1 Field citation is mandatory
Every claim the LLM makes about why a profile is/isn't a honeypot must reference the specific field(s) and value(s) in the profile that justify it. A judgment with no cited field is treated as unreliable, not as a finding. This is the only real check against hallucinated contradictions, since there's no ground truth to verify against directly.

### 3.2 Structured output per candidate
For each candidate, capture:
- `candidate_id`
- `verdict`: honeypot / not honeypot / uncertain
- `confidence`: low / medium / high
- `contradiction_type`: free text — what kind of impossibility, if any (tenure/founding mismatch, proficiency/duration mismatch, timeline arithmetic, skill-breadth implausibility, narrative incoherence, title/description mismatch, skills-array unsupported-by-narrative, other, none)
- `title_description_consistent`: yes/no/partial — explicit separate field asking whether the candidate's `career_history` titles match their own `description` text, scored per-record across all roles. Tracked separately from the honeypot verdict because initial sampling suggests this mismatch may be common dataset-wide rather than honeypot-specific, and conflating it with the verdict would make the verdict noisy.
- `cited_fields`: the exact field names/values that drove the verdict
- `reasoning`: full free-text explanation, written for a human reader — this is the actual research output, not the verdict column

### 3.3 Prompt structure guidance
- One candidate per call. No batching — batching invites relative judgment ("this one's more suspicious than the last one") instead of absolute judgment against the profile's own internal logic.
- Give the model the full profile (all fields, not a curated subset) plus a short explanation of what a honeypot is in this dataset's context (echo the JD's own framing: internally impossible profiles, e.g. tenure exceeding company age, expert-level skill claims with zero years used).
- Explicitly instruct the model *not* to use technical impressiveness, AI keyword density, or skill-list length as honeypot signal — the point is to test whether the model can tell apart "genuinely strong profile" from "suspiciously perfect profile," which is precisely the distinction the hackathon's keyword-stuffer trap is designed to probe.
- Ask the model to flag if it has **no contradiction to cite** but the profile still "feels" engineered/synthetic, as a separate lower-confidence category — this captures pattern-level synthetic-data tells that don't reduce to a single field comparison, which is exactly the category the rule-based filter is structurally blind to.
- Explicitly instruct the model to check each `career_history` entry's `title` against its own `description` for topical consistency, and report this as the separate `title_description_consistent` field (Section 3.2) — **not** folded into the honeypot verdict by default, since initial sampling suggests this mismatch may be widespread and not honeypot-specific. Only let it influence the honeypot verdict if it's unusually severe or stacks with other contradictions.
- Explicitly instruct the model to cross-check the `skills` array against the career narrative (`summary` + all `description` fields) — a skill present in the array with no support anywhere in the narrative is a softer signal than a numeric contradiction, but should still be logged via `contradiction_type: skills-array unsupported-by-narrative`.
- Explicitly caution the model that internal messiness (mismatched titles, scattered unsupported skills) appears common across this dataset and is not, by itself, sufficient to call a profile a honeypot — reserve the honeypot verdict for genuine impossibility (timeline arithmetic that cannot be true, tenure exceeding plausible company age, expert-claimed skills with near-zero duration_months) or clearly synthetic/templated narrative content, not general profile quality.

---

## 4. Execution procedure

1. Build the stratified sample (Section 2). Store the candidate ID list before running anything, so the sample is fixed and auditable.
2. Run the prompt once per candidate, full profile in, structured output out (Section 3.2 schema).
3. Run a second pass on a subset — at minimum, every candidate the first pass marked `uncertain` or `honeypot` with `confidence: low` — to see whether the verdict is stable or flips. Disagreement is itself a research finding (these are exactly the cases that matter for understanding where the LLM's honeypot judgment breaks down).
4. Log everything: candidate_id, both passes' verdicts, cited fields, full reasoning text. Nothing gets thrown away at this stage — this is a study, not a filter, so over-logging costs nothing and under-logging loses the point of the exercise.

---

## 5. What to do with the output

This step is intentionally yours to drive — flagging the decision points rather than prescribing them:

- Read through the `reasoning` text for every flagged honeypot, not just the verdict. Look for clusters of `contradiction_type` — if most flags fall into 2-3 categories, that tells you which rule-based checks in `precompute.py` are well-justified and which (if any) are missing entirely.
- Compare against the rule-based filter's hypothesized categories (tenure/founding, proficiency/years, timeline arithmetic). Where this study's findings overlap, that's validation. Where they diverge — new contradiction types the rules don't catch, or rule-flagged categories the LLM doesn't independently surface as suspicious — that's the actually useful signal.
- Pay specific attention to Stratum B results: did the LLM correctly *not* flag Tier 5 plain-language candidates as honeypots (a real false-positive risk, since "doesn't use expected jargon" can look superficially like "doesn't know the field")? Did it correctly *not* get fooled by keyword-stuffers with mismatched titles?
- Decide, based on what you find, whether any new rule gets added to the production filter, whether existing rule weights should shift, or whether this stays purely as background understanding that informs how cautious you are about trusting the rule-based filter's edge cases.

---

## 6. Constraints reminder (so this doesn't accidentally leak into the graded pipeline)

- This is allowed to call hosted LLM APIs freely — it's development-time research, not the ranking step.
- This is allowed to take however long and cost however much is reasonable — no 5-minute budget applies.
- Its output (lists, reasoning logs) must not be silently imported as a precomputed feature into Track B without being re-derived through a documented, reproducible offline step if it ends up informing the actual submission — otherwise the "no hidden steps" requirement in the submission spec's code repo section is at risk, and you won't be able to defend it cleanly at Stage 5.
