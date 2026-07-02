# CODING AGENT IMPLEMENTATION GUIDE — Honeypot Research Pipeline

**Read this entire document before writing any code.** This guide is self-contained — it does not assume you have access to any prior conversation. If anything here conflicts with [`plan.md`](../plans/plan.md) (the production hackathon architecture) or [`honeypot_research_plan (1).md`](../plans/honeypot_research_plan%20(1).md) (the research design rationale), this document's scope note in Section 0 governs which one wins.

---

## 0. Scope — what this is and what it is not

You are building a **research/study script**, not part of the graded hackathon submission. Its purpose is to have an LLM read individual candidate profiles and judge whether each is a "honeypot" (an intentionally fake/impossible profile planted in the dataset), then log full structured reasoning so a human can study what honeypot patterns actually exist in this dataset.

**This script must never be imported, called, or referenced by `rank.py`.** The hackathon's online ranking step (`rank.py`) is constrained to ≤5 minutes wall-clock, ≤16GB RAM, CPU-only, **no network access, no hosted LLM API calls** — see `submission_spec.docx` Section 3, "Compute constraints." This research script makes hosted LLM API calls freely and has no runtime budget. Keep it in a clearly separate directory (e.g., `research/` or `honeypot_study/`), not inside the production pipeline's directory structure, and do not let its output silently become an input to `precompute.py` without a human explicitly deciding to encode a specific finding as a new, separately-documented, reproducible rule.

If you find yourself about to write code that calls this script from `rank.py` or `precompute.py`, stop — that's out of scope for this document. Production honeypot filtering is rule-based and lives in `precompute.py`; this is a separate, independent study used to validate or improve that rule-based filter's design.

---

## 1. Background — why this exists

The hackathon (see [`job_description.docx`](job_description.docx) and `submission_spec.docx` in this folder) asks teams to rank 100,000 candidates for a "Senior AI Engineer" role. The dataset contains ~80 intentionally fake "honeypot" candidate profiles with subtly impossible details (e.g., claiming years of experience at a company that didn't exist that long ago, or "expert" proficiency in a skill with no actual time spent using it). If a submission ranks too many honeypots in its top 100, it is disqualified at Stage 3 regardless of score.

The production approach (documented in [`plan.md`](../plans/plan.md) and discussed separately) is a deterministic, rule-based honeypot filter that runs offline during pre-computation — cheap, auditable, defensible in a code-reproduction/interview setting. This research script exists to inform that filter: by having an LLM independently examine real profiles with comparatively open-ended instructions (rather than a fixed rule list), you can discover honeypot patterns the rule-based hypotheses might be missing, and equally important, discover what *isn't* actually a honeypot signal despite looking suspicious.

This script's output is reasoning text to be read by a human, not a list to be consumed by a downstream filter automatically.

---

## 2. Data schema — confirmed from real sample records

Two real candidate records were inspected before writing this guide. The schema below is confirmed from those records, not guessed. If you encounter fields not listed here, treat the dataset as having those fields too — this list documents what's confirmed present, not an exhaustive schema lock.

```
{
  "candidate_id": "CAND_XXXXXXX",
  "profile": {
    "anonymized_name": str,
    "headline": str,
    "summary": str,                          // free-text career summary, often self-describing honestly
    "location": str,
    "country": str,
    "years_of_experience": float,
    "current_title": str,
    "current_company": str,
    "current_company_size": str,             // e.g. "201-500", "10001+"
    "current_industry": str
  },
  "career_history": [                        // list of role objects, most recent first observed
    {
      "company": str,
      "title": str,
      "start_date": "YYYY-MM-DD",
      "end_date": "YYYY-MM-DD" | null,        // null means current role
      "duration_months": int,
      "is_current": bool,
      "industry": str,
      "company_size": str,
      "description": str                     // free-text role description — SEE WARNING BELOW
    },
    ...
  ],
  "education": [
    {
      "institution": str,
      "degree": str,
      "field_of_study": str,
      "start_year": int,
      "end_year": int,
      "grade": str,                          // e.g. "8.85 CGPA", "77%"
      "tier": str                            // e.g. "tier_3", "tier_4"
    }
  ],
  "skills": [
    {
      "name": str,
      "proficiency": str,                    // e.g. "beginner", "intermediate", "expert" (expert not yet confirmed but assume exists)
      "endorsements": int,
      "duration_months": int                 // closest analog to "years used" — no separate years-used field exists
    }
  ],
  "certifications": [...],                   // empty array seen in samples, structure unconfirmed
  "languages": [
    { "language": str, "proficiency": str }
  ],
  "redrob_signals": {
    "profile_completeness_score": float (0-100),
    "signup_date": "YYYY-MM-DD",
    "last_active_date": "YYYY-MM-DD",
    "open_to_work_flag": bool,
    "profile_views_received_30d": int,
    "applications_submitted_30d": int,
    "recruiter_response_rate": float (0.0-1.0),
    "avg_response_time_hours": float,
    "skill_assessment_scores": dict,         // OFTEN EMPTY {} — confirmed in both sampled records
    "connection_count": int,
    "endorsements_received": int,
    "notice_period_days": int (0-180),
    "expected_salary_range_inr_lpa": { "min": float, "max": float },
    "preferred_work_mode": str,              // "flexible" etc.
    "willing_to_relocate": bool,
    "github_activity_score": int,            // -1 SENTINEL VALUE means "no GitHub linked" — confirmed in both samples
    "search_appearance_30d": int,
    "saved_by_recruiters_30d": int,
    "interview_completion_rate": float (0.0-1.0),
    "offer_acceptance_rate": float,          // -1 SENTINEL VALUE means "no prior offers" — confirmed in both samples
    "verified_email": bool,
    "verified_phone": bool,
    "linkedin_connected": bool
  }
}
```

### 2.1 Critical data quirk — title/description mismatch (confirmed, not hypothetical)

In both sampled records, **every** `career_history` entry's `description` text describes work that does not match that entry's own `title`. Example: a role titled "DevOps Engineer" had a `description` entirely about Android/Kotlin mobile app development. A role titled "Operations Manager" had a `description` about mechanical engineering hardware design (SolidWorks, ANSYS).

Each `description` is internally coherent as its own piece of text — it reads like a real, specific role description — it is simply attached to the wrong `title`. This appeared in **100% of entries across both sampled records (2 records, ~7 role entries total)**, which is too small a sample to call definitive, but is strong enough that you must design around it rather than assume it's rare:

- **Do not trust `title` or `current_title` in isolation** for any judgment (fit scoring, honeypot detection, hard filtering). Always read the paired `description`.
- **Do not assume title/description mismatch alone indicates a honeypot.** Initial evidence suggests this may be a dataset-wide generation artifact, not a deliberate trap. The research prompt (Section 4) tracks this as a separate field specifically so it doesn't contaminate the honeypot verdict.
- When extracting "what does this candidate actually do," prioritize `description` and `summary` text over `title` fields.

### 2.2 Critical data quirk — skills array can contain unsupported entries

In one sampled record, the `skills` array contained entries (e.g., "Kafka", "Feature Engineering") with no corresponding support anywhere in `summary` or any `career_history.description`. The candidate's actual career narrative was marketing/operations/hardware/design — entirely non-technical — while two `skills` entries looked technical/ML-adjacent in isolation.

This is the job description's explicitly-named "keyword stuffer" trap (see [`job_description.docx`](job_description.docx): "A candidate who has all the AI keywords listed as skills but whose title is 'Marketing Manager' is not a fit"), but implemented via the structured `skills` array rather than padded into prose. **Any logic — rule-based or LLM-based — that scores `skills` array entries without cross-checking them against the career narrative will be fooled by this.**

### 2.3 Sentinel values — confirmed

`github_activity_score: -1` and `offer_acceptance_rate: -1` are confirmed "no data" sentinels (consistent with `redrob_signals_doc.docx`'s documented convention). Any code that consumes these fields (including the LLM prompt, if you choose to pass them) must handle -1 as "absent," not as a literal low/negative score. Do not let a numeric aggregation or average accidentally treat -1 as real data.

`skill_assessment_scores` was `{}` (empty dict) in both samples. Do not build logic — prompt instructions or downstream rules — that assumes this field is reliably populated.

---

## 3. Sampling strategy

Do not sample the first N records or a single pure-random N records — both produce a misleading study. Build three strata:

**Stratum A — broad AI/ML-adjacent pool.** Candidates with at least some technical/AI signal, cast as a wide net (not a tight keyword filter — err toward over-inclusion). This is where honeypots are concentrated, since they're designed to look plausible specifically within this population.

**Stratum B — known trap categories, sampled deliberately.** Pull examples matching each of these patterns (use a cheap heuristic to locate candidates, it doesn't need to be precise — the goal is coverage, not a clean filter):
- Keyword-stuffer pattern: technical-sounding terms present in `skills` or prose, but `current_title`/`career_history` shows an unrelated field (e.g., marketing, operations, design — see Section 2.2 for a confirmed real example)
- Plain-language / Tier 5 pattern: career history describing system-building work (recommendation systems, ranking, search) without canonical AI vocabulary ("RAG," "embeddings," "Pinecone," etc.)
- Behavioral-rescue pattern: strong `redrob_signals` (high `recruiter_response_rate`, high `profile_views_received_30d`) paired with weak/absent technical skill signal
- Consulting-only career pattern: all `career_history.company` entries are from TCS, Infosys, Wipro, Accenture, Cognizant, or Capgemini
- CV/speech/robotics-only pattern: technical career history with no NLP/IR/retrieval exposure

**Stratum C — small unfiltered random sample** from the full candidate pool, as a control group to catch anything strata A/B's construction logic might structurally miss.

Sample size per stratum is an open parameter — do not hardcode a "final" number. Start with a small batch (suggest starting around 50-100 per stratum as an initial pass) to validate the prompt and pipeline mechanics work correctly, then scale up. Stop scaling once new batches stop surfacing new `contradiction_type` patterns in the output (saturation) rather than targeting a fixed total. Surface this as a configurable parameter (CLI flag or config value), not a literal constant buried in code.

Before running the LLM pass, write the selected `candidate_id` list to disk (e.g., a JSON or CSV manifest) so the sample is fixed, reproducible, and auditable independent of the LLM responses.

---

## 4. LLM prompt design

### 4.1 Execution shape

- **One candidate per API call. Do not batch multiple candidates into a single prompt.** Batching invites the model to judge candidates relative to each other within the batch rather than against the profile's own internal logic and the JD's stated criteria — this would corrupt the independence of the study.
- Pass the **full profile JSON** for the candidate (all fields from Section 2), not a curated subset. Field-level citation (Section 4.3) requires the model to have access to everything it might cite.
- Run a **second pass** on every candidate whose first-pass verdict is `uncertain`, or `honeypot` with `confidence: low`. Record both passes. Disagreement between passes is itself a useful research finding — log it, don't resolve it silently by re-running until you get one answer you like.

### 4.2 What the prompt must instruct the model to do

Compose a system/instruction prompt that does all of the following:

1. Explain what a honeypot is in this dataset's specific context: an intentionally fake profile containing internal impossibilities, e.g. claimed tenure at a company exceeding the company's plausible founding-to-now age, or expert-level proficiency claimed in a skill paired with near-zero `duration_months` for that skill, or career-history date arithmetic that cannot be true (overlapping full-time roles, experience exceeding what age/graduation year permits).

2. Explicitly instruct the model **not** to use technical impressiveness, AI-keyword density, or skill-list length as a honeypot signal on its own. A profile can be a strong, legitimate candidate and look "impressive" — that is not suspicious by itself. The model's job is to distinguish "genuinely strong" from "internally impossible," not "strong" from "weak."

3. Explicitly instruct the model to check, for every `career_history` entry, whether `title` and `description` are topically consistent — and to report this as its own separate field (Section 4.3), **not** to fold a mismatch directly into the honeypot verdict. State plainly in the prompt: initial sampling suggests title/description mismatches may be common across this dataset and are not by themselves evidence of a honeypot; only let this influence the honeypot verdict if it's unusually severe (e.g., title and description are in completely unrelated fields *and* this pattern stacks with other red flags) or where it represents a different problem worth flagging (e.g., looks like deliberate misrepresentation rather than a data quirk).

4. Explicitly instruct the model to cross-check every entry in the `skills` array against the full career narrative (`summary` plus all `career_history.description` fields). A skill with no support anywhere in the narrative should be reported via `contradiction_type: skills-array-unsupported-by-narrative` — this is a real signal worth tracking (see Section 2.2) but is softer than a numeric impossibility, so it should not automatically drive an honeypot verdict on its own.

5. Require that every claim the model makes is tied to a specific field name and value from the profile. A judgment without a cited field is not acceptable output — this is the only check available against hallucinated contradictions, since there is no labeled ground truth to verify against.

6. Provide a lower-confidence escape hatch: if the model has no specific contradiction to cite but the profile still reads as synthetic or templated (generic phrasing, suspiciously uniform numbers, vague "achievements" with no specifics), it should say so explicitly via a separate flag rather than either forcing a binary honeypot/not-honeypot call or silently ignoring the instinct. This captures a category of signal that field-by-field rule logic structurally cannot catch.

7. Instruct the model to explicitly note when it observes -1 sentinel values (`github_activity_score`, `offer_acceptance_rate`) and treat them as "no data," not as literal scores.

### 4.3 Required structured output schema, per candidate

```json
{
  "candidate_id": "string",
  "verdict": "honeypot | not_honeypot | uncertain",
  "confidence": "low | medium | high",
  "contradiction_type": "tenure_founding_mismatch | proficiency_duration_mismatch | timeline_arithmetic | skill_breadth_implausibility | narrative_incoherence | title_description_mismatch | skills_array_unsupported | synthetic_pattern_no_specific_contradiction | other | none",
  "cited_fields": [
    { "field": "string (e.g. career_history[1].title)", "value": "string", "relevance": "string — why this field supports the verdict" }
  ],
  "title_description_consistent": "yes | no | partial",
  "reasoning": "free text — full explanation written for a human reader; this is the primary research output, not a formality"
}
```

Persist every field for every candidate, both passes where applicable. Do not discard low-confidence or "none" verdicts — they're part of the dataset you're studying.

---

## 5. Implementation requirements

- **Language/runtime:** match whatever the rest of the team's tooling uses (Python is a safe default given `plan.md`'s dependencies). No specific framework is mandated for this script since it's outside the graded pipeline's constraints.
- **LLM provider:** use the Anthropic API (or whichever provider the team has credentials for) via standard API calls — this is explicitly allowed to be a hosted LLM call, since this script is never reproduced inside the hackathon's constrained sandbox.
- **Persistence:** write results incrementally (e.g., append to a JSONL file per candidate as each completes) rather than holding everything in memory and writing once at the end — this protects against losing partial progress on a long-running batch, and there is no reason to optimize for single-shot completion given there's no runtime budget here.
- **Rate limiting / retries:** handle API rate limits and transient errors with backoff and retry; log failures per-candidate rather than letting one failure abort the whole batch.
- **Idempotency:** before calling the API for a candidate, check whether a result already exists for that `candidate_id` (and pass number) in the output file, and skip if so — this allows the script to be safely re-run/resumed after interruption without re-spending API calls on already-completed candidates.
- **No integration with `rank.py` or `precompute.py`.** Re-stating Section 0: this script's output directory should not be referenced anywhere in the production pipeline's code path. If a human later decides a specific finding from this study should become a production rule, that should be implemented as new, explicit, documented logic in `precompute.py` — not as a runtime or import-time dependency on this script's output files.

---

## 6. Output / deliverable

The deliverable is a structured log (e.g., JSONL, one record per candidate per pass) containing every field in Section 4.3, plus the sample manifest from Section 3. This is meant to be read and analyzed by a human afterward — there is no requirement to build a summary dashboard or aggregate report as part of this script, though one may be added later if useful for analysis.

Do not treat a "Top N honeypots" list as the deliverable. The reasoning and contradiction-type distribution across the full sample is the actual research output; the verdict column alone discards the reason this study exists.
