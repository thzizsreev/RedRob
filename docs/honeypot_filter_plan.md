# HONEYPOT FILTER PLAN — Deterministic Timeline Rules

**Status: Production component — feeds into `precompute.py` as the honeypot exclusion bitmask.**
This is not the LLM-based research study (`honeypot_research_plan.md`). This is the offline rule engine that runs once during pre-computation, flags candidate IDs as honeypots, and excludes them from the FAISS index and Track B Polars DataFrame before any retrieval or ranking happens.

**Design principle: deterministic only.** Every rule in this file produces a binary flag from structured fields alone — no LLM calls, no embeddings, no fuzzy matching. A rule either fires or it doesn't. If it fires, the candidate is excluded. No scoring, no confidence levels, no thresholds that require tuning against labeled data. The goal is 100% precision (zero false positives on legitimate candidates) even at the cost of missing some honeypots — missing a honeypot is recoverable (it may still rank poorly on fit); falsely excluding a real candidate is not.

**Why not LLM-based:** Pass1 research results demonstrated that LLM-based honeypot detection produces systematic false positives on date arithmetic — the model flagged past dates (2024, 2025) as "future" because it lacked reliable current-date grounding. Deterministic Python date math has none of this failure mode.

---

## CURRENT DATE ANCHOR

```python
CURRENT_DATE = date(2026, 6, 22)
```

This must be defined once at the top of the script and referenced by every rule. Never hardcode the date inside individual rule functions — if the date needs updating, it should change in exactly one place.

Do not use `date.today()` in production — the sandboxed Docker environment at Stage 3 may have a different system clock. Use the explicit constant above.

---

## RULE SET

Five rules, all hard flags. Any single rule firing on any career_history entry or education entry for a candidate → that candidate is marked `is_honeypot = True` and excluded from all downstream processing.

---

### Rule 1 — Future Start Date

**Definition:** A role's `start_date` is strictly after `CURRENT_DATE`.

**Logic:**
```
for each role in career_history:
    if role.start_date > CURRENT_DATE:
        flag
```

**Why this is a hard impossibility:** A candidate cannot currently hold or have held a role that hasn't started yet. No edge case exists — even a signed offer with a future start date does not make the role part of current career history.

**Precision:** 100%. No legitimate candidate has a career_history entry with a future start_date.

**Evidence from pass1:** This was the most common pattern flagged — though pass1 incorrectly flagged 2024/2025 dates as "future" due to wrong date anchor. With `CURRENT_DATE = 2026-06-22`, only genuine future dates (post June 2026) will fire.

**Fields used:** `career_history[i].start_date`

---

### Rule 2 — Duration Overshoot

**Definition:** A role's `start_date` plus its `duration_months` places the end of that role after `CURRENT_DATE`.

**Logic:**
```
for each role in career_history:
    implied_end = role.start_date + relativedelta(months=role.duration_months)
    if implied_end > CURRENT_DATE:
        flag
```

**No exemption for `is_current: true` roles.** Even a current role's duration cannot imply an end date past today — duration_months for a current role should reflect months elapsed so far, not projected future months. If it overshoots today, the duration value itself is impossible.

**Precision:** High. The only false positive risk is a data entry where `duration_months` was rounded up aggressively (e.g., a role started May 2026 with `duration_months: 3` implying August 2026). To buffer this, allow a **30-day grace period** — flag only if `implied_end > CURRENT_DATE + 30 days`. This is the one configurable tolerance in the entire rule set.

```python
DURATION_OVERSHOOT_GRACE_DAYS = 30  # configurable
```

**Fields used:** `career_history[i].start_date`, `career_history[i].duration_months`

---

### Rule 3 — Role Overlap

**Definition:** Two consecutive roles in `career_history` overlap — i.e., a role's `start_date` is before the previous role's `end_date`.

**Logic:**
```
sort career_history by start_date ascending
for each consecutive pair (role[i], role[i+1]):
    if role[i+1].start_date < role[i].end_date:
        flag
```

**Zero tolerance.** No buffer, no grace period. Any overlap between two closed roles is an impossibility — the schema has no concept of part-time or consulting roles, so all roles are treated as full-time mutually exclusive positions.

**Handling null end_date:** A role with `end_date: null` is the current role. When sorting, treat `null` end_date as `CURRENT_DATE` for overlap computation purposes. If a non-current role has `end_date: null`, that is itself a data anomaly — flag it separately or treat it as `CURRENT_DATE` conservatively.

**Sorting note:** `career_history` in the dataset is not guaranteed to be in chronological order. Always sort by `start_date` before checking consecutive pairs, not by array index.

**Fields used:** `career_history[i].start_date`, `career_history[i].end_date`, `career_history[i].is_current`

---

### Rule 4 — Timeline Sum vs Claimed Experience

**Definition:** The sum of `duration_months` across all `career_history` entries, when converted to years, substantially exceeds the candidate's stated `years_of_experience`.

**Logic:**
```
total_months = sum(role.duration_months for role in career_history)
total_years_from_roles = total_months / 12
claimed_years = profile.years_of_experience
if total_years_from_roles > claimed_years + EXPERIENCE_OVERAGE_TOLERANCE_YEARS:
    flag
```

**Direction of check:** Flag when role durations *exceed* claimed experience by a large margin — this implies the individual role durations were inflated beyond what the candidate's own stated total experience allows. The reverse (claimed experience exceeding role durations) is normal — gaps between jobs are expected and not suspicious.

**Configurable tolerance:**
```python
EXPERIENCE_OVERAGE_TOLERANCE_YEARS = 2  # configurable — default 2 years
```

Two years is generous enough to avoid false positives from legitimate rounding (a candidate who says "5 years experience" but whose roles sum to 6.5 years is not suspicious). Tighten if false positives appear on real data; loosen if false negatives on honeypots are confirmed. Do not set below 1 year without validating on a real sample first.

**Precision note:** Lower precision than Rules 1-3. A candidate with genuinely overlapping consulting/part-time roles (not represented as distinct entries) might legitimately have role durations summing to more than stated experience. Flag and review rather than hard-exclude if this rule fires alone without corroboration from another rule. Consider treating Rule 4 as a **soft flag** that only produces an exclusion when combined with at least one other rule firing on the same candidate.

**Fields used:** `career_history[i].duration_months`, `profile.years_of_experience`

---

### Rule 5 — Graduation Year vs Claimed Experience

**Definition:** The candidate's stated `years_of_experience` exceeds the maximum possible working years since their latest education completion.

**Logic:**
```
latest_graduation_year = max(edu.end_year for edu in education)
max_possible_experience = CURRENT_DATE.year - latest_graduation_year + GRAD_TO_WORK_BUFFER_YEARS
if profile.years_of_experience > max_possible_experience:
    flag
```

**Configurable buffer:**
```python
GRAD_TO_WORK_BUFFER_YEARS = 1  # configurable — accounts for gap year, delayed start, etc.
```

One year is a reasonable buffer — most candidates start working within a year of graduation. Do not set this above 2 without a specific justification, as a large buffer makes the rule too permissive to catch anything.

**Example:** `education.end_year = 2018`, `CURRENT_DATE.year = 2026`, `GRAD_TO_WORK_BUFFER_YEARS = 1` → `max_possible_experience = 2026 - 2018 + 1 = 9 years`. If `years_of_experience = 14`, flag.

**Handling multiple education entries:** Use `max(end_year)` — the latest degree is the one that determines when formal education ended. Do not use earliest degree.

**Handling missing education:** If `education` is empty or all `end_year` values are null/absent, skip this rule for that candidate — do not flag on missing data.

**Precision note:** Similar caveat to Rule 4. Some candidates do part-time work during education, or have non-degree work experience before formal education, making their stated experience legitimately higher than this formula allows. Treat Rule 5 as a **soft flag** as well — only hard-exclude if it fires alongside at least one other rule, or if the overshoot is extreme (e.g., more than 5 years beyond the maximum).

**Fields used:** `education[i].end_year`, `profile.years_of_experience`

---

## RULE CLASSIFICATION SUMMARY

| Rule | Type | Fires on | Precision | Hard or Soft exclude |
|------|------|----------|-----------|----------------------|
| 1 — Future start_date | Per-role | start_date > CURRENT_DATE | 100% | Hard — exclude alone |
| 2 — Duration overshoot | Per-role | start_date + duration_months > CURRENT_DATE + 30d | High | Hard — exclude alone |
| 3 — Role overlap | Per-role pair | any consecutive overlap | High | Hard — exclude alone |
| 4 — Timeline sum vs experience | Per-candidate | role sum > claimed exp + 2yr | Medium | Soft — exclude only if combined with another rule |
| 5 — Graduation vs experience | Per-candidate | exp > (2026 - grad_year + 1) | Medium | Soft — exclude only if combined with another rule |

**Exclusion logic:**
```
is_honeypot = (Rule1 OR Rule2 OR Rule3) OR (Rule4 AND Rule5) OR (Rule4 AND (Rule1 OR Rule2 OR Rule3)) OR (Rule5 AND (Rule1 OR Rule2 OR Rule3))
```

In plain terms: any hard rule alone is sufficient. Soft rules only exclude when they fire together or alongside a hard rule.

---

## IMPLEMENTATION NOTES

**Date parsing:** Parse all date strings as `datetime.date` objects immediately on data load — do not do string comparison on date values anywhere in the rule logic. `"2026-07-01" > "2026-06-22"` works by accident in ISO format but is not safe as a general pattern and will silently break on any non-ISO date.

**Missing fields:** Every rule must handle missing/null field values without raising exceptions — skip the rule for that candidate if the required field is absent, do not flag on missing data. Log skipped fields for auditing.

**Output format:** Produce a flat set of excluded `candidate_id` strings. Optionally also produce a per-candidate log of which rule(s) fired and on which fields, for audit/debugging. The exclusion set is what feeds into `precompute.py`'s `IDSelector` bitmask — the log is for your own use.

**Performance:** This runs offline during pre-computation — no runtime budget constraint. But with 100K candidates and typically 2-4 roles each, the entire rule engine should complete in well under a minute even in pure Python. No vectorization needed.

**Ordering:** Run rules in order 1 → 5. Short-circuit per candidate — once any hard rule fires, mark as honeypot and skip remaining rules for that candidate (saves compute, avoids redundant log entries).

---

## CONFIGURABLE PARAMETERS (all in one place)

```python
CURRENT_DATE = date(2026, 6, 22)
DURATION_OVERSHOOT_GRACE_DAYS = 30
EXPERIENCE_OVERAGE_TOLERANCE_YEARS = 2
GRAD_TO_WORK_BUFFER_YEARS = 1
```

All four live at the top of the script. No magic numbers inside rule functions.

---

## VALIDATION APPROACH (no labeled data)

Since there is no labeled ground truth:

1. **Size check:** Flag count across the full 100K should be in the ballpark of ~80 (the spec's stated honeypot count). Significantly more than ~200 suggests a rule is over-firing — tighten tolerances. Significantly fewer than ~30 suggests rules are under-firing or honeypots don't use these patterns — revisit.

2. **Manual spot-check:** Read 10-15 flagged candidates in full. Every flag should be obviously impossible on inspection — if you have to reason about whether it's actually wrong, the rule is too aggressive.

3. **Legitimate candidate check:** Pull 10-15 strong legitimate candidates (high fit scores from your ranking) and confirm none of them are in the honeypot exclusion set.

4. **Cross-check against pass1 (after re-running with correct date):** Any candidate flagged by both the deterministic rules and a corrected LLM pass is high-confidence. Any candidate flagged by rules but not LLM (or vice versa) is worth manually inspecting.
