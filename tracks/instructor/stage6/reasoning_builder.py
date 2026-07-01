"""Stage 6 — 3-sentence reasoning builder per reasoning_builder_plan.md."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable
from typing import Any

VERB_POOL = [
    "built and owned",
    "designed and deployed",
    "developed and scaled",
    "architected and shipped",
    "constructed and maintained",
]

VERB_GERUND_MAP = {
    "built and owned": "building and owning",
    "designed and deployed": "designing and deploying",
    "developed and scaled": "developing and scaling",
    "architected and shipped": "architecting and shipping",
    "constructed and maintained": "constructing and maintaining",
}

PARTICIPLE_MAP = {
    "improved": "improving",
    "increased": "increasing",
    "reduced": "reducing",
    "boosted": "boosting",
    "lifted": "lifting",
}

METRIC_VERB_RE = re.compile(
    r"(improved|increased|reduced|boosted|lifted)\s+([\w\s\-]{2,25})\s+by\s+(\d+(?:\.\d+)?%)",
    re.I,
)

METRIC_SCALE_RE = re.compile(
    r"\d+(?:\.\d+)?[KMB]\+?\s*(users|documents|records|queries|candidates)",
    re.I,
)

METRIC_LATENCY_RE = re.compile(r"(\d+ms)\s+to\s+(\d+ms)", re.I)

METRIC_PLAIN_PCT_RE = re.compile(r"(\d+(?:\.\d+)?%)")

DESCRIPTION_TECH_RE = re.compile(
    r"\b(FAISS|Qdrant|Weaviate|Milvus|Pinecone|Elasticsearch|OpenSearch|"
    r"BM25|sentence-transformers|sentence_transformers|bge-base|all-MiniLM|"
    r"XGBoost|LightGBM|scikit-learn|MLflow|Kubeflow|LangChain|LlamaIndex|"
    r"PEFT|LoRA|QLoRA|Hugging\s*Face|HNSW|ANNOY|ScaNN)\b",
    re.I,
)

IRRELEVANT_SKILL_SET = {
    "image classification",
    "computer vision",
    "object detection",
    "speech recognition",
    "deep learning",
    "reinforcement learning",
    "time series",
    "robotics",
    "nlp",
    "recommendation systems",
    "machine learning",
    "data science",
    "feature engineering",
    "statistical modeling",
    "information retrieval",
}

SYSTEM_TYPE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\branking\b.*\b(layer|pipeline|model|system)\b", re.I), "ranking pipeline"),
    (re.compile(r"\brecommendation\b.*\bsystem\b", re.I), "recommendation system"),
    (re.compile(r"\bsemantic\s+search\b", re.I), "semantic search system"),
    (re.compile(r"\bdiscovery\s+feed\b|\branking\s+model\b", re.I), "discovery ranking system"),
    (re.compile(r"\bML\s+pipeline\b|\bMLflow\b|\bKubeflow\b", re.I), "ML infrastructure pipeline"),
    (re.compile(r"\bRAG\b|\bretrieval.augmented\b", re.I), "RAG-based retrieval pipeline"),
    (re.compile(r"\bwhat\s+users\s+are\s+looking\b", re.I), "relevance matching system"),
    (re.compile(r"\bsearch\s+and\s+discovery\b", re.I), "search and discovery system"),
    (re.compile(r"\bembedding.based\s+search\b", re.I), "embedding-based search system"),
    (re.compile(r"\bmigration\b.*\bsearch\b", re.I), "search migration"),
]

SKILL_CATEGORIES: list[tuple[str, set[str]]] = [
    (
        "VECTOR_DB",
        {"weaviate", "qdrant", "milvus", "pinecone", "faiss", "chroma", "pgvector"},
    ),
    (
        "SEARCH",
        {"elasticsearch", "opensearch", "solr", "algolia", "typesense"},
    ),
    (
        "RANKING_ML",
        {"scikit-learn", "xgboost", "lightgbm", "learning to rank", "catboost"},
    ),
    (
        "LLM",
        {"langchain", "llamaindex", "fine-tuning llms", "haystack", "peft", "lora"},
    ),
]

CATEGORY_KEYWORDS: list[tuple[str, list[str]]] = [
    ("VECTOR_DB", ["embedding", "vector search", "ann", "hnsw", "faiss"]),
    ("SEARCH", ["inverted index", "bm25", "full-text search", "opensearch", "elasticsearch"]),
    ("RANKING_ML", ["ranking", "gradient-boost", "learning-to-rank"]),
    ("LLM", ["rag", "fine-tun", "langchain", "llm", "prompt"]),
]

TEMPERATURES = [0.3, 0.5, 0.7, 0.9, 1.0]


def stable_seed(text: str, modulus: int) -> int:
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % modulus


def pick_temperature(candidate_id: str, slot: str) -> float:
    index = stable_seed(f"{candidate_id}:{slot}:temp", len(TEMPERATURES))
    return TEMPERATURES[index]


def select_verb(candidate_id: str) -> str:
    return VERB_POOL[stable_seed(candidate_id + ":verb", len(VERB_POOL))]


def _truncate_subject(subject: str, max_words: int = 4) -> str:
    words = subject.strip().split()
    if len(words) <= max_words:
        return subject.strip()
    return " ".join(words[:max_words])


def _metric_dict(
    raw: str,
    participle: str,
    noun: str,
) -> dict[str, str]:
    return {
        "raw": raw,
        "converted_participle": participle,
        "converted_noun": noun,
    }


def calculate_tech_cat(cross_encoder_score: float) -> str:
    if cross_encoder_score >= 3.5:
        return "DEEP"
    if cross_encoder_score >= 2.5:
        return "STRONG"
    if cross_encoder_score >= 1.5:
        return "MODERATE"
    return "SURFACE"


def calculate_years_label(total_years_exp: float) -> str:
    if total_years_exp >= 8.0:
        return f"{int(total_years_exp)}-year"
    if total_years_exp >= 7.0:
        return f"nearly {math.ceil(total_years_exp)}-year"
    if total_years_exp >= 6.0:
        return f"{int(total_years_exp)}-plus-year"
    if total_years_exp >= 5.0:
        return "5-plus-year"
    return f"{int(total_years_exp)}-year"


def calculate_experience_type(pre_llm: bool) -> str:
    return "production ML" if pre_llm else "ML engineering"


def calculate_notice_days_label(notice_days: int) -> str:
    if notice_days == 0:
        return "immediate availability"
    return f"{notice_days}-day notice period"


def calculate_availability_assessment(
    days_since_active: int,
    notice_days: int,
    open_to_work: bool,
    applications_30d: int,
    offer_acceptance: float,
) -> str:
    if open_to_work and days_since_active <= 30 and notice_days <= 30:
        return "immediate availability and low outreach friction"

    if open_to_work and (days_since_active > 60 or notice_days > 60):
        return "moderate friction — timeline confirmation needed"

    if applications_30d >= 10 and offer_acceptance >= 0.80:
        return "active job-seeking with strong commitment signals"

    if (
        open_to_work
        and days_since_active <= 60
        and notice_days <= 60
        and (applications_30d >= 5 or offer_acceptance >= 0.70)
    ):
        return "active availability with manageable timeline"

    if not open_to_work or days_since_active > 90:
        return "low availability signal — confirm interest before outreach"

    return "low availability signal — confirm interest before outreach"


def calculate_outreach_recommendation(tech_cat: str, availability_assessment: str) -> str:
    is_low_friction = (
        "immediate" in availability_assessment
        or "active availability" in availability_assessment
        or "strong commitment" in availability_assessment
    )
    is_friction = "moderate friction" in availability_assessment
    is_low_signal = "low availability" in availability_assessment

    if tech_cat == "DEEP":
        if is_low_friction:
            return "strong outreach candidate"
        if is_friction:
            return "recommended for outreach with timeline negotiation"
        if is_low_signal:
            return "outreach recommended but availability must be confirmed"
        return "recommended for outreach with timeline negotiation"

    if tech_cat == "STRONG":
        if is_low_friction:
            return "solid outreach candidate"
        return "worth outreach with timeline confirmation"

    if tech_cat == "MODERATE":
        return "worth outreach pending direct technical evaluation"

    return "low outreach priority unless screening confirms depth"


def calculate_career_characterization(product_frac: float, consulting_count: int) -> str:
    if product_frac == 1.0 and consulting_count == 0:
        return "career is entirely at product companies"
    if product_frac >= 0.7 and consulting_count == 0:
        return "career is predominantly at product companies"
    if consulting_count >= 1 and product_frac >= 0.5:
        return "career combines product company and consulting firm experience"
    if consulting_count >= 1 and product_frac < 0.5:
        return "career is primarily at consulting firms"
    return "career spans mixed company types"


def calculate_disqualifier_statement(
    consulting_count: int,
    llm_framework_only: bool,
    recent_ai_only: bool,
) -> str:
    flags: list[str] = []
    if consulting_count >= 1:
        flags.append("consulting firm history is an explicit JD disqualifier")
    if llm_framework_only:
        flags.append("LLM framework-only experience falls below the JD's threshold")
    if recent_ai_only:
        flags.append("production ML experience is primarily post-2022")

    if not flags:
        return "clearing the JD's explicit disqualifiers"
    if len(flags) == 1:
        return f"though {flags[0]}"
    return f"though {flags[0]} and {flags[1]}"


def extract_system_type(description: str | None) -> str:
    if not description:
        return "production ML system"
    for pattern, label in SYSTEM_TYPE_PATTERNS:
        if pattern.search(description):
            return label
    return "production ML system"


def extract_primary_metric(description: str | None) -> dict[str, str] | None:
    if not description:
        return None

    match = METRIC_VERB_RE.search(description)
    if match:
        verb_word = match.group(1).lower()
        subject = _truncate_subject(match.group(2).strip())
        pct = match.group(3)
        raw = match.group(0).strip()
        participle_verb = PARTICIPLE_MAP.get(verb_word, verb_word + "ing")
        participle = f"{participle_verb} {subject} by {pct}"
        noun = f"a {pct} {subject} improvement"
        return _metric_dict(raw, participle, noun)

    scale_match = METRIC_SCALE_RE.search(description)
    if scale_match:
        value = scale_match.group(0)
        raw = value
        participle = f"serving {value}" if not value.lower().startswith("serving") else value
        noun = participle
        return _metric_dict(raw, participle, noun)

    latency_match = METRIC_LATENCY_RE.search(description)
    if latency_match:
        ms_from = latency_match.group(1)
        ms_to = latency_match.group(2)
        raw = f"{ms_from} to {ms_to}"
        participle = f"reducing latency from {ms_from} to {ms_to}"
        noun = f"a latency reduction from {ms_from} to {ms_to}"
        return _metric_dict(raw, participle, noun)

    pct_match = METRIC_PLAIN_PCT_RE.search(description)
    if pct_match:
        pct = pct_match.group(1)
        start = max(0, pct_match.start() - 25)
        raw = description[start : pct_match.end()].strip()
        participle = f"delivering a {pct} improvement"
        noun = f"a {pct} improvement"
        return _metric_dict(raw, participle, noun)

    return None


def extract_description_tech(description: str | None) -> list[str]:
    if not description:
        return []

    seen: set[str] = set()
    items: list[str] = []
    for match in DESCRIPTION_TECH_RE.finditer(description):
        normalized = match.group(0).lower().replace("_", "-")
        if normalized in seen:
            continue
        seen.add(normalized)
        items.append(match.group(0))
        if len(items) >= 2:
            break
    return items


def extract_company_scope(career_history: list) -> tuple[list[str], str]:
    seen: set[str] = set()
    companies: list[str] = []
    for role in career_history:
        if not isinstance(role, dict):
            continue
        name = str(role.get("company") or "").strip()
        if name and name not in seen:
            seen.add(name)
            companies.append(name)

    if len(companies) == 0:
        return [], ""
    if len(companies) == 1:
        return companies, f"at {companies[0]}"
    if len(companies) == 2:
        return companies, f"at {companies[0]} and {companies[1]}"
    if len(companies) == 3:
        return companies, f"across {companies[0]}, {companies[1]}, and {companies[2]}"
    return companies, f"across {companies[0]}, {companies[1]}, and prior companies"


def _surface_company_list(companies: list[str]) -> str:
    if len(companies) == 0:
        return ""
    if len(companies) == 1:
        return companies[0]
    if len(companies) == 2:
        return f"{companies[0]} and {companies[1]}"
    if len(companies) == 3:
        return f"{companies[0]}, {companies[1]}, and {companies[2]}"
    return f"{companies[0]}, {companies[1]}, and prior companies"


def _sorted_skills(skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        skills,
        key=lambda item: int(item.get("endorsements") or 0),
        reverse=True,
    )


def _skill_name(skill: dict[str, Any]) -> str:
    return str(skill.get("name") or "").strip()


def extract_named_tech(skills: list, descriptions: list[str]) -> str:
    if not skills:
        return "unknown"

    combined = " ".join(d.lower() for d in descriptions if d)
    sorted_skills = _sorted_skills(skills)

    for skill in sorted_skills:
        name = _skill_name(skill)
        if name and name.lower() in combined:
            return name

    triggered: list[str] = []
    for category, keywords in CATEGORY_KEYWORDS:
        if any(keyword in combined for keyword in keywords):
            triggered.append(category)

    category_map = {cat: members for cat, members in SKILL_CATEGORIES}
    for category in triggered:
        members = category_map.get(category, set())
        for skill in sorted_skills:
            name = _skill_name(skill)
            if name.lower() in members:
                return name

    return _skill_name(sorted_skills[0]) or "unknown"


def extract_verified_skill(skill_assessment_scores: dict) -> str | None:
    if not skill_assessment_scores:
        return None

    filtered = {
        key: value
        for key, value in skill_assessment_scores.items()
        if str(key).lower() not in IRRELEVANT_SKILL_SET
    }

    if filtered:
        return str(max(filtered, key=filtered.get))

    return str(max(skill_assessment_scores, key=skill_assessment_scores.get))


def extract_best_metric_from_prior_roles(career_history: list) -> dict[str, str] | None:
    for role in career_history[1:]:
        if not isinstance(role, dict):
            continue
        metric = extract_primary_metric(role.get("description"))
        if metric:
            return metric
    return None


def score_tenure_signal(avg_tenure: float) -> str | None:
    if avg_tenure >= 3.0:
        return "stable tenure"
    if avg_tenure <= 1.5:
        return "short average tenure raising ownership depth questions"
    return None


def score_pre_llm_signal(pre_llm: bool) -> str | None:
    if pre_llm:
        return "pre-LLM production ML ownership"
    return None


def score_sweet_spot_signal(in_sweet_spot: bool) -> str | None:
    if in_sweet_spot:
        return "placing them in the role's target experience band"
    return None


def score_tech_depth_caveat(tech_cat: str) -> str | None:
    if tech_cat == "MODERATE":
        return "cross-encoder alignment is partial — direct technical evaluation recommended"
    if tech_cat == "SURFACE":
        return "semantic depth against the role's core requirements is limited"
    return None


def score_activity_signal(days_since_active: int) -> str | None:
    if days_since_active > 45:
        return f"{days_since_active} days since last platform login"
    return None


def score_response_signal(recruiter_response_rate: float) -> str | None:
    if recruiter_response_rate >= 0.80:
        return "strong recruiter responsiveness"
    if recruiter_response_rate < 0.50:
        return "low response rate — outreach may face delays"
    return None


def score_github_signal(github_activity_score: float) -> str | None:
    if github_activity_score >= 70:
        return "active GitHub presence supporting technical claims"
    return None


def score_applications_signal(applications_30d: int, open_to_work: bool) -> str | None:
    if applications_30d >= 10 and open_to_work:
        return "actively applying to roles"
    if applications_30d == 0 and not open_to_work:
        return "no recent applications"
    return None


def assemble_sentence_1(
    name: str,
    years_label: str,
    experience_type: str,
    company_scope: str,
    verb_gerund: str,
    system_type: str,
    company_0: str,
    metric: dict[str, str] | None,
    desc_tech: list[str],
    verified_skill: str | None,
    named_tech: str,
    tech_cat: str,
    surface_company_list: str = "",
) -> str:
    if tech_cat == "SURFACE":
        tech_clause = f" including {named_tech} exposure" if named_tech else ""
        return (
            f"{name}'s profile, spanning {surface_company_list}, shows keyword-level "
            f"alignment to retrieval systems{tech_clause}, but limited semantic depth "
            f"against the role's ranking and embedding requirements"
        )

    if tech_cat in ("DEEP", "STRONG"):
        metric_clause = f", {metric['converted_participle']}" if metric else ""
    elif tech_cat == "MODERATE":
        metric_clause = f", with {metric['converted_noun']} achieved" if metric else ""
    else:
        metric_clause = ""

    if len(desc_tech) == 2:
        desc_tech_clause = f", using {desc_tech[0]} and {desc_tech[1]}"
    elif len(desc_tech) == 1:
        desc_tech_clause = f", using {desc_tech[0]}"
    else:
        desc_tech_clause = ""

    if verified_skill:
        skill_clause = f"backed by verified {verified_skill} depth"
    else:
        skill_clause = f"with {named_tech} experience"

    jd_alignment = {
        "DEEP": "directly matching the role's retrieval and ranking requirements",
        "STRONG": "aligning with the role's retrieval and ranking requirements",
        "MODERATE": "showing partial alignment to the role's retrieval and ranking focus",
    }[tech_cat]

    return (
        f"{name} brings {years_label} years of {experience_type} experience "
        f"{company_scope}, most recently {verb_gerund} the {system_type} "
        f"at {company_0}{metric_clause}{desc_tech_clause}, "
        f"{skill_clause} {jd_alignment}"
    )


def assemble_sentence_2(
    career_characterization: str,
    tenure_signal: str | None,
    pre_llm_signal: str | None,
    disqualifier_statement: str,
    sweet_spot_signal: str | None,
    tech_depth_caveat: str | None,
) -> str:
    qualifiers: list[str] = []
    if tenure_signal:
        qualifiers.append(tenure_signal)
    if pre_llm_signal:
        qualifiers.append(pre_llm_signal)

    if len(qualifiers) == 1:
        optional_qualifiers = f", with {qualifiers[0]}"
    elif len(qualifiers) == 2:
        optional_qualifiers = f", with {qualifiers[0]} and {qualifiers[1]}"
    else:
        optional_qualifiers = ""

    sweet = f", {sweet_spot_signal}" if sweet_spot_signal else ""
    caveat = f"; {tech_depth_caveat}" if tech_depth_caveat else ""

    sentence = (
        f"{career_characterization}{optional_qualifiers}, "
        f"{disqualifier_statement}{sweet}{caveat}"
    )
    return sentence[0].upper() + sentence[1:] if sentence else sentence


def assemble_sentence_3(
    activity_signal: str | None,
    notice_label: str,
    availability_assessment: str,
    response_signal: str | None,
    github_signal: str | None,
    applications_signal: str | None,
    outreach_recommendation: str,
) -> str:
    parts: list[str] = []
    if activity_signal:
        parts.append(activity_signal)
    parts.append(f"{notice_label} with {availability_assessment}")
    if response_signal:
        parts.append(response_signal)
    if github_signal:
        parts.append(github_signal)
    if applications_signal:
        parts.append(applications_signal)
    body = "; ".join(parts)
    return f"{body} — {outreach_recommendation}"


def _field_values(candidate: dict[str, Any]) -> dict[str, Any]:
    pipeline = candidate.get("pipeline") or {}
    retrieval = pipeline.get("retrieval_scores") or {}
    gates = pipeline.get("gates_and_career") or {}
    behavioral = pipeline.get("behavioral_signals") or {}
    stage5 = pipeline.get("stage5_scoring") or {}
    profile = candidate.get("profile") or {}

    notice_days = gates.get("notice_period_days")
    if notice_days is None:
        notice_days = behavioral.get("notice_period_days", 0)

    return {
        "candidate_id": str(candidate["candidate_id"]),
        "name": str(profile.get("anonymized_name") or ""),
        "current_company": str(profile.get("current_company") or ""),
        "cross_encoder_score": float(retrieval.get("cross_encoder_score", 0)),
        "skill_assessment_scores": pipeline.get("skill_assessment_scores") or {},
        "total_years_exp": float(gates.get("total_years_exp", 0)),
        "pre_llm_production_ml": bool(gates.get("pre_llm_production_ml")),
        "product_company_fraction": float(gates.get("product_company_fraction", 0)),
        "consulting_company_count": int(gates.get("consulting_company_count", 0)),
        "avg_tenure_per_employer": float(gates.get("avg_tenure_per_employer", 0)),
        "llm_framework_only": bool(gates.get("llm_framework_only")),
        "recent_ai_only": bool(gates.get("recent_ai_only")),
        "in_sweet_spot": bool(gates.get("in_sweet_spot")),
        "notice_period_days": int(notice_days or 0),
        "days_since_active": int(stage5.get("days_since_active", 0)),
        "open_to_work_flag": bool(behavioral.get("open_to_work_flag")),
        "applications_submitted_30d": int(behavioral.get("applications_submitted_30d", 0)),
        "recruiter_response_rate": float(behavioral.get("recruiter_response_rate", 0)),
        "offer_acceptance_rate": float(behavioral.get("offer_acceptance_rate", 0)),
        "github_activity_score": float(behavioral.get("github_activity_score", -1)),
        "career_history": candidate.get("career_history") or [],
        "skills": candidate.get("skills") or [],
    }


def _compute_s3_parts(fields: dict[str, Any], tech_cat: str) -> dict[str, Any]:
    notice_label = calculate_notice_days_label(fields["notice_period_days"])
    availability_assessment = calculate_availability_assessment(
        fields["days_since_active"],
        fields["notice_period_days"],
        fields["open_to_work_flag"],
        fields["applications_submitted_30d"],
        fields["offer_acceptance_rate"],
    )
    outreach_recommendation = calculate_outreach_recommendation(tech_cat, availability_assessment)
    return {
        "notice_label": notice_label,
        "availability_assessment": availability_assessment,
        "outreach_recommendation": outreach_recommendation,
        "activity_signal": score_activity_signal(fields["days_since_active"]),
        "response_signal": score_response_signal(fields["recruiter_response_rate"]),
        "github_signal": score_github_signal(fields["github_activity_score"]),
        "applications_signal": score_applications_signal(
            fields["applications_submitted_30d"],
            fields["open_to_work_flag"],
        ),
    }


def build_s3_raw(candidate: dict[str, Any], tech_cat: str | None = None) -> str:
    """Rebuild sentence 3 from volatile behavioral fields only."""
    fields = _field_values(candidate)
    cat = tech_cat or calculate_tech_cat(fields["cross_encoder_score"])
    parts = _compute_s3_parts(fields, cat)
    return assemble_sentence_3(
        activity_signal=parts["activity_signal"],
        notice_label=parts["notice_label"],
        availability_assessment=parts["availability_assessment"],
        response_signal=parts["response_signal"],
        github_signal=parts["github_signal"],
        applications_signal=parts["applications_signal"],
        outreach_recommendation=parts["outreach_recommendation"],
    )


def build_raw_sentences(candidate: dict[str, Any]) -> dict[str, Any]:
    """Sections 1–4: assemble s1/s2/s3 raw sentences without paraphrasing."""
    fields = _field_values(candidate)
    cid = fields["candidate_id"]

    tech_cat = calculate_tech_cat(fields["cross_encoder_score"])
    years_label = calculate_years_label(fields["total_years_exp"])
    experience_type = calculate_experience_type(fields["pre_llm_production_ml"])
    career_characterization = calculate_career_characterization(
        fields["product_company_fraction"],
        fields["consulting_company_count"],
    )
    disqualifier_statement = calculate_disqualifier_statement(
        fields["consulting_company_count"],
        fields["llm_framework_only"],
        fields["recent_ai_only"],
    )

    career_history = fields["career_history"]
    descriptions = [
        str(role.get("description") or "")
        for role in career_history
        if isinstance(role, dict) and role.get("description")
    ]
    desc_0 = descriptions[0] if descriptions else None

    companies, company_scope = extract_company_scope(career_history)
    surface_company_list = _surface_company_list(companies)
    system_type = extract_system_type(desc_0)
    metric = extract_primary_metric(desc_0)
    desc_tech = extract_description_tech(desc_0)
    named_tech = extract_named_tech(fields["skills"], descriptions)
    verified_skill = extract_verified_skill(fields["skill_assessment_scores"])

    tenure_signal = score_tenure_signal(fields["avg_tenure_per_employer"])
    pre_llm_signal = score_pre_llm_signal(fields["pre_llm_production_ml"])
    sweet_spot_signal = score_sweet_spot_signal(fields["in_sweet_spot"])
    tech_depth_caveat = score_tech_depth_caveat(tech_cat)

    verb_gerund = VERB_GERUND_MAP[select_verb(cid)]
    s3_parts = _compute_s3_parts(fields, tech_cat)

    s1_raw = assemble_sentence_1(
        name=fields["name"],
        years_label=years_label,
        experience_type=experience_type,
        company_scope=company_scope,
        verb_gerund=verb_gerund,
        system_type=system_type,
        company_0=fields["current_company"],
        metric=metric,
        desc_tech=desc_tech,
        verified_skill=verified_skill,
        named_tech=named_tech,
        tech_cat=tech_cat,
        surface_company_list=surface_company_list,
    )

    s2_raw = assemble_sentence_2(
        career_characterization=career_characterization,
        tenure_signal=tenure_signal,
        pre_llm_signal=pre_llm_signal,
        disqualifier_statement=disqualifier_statement,
        sweet_spot_signal=sweet_spot_signal,
        tech_depth_caveat=tech_depth_caveat,
    )

    s3_raw = assemble_sentence_3(
        activity_signal=s3_parts["activity_signal"],
        notice_label=s3_parts["notice_label"],
        availability_assessment=s3_parts["availability_assessment"],
        response_signal=s3_parts["response_signal"],
        github_signal=s3_parts["github_signal"],
        applications_signal=s3_parts["applications_signal"],
        outreach_recommendation=s3_parts["outreach_recommendation"],
    )

    return {
        "candidate_id": cid,
        "tech_cat": tech_cat,
        "s1_raw": s1_raw,
        "s2_raw": s2_raw,
        "s3_raw": s3_raw,
        "temperature_s1": pick_temperature(cid, "s1"),
        "temperature_s2": pick_temperature(cid, "s2"),
        "temperature_s3": pick_temperature(cid, "s3"),
    }


def reconstruct_reasoning(s1: str, s2: str, s3: str) -> str:
    return s1.rstrip(".") + ". " + s2.rstrip(".") + ". " + s3.rstrip(".") + "."


def paraphrase_and_reconstruct(
    raw: dict[str, Any],
    paraphrase_fn: Callable[[str, float], str],
) -> dict[str, Any]:
    s1_paraphrased = paraphrase_fn(raw["s1_raw"], raw["temperature_s1"])
    s2_paraphrased = paraphrase_fn(raw["s2_raw"], raw["temperature_s2"])
    s3_paraphrased = paraphrase_fn(raw["s3_raw"], raw["temperature_s3"])
    reasoning = reconstruct_reasoning(s1_paraphrased, s2_paraphrased, s3_paraphrased)
    return {
        **raw,
        "s1_paraphrased": s1_paraphrased,
        "s2_paraphrased": s2_paraphrased,
        "s3_paraphrased": s3_paraphrased,
        "s1_selected": s1_paraphrased,
        "s2_selected": s2_paraphrased,
        "s3_selected": s3_paraphrased,
        "reasoning": reasoning,
    }


def merge_precomputed_raw(
    candidate: dict[str, Any],
    precomputed: dict[str, Any] | None,
) -> dict[str, Any]:
    """Use cached s1/s2 when available; always refresh s3."""
    if precomputed is None:
        return build_raw_sentences(candidate)

    tech_cat = precomputed.get("tech_cat") or calculate_tech_cat(
        float((candidate.get("pipeline") or {}).get("retrieval_scores", {}).get("cross_encoder_score", 0))
    )
    cid = str(candidate["candidate_id"])
    raw = {
        "candidate_id": cid,
        "tech_cat": tech_cat,
        "s1_raw": str(precomputed["s1_raw"]),
        "s2_raw": str(precomputed["s2_raw"]),
        "s3_raw": build_s3_raw(candidate, tech_cat),
        "temperature_s1": float(precomputed.get("temperature_s1", pick_temperature(cid, "s1"))),
        "temperature_s2": float(precomputed.get("temperature_s2", pick_temperature(cid, "s2"))),
        "temperature_s3": float(precomputed.get("temperature_s3", pick_temperature(cid, "s3"))),
    }
    return raw


def build_reasoning(candidate: dict, paraphrase_fn: Callable[[str, float], str]) -> dict:
    """Build raw sentences, paraphrase once per slot, reconstruct final reasoning."""
    raw = build_raw_sentences(candidate)
    return paraphrase_and_reconstruct(raw, paraphrase_fn)
