"""Load Stage 2 configuration from config.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml


@dataclass(frozen=True)
class HoneypotConfig:
    duration_overshoot_grace_days: int
    experience_overage_tolerance_years: float
    grad_to_work_buffer_years: int
    max_skill_years_absolute: float
    max_skill_overshoot_factor: float
    min_yoe_for_factor_check: float
    early_career_skill_buffer_years: float


@dataclass(frozen=True)
class ConsultingConfig:
    named_firms: list[str]
    consulting_signal_words: list[str]
    min_employers_to_classify: int
    product_fraction_threshold: float


@dataclass(frozen=True)
class ResearchConfig:
    research_title_signals: list[str]
    academic_employer_signals: list[str]
    production_title_signals: list[str]
    research_heavy_threshold: float
    min_roles_to_classify: int


@dataclass(frozen=True)
class CodingRecencyConfig:
    stale_coding_window_months: int
    management_title_signals: list[str]
    hands_on_title_signals: list[str]


@dataclass(frozen=True)
class ShallowAiConfig:
    llm_era_start_year: int
    recent_ai_window_months: int
    llm_framework_signals: list[str]
    pre_llm_skill_signals: list[str]
    min_llm_framework_skills: int
    ml_title_signals: list[str]


@dataclass(frozen=True)
class CareerShapeConfig:
    short_hop_threshold_years: float


@dataclass(frozen=True)
class LogisticsConfig:
    preferred_locations: list[str]
    acceptable_locations: list[str]
    india_signals: list[str]


@dataclass(frozen=True)
class Stage2Config:
    hard_min: float
    hard_max: float
    soft_tolerance: float
    sweet_low: float
    sweet_high: float
    stuffer_density: float
    expert_zero_threshold: int
    skill_years_slack: float
    stale_days: int
    min_response_rate: float
    enable_isolation_forest: bool
    current_date: date
    expected_input_count: int
    expected_survivor_min: int
    expected_survivor_max: int
    honeypot: HoneypotConfig
    title_families: dict[str, list[str]]
    jd_keywords: list[str]
    consulting: ConsultingConfig
    research: ResearchConfig
    coding_recency: CodingRecencyConfig
    shallow_ai: ShallowAiConfig
    career_shape: CareerShapeConfig
    logistics: LogisticsConfig


def _parse_date(value: str) -> date:
    year, month, day = value.split("-")
    return date(int(year), int(month), int(day))


def _lower_list(items: list) -> list[str]:
    return [str(x).lower() for x in items]


def _default_consulting_config() -> ConsultingConfig:
    return ConsultingConfig(
        named_firms=_lower_list(
            [
                "tcs",
                "tata consultancy",
                "infosys",
                "wipro",
                "accenture",
                "cognizant",
                "capgemini",
                "hcl",
                "tech mahindra",
                "mphasis",
                "hexaware",
                "mindtree",
                "l&t infotech",
                "ltimindtree",
                "niit technologies",
                "patni",
                "mastech",
                "kforce",
                "igate",
            ]
        ),
        consulting_signal_words=_lower_list(
            ["consulting", "consultancy", "outsourcing", "staffing", "it services"]
        ),
        min_employers_to_classify=2,
        product_fraction_threshold=0.0,
    )


def _default_research_config() -> ResearchConfig:
    return ResearchConfig(
        research_title_signals=_lower_list(
            [
                "phd",
                "postdoc",
                "post-doctoral",
                "doctoral",
                "research intern",
                "research fellow",
                "research assistant",
                "research associate",
                "principal investigator",
                "professor",
                "lecturer",
                "teaching assistant",
                "graduate researcher",
            ]
        ),
        academic_employer_signals=_lower_list(
            [
                "university",
                "college",
                "institute of technology",
                "iit",
                "iisc",
                "research lab",
                "research institute",
                "national lab",
            ]
        ),
        production_title_signals=_lower_list(
            [
                "engineer",
                "developer",
                "sde",
                "mle",
                "applied scientist",
                "architect",
                "tech lead",
                "staff",
            ]
        ),
        research_heavy_threshold=0.8,
        min_roles_to_classify=1,
    )


def _default_coding_recency_config() -> CodingRecencyConfig:
    return CodingRecencyConfig(
        stale_coding_window_months=18,
        management_title_signals=_lower_list(
            [
                "chief architect",
                "enterprise architect",
                "solution architect",
                "engineering manager",
                "vp engineering",
                "director of engineering",
                "head of engineering",
                "cto",
                "technical director",
                "principal architect",
            ]
        ),
        hands_on_title_signals=_lower_list(
            [
                "software engineer",
                "ml engineer",
                "machine learning engineer",
                "applied scientist",
                "research engineer",
                "sde",
                "data engineer",
                "backend engineer",
                "staff engineer",
                "principal engineer",
            ]
        ),
    )


def _default_shallow_ai_config() -> ShallowAiConfig:
    return ShallowAiConfig(
        llm_era_start_year=2022,
        recent_ai_window_months=12,
        llm_framework_signals=_lower_list(
            [
                "langchain",
                "llamaindex",
                "llama-index",
                "autogpt",
                "crewai",
                "openai api",
                "anthropic api",
                "azure openai",
                "chatgpt api",
            ]
        ),
        pre_llm_skill_signals=_lower_list(
            [
                "scikit-learn",
                "sklearn",
                "xgboost",
                "lightgbm",
                "word2vec",
                "fasttext",
                "spacy",
                "nltk",
                "traditional nlp",
                "recommendation system",
                "collaborative filtering",
                "classical ml",
                "feature engineering",
                "pytorch",
                "tensorflow",
            ]
        ),
        min_llm_framework_skills=2,
        ml_title_signals=_lower_list(
            [
                "ml",
                "machine learning",
                "ai engineer",
                "data scientist",
                "applied scientist",
                "research engineer",
                "nlp",
            ]
        ),
    )


def _default_career_shape_config() -> CareerShapeConfig:
    return CareerShapeConfig(short_hop_threshold_years=1.5)


def _default_logistics_config() -> LogisticsConfig:
    return LogisticsConfig(
        preferred_locations=_lower_list(
            [
                "noida",
                "pune",
                "delhi",
                "delhi ncr",
                "ncr",
                "gurgaon",
                "gurugram",
                "faridabad",
                "new delhi",
            ]
        ),
        acceptable_locations=_lower_list(
            ["hyderabad", "mumbai", "bangalore", "bengaluru", "chennai", "kolkata"]
        ),
        india_signals=_lower_list(["india"]),
    )


def _load_consulting(raw: dict | None) -> ConsultingConfig:
    default = _default_consulting_config()
    if not raw:
        return default
    return ConsultingConfig(
        named_firms=_lower_list(raw.get("named_firms", default.named_firms)),
        consulting_signal_words=_lower_list(
            raw.get("consulting_signal_words", default.consulting_signal_words)
        ),
        min_employers_to_classify=int(
            raw.get("min_employers_to_classify", default.min_employers_to_classify)
        ),
        product_fraction_threshold=float(
            raw.get("product_fraction_threshold", default.product_fraction_threshold)
        ),
    )


def _load_research(raw: dict | None) -> ResearchConfig:
    default = _default_research_config()
    if not raw:
        return default
    return ResearchConfig(
        research_title_signals=_lower_list(
            raw.get("research_title_signals", default.research_title_signals)
        ),
        academic_employer_signals=_lower_list(
            raw.get("academic_employer_signals", default.academic_employer_signals)
        ),
        production_title_signals=_lower_list(
            raw.get("production_title_signals", default.production_title_signals)
        ),
        research_heavy_threshold=float(
            raw.get("research_heavy_threshold", default.research_heavy_threshold)
        ),
        min_roles_to_classify=int(
            raw.get("min_roles_to_classify", default.min_roles_to_classify)
        ),
    )


def _load_coding_recency(raw: dict | None) -> CodingRecencyConfig:
    default = _default_coding_recency_config()
    if not raw:
        return default
    return CodingRecencyConfig(
        stale_coding_window_months=int(
            raw.get("stale_coding_window_months", default.stale_coding_window_months)
        ),
        management_title_signals=_lower_list(
            raw.get("management_title_signals", default.management_title_signals)
        ),
        hands_on_title_signals=_lower_list(
            raw.get("hands_on_title_signals", default.hands_on_title_signals)
        ),
    )


def _load_shallow_ai(raw: dict | None) -> ShallowAiConfig:
    default = _default_shallow_ai_config()
    if not raw:
        return default
    return ShallowAiConfig(
        llm_era_start_year=int(raw.get("llm_era_start_year", default.llm_era_start_year)),
        recent_ai_window_months=int(
            raw.get("recent_ai_window_months", default.recent_ai_window_months)
        ),
        llm_framework_signals=_lower_list(
            raw.get("llm_framework_signals", default.llm_framework_signals)
        ),
        pre_llm_skill_signals=_lower_list(
            raw.get("pre_llm_skill_signals", default.pre_llm_skill_signals)
        ),
        min_llm_framework_skills=int(
            raw.get("min_llm_framework_skills", default.min_llm_framework_skills)
        ),
        ml_title_signals=_lower_list(
            raw.get("ml_title_signals", default.ml_title_signals)
        ),
    )


def _load_career_shape(raw: dict | None) -> CareerShapeConfig:
    default = _default_career_shape_config()
    if not raw:
        return default
    return CareerShapeConfig(
        short_hop_threshold_years=float(
            raw.get("short_hop_threshold_years", default.short_hop_threshold_years)
        ),
    )


def _load_logistics(raw: dict | None) -> LogisticsConfig:
    default = _default_logistics_config()
    if not raw:
        return default
    return LogisticsConfig(
        preferred_locations=_lower_list(
            raw.get("preferred_locations", default.preferred_locations)
        ),
        acceptable_locations=_lower_list(
            raw.get("acceptable_locations", default.acceptable_locations)
        ),
        india_signals=_lower_list(raw.get("india_signals", default.india_signals)),
    )


def load_stage2_config(config_path: Path) -> Stage2Config:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw or "stage2" not in raw:
        raise ValueError(f"Missing 'stage2' namespace in {config_path}")

    s2 = raw["stage2"]
    hp = s2.get("honeypot", {})

    return Stage2Config(
        hard_min=float(s2["hard_min"]),
        hard_max=float(s2["hard_max"]),
        soft_tolerance=float(s2["soft_tolerance"]),
        sweet_low=float(s2["sweet_low"]),
        sweet_high=float(s2["sweet_high"]),
        stuffer_density=float(s2["stuffer_density"]),
        expert_zero_threshold=int(s2["expert_zero_threshold"]),
        skill_years_slack=float(s2["skill_years_slack"]),
        stale_days=int(s2["stale_days"]),
        min_response_rate=float(s2["min_response_rate"]),
        enable_isolation_forest=bool(s2.get("enable_isolation_forest", False)),
        current_date=_parse_date(str(s2["current_date"])),
        expected_input_count=int(s2.get("expected_input_count", 6000)),
        expected_survivor_min=int(s2.get("expected_survivor_min", 2000)),
        expected_survivor_max=int(s2.get("expected_survivor_max", 5000)),
        honeypot=HoneypotConfig(
            duration_overshoot_grace_days=int(hp.get("duration_overshoot_grace_days", 30)),
            experience_overage_tolerance_years=float(
                hp.get("experience_overage_tolerance_years", 2)
            ),
            grad_to_work_buffer_years=int(hp.get("grad_to_work_buffer_years", 1)),
            max_skill_years_absolute=float(hp.get("max_skill_years_absolute", 30)),
            max_skill_overshoot_factor=float(hp.get("max_skill_overshoot_factor", 1.4)),
            min_yoe_for_factor_check=float(hp.get("min_yoe_for_factor_check", 3)),
            early_career_skill_buffer_years=float(
                hp.get("early_career_skill_buffer_years", 3)
            ),
        ),
        title_families={
            k: [str(x).lower() for x in v]
            for k, v in s2.get("title_families", {}).items()
        },
        jd_keywords=[str(k).lower() for k in s2.get("jd_keywords", [])],
        consulting=_load_consulting(s2.get("consulting")),
        research=_load_research(s2.get("research")),
        coding_recency=_load_coding_recency(s2.get("coding_recency")),
        shallow_ai=_load_shallow_ai(s2.get("shallow_ai")),
        career_shape=_load_career_shape(s2.get("career_shape")),
        logistics=_load_logistics(s2.get("logistics")),
    )
