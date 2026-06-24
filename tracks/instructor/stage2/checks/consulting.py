"""Check E — consulting-only career hard remove + career composition features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tracks.instructor.stage2.checks._history import normalize_text, unique_employers
from tracks.instructor.stage2.config import ConsultingConfig, Stage2Config

EmployerType = Literal["consulting", "product", "unknown"]


@dataclass(frozen=True)
class ConsultingResult:
    remove: bool
    reason: str | None
    product_company_count: int
    consulting_company_count: int
    product_company_fraction: float
    career_type: str


def classify_employer(company: str, config: ConsultingConfig) -> EmployerType:
    normalized = normalize_text(company)
    if not normalized:
        return "unknown"

    for firm in config.named_firms:
        if firm in normalized:
            return "consulting"

    for signal in config.consulting_signal_words:
        if signal in normalized:
            return "consulting"

    return "product"


def _derive_career_type(
    product_count: int,
    consulting_count: int,
    total: int,
) -> str:
    if total < 2:
        return "unknown"
    fraction = product_count / total
    if fraction > 0.6:
        return "product_heavy"
    if fraction >= 0.2:
        return "mixed"
    if consulting_count > 0:
        return "consulting_heavy"
    return "unknown"


def evaluate_consulting(record: dict, config: Stage2Config) -> ConsultingResult:
    career_history = record.get("career_history") or []
    employers = unique_employers(career_history)
    total = len(employers)

    product_count = 0
    consulting_count = 0
    unknown_count = 0

    for employer in employers:
        kind = classify_employer(employer, config.consulting)
        if kind == "consulting":
            consulting_count += 1
        elif kind == "product":
            product_count += 1
        else:
            unknown_count += 1

    fraction = product_count / total if total > 0 else 0.0
    career_type = _derive_career_type(product_count, consulting_count, total)

    remove = False
    reason: str | None = None

    if product_count > 0:
        pass
    elif total < config.consulting.min_employers_to_classify:
        pass
    elif unknown_count == total:
        pass
    elif consulting_count >= 1 and product_count == 0:
        remove = True
        reason = "consulting_only_career"

    return ConsultingResult(
        remove=remove,
        reason=reason,
        product_company_count=product_count,
        consulting_company_count=consulting_count,
        product_company_fraction=fraction,
        career_type=career_type,
    )
