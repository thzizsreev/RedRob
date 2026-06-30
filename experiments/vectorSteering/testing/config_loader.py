"""Load and validate test harness YAML config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EncoderConfig:
    model_name: str


@dataclass(frozen=True)
class DecoderConfig:
    embedder_id: str
    num_steps: int
    sequence_beam_width: int


@dataclass(frozen=True)
class PhaseASentence:
    id: str
    text: str


@dataclass(frozen=True)
class PhaseAConfig:
    decode_repeats: int
    sentences: list[PhaseASentence]


@dataclass(frozen=True)
class PhaseBConfig:
    sweep_repeats: int
    s_values: list[float]
    anchors: dict[str, str]


@dataclass(frozen=True)
class TestConfig:
    encoder: EncoderConfig
    decoder: DecoderConfig
    phase_a: PhaseAConfig
    phase_b: PhaseBConfig


def load_config(path: Path) -> TestConfig:
    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    decoder_raw = raw["decoder"]
    phase_a_raw = raw["phase_a"]
    phase_b_raw = raw["phase_b"]

    sentences = [
        PhaseASentence(id=s["id"], text=s["text"].strip())
        for s in phase_a_raw["sentences"]
    ]

    return TestConfig(
        encoder=EncoderConfig(model_name=raw["encoder"]["model_name"]),
        decoder=DecoderConfig(
            embedder_id=decoder_raw["embedder_id"],
            num_steps=int(decoder_raw["num_steps"]),
            sequence_beam_width=int(decoder_raw["sequence_beam_width"]),
        ),
        phase_a=PhaseAConfig(
            decode_repeats=int(phase_a_raw["decode_repeats"]),
            sentences=sentences,
        ),
        phase_b=PhaseBConfig(
            sweep_repeats=int(phase_b_raw["sweep_repeats"]),
            s_values=[float(v) for v in phase_b_raw["s_values"]],
            anchors={k: v.strip() for k, v in phase_b_raw["anchors"].items()},
        ),
    )
