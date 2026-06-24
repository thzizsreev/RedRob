"""Stage 5 — deterministic composite scorer."""

from tracks.instructor.stage5.config import Stage5Config, load_stage5_config
from tracks.instructor.stage5.score import Stage5Result, print_stage5_summary, run

__all__ = [
    "Stage5Config",
    "Stage5Result",
    "load_stage5_config",
    "print_stage5_summary",
    "run",
]
