"""Re-export Stage 6 reasoning builder for experiment harness."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tracks.instructor.stage6.reasoning_builder import *  # noqa: F403
