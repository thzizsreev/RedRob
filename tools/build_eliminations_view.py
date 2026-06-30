#!/usr/bin/env python3
"""Convenience wrapper — see tools/eliminations_view/build.py."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TOOLS = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from eliminations_view.build import main

if __name__ == "__main__":
    main()
