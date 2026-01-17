from __future__ import annotations

import sys
from pathlib import Path


# Ensure the src/ layout package is importable when the project is not installed.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
