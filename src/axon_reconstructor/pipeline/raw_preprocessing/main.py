"""Raw preprocessing public API.

The heavy concatenation implementation lives in .runner to keep this module orchestration-only.
"""

from __future__ import annotations

from .concatenation import find_common_electrodes_from_segments
from .planning import RawPreprocessPlan, build_preprocess_plan, discover_cfg_files, parse_cfg_channel_locations
from .runner import build_concatenated_recording

__all__ = [
    "RawPreprocessPlan",
    "discover_cfg_files",
    "parse_cfg_channel_locations",
    "build_preprocess_plan",
    "find_common_electrodes_from_segments",
    "build_concatenated_recording",
]
