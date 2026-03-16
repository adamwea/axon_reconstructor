"""Compatibility wrappers for stg1 H5 timing helpers.

Canonical functional ownership has moved to
`MEA_Analysis.IPNAnalysis.multiseg_utils.preprocess_multiseg_h5.runtime`.
"""

from __future__ import annotations

import importlib
from pathlib import Path


def _read_well_rec_frame_nos_and_trigger_settings(*, h5_path: Path, stream_id: str, rec_name: str):
    mod = importlib.import_module("MEA_Analysis.IPNAnalysis.multiseg_utils.preprocess_multiseg_h5.runtime")
    fn = getattr(mod, "_read_well_rec_frame_nos_and_trigger_settings")
    return fn(h5_path=h5_path, stream_id=stream_id, rec_name=rec_name)


__all__ = [
    "_read_well_rec_frame_nos_and_trigger_settings",
]
