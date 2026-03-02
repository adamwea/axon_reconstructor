"""Raw preprocessing for Maxwell `.h5` recordings.

This package is a refactor of the original monolithic module
`axon_reconstructor.pipeline.raw_preprocessing`.

We split responsibilities into:
- `main.py`: main preprocessing + concatenation logic
- `h5_helpers.py`: HDF5/assay timing and metadata debugging helpers
- `plotting.py`: plotting and electrode-layout utilities

The package re-exports the primary public entry points used by the
preprocess stage service.
"""

from __future__ import annotations

from .main import (
    RawPreprocessPlan,
    build_concatenated_recording,
    build_preprocess_plan,
    discover_cfg_files,
    find_common_electrodes_from_segments,
    parse_cfg_channel_locations,
    run_preprocess_stage,
)

__all__ = [
    "RawPreprocessPlan",
    "discover_cfg_files",
    "parse_cfg_channel_locations",
    "build_preprocess_plan",
    "find_common_electrodes_from_segments",
    "build_concatenated_recording",
    "run_preprocess_stage",
]
