"""Waveforms step.

This package contains waveform extraction, curation, exclusion logging/artifacts,
and plotting.

Primary entry points are re-exported from .main for compatibility.
"""

from __future__ import annotations

from .main import (
    WAVEFORMS_OUTPUTS_DIRNAME,
    WaveformExtractInputs,
    WaveformExtractOutputs,
    extract_waveforms,
)

from .exclusions import (  # noqa: F401
    WF_EXCLUSIONS_NPZ_NAME,
    TemplateFromWaveformsResult,
    compute_unit_template_from_waveforms,
    init_wf_exclusion_report,
    load_wf_exclusions_by_source,
    normalize_unit_id,
    update_wf_exclusion_report,
    write_wf_exclusions_npz,
)

__all__ = [
    "WAVEFORMS_OUTPUTS_DIRNAME",
    "WaveformExtractInputs",
    "WaveformExtractOutputs",
    "extract_waveforms",
    "WF_EXCLUSIONS_NPZ_NAME",
    "TemplateFromWaveformsResult",
    "compute_unit_template_from_waveforms",
    "init_wf_exclusion_report",
    "load_wf_exclusions_by_source",
    "normalize_unit_id",
    "update_wf_exclusion_report",
    "write_wf_exclusions_npz",
]
