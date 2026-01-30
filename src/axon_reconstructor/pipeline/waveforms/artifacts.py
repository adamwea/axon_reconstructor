from __future__ import annotations

from pathlib import Path
from typing import Any

from .reporting import _write_wf_rejection_log_xlsx
from .utils import _write_json


def _write_waveform_extraction_params(
    *,
    params_json: Path,
    inputs,
    sorter_output_dir: Path,
    window,
) -> None:
    _write_json(
        params_json,
        {
            "h5_path": str(inputs.h5_path),
            "stream_id": inputs.stream_id,
            "sorter": inputs.sorter,
            "sorter_output_dir": str(sorter_output_dir),
            "ms_before": float(window.ms_before),
            "ms_after": float(window.ms_after),
            "n_jobs": int(inputs.n_jobs),
            "max_spikes_per_unit": inputs.max_spikes_per_unit,
            "per_segment": bool(inputs.per_segment),
            "per_segment_recording_source": "raw_maxwell_full_channels" if inputs.per_segment else None,
            "per_segment_only_additional_channels": bool(inputs.per_segment_only_additional_channels),
        },
    )


def _persist_filtering_and_exclusions(
    *,
    filtering_json: Path,
    waveforms_out_dir: Path,
    wf_rejection_rows: list[dict[str, Any]],
    filtering_summary: dict[str, Any],
    inputs,
    logger: Any,
) -> None:
    _write_json(filtering_json, filtering_summary)

    try:
        wf_rejection_log_xlsx = waveforms_out_dir / "wf_rejection_log.xlsx"
        _write_wf_rejection_log_xlsx(
            wf_rejection_log_xlsx=wf_rejection_log_xlsx,
            rows=wf_rejection_rows,
            force_restart=bool(inputs.force_restart),
            logger=logger,
        )
    except Exception as e:
        logger.warning("Failed to write wf_rejection_log.xlsx: %s", e)

    try:
        from .exclusions import WF_EXCLUSIONS_NPZ_NAME, write_wf_exclusions_npz

        write_wf_exclusions_npz(
            wf_exclusions_npz=waveforms_out_dir / WF_EXCLUSIONS_NPZ_NAME,
            rows=wf_rejection_rows,
            force_restart=bool(inputs.force_restart),
            logger=logger,
        )
    except Exception as e:
        logger.warning("Failed to write wf_exclusions.npz: %s", e)


__all__ = [
    "_persist_filtering_and_exclusions",
    "_write_waveform_extraction_params",
]
