#!/usr/bin/env python3
from __future__ import annotations

import json
import importlib
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from axon_reconstructor.pipeline.stg4_templates.extraction import _gather_template_sources_for_unit
from axon_reconstructor.pipeline.stg4_templates.multi_source_utils import (
    _get_unit_template_from_extension,
    _load_waveforms_analyzers,
    _sparsity_unit_channel_indices,
    _try_get_electrode_ids,
)
from axon_reconstructor.pipeline.stg4_templates.overlaps import (
    _resolve_waveforms_channel_axis_index,
    _try_get_waveforms_one_unit,
)
from axon_reconstructor.pipeline.stg4_templates.utils import (
    _build_merged_contributing_template_for_unit,
)


# -----------------------------------------------------------------------------
# Hardcoded probe target for VS Code Run/Debug (no CLI args required).
# Edit these constants directly if you want to inspect a different run/unit.
# -----------------------------------------------------------------------------
PROBE_UNIT_ID = 94
PROBE_WELL_OUT_DIR = Path(
    "/home/adamm/dev/symlinks/local_RBS_data/outputs/Media_Density_T5_02182026_AR/260313/M07036/AxonTracking/000147/well001"
)
PROBE_WAVEFORMS_VARIANT_NAME = ""
PROBE_INCLUDE_CONCAT = True
PROBE_INCLUDE_SEGMENTS = True
PROBE_MAX_SPIKES_PER_CONTRIBUTION = 500
PROBE_DEBUG = True

# If None, writes to tools/debug/logs/unit_<id>_overlap_probe_report.json
PROBE_REPORT_JSON: Optional[Path] = None


@dataclass(frozen=True)
class ContributionDiag:
    source_name: str
    unit_id: Any
    channel_ref: Any
    waveforms_shape: Any
    status: str
    detail: str
    channel_axis_index: Optional[int] = None
    selected_n_spikes: Optional[int] = None
    selected_n_samples: Optional[int] = None


@dataclass(frozen=True)
class MergeCallDiag:
    n_contributions: int
    merged_is_none: bool
    merged_n_samples: Optional[int]
    contribution_diagnostics: list[dict[str, Any]]


def _build_logger(debug: bool) -> logging.Logger:
    logger = logging.getLogger("debug_templates_overlap_probe")
    logger.handlers = []
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if debug else logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)
    return logger


def _diagnose_contribution(*, c: Any, max_spikes_per_contribution: int) -> ContributionDiag:
    try:
        wfs_raw = _try_get_waveforms_one_unit(analyzer=c.analyzer, unit_id=c.unit_id)
    except Exception as exc:
        return ContributionDiag(
            source_name=str(c.source_name),
            unit_id=c.unit_id,
            channel_ref=c.channel_ref,
            waveforms_shape=None,
            status="error",
            detail=f"waveforms_load_exception:{exc}",
        )

    if wfs_raw is None:
        return ContributionDiag(
            source_name=str(c.source_name),
            unit_id=c.unit_id,
            channel_ref=c.channel_ref,
            waveforms_shape=None,
            status="skipped",
            detail="no_waveforms",
        )

    try:
        wfs = np.asarray(wfs_raw)
    except Exception as exc:
        return ContributionDiag(
            source_name=str(c.source_name),
            unit_id=c.unit_id,
            channel_ref=c.channel_ref,
            waveforms_shape=None,
            status="skipped",
            detail=f"waveforms_to_array_failed:{exc}",
        )

    if wfs.ndim != 3:
        return ContributionDiag(
            source_name=str(c.source_name),
            unit_id=c.unit_id,
            channel_ref=c.channel_ref,
            waveforms_shape=tuple(wfs.shape),
            status="skipped",
            detail="waveforms_not_3d",
        )

    n_spikes, n_samples, n_ch = [int(x) for x in wfs.shape]
    if n_spikes <= 0 or n_samples <= 0 or n_ch <= 0:
        return ContributionDiag(
            source_name=str(c.source_name),
            unit_id=c.unit_id,
            channel_ref=c.channel_ref,
            waveforms_shape=tuple(wfs.shape),
            status="skipped",
            detail="waveforms_invalid_dims",
        )

    ch_i = _resolve_waveforms_channel_axis_index(
        analyzer=c.analyzer,
        unit_id=c.unit_id,
        channel_ref=c.channel_ref,
    )
    if ch_i is None or int(ch_i) < 0 or int(ch_i) >= n_ch:
        return ContributionDiag(
            source_name=str(c.source_name),
            unit_id=c.unit_id,
            channel_ref=c.channel_ref,
            waveforms_shape=tuple(wfs.shape),
            status="skipped",
            detail="channel_ref_unresolved_or_oob",
            channel_axis_index=(None if ch_i is None else int(ch_i)),
        )

    selected_n_spikes = n_spikes
    if max_spikes_per_contribution >= 0:
        selected_n_spikes = min(int(max_spikes_per_contribution), n_spikes)

    return ContributionDiag(
        source_name=str(c.source_name),
        unit_id=c.unit_id,
        channel_ref=c.channel_ref,
        waveforms_shape=tuple(wfs.shape),
        status="accepted",
        detail="ok",
        channel_axis_index=int(ch_i),
        selected_n_spikes=int(selected_n_spikes),
        selected_n_samples=int(n_samples),
    )


def _summarize_reasons_from_merge_calls(calls: list[MergeCallDiag]) -> dict[str, int]:
    counts: dict[str, int] = {
        "merge_returned_none:no_accepted_contributions": 0,
        "merge_returned_none:accepted_contributions_present": 0,
        "merge_succeeded": 0,
    }
    for call in calls:
        accepted = sum(1 for d in call.contribution_diagnostics if d.get("status") == "accepted")
        if call.merged_is_none:
            if accepted == 0:
                counts["merge_returned_none:no_accepted_contributions"] += 1
            else:
                counts["merge_returned_none:accepted_contributions_present"] += 1
        else:
            counts["merge_succeeded"] += 1
    return counts


def main() -> int:
    logger = _build_logger(bool(PROBE_DEBUG))

    well_out_dir = Path(PROBE_WELL_OUT_DIR).expanduser().resolve()
    waveforms_dirname = "stg3_waveforms_outputs" + (
        f"_{PROBE_WAVEFORMS_VARIANT_NAME}" if str(PROBE_WAVEFORMS_VARIANT_NAME).strip() else ""
    )
    unit_id = int(PROBE_UNIT_ID)

    if not well_out_dir.exists():
        raise FileNotFoundError(f"Probe well output dir not found: {well_out_dir}")

    logger.info("well_out_dir=%s", well_out_dir)
    logger.info("waveforms_dirname=%s", waveforms_dirname)
    logger.info("unit_id=%s", unit_id)

    analyzers = _load_waveforms_analyzers(
        well_out_dir=well_out_dir,
        waveforms_dirname=waveforms_dirname,
        include_concat=bool(PROBE_INCLUDE_CONCAT),
        include_segments=bool(PROBE_INCLUDE_SEGMENTS),
        logger=logger,
    )

    sources_for_unit = _gather_template_sources_for_unit(
        uid=unit_id,
        analyzers=analyzers,
        get_template_from_extension=_get_unit_template_from_extension,
        sparsity_unit_channel_indices=_sparsity_unit_channel_indices,
        try_get_electrode_ids=_try_get_electrode_ids,
    )
    if not sources_for_unit:
        raise RuntimeError(f"No template sources found for unit {unit_id}")

    n_samples = int(np.asarray(sources_for_unit[0]["template"]).shape[0])

    # Run the real production merge function, but record which overlap keys were
    # successfully resolved by monkeypatching the imported helper it uses.
    # `_build_merged_contributing_template_for_unit` imports this helper from
    # `axon_reconstructor.pipeline.templates.overlaps` at runtime.
    try:
        overlaps_module = importlib.import_module("axon_reconstructor.pipeline.templates.overlaps")
    except Exception:
        overlaps_module = importlib.import_module("axon_reconstructor.pipeline.stg4_templates.overlaps")

    original_mean = overlaps_module.mean_waveform_from_contributions
    merge_calls: list[MergeCallDiag] = []

    def wrapped_mean_waveform_with_diagnostics(*, contributions, logger=None, **kwargs):
        contrib_diags = [
            asdict(_diagnose_contribution(c=c, max_spikes_per_contribution=int(PROBE_MAX_SPIKES_PER_CONTRIBUTION)))
            for c in list(contributions or [])
        ]
        merged = original_mean(contributions=contributions, logger=logger, **kwargs)
        merged_n_samples = None
        if merged is not None:
            try:
                merged_n_samples = int(np.asarray(merged).shape[0])
            except Exception:
                merged_n_samples = None
        merge_calls.append(
            MergeCallDiag(
                n_contributions=int(len(list(contributions or []))),
                merged_is_none=bool(merged is None),
                merged_n_samples=merged_n_samples,
                contribution_diagnostics=contrib_diags,
            )
        )
        return merged

    overlaps_module.mean_waveform_from_contributions = wrapped_mean_waveform_with_diagnostics
    try:
        merged = _build_merged_contributing_template_for_unit(
            sources_for_unit=sources_for_unit,
            unit_id=unit_id,
            logger=logger,
        )
    finally:
        overlaps_module.mean_waveform_from_contributions = original_mean

    if merged is None:
        raise RuntimeError("Production merged_contributing returned None")

    stats = dict(merged.get("stats", {}))
    overlap_encountered = int(stats.get("overlap_encountered", 0) or 0)
    overlap_resolved = int(stats.get("overlap_resolved", 0) or 0)
    overlap_unresolved = max(0, overlap_encountered - overlap_resolved)

    reason_counts = _summarize_reasons_from_merge_calls(merge_calls)

    logger.info(
        "unit=%s sources=%d overlap_encountered=%d overlap_resolved=%d overlap_unresolved=%d merge_call_count=%d",
        unit_id,
        int(len(sources_for_unit)),
        overlap_encountered,
        overlap_resolved,
        overlap_unresolved,
        int(len(merge_calls)),
    )

    report = {
        "unit_id": int(unit_id),
        "well_out_dir": str(well_out_dir),
        "waveforms_dirname": str(waveforms_dirname),
        "n_sources": int(len(sources_for_unit)),
        "n_samples": int(n_samples),
        "resolved_overlap_keys": int(overlap_resolved),
        "unresolved_overlap_keys": int(overlap_unresolved),
        "unresolved_reason_counts": reason_counts,
        "merged_stats": stats,
        "resolved_overlap_details": merged.get("overlap", {}),
        "mean_waveform_merge_calls": [asdict(c) for c in merge_calls],
    }

    report_path = PROBE_REPORT_JSON
    if report_path is None:
        report_dir = Path(__file__).resolve().parent / "logs"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"unit_{int(unit_id)}_overlap_probe_report.json"

    report_path = Path(report_path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    logger.info("resolved_overlap_keys=%d unresolved_overlap_keys=%d", int(overlap_resolved), int(overlap_unresolved))
    logger.info("unresolved_reason_counts=%s", reason_counts)
    logger.info("Wrote report: %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
