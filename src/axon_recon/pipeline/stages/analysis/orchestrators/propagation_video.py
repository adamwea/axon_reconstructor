"""analysis.propagation_video phase orchestrator.

Slice 2 of `analysis_propagation_video_plan.md`: scaffold-only — the
phase is wired into the analysis stage with `enabled: false` defaults
in both runtime YAMLs; per-target invocation returns a noop marker
indicating the phase isn't implemented yet.

Slice 3 will add the inputs resolver; slice 4 ports the core impl from
the archeology audit (`dev/notes/refs/propagation_video_audit.md`)
that wraps `axon_velocity.plotting.play_template_map` with cropping +
clipping + colorbar overlays.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from ....output_paths import compute_mea_analysis_output_dir
from .compute_metrics import (
	_print_analysis_aggregate,
	_target_datasets_override_from_args,
)


LOGGER = logging.getLogger("axon_recon.analysis.propagation_video")


def _stage_output_root(
	*,
	mea_output_root: Path,
	h5_path: Path,
	well_id: str,
	output_rel_root: str,
) -> Path:
	well_dir = compute_mea_analysis_output_dir(
		output_root=Path(mea_output_root),
		data_file=Path(h5_path),
		well=str(well_id),
	)
	return well_dir / str(output_rel_root)


def _per_target_summary_path(stage_output_root: Path) -> Path:
	return stage_output_root / "context" / "propagation_video_summary.json"


def _write_phase_summary(
	*,
	target_summary_path: Path,
	status: str,
	well_id: str,
	dataset_index: int,
	reason: str,
) -> dict[str, Any]:
	payload: dict[str, Any] = {
		"phase": "propagation_video",
		"status": status,
		"dataset_index": int(dataset_index),
		"well_id": str(well_id),
		"reason": reason,
	}
	target_summary_path.parent.mkdir(parents=True, exist_ok=True)
	target_summary_path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
	)
	return payload


def run_analysis_propagation_video(
	*,
	dataset_index: int,
	dataset_id: str | None,
	h5_path: Path,
	stream_id: str,
	mea_output_root: Path,
	output_rel_root: str,
	stage_config: Any,
	force_restart: bool,
) -> dict[str, Any]:
	"""Per-target entry point for the analysis.propagation_video phase.

	Slice 2 scaffold: returns a noop marker when disabled (the YAML
	default) OR a "skipped: not_implemented_yet" marker when enabled
	but the core impl (slice 4) hasn't shipped. Subsequent slices
	replace this body with the real video-generation logic.
	"""

	stage_output_root = _stage_output_root(
		mea_output_root=mea_output_root,
		h5_path=h5_path,
		well_id=str(stream_id),
		output_rel_root=output_rel_root,
	)
	target_summary_path = _per_target_summary_path(stage_output_root)

	enabled = bool(getattr(stage_config, "propagation_video_enabled", False))
	if not enabled:
		return _write_phase_summary(
			target_summary_path=target_summary_path,
			status="noop",
			well_id=str(stream_id),
			dataset_index=int(dataset_index),
			reason="phase disabled in YAML",
		)

	# Slice 2 scaffold: when enabled, surface that the core impl is
	# still pending. Slice 4 replaces this branch with the real video
	# generation pipeline.
	return _write_phase_summary(
		target_summary_path=target_summary_path,
		status="skipped",
		well_id=str(stream_id),
		dataset_index=int(dataset_index),
		reason="not_implemented_yet (slice 4 of analysis_propagation_video_plan)",
	)


def _run_propagation_video_from_args(args: argparse.Namespace) -> int:
	"""CLI handler for `axon-recon stages analysis.propagation_video`."""

	from ....runner import run_analysis_propagation_video_from_runtime

	target_datasets_override = _target_datasets_override_from_args(args)
	return _print_analysis_aggregate(
		run_analysis_propagation_video_from_runtime(
			config_path=str(args.config),
			limit_segments_override=getattr(args, "limit_segments", None),
			limit_datasets_override=getattr(args, "limit_datasets", None),
			target_datasets_override=target_datasets_override,
			limit_wells_per_dataset_override=getattr(args, "limit_wells_per_dataset", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			replot_override=(True if bool(getattr(args, "replot", False)) else None),
			task_allocation_override=getattr(args, "task_allocation_override", None),
		)
	)


__all__ = ["run_analysis_propagation_video", "_run_propagation_video_from_args"]
