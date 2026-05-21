"""analysis.propagation_video phase orchestrator.

Slice 7 of `analysis_propagation_video_plan.md`: per-target orchestrator
fans out across the target's units, calls the slice-3 inputs resolver +
slice-4 renderer for each, and writes a per-target summary marker
aggregating per-unit results.

Soft-imports `axon_velocity` at render-time via the slice-4 renderer's
soft-import gate, so the analysis stage's other phases keep loading
cleanly when this optional dep isn't available.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from ....output_paths import compute_mea_analysis_output_dir
from ..core.propagation_video_inputs import (
	PropagationVideoInputsMissing,
	discover_unit_ids_for_target,
	resolve_propagation_video_inputs,
)
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


def _phase_video_output_dir(
	*,
	stage_output_root: Path,
	rel_output_root: str,
) -> Path:
	"""Per-target video output dir at
	``<analysis_outputs>/<rel_output_root>/`` (one GIF per unit_id)."""

	return stage_output_root / str(rel_output_root or "propagation_video")


def _write_phase_summary(
	*,
	target_summary_path: Path,
	status: str,
	well_id: str,
	dataset_index: int,
	reason: str,
	units_processed: list[dict[str, Any]] | None = None,
	video_output_dir: Path | None = None,
) -> dict[str, Any]:
	payload: dict[str, Any] = {
		"phase": "propagation_video",
		"status": status,
		"dataset_index": int(dataset_index),
		"well_id": str(well_id),
		"reason": reason,
	}
	if units_processed is not None:
		payload["units_processed"] = list(units_processed)
		payload["n_units_processed"] = int(len(units_processed))
		payload["n_units_ok"] = int(sum(1 for u in units_processed if u.get("status") == "ok"))
		payload["n_units_skipped"] = int(sum(1 for u in units_processed if u.get("status") == "skipped"))
		payload["n_units_error"] = int(sum(1 for u in units_processed if u.get("status") == "error"))
	if video_output_dir is not None:
		payload["video_output_dir"] = str(video_output_dir)
	target_summary_path.parent.mkdir(parents=True, exist_ok=True)
	target_summary_path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
	)
	return payload


def _write_dry_run_summary(
	*,
	target_summary_path: Path,
	dataset_index: int,
	stream_id: str,
	h5_path: Path,
	mea_output_root: Path,
	stage_output_root: Path,
	rel_output_root: str,
) -> dict[str, Any]:
	"""Per slice 6: write `status: dry_run_ok` summary.

	Per `guardrails/dry_run.md`, this:
	- Resolves inputs (paths, prerequisites) the same way a real run
	  would.
	- Validates that prerequisites are present and readable.
	- Skips the expensive render work.
	- Reports findings under `inputs_resolved` / `outputs_would_produce`
	  / `validation.missing_prerequisites`.
	"""

	video_output_dir = stage_output_root / rel_output_root
	unit_ids = discover_unit_ids_for_target(
		well_id=str(stream_id),
		h5_path=h5_path,
		mea_output_root=mea_output_root,
	)
	inputs_resolved: list[dict[str, Any]] = []
	outputs_would_produce: list[dict[str, Any]] = []
	missing_prerequisites: list[str] = []

	for unit_id in unit_ids:
		try:
			inputs = resolve_propagation_video_inputs(
				dataset_index=int(dataset_index),
				well_id=str(stream_id),
				unit_id=int(unit_id),
				h5_path=h5_path,
				mea_output_root=mea_output_root,
				require_exists=False,  # dry-run gathers paths even if missing
			)
		except Exception as exc:  # pragma: no cover — require_exists=False shouldn't raise
			missing_prerequisites.append(
				f"unit_id={unit_id}: input resolution raised {exc!r}"
			)
			continue
		for name, path in (
			("merged_template_npy", inputs.merged_template_npy),
			("merged_locations_npy", inputs.merged_locations_npy),
			("gtr_pkl", inputs.gtr_pkl),
		):
			exists = bool(Path(path).is_file())
			inputs_resolved.append({
				"name": f"unit_{int(unit_id):04d}.{name}",
				"path": str(path),
				"exists": exists,
			})
			if not exists:
				missing_prerequisites.append(
					f"unit_id={unit_id}: {name} missing at {path}"
				)
		outputs_would_produce.append({
			"name": f"unit_{int(unit_id):04d}_video",
			"path": str(video_output_dir / f"unit_{int(unit_id):04d}.gif"),
		})

	payload: dict[str, Any] = {
		"phase": "propagation_video",
		"status": "dry_run_ok",
		"dataset_index": int(dataset_index),
		"well_id": str(stream_id),
		"reason": "dry_run: skipped renders",
		"stage_output_root_dir": str(stage_output_root),
		"video_output_dir": str(video_output_dir),
		"n_units_discovered": len(unit_ids),
		"inputs_resolved": inputs_resolved,
		"outputs_would_produce": outputs_would_produce,
		"validation": {
			"missing_prerequisites": missing_prerequisites,
			"warnings": [],
		},
	}
	target_summary_path.parent.mkdir(parents=True, exist_ok=True)
	target_summary_path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
	)
	return payload


def _resolve_render_callable(stage_config: Any) -> Any:
	"""Honor a test-only `_propagation_video_render_override` on
	stage_config so slice-7 tests can monkey-patch the renderer without
	requiring axon_velocity in the env. Default returns the slice-4
	renderer."""

	override = getattr(stage_config, "_propagation_video_render_override", None)
	if callable(override):
		return override
	from ..core.propagation_video_render import render_unit_propagation_video

	return render_unit_propagation_video


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

	When disabled (YAML default), writes a `noop` per-target marker.
	When enabled, fans out across discovered unit_ids and calls the
	slice-4 renderer for each. Aggregates per-unit results into a
	``units_processed`` list on the per-target summary.
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

	# Slice 6 of analysis_propagation_video_plan: --dry-run support.
	# Per `guardrails/dry_run.md`: resolve inputs + validate
	# prerequisites + skip the expensive render call + write a
	# `status: dry_run_ok` summary with `inputs_resolved` /
	# `outputs_would_produce` / `validation` fields.
	#
	# Honors EITHER the per-stage_config `dry_run` flag (legacy slice-6
	# mechanism) OR the process-wide `get_dry_run_override()` override
	# (the CLI `--dry-run` flag wired in dry_run_rollout slice 1a).
	from ....config import get_dry_run_override

	if bool(getattr(stage_config, "dry_run", False)) or bool(get_dry_run_override()):
		return _write_dry_run_summary(
			target_summary_path=target_summary_path,
			dataset_index=int(dataset_index),
			stream_id=str(stream_id),
			h5_path=h5_path,
			mea_output_root=mea_output_root,
			stage_output_root=stage_output_root,
			rel_output_root=str(
				getattr(stage_config, "propagation_video_rel_output_root", "propagation_video")
				or "propagation_video"
			),
		)

	rel_output_root = str(
		getattr(stage_config, "propagation_video_rel_output_root", "propagation_video")
		or "propagation_video"
	)
	video_output_dir = _phase_video_output_dir(
		stage_output_root=stage_output_root,
		rel_output_root=rel_output_root,
	)

	# Discover units the recon stage produced templates for.
	unit_ids = discover_unit_ids_for_target(
		well_id=str(stream_id),
		h5_path=h5_path,
		mea_output_root=mea_output_root,
	)
	if not unit_ids:
		return _write_phase_summary(
			target_summary_path=target_summary_path,
			status="error",
			well_id=str(stream_id),
			dataset_index=int(dataset_index),
			reason=(
				"no unit_ids discovered under recon-stage merged templates "
				"dir; run `axon-recon stages reconstruct` first"
			),
			units_processed=[],
			video_output_dir=video_output_dir,
		)

	render_callable = _resolve_render_callable(stage_config)

	# Slice 5: per-phase YAML knobs forwarded to every render call.
	render_kwargs: dict[str, Any] = {
		"fps": int(getattr(stage_config, "propagation_video_fps", 20) or 20),
		"skip_frames": int(getattr(stage_config, "propagation_video_skip_frames", 2) or 2),
		"cmap": str(getattr(stage_config, "propagation_video_cmap", "coolwarm") or "coolwarm"),
	}

	units_processed: list[dict[str, Any]] = []
	for unit_id in unit_ids:
		try:
			inputs = resolve_propagation_video_inputs(
				dataset_index=int(dataset_index),
				well_id=str(stream_id),
				unit_id=int(unit_id),
				h5_path=h5_path,
				mea_output_root=mea_output_root,
				require_exists=True,
			)
		except PropagationVideoInputsMissing as exc:
			units_processed.append({
				"unit_id": int(unit_id),
				"status": "error",
				"reason": f"missing inputs: {exc}",
			})
			continue

		out_path = video_output_dir / f"unit_{int(unit_id):04d}.gif"
		try:
			render_result = render_callable(
				inputs=inputs,
				out_path=out_path,
				force_restart=bool(force_restart),
				**render_kwargs,
			)
		except Exception as exc:
			LOGGER.exception(
				"propagation_video render failed unit_id=%s well=%s",
				unit_id,
				stream_id,
			)
			units_processed.append({
				"unit_id": int(unit_id),
				"status": "error",
				"reason": f"render raised: {exc!r}",
				"out_path": str(out_path),
			})
			continue

		units_processed.append({
			"unit_id": int(unit_id),
			"status": str(render_result.get("status", "ok")),
			"reason": str(render_result.get("reason", "rendered")),
			"out_path": str(render_result.get("out_path", out_path)),
			**{
				k: v
				for k, v in render_result.items()
				if k in ("frames", "cmap", "fps", "skip_frames")
			},
		})

	# Aggregate status: ok iff every unit succeeded or was idempotently
	# skipped; error iff any unit's render raised; partial when mixed.
	statuses = {u.get("status") for u in units_processed}
	if statuses == {"ok"} or statuses == {"skipped"} or statuses == {"ok", "skipped"}:
		aggregate_status = "ok"
		aggregate_reason = (
			f"rendered {sum(1 for u in units_processed if u['status'] == 'ok')} / "
			f"skipped {sum(1 for u in units_processed if u['status'] == 'skipped')} / "
			f"{len(units_processed)} units"
		)
	elif "error" in statuses and ("ok" in statuses or "skipped" in statuses):
		aggregate_status = "partial"
		aggregate_reason = (
			f"{sum(1 for u in units_processed if u['status'] == 'error')} unit "
			f"render(s) failed out of {len(units_processed)}"
		)
	else:
		aggregate_status = "error"
		aggregate_reason = "all unit renders failed"

	return _write_phase_summary(
		target_summary_path=target_summary_path,
		status=aggregate_status,
		well_id=str(stream_id),
		dataset_index=int(dataset_index),
		reason=aggregate_reason,
		units_processed=units_processed,
		video_output_dir=video_output_dir,
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
