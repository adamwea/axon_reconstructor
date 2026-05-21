"""analysis.unitmatch phase orchestrator.

Slices 1-3 of `unitmatch_phase_plan.md`:
- Slice 1 wired the phase into the analysis stage (scaffold; disabled
  default).
- Slice 2 added `core/unitmatch_groups.py` (group discovery + per-session
  path resolution).
- Slice 3 (this commit): the per-target orchestrator now identifies its
  (chip, well) GROUP, resolves all sessions in that group, calls
  ``unitlink.match`` once per group, and lands outputs at
  ``<analysis_outputs>/unitmatch/<chip>/<well>/``. Subsequent targets in
  the same group see the existing summary on disk and short-circuit
  (idempotent).

Slice 4 will wire ``--targets`` to chip-well groups directly; slice 5
flips ``enabled: true`` after a real-data smoke once the recon-stage
kssynth phase has produced inputs.

unitlink is a soft dependency — the import is deferred to call time so
``axon_recon`` loads cleanly even if the user hasn't installed unitlink.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from ....output_paths import compute_mea_analysis_output_dir
from ..core.unitmatch_groups import (
	UnitmatchSessionInputMissing,
	discover_chip_well_groups,
	resolve_group_session_inputs,
)
from .compute_metrics import (
	_print_analysis_aggregate,
	_target_datasets_override_from_args,
)


LOGGER = logging.getLogger("axon_recon.analysis.unitmatch")


def _stage_output_root(*, mea_output_root: Path, h5_path: Path, well_id: str, output_rel_root: str) -> Path:
	"""Compute ``<well_out_dir>/<output_rel_root>/`` for the calling target."""

	well_dir = compute_mea_analysis_output_dir(
		output_root=Path(mea_output_root),
		data_file=Path(h5_path),
		well=str(well_id),
	)
	return well_dir / str(output_rel_root)


def _group_output_dir(
	*,
	output_root: Path,
	unitmatch_rel_output_root: str,
	chip_id: str,
	well_id: str,
) -> Path:
	"""Stable per-group output dir at the project-level ``output_root``.

	Per `unitmatch_phase_plan.md` §4 Slice 3: outputs land at
	``<output_root>/<unitmatch_rel_output_root>/<chip>/<well>/`` so every
	session in the group resolves to the SAME directory regardless of
	which (dataset, well) target invoked the phase.
	"""

	return Path(output_root) / str(unitmatch_rel_output_root) / str(chip_id) / str(well_id)


def _group_summary_path(group_dir: Path) -> Path:
	return group_dir / "summary.json"


def _resolve_unitlink_call(stage_config: Any) -> Any:
	"""Return the bound ``unitlink.match`` callable, or raise actionably.

	Honors a ``stage_config._unitmatch_call_override`` test hook so the
	slice-3 unit tests can monkeypatch the orchestrator's library
	invocation without monkeypatching ``unitlink`` at the module level.
	"""

	override = getattr(stage_config, "_unitmatch_call_override", None)
	if callable(override):
		return override

	try:
		import unitlink  # type: ignore[import-not-found]
	except Exception as exc:
		raise SystemExit(
			"analysis.unitmatch: unitlink is not importable. Install with "
			"`pip install -e ~/dev/pkgs/unitlink/` first."
		) from exc
	return unitlink.match


def _group_already_done(group_dir: Path, *, force_restart: bool) -> bool:
	if force_restart:
		return False
	summary_path = _group_summary_path(group_dir)
	if not summary_path.is_file():
		return False
	try:
		summary = json.loads(summary_path.read_text(encoding="utf-8"))
	except Exception:
		return False
	return str(summary.get("status", "")).lower() == "ok"


def _write_phase_summary(
	*,
	target_summary_path: Path,
	status: str,
	chip_id: str | None,
	well_id: str,
	dataset_index: int,
	group_dataset_indices: list[int] | None,
	group_dir: Path | None,
	match_table_rows: int | None,
	uid_assignment_rows: int | None,
	reason: str,
) -> dict[str, Any]:
	"""Write the per-target unitmatch summary marker."""

	payload: dict[str, Any] = {
		"phase": "unitmatch",
		"status": status,
		"dataset_index": int(dataset_index),
		"well_id": str(well_id),
		"chip_id": chip_id,
		"group_dataset_indices": list(group_dataset_indices) if group_dataset_indices is not None else None,
		"group_output_dir": str(group_dir) if group_dir is not None else None,
		"match_table_rows": match_table_rows,
		"uid_assignment_rows": uid_assignment_rows,
		"reason": reason,
	}
	target_summary_path.parent.mkdir(parents=True, exist_ok=True)
	target_summary_path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
	)
	return payload


def _per_target_summary_path(stage_output_root: Path) -> Path:
	return stage_output_root / "context" / "unitmatch_summary.json"


def run_analysis_unitmatch(
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
	"""Per-target entry point for the analysis.unitmatch phase.

	Looks up the (chip, well) group for this (dataset_index, well_id),
	resolves all sessions in the group, calls ``unitlink.match`` once,
	and lands outputs at ``<analysis_outputs>/<unitmatch_rel_output_root>/<chip>/<well>/``.

	When the phase is disabled in YAML, returns a noop marker without
	touching disk. When the group has already been processed (group
	summary on disk + not force_restart), returns a "skipped" marker —
	subsequent targets in the same group thus short-circuit cleanly.
	"""

	stage_output_root = _stage_output_root(
		mea_output_root=mea_output_root,
		h5_path=h5_path,
		well_id=str(stream_id),
		output_rel_root=output_rel_root,
	)
	target_summary_path = _per_target_summary_path(stage_output_root)

	# Dry-run short-circuit (dry_run_rollout slice 6 — unitmatch phase).
	# Per `guardrails/dry_run.md`: skip the heavy unitlink.match() call and
	# the per-group session resolution. Reports the per-target summary
	# path + group-output-dir location so the caller can see what would be
	# produced. Honors EITHER the per-stage_config `dry_run` flag (mirrors
	# propagation_video's pattern) OR the process-wide --dry-run override.
	from ....config import get_dry_run_override
	from ....dry_run import write_dry_run_summary

	if bool(getattr(stage_config, "dry_run", False)) or bool(get_dry_run_override()):
		stage_output_root.mkdir(parents=True, exist_ok=True)
		well_out_dir_for_summary = compute_mea_analysis_output_dir(
			output_root=Path(mea_output_root),
			data_file=Path(h5_path),
			well=str(stream_id),
		)
		unitmatch_rel = str(
			getattr(stage_config, "unitmatch_rel_output_root", "unitmatch")
		)
		well_metadata = getattr(stage_config, "well_metadata_lookup", {}) or {}
		entry = well_metadata.get((int(dataset_index), str(stream_id))) or {}
		chip_id = entry.get("chip_id", "unknown")
		group_dir = _group_output_dir(
			output_root=Path(mea_output_root),
			unitmatch_rel_output_root=unitmatch_rel,
			chip_id=str(chip_id),
			well_id=str(stream_id),
		)
		validation_warnings: list[str] = []
		if not bool(getattr(stage_config, "unitmatch_enabled", False)):
			validation_warnings.append(
				"unitmatch_enabled=False in YAML; a real run would skip this phase. "
				"Force-enable via --force-enable unitmatch if you want this to run."
			)
		if not entry:
			validation_warnings.append(
				f"no well_metadata_lookup entry for (dataset={dataset_index}, well={stream_id!r})"
			)
		write_dry_run_summary(
			phase_name="analysis.unitmatch",
			well_out_dir=well_out_dir_for_summary,
			stage_output_root_dir=stage_output_root,
			summary_json_path=target_summary_path,
			inputs_resolved=[
				{
					"name": "well_metadata_chip_id",
					"path": str(chip_id),
					"exists": bool(entry),
				},
			],
			outputs_would_produce=[
				{"name": "target_summary_json", "path": str(target_summary_path)},
				{"name": "group_dir", "path": str(group_dir)},
			],
			validation={
				"missing_prerequisites": [],
				"warnings": validation_warnings,
			},
			extra_fields={
				"unitmatch_enabled": bool(
					getattr(stage_config, "unitmatch_enabled", False)
				),
			},
		)
		return {
			"phase": "analysis.unitmatch",
			"status": "dry_run_ok",
			"well_id": str(stream_id),
			"dataset_index": int(dataset_index),
			"stage_output_root": str(stage_output_root),
			"target_summary_path": str(target_summary_path),
		}

	enabled = bool(getattr(stage_config, "unitmatch_enabled", False))
	if not enabled:
		return _write_phase_summary(
			target_summary_path=target_summary_path,
			status="noop",
			chip_id=None,
			well_id=str(stream_id),
			dataset_index=int(dataset_index),
			group_dataset_indices=None,
			group_dir=None,
			match_table_rows=None,
			uid_assignment_rows=None,
			reason="phase disabled in YAML",
		)

	# Group lookup.
	well_metadata = getattr(stage_config, "well_metadata_lookup", {}) or {}
	entry = well_metadata.get((int(dataset_index), str(stream_id)))
	chip_id = (entry or {}).get("chip_id")
	if not chip_id:
		return _write_phase_summary(
			target_summary_path=target_summary_path,
			status="error",
			chip_id=None,
			well_id=str(stream_id),
			dataset_index=int(dataset_index),
			group_dataset_indices=None,
			group_dir=None,
			match_table_rows=None,
			uid_assignment_rows=None,
			reason=(
				f"no chip_id for (dataset={dataset_index}, well={stream_id!r}) — "
				"either the data config is missing a parseable raw_data_h5_path or "
				"well_metadata_lookup wasn't built from a data config."
			),
		)

	groups = discover_chip_well_groups(well_metadata=well_metadata)
	group_indices = groups.get((str(chip_id), str(stream_id)), [int(dataset_index)])

	# Locate the group's output dir + check if it's already done.
	unitmatch_rel = str(getattr(stage_config, "unitmatch_rel_output_root", "unitmatch"))
	group_dir = _group_output_dir(
		output_root=Path(mea_output_root),
		unitmatch_rel_output_root=unitmatch_rel,
		chip_id=str(chip_id),
		well_id=str(stream_id),
	)

	if _group_already_done(group_dir, force_restart=bool(force_restart)):
		# Idempotent skip — another target in this group already produced
		# the output. We still write our per-target marker so the
		# per-well status reporter sees a coherent state.
		return _write_phase_summary(
			target_summary_path=target_summary_path,
			status="skipped",
			chip_id=str(chip_id),
			well_id=str(stream_id),
			dataset_index=int(dataset_index),
			group_dataset_indices=list(group_indices),
			group_dir=group_dir,
			match_table_rows=None,
			uid_assignment_rows=None,
			reason="group output already on disk; another target processed it first",
		)

	# Resolve all session inputs for the group.
	try:
		sessions = resolve_group_session_inputs(
			group_dataset_indices=group_indices,
			well_id=str(stream_id),
			well_metadata=well_metadata,
			output_root=Path(mea_output_root),
			require_exists=True,
		)
	except UnitmatchSessionInputMissing as exc:
		return _write_phase_summary(
			target_summary_path=target_summary_path,
			status="error",
			chip_id=str(chip_id),
			well_id=str(stream_id),
			dataset_index=int(dataset_index),
			group_dataset_indices=list(group_indices),
			group_dir=group_dir,
			match_table_rows=None,
			uid_assignment_rows=None,
			reason=f"missing session inputs: {exc}",
		)

	# Call unitlink.match for the group.
	match_threshold = float(getattr(stage_config, "unitmatch_match_threshold", 0.5))
	match_callable = _resolve_unitlink_call(stage_config)
	LOGGER.info(
		"unitmatch group=%s/%s sessions=%d output=%s",
		str(chip_id),
		str(stream_id),
		len(sessions),
		str(group_dir),
	)
	try:
		result = match_callable(
			sorter_outputs=[s.sorter_output for s in sessions],
			out_folder=group_dir,
			match_threshold=match_threshold,
		)
	except Exception as exc:
		LOGGER.exception(
			"unitmatch group %s/%s failed during unitlink.match", chip_id, stream_id
		)
		return _write_phase_summary(
			target_summary_path=target_summary_path,
			status="error",
			chip_id=str(chip_id),
			well_id=str(stream_id),
			dataset_index=int(dataset_index),
			group_dataset_indices=list(group_indices),
			group_dir=group_dir,
			match_table_rows=None,
			uid_assignment_rows=None,
			reason=f"unitlink.match raised: {exc!r}",
		)

	match_table_rows = (
		int(len(result.match_table)) if getattr(result, "match_table", None) is not None else None
	)
	uid_assignment_rows = (
		int(len(result.uid_assignment))
		if getattr(result, "uid_assignment", None) is not None
		else None
	)

	# Group-level summary lives alongside unitlink's output. unitlink
	# writes summary.json already; we overlay a status:ok marker so the
	# orchestrator's "already done" check on the next target invocation
	# in the same group can see it.
	group_summary_path = _group_summary_path(group_dir)
	group_summary_payload: dict[str, Any] = {}
	if group_summary_path.is_file():
		try:
			group_summary_payload = json.loads(group_summary_path.read_text(encoding="utf-8"))
		except Exception:
			group_summary_payload = {}
	group_summary_payload.update(
		{
			"status": "ok",
			"phase": "unitmatch",
			"chip_id": str(chip_id),
			"well_id": str(stream_id),
			"group_dataset_indices": list(group_indices),
			"match_table_rows": match_table_rows,
			"uid_assignment_rows": uid_assignment_rows,
		}
	)
	group_summary_path.parent.mkdir(parents=True, exist_ok=True)
	group_summary_path.write_text(
		json.dumps(group_summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
	)

	return _write_phase_summary(
		target_summary_path=target_summary_path,
		status="ok",
		chip_id=str(chip_id),
		well_id=str(stream_id),
		dataset_index=int(dataset_index),
		group_dataset_indices=list(group_indices),
		group_dir=group_dir,
		match_table_rows=match_table_rows,
		uid_assignment_rows=uid_assignment_rows,
		reason=f"unitlink.match completed across {len(sessions)} session(s)",
	)


def _run_unitmatch_from_args(args: argparse.Namespace) -> int:
	"""CLI handler for `axon-recon stages analysis.unitmatch`."""

	from ....runner import run_analysis_unitmatch_from_runtime

	target_datasets_override = _target_datasets_override_from_args(args)
	return _print_analysis_aggregate(
		run_analysis_unitmatch_from_runtime(
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


__all__ = ["run_analysis_unitmatch", "_run_unitmatch_from_args"]
