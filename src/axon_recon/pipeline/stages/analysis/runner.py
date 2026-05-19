from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from axon_recon.pipeline.checkpoint import with_checkpoint_marker
from axon_recon.pipeline.output_paths import compute_mea_analysis_output_dir

from .core.labels_io import (
	lookup_label,
	lookup_spike_count,
	read_bombcell_labels,
	read_per_cluster_spike_counts,
)
from .core.metrics import (
	WELL_SUMMARY_METRIC_COLUMNS,
	compute_unit_metrics,
	compute_well_summary,
)
from .core.recon_io import (
	get_recon_status,
	get_template_status,
	get_unit_id_from_branches,
	iter_unit_dirs,
	read_branches,
	read_merged_contributing_electrode_ids,
	read_unit_reconstruction_summary,
	read_unit_templates_summary,
)
from .core.spikesort_metrics import SPIKESORT_AGG_COLUMNS, collect_spikesort_unit_counts
from .models.results import AnalysisResult


LOGGER = logging.getLogger("axon_recon.analysis.runner")


_UNITS_TABLE_COLUMNS: tuple[str, ...] = (
	# Identity
	"project",
	"recording_date",
	"chip_id",
	"scan_type",
	"run_id",
	"well_id",
	"dataset_id",
	"unit_id",
	"DIV",
	"genotype",
	"media",
	"plating_density",
	"treatment",
	# Filter columns (see plan §4)
	"recon_status",
	"template_status",
	"bombcell_label",
	"num_spikes",
	"num_branches",
	"recon_quality_score",
	# Starter metrics
	"branch_count",
	"total_branch_length_um",
	"mean_branch_length_um",
	"template_density",
	"recon_density",
	# Handy passthroughs
	"max_amplitude_uv",
	"max_ptp_uv",
	"max_delay_ms",
	"unit_location_x_um",
	"unit_location_y_um",
)


_WELL_SUMMARY_IDENTITY_COLUMNS: tuple[str, ...] = (
	"project",
	"recording_date",
	"chip_id",
	"scan_type",
	"run_id",
	"well_id",
	"dataset_id",
	"DIV",
	"genotype",
	"media",
	"plating_density",
	"treatment",
)


_WELL_SUMMARY_AGG_COLUMNS: tuple[str, ...] = (
	"unit_count_total",
	"unit_count_recon_ok",
	"unit_count_template_ok",
	"unit_count_bombcell_good",
	"unit_count_bombcell_non_soma_good",
	*SPIKESORT_AGG_COLUMNS,
	*(f"mean_{m}" for m in WELL_SUMMARY_METRIC_COLUMNS),
	*(f"median_{m}" for m in WELL_SUMMARY_METRIC_COLUMNS),
)


_WELL_SUMMARY_TABLE_COLUMNS: tuple[str, ...] = (
	*_WELL_SUMMARY_IDENTITY_COLUMNS,
	*_WELL_SUMMARY_AGG_COLUMNS,
)


def _resolve_under_well(*, well_out_dir: Path, relpath: str) -> Path:
	candidate = Path(str(relpath).strip()).expanduser()
	if candidate.is_absolute():
		return candidate.resolve()
	return (well_out_dir / str(relpath).lstrip("/")).resolve()


def target_analysis_phase_summary_ok(
	*,
	phase_name: str,
	target: Any,
	stage_config: Any,
) -> bool:
	"""Slice 14c (target-level auto-restart skip): True iff the per-target
	per-phase summary for ``phase_name`` already shows the phase is
	complete on disk.

	``compute_metrics`` writes a per-target ``manifest.json`` at
	``<analysis_outputs>/`` whose top-level ``status`` is ``ok`` on
	success. ``unitmatch`` writes per-target ``context/unitmatch_summary.json``
	whose ``status`` is one of ``ok`` / ``skipped`` / ``noop`` when the
	group is already done or the phase is disabled. Both count as
	"already ok" for slice 14c purposes.

	Returns False for unknown phase names so the caller proceeds with
	normal dispatch. Falsy / unreadable summaries also return False.
	"""

	output_rel_root = str(getattr(stage_config, "output_rel_root", "analysis_outputs") or "analysis_outputs")
	well_out_dir = compute_mea_analysis_output_dir(
		output_root=Path(getattr(target, "mea_output_root", Path("."))),
		data_file=Path(getattr(target, "h5_path", Path("."))),
		well=str(getattr(target, "stream_id", "")),
	)
	stage_output_root = (well_out_dir / output_rel_root).resolve()
	if str(phase_name).strip() == "compute_metrics":
		manifest_rel = str(getattr(stage_config, "manifest_relpath", "manifest.json") or "manifest.json")
		summary_path = stage_output_root / Path(manifest_rel)
	elif str(phase_name).strip() == "unitmatch":
		summary_path = stage_output_root / "context" / "unitmatch_summary.json"
	elif str(phase_name).strip() == "propagation_video":
		summary_path = stage_output_root / "context" / "propagation_video_summary.json"
	else:
		return False
	if not summary_path.is_file():
		return False
	try:
		payload = json.loads(summary_path.read_text(encoding="utf-8"))
	except Exception:
		return False
	if not isinstance(payload, dict):
		return False
	return str(payload.get("status", "")).strip().lower() in ("ok", "skipped", "noop")


def _resolve_under_analysis_output_root(
	*,
	well_out_dir: Path,
	output_rel_root: str,
	relpath: str,
) -> Path:
	stage_root_rel = str(output_rel_root).strip().lstrip("/") or "analysis_outputs"
	resolved_stage_root = (well_out_dir / stage_root_rel).resolve()
	rel = str(relpath).strip().lstrip("/")
	if rel == stage_root_rel:
		rel = ""
	elif rel.startswith(f"{stage_root_rel}/"):
		rel = rel[len(stage_root_rel) + 1 :]
	if not rel:
		return resolved_stage_root
	return (resolved_stage_root / rel).resolve()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _identity_for_target(
	*,
	dataset_index: int,
	stream_id: str,
	stage_config: Any,
) -> dict[str, Any]:
	lookup = getattr(stage_config, "well_metadata_lookup", {}) or {}
	key = (int(dataset_index), str(stream_id))
	entry = lookup.get(key)
	if entry is None:
		return {
			"project": None,
			"recording_date": None,
			"chip_id": None,
			"scan_type": None,
			"run_id": None,
			"dataset_id": None,
			"DIV": None,
			"well_attributes": {},
		}
	return dict(entry)


def _unit_id_from_dir_name(unit_dir: Path) -> int | None:
	name = unit_dir.name
	try:
		return int(name)
	except (TypeError, ValueError):
		return None


def _row_for_unit(
	*,
	unit_dir: Path,
	identity_cols: dict[str, Any],
	bombcell_labels: dict[int, str],
	spike_counts: dict[int, int],
) -> dict[str, Any]:
	unit_summary = read_unit_reconstruction_summary(unit_dir)
	branches_payload = read_branches(unit_dir)
	merged_payload = read_merged_contributing_electrode_ids(unit_dir)
	templates_payload = read_unit_templates_summary(unit_dir)

	recon_status = get_recon_status(unit_summary)
	template_status = get_template_status(templates_payload)
	unit_id = get_unit_id_from_branches(branches_payload)
	if unit_id is None:
		unit_id = _unit_id_from_dir_name(unit_dir)

	if recon_status == "ok":
		metrics = compute_unit_metrics(
			unit_summary_payload=unit_summary,
			branches_payload=branches_payload,
			merged_payload=merged_payload,
			templates_payload=templates_payload,
		)
	else:
		# Non-ok units carry NaN metrics so the dashboard's recon_status filter
		# does the gating instead of metric-level None.
		metrics = compute_unit_metrics(
			unit_summary_payload=None,
			branches_payload=None,
			merged_payload=None,
			templates_payload=None,
		)

	bombcell_label = lookup_label(bombcell_labels, unit_id) if unit_id is not None else None
	num_spikes = lookup_spike_count(spike_counts, unit_id) if unit_id is not None else None

	row: dict[str, Any] = {
		**identity_cols,
		"unit_id": int(unit_id) if unit_id is not None else None,
		"recon_status": str(recon_status),
		"template_status": str(template_status),
		"bombcell_label": bombcell_label,
		"num_spikes": int(num_spikes) if num_spikes is not None else None,
		"num_branches": int(metrics["branch_count"])
		if not _is_nan(metrics["branch_count"])
		else None,
		"recon_quality_score": None,
		"branch_count": metrics["branch_count"],
		"total_branch_length_um": metrics["total_branch_length_um"],
		"mean_branch_length_um": metrics["mean_branch_length_um"],
		"template_density": metrics["template_density"],
		"recon_density": metrics["recon_density"],
		"max_amplitude_uv": metrics["max_amplitude_uv"],
		"max_ptp_uv": metrics["max_ptp_uv"],
		"max_delay_ms": metrics["max_delay_ms"],
		"unit_location_x_um": metrics["unit_location_x_um"],
		"unit_location_y_um": metrics["unit_location_y_um"],
	}
	return row


def _is_nan(value: Any) -> bool:
	try:
		return value != value  # type: ignore[no-any-return]
	except Exception:
		return False


def _build_identity_cols(
	*,
	identity: dict[str, Any],
	stream_id: str,
	resolved_dataset_id: Any,
) -> dict[str, Any]:
	well_attributes = identity.get("well_attributes", {}) or {}
	return {
		"project": identity.get("project"),
		"recording_date": identity.get("recording_date"),
		"chip_id": identity.get("chip_id"),
		"scan_type": identity.get("scan_type"),
		"run_id": identity.get("run_id"),
		"well_id": str(stream_id),
		"dataset_id": resolved_dataset_id,
		"DIV": identity.get("DIV"),
		"genotype": well_attributes.get("genotype", None),
		"media": well_attributes.get("media", None),
		"plating_density": well_attributes.get("plating_density", None),
		# Descriptive string for treatment condition; null/absent means no
		# treatment context. Use specific strings like "drug_X_post_2h" when
		# known, generic ones like "post_treatment_2h_unspecified" when not.
		"treatment": well_attributes.get("treatment", None),
	}


def _write_units_parquet(
	*,
	rows: list[dict[str, Any]],
	parquet_path: Path,
) -> tuple[int, Any]:
	import pandas as pd  # local import — keeps test imports cheap

	parquet_path.parent.mkdir(parents=True, exist_ok=True)
	# Always emit the full column set in the documented order, even when rows
	# is empty (so downstream schema asserts remain stable across wells).
	if rows:
		df = pd.DataFrame(rows, columns=list(_UNITS_TABLE_COLUMNS))
	else:
		df = pd.DataFrame({column: [] for column in _UNITS_TABLE_COLUMNS})
	df.to_parquet(parquet_path, engine="pyarrow", index=False)
	return int(len(df)), df


def _write_well_summary_parquet(
	*,
	units_df: Any,
	identity_cols: dict[str, Any],
	parquet_path: Path,
	well_out_dir: Path | None = None,
	spikesort_output_rel_root: str = "spikesort_outputs",
) -> dict[str, Any]:
	"""Aggregate units_df into a single-row well_summary table and persist it.

	When `well_out_dir` is provided, additionally read well-level spikesort
	artifacts (pre/post-merge metadata, bombcell label histogram, SLAy
	candidate / merge counts) and merge them into the summary row. Columns
	come from `SPIKESORT_AGG_COLUMNS`; missing source files leave them None.
	"""
	import pandas as pd  # local import for symmetry with _write_units_parquet

	summary_row = compute_well_summary(units_df=units_df, identity_cols=identity_cols)
	if well_out_dir is not None:
		summary_row.update(
			collect_spikesort_unit_counts(
				well_out_dir=well_out_dir,
				output_rel_root=spikesort_output_rel_root,
			)
		)
	parquet_path.parent.mkdir(parents=True, exist_ok=True)
	df = pd.DataFrame([summary_row], columns=list(_WELL_SUMMARY_TABLE_COLUMNS))
	df.to_parquet(parquet_path, engine="pyarrow", index=False)
	return summary_row


def run_analysis_compute_metrics_stage(
	*,
	dataset_index: int,
	dataset_id: str | None,
	h5_path: Path,
	stream_id: str,
	mea_output_root: Path,
	output_rel_root: str,
	stage_config: Any,
	force_restart: bool,
) -> AnalysisResult:
	"""Per-well runner for analysis.compute_metrics.

	Reads each recon unit dir under `<well>/recon_outputs/units/`, computes the
	starter metrics + filter columns, writes `tables/units.parquet`, and
	publishes the manifest with `tables.units` populated and `unit_count` set.
	"""
	well_out_dir = compute_mea_analysis_output_dir(
		output_root=mea_output_root,
		data_file=h5_path,
		well=stream_id,
	)
	stage_output_root_dir = _resolve_under_well(
		well_out_dir=well_out_dir,
		relpath=(str(output_rel_root).strip() or "analysis_outputs"),
	)
	stage_output_root_dir.mkdir(parents=True, exist_ok=True)

	manifest_relpath = str(getattr(stage_config, "manifest_relpath", "manifest.json") or "manifest.json")
	manifest_path = _resolve_under_analysis_output_root(
		well_out_dir=well_out_dir,
		output_rel_root=output_rel_root,
		relpath=manifest_relpath,
	)

	# Lay down the in_progress marker BEFORE any phase work. The final
	# manifest write below overwrites this on success; on uncaught
	# exception the context manager overwrites with status=error.
	with with_checkpoint_marker(
		manifest_path,
		phase_name="compute_metrics",
		stage_name="analysis",
	):
		enabled = bool(getattr(stage_config, "compute_metrics_enabled", True))
		identity = _identity_for_target(
			dataset_index=int(dataset_index),
			stream_id=str(stream_id),
			stage_config=stage_config,
		)
		resolved_dataset_id = dataset_id if dataset_id is not None else identity.get("dataset_id")

		tables_relpath = str(getattr(stage_config, "tables_relpath", "tables") or "tables")
		tables_dir = _resolve_under_analysis_output_root(
			well_out_dir=well_out_dir,
			output_rel_root=output_rel_root,
			relpath=tables_relpath,
		)
		units_parquet_path = tables_dir / "units.parquet"
		well_summary_parquet_path = tables_dir / "well_summary.parquet"

		tables_section: dict[str, Any] = {}
		unit_count = 0
		if enabled:
			recon_output_rel_root = str(
				getattr(stage_config, "recon_output_rel_root", "recon_outputs") or "recon_outputs"
			)
			recon_outputs_dir = _resolve_under_well(well_out_dir=well_out_dir, relpath=recon_output_rel_root)
			spikesort_outputs_dir = _resolve_under_well(well_out_dir=well_out_dir, relpath="spikesort_outputs")

			bombcell_labels = read_bombcell_labels(spikesort_outputs_dir)
			spike_counts = read_per_cluster_spike_counts(spikesort_outputs_dir)

			identity_cols = _build_identity_cols(
				identity=identity,
				stream_id=stream_id,
				resolved_dataset_id=resolved_dataset_id,
			)
			rows: list[dict[str, Any]] = []
			for unit_dir in iter_unit_dirs(recon_outputs_dir):
				rows.append(
					_row_for_unit(
						unit_dir=unit_dir,
						identity_cols=identity_cols,
						bombcell_labels=bombcell_labels,
						spike_counts=spike_counts,
					)
				)
			unit_count, units_df = _write_units_parquet(rows=rows, parquet_path=units_parquet_path)
			tables_section["units"] = f"{tables_relpath}/units.parquet"

			well_summary_identity = {
				column: identity_cols.get(column)
				for column in _WELL_SUMMARY_IDENTITY_COLUMNS
			}
			_write_well_summary_parquet(
				units_df=units_df,
				identity_cols=well_summary_identity,
				parquet_path=well_summary_parquet_path,
				well_out_dir=well_out_dir,
			)
			tables_section["well_summary"] = f"{tables_relpath}/well_summary.parquet"

		manifest_payload: dict[str, Any] = {
			"artifact_type": "axon_recon_well_analysis",
			"schema_version": "axon_analysis_v1",
			"pipeline_version": str(getattr(stage_config, "pipeline_version", "unknown")),
			"project": identity.get("project"),
			"recording_date": identity.get("recording_date"),
			"chip_id": identity.get("chip_id"),
			"scan_type": identity.get("scan_type"),
			"run_id": identity.get("run_id"),
			"well_id": str(stream_id),
			"dataset_id": resolved_dataset_id,
			"DIV": identity.get("DIV"),
			"well_attributes": identity.get("well_attributes", {}),
			"written_at": datetime.now(timezone.utc).isoformat(),
			"tables": tables_section,
			"unit_count": int(unit_count),
			"status": "ok" if enabled else "skipped",
			"reason": None if enabled else "compute_metrics_disabled",
			"force_restart": bool(force_restart),
		}

		_write_json(manifest_path, manifest_payload)

		outputs = {"manifest_json": str(manifest_path)}
		if "units" in tables_section:
			outputs["units_parquet"] = str(units_parquet_path)
		if "well_summary" in tables_section:
			outputs["well_summary_parquet"] = str(well_summary_parquet_path)
		return AnalysisResult(
			well_out_dir=well_out_dir,
			analysis_out_dir=stage_output_root_dir,
			manifest_json=manifest_path,
			outputs=outputs,
		)
