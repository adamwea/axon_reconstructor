from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from axon_recon.pipeline.output_paths import compute_mea_analysis_output_dir

from .models.results import AnalysisResult


LOGGER = logging.getLogger("axon_recon.analysis.runner")


def _resolve_under_well(*, well_out_dir: Path, relpath: str) -> Path:
	candidate = Path(str(relpath).strip()).expanduser()
	if candidate.is_absolute():
		return candidate.resolve()
	return (well_out_dir / str(relpath).lstrip("/")).resolve()


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

	Slice 1 scope: writes <well>/analysis_outputs/manifest.json with identity
	fields populated. tables: {} (slice 2 will populate units.parquet, slice 3
	well_summary.parquet).
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

	enabled = bool(getattr(stage_config, "compute_metrics_enabled", True))
	identity = _identity_for_target(
		dataset_index=int(dataset_index),
		stream_id=str(stream_id),
		stage_config=stage_config,
	)
	# Prefer caller-provided dataset_id; otherwise the data-config-derived one.
	resolved_dataset_id = dataset_id if dataset_id is not None else identity.get("dataset_id")

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
		"tables": {},
		"status": "ok" if enabled else "skipped",
		"reason": None if enabled else "compute_metrics_disabled",
		"force_restart": bool(force_restart),
	}

	_write_json(manifest_path, manifest_payload)

	outputs = {"manifest_json": str(manifest_path)}
	return AnalysisResult(
		well_out_dir=well_out_dir,
		analysis_out_dir=stage_output_root_dir,
		manifest_json=manifest_path,
		outputs=outputs,
	)
