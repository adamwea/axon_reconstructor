from __future__ import annotations

import json
from pathlib import Path

from ..config import AnalysisStageConfig
from ..runner import run_analysis_compute_metrics_stage


def _make_stage_config(
	*,
	well_metadata_lookup: dict[tuple[int, str], dict],
	output_rel_root: str = "analysis_outputs",
	enabled: bool = True,
) -> AnalysisStageConfig:
	return AnalysisStageConfig(
		output_rel_root=output_rel_root,
		phase_sequence=("compute_metrics",),
		compute_metrics_enabled=bool(enabled),
		compute_metrics_resource_class=None,
		debug_mode_enabled=False,
		debug_limit_datasets=None,
		debug_limit_wells=None,
		debug_limit_wells_per_dataset=None,
		compute_metrics_debug_mode_enabled=False,
		compute_metrics_debug_limit_datasets=None,
		compute_metrics_debug_limit_wells=None,
		compute_metrics_debug_limit_wells_per_dataset=None,
		recon_output_rel_root="recon_outputs",
		manifest_relpath="manifest.json",
		tables_relpath="tables",
		pipeline_version="test-version",
		well_metadata_lookup=well_metadata_lookup,
		force_restart=False,
		force_replot=False,
	)


def test_run_compute_metrics_writes_manifest_with_identity_fields(tmp_path: Path) -> None:
	mea_output_root = tmp_path / "scratch_outputs"
	mea_output_root.mkdir(parents=True, exist_ok=True)
	h5_path = (
		mea_output_root / "Media_Density_T5_02182026_AR" / "260326" / "M08073"
		/ "AxonTracking" / "000208" / "data.raw.h5"
	)
	h5_path.parent.mkdir(parents=True, exist_ok=True)
	h5_path.write_bytes(b"")

	lookup = {
		(0, "well000"): {
			"DIV": 36,
			"dataset_id": "ds-11",
			"project": "Media_Density_T5_02182026_AR",
			"recording_date": "2026-03-26",
			"chip_id": "M08073",
			"scan_type": "AxonTracking",
			"run_id": "000208",
			"well_attributes": {
				"plating_density": 80000,
				"media": "DMEM",
				"genotype": "WT",
			},
		}
	}
	stage_config = _make_stage_config(well_metadata_lookup=lookup)

	result = run_analysis_compute_metrics_stage(
		dataset_index=0,
		dataset_id="ds-11",
		h5_path=h5_path,
		stream_id="well000",
		mea_output_root=mea_output_root,
		output_rel_root="analysis_outputs",
		stage_config=stage_config,
		force_restart=False,
	)

	manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
	assert manifest["artifact_type"] == "axon_recon_well_analysis"
	assert manifest["schema_version"] == "axon_analysis_v1"
	assert manifest["pipeline_version"] == "test-version"
	assert manifest["project"] == "Media_Density_T5_02182026_AR"
	assert manifest["recording_date"] == "2026-03-26"
	assert manifest["chip_id"] == "M08073"
	assert manifest["scan_type"] == "AxonTracking"
	assert manifest["run_id"] == "000208"
	assert manifest["well_id"] == "well000"
	assert manifest["dataset_id"] == "ds-11"
	assert manifest["DIV"] == 36
	assert manifest["well_attributes"] == {
		"plating_density": 80000,
		"media": "DMEM",
		"genotype": "WT",
	}
	assert manifest["tables"] == {}
	assert manifest["status"] == "ok"
	assert manifest["reason"] is None
	assert manifest["force_restart"] is False
	assert "written_at" in manifest

	# Stage output dir matches output_rel_root.
	assert result.analysis_out_dir.name == "analysis_outputs"
	# Manifest sits under analysis_outputs/ by default.
	assert result.manifest_json.parent == result.analysis_out_dir


def test_run_compute_metrics_disabled_phase_writes_skipped_manifest(tmp_path: Path) -> None:
	mea_output_root = tmp_path / "scratch_outputs"
	h5_path = (
		mea_output_root / "ProjectX" / "260101" / "ChipA"
		/ "AxonTracking" / "000001" / "data.raw.h5"
	)
	h5_path.parent.mkdir(parents=True, exist_ok=True)
	h5_path.write_bytes(b"")

	stage_config = _make_stage_config(well_metadata_lookup={}, enabled=False)

	result = run_analysis_compute_metrics_stage(
		dataset_index=0,
		dataset_id=None,
		h5_path=h5_path,
		stream_id="well000",
		mea_output_root=mea_output_root,
		output_rel_root="analysis_outputs",
		stage_config=stage_config,
		force_restart=False,
	)
	manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
	assert manifest["status"] == "skipped"
	assert manifest["reason"] == "compute_metrics_disabled"


def test_run_compute_metrics_idempotent_on_rerun(tmp_path: Path) -> None:
	mea_output_root = tmp_path / "scratch_outputs"
	h5_path = (
		mea_output_root / "ProjectX" / "260101" / "ChipA"
		/ "AxonTracking" / "000001" / "data.raw.h5"
	)
	h5_path.parent.mkdir(parents=True, exist_ok=True)
	h5_path.write_bytes(b"")

	stage_config = _make_stage_config(well_metadata_lookup={})

	# Run twice; both must succeed and the manifest must remain readable.
	for _ in range(2):
		result = run_analysis_compute_metrics_stage(
			dataset_index=0,
			dataset_id=None,
			h5_path=h5_path,
			stream_id="well000",
			mea_output_root=mea_output_root,
			output_rel_root="analysis_outputs",
			stage_config=stage_config,
			force_restart=False,
		)
		manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
		assert manifest["status"] == "ok"


def test_run_compute_metrics_respects_alternate_output_rel_root(tmp_path: Path) -> None:
	mea_output_root = tmp_path / "scratch_outputs"
	h5_path = (
		mea_output_root / "ProjectX" / "260101" / "ChipA"
		/ "AxonTracking" / "000001" / "data.raw.h5"
	)
	h5_path.parent.mkdir(parents=True, exist_ok=True)
	h5_path.write_bytes(b"")

	stage_config = _make_stage_config(well_metadata_lookup={}, output_rel_root="alt_analysis")

	result = run_analysis_compute_metrics_stage(
		dataset_index=0,
		dataset_id=None,
		h5_path=h5_path,
		stream_id="well000",
		mea_output_root=mea_output_root,
		output_rel_root="alt_analysis",
		stage_config=stage_config,
		force_restart=False,
	)
	assert result.analysis_out_dir.name == "alt_analysis"
