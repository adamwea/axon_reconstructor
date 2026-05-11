from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import AnalysisStageConfig
from ..runner import _UNITS_TABLE_COLUMNS, run_analysis_compute_metrics_stage


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
	# Slice 2: tables.units always emitted when compute_metrics is enabled (even
	# when the recon_outputs/units/ tree is empty — units.parquet is then a
	# zero-row Parquet with the documented schema).
	assert manifest["tables"] == {"units": "tables/units.parquet"}
	assert manifest["unit_count"] == 0
	assert manifest["status"] == "ok"
	assert manifest["reason"] is None
	assert manifest["force_restart"] is False
	assert "written_at" in manifest

	# Stage output dir matches output_rel_root.
	assert result.analysis_out_dir.name == "analysis_outputs"
	# Manifest sits under analysis_outputs/ by default.
	assert result.manifest_json.parent == result.analysis_out_dir
	# units.parquet must exist on disk (even when empty).
	expected_parquet = result.analysis_out_dir / "tables" / "units.parquet"
	assert expected_parquet.exists()
	assert result.outputs["units_parquet"] == str(expected_parquet)


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


# ---------- slice 2: units.parquet integration ----------


def _make_unit_dir(
	well_dir: Path,
	*,
	unit_id: int,
	status: str = "ok",
	template_density: float = 0.5,
	max_amplitude: float = -200.0,
	max_ptp: float = 350.0,
	max_delay: float = 1.25,
	branches: list[dict] | None = None,
	electrode_ids: list[str] | None = None,
	unit_location_xy: tuple[float, float] | None = (100.0, 200.0),
) -> Path:
	unit_dir = well_dir / "recon_outputs" / "units" / f"{unit_id:04d}"
	unit_dir.mkdir(parents=True, exist_ok=True)
	(unit_dir / "unit_reconstruction_summary.json").write_text(
		json.dumps(
			{
				"unit_id": unit_id,
				"status": status,
				"grid_sort_metrics": {
					"template_density": template_density,
					"max_amplitude": max_amplitude,
					"max_ptp": max_ptp,
					"max_delay": max_delay,
				},
			}
		),
		encoding="utf-8",
	)
	if branches is not None:
		(unit_dir / "branches.json").write_text(
			json.dumps({"unit_id": unit_id, "branches": branches}), encoding="utf-8"
		)
	if electrode_ids is not None:
		(unit_dir / "merged_contributing_electrode_ids.json").write_text(
			json.dumps({"electrode_ids": electrode_ids}), encoding="utf-8"
		)
	if unit_location_xy is not None:
		x, y = unit_location_xy
		(unit_dir / "unit_templates_summary.json").write_text(
			json.dumps({"unit_id": unit_id, "unit_location": {"x_um": x, "y_um": y}}),
			encoding="utf-8",
		)
	return unit_dir


def _make_well_dir(tmp_path: Path) -> tuple[Path, Path, Path]:
	mea_output_root = tmp_path / "scratch_outputs"
	well_dir = (
		mea_output_root / "Media_Density_T5_02182026_AR" / "260326" / "M08073"
		/ "AxonTracking" / "000208" / "well000"
	)
	well_dir.mkdir(parents=True, exist_ok=True)
	h5_path = (
		mea_output_root / "Media_Density_T5_02182026_AR" / "260326" / "M08073"
		/ "AxonTracking" / "000208" / "data.raw.h5"
	)
	h5_path.parent.mkdir(parents=True, exist_ok=True)
	h5_path.write_bytes(b"")
	return mea_output_root, well_dir, h5_path


def _lookup() -> dict[tuple[int, str], dict[str, Any]]:
	return {
		(0, "well000"): {
			"DIV": 36,
			"dataset_id": "ds-fixture",
			"project": "Media_Density_T5_02182026_AR",
			"recording_date": "2026-03-26",
			"chip_id": "M08073",
			"scan_type": "AxonTracking",
			"run_id": "000208",
			"well_attributes": {"plating_density": 80000, "media": "DMEM", "genotype": "WT"},
		}
	}


def test_units_parquet_emits_documented_schema(tmp_path: Path) -> None:
	import pandas as pd

	mea_output_root, well_dir, h5_path = _make_well_dir(tmp_path)
	# One ok unit with branches/electrode_ids to make recon_density computable.
	_make_unit_dir(
		well_dir,
		unit_id=1,
		branches=[
			{"distances": [3.0, 4.0], "polyline_xy": [[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]]}
		],
		electrode_ids=["0", "1", "2", "3"],
	)
	stage_config = _make_stage_config(well_metadata_lookup=_lookup())

	result = run_analysis_compute_metrics_stage(
		dataset_index=0,
		dataset_id="ds-fixture",
		h5_path=h5_path,
		stream_id="well000",
		mea_output_root=mea_output_root,
		output_rel_root="analysis_outputs",
		stage_config=stage_config,
		force_restart=False,
	)
	manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
	assert manifest["tables"] == {"units": "tables/units.parquet"}
	assert manifest["unit_count"] == 1

	parquet_path = result.analysis_out_dir / "tables" / "units.parquet"
	assert parquet_path.exists()
	df = pd.read_parquet(parquet_path)
	assert list(df.columns) == list(_UNITS_TABLE_COLUMNS)
	assert len(df) == 1

	row = df.iloc[0]
	# Identity stamping
	assert row["project"] == "Media_Density_T5_02182026_AR"
	assert row["recording_date"] == "2026-03-26"
	assert row["chip_id"] == "M08073"
	assert row["scan_type"] == "AxonTracking"
	assert row["run_id"] == "000208"
	assert row["well_id"] == "well000"
	assert row["dataset_id"] == "ds-fixture"
	assert int(row["unit_id"]) == 1
	assert int(row["DIV"]) == 36
	assert row["genotype"] == "WT"
	assert row["media"] == "DMEM"
	assert int(row["plating_density"]) == 80000
	# Filter cols
	assert row["recon_status"] == "ok"
	assert row["bombcell_label"] is None  # no spikesort fixture in this test
	assert row["num_spikes"] is None
	assert int(row["num_branches"]) == 1
	assert row["recon_quality_score"] is None
	# Metrics (distances=[3,4] sums to 7)
	assert float(row["branch_count"]) == 1.0
	assert abs(float(row["total_branch_length_um"]) - 7.0) < 1e-6
	assert abs(float(row["template_density"]) - 0.5) < 1e-9
	# bbox: x in [0,3], y in [0,4] -> area = 12; 4 electrodes / 12 = 0.333...
	assert abs(float(row["recon_density"]) - (4.0 / 12.0)) < 1e-9
	# Passthroughs
	assert abs(float(row["max_amplitude_uv"]) - -200.0) < 1e-6
	assert abs(float(row["max_ptp_uv"]) - 350.0) < 1e-6
	assert abs(float(row["max_delay_ms"]) - 1.25) < 1e-9
	assert abs(float(row["unit_location_x_um"]) - 100.0) < 1e-6
	assert abs(float(row["unit_location_y_um"]) - 200.0) < 1e-6


def test_units_parquet_keeps_non_ok_recon_status_rows_with_nan_metrics(tmp_path: Path) -> None:
	import pandas as pd

	mea_output_root, well_dir, h5_path = _make_well_dir(tmp_path)
	# One ok + one error unit.
	_make_unit_dir(
		well_dir,
		unit_id=1,
		branches=[{"distances": [5.0], "polyline_xy": [[0.0, 0.0], [5.0, 0.0]]}],
		electrode_ids=["0", "1"],
	)
	_make_unit_dir(
		well_dir,
		unit_id=2,
		status="error",
		branches=None,
		electrode_ids=None,
		unit_location_xy=None,
	)
	stage_config = _make_stage_config(well_metadata_lookup=_lookup())

	result = run_analysis_compute_metrics_stage(
		dataset_index=0,
		dataset_id="ds-fixture",
		h5_path=h5_path,
		stream_id="well000",
		mea_output_root=mea_output_root,
		output_rel_root="analysis_outputs",
		stage_config=stage_config,
		force_restart=False,
	)
	df = pd.read_parquet(result.analysis_out_dir / "tables" / "units.parquet")
	assert len(df) == 2
	statuses = dict(zip(df["unit_id"], df["recon_status"]))
	assert statuses == {1: "ok", 2: "error"}

	# ok row metrics are non-null; error row metrics are NaN.
	ok_row = df[df["unit_id"] == 1].iloc[0]
	err_row = df[df["unit_id"] == 2].iloc[0]
	assert not _is_nan_scalar(ok_row["branch_count"])
	assert not _is_nan_scalar(ok_row["total_branch_length_um"])
	assert not _is_nan_scalar(ok_row["template_density"])
	assert _is_nan_scalar(err_row["branch_count"])
	assert _is_nan_scalar(err_row["total_branch_length_um"])
	assert _is_nan_scalar(err_row["template_density"])


def test_units_parquet_uses_bombcell_labels_and_spike_counts(tmp_path: Path) -> None:
	import pandas as pd

	mea_output_root, well_dir, h5_path = _make_well_dir(tmp_path)
	_make_unit_dir(
		well_dir,
		unit_id=1,
		branches=[{"distances": [3.0], "polyline_xy": [[0.0, 0.0], [3.0, 0.0]]}],
		electrode_ids=["0", "1"],
	)
	# Fixture spikesort outputs: cluster_group.tsv + spike_clusters.npy
	sorter_out = well_dir / "spikesort_outputs" / "sorter_output"
	sorter_out.mkdir(parents=True, exist_ok=True)
	(sorter_out / "cluster_group.tsv").write_text(
		"cluster_id\tKSLabel\tlabel\tlabel_reason\n1\tgood\tnon_soma_good\t\n",
		encoding="utf-8",
	)
	np.save(sorter_out / "spike_clusters.npy", np.array([1, 1, 1, 0, 0], dtype=np.int64))

	stage_config = _make_stage_config(well_metadata_lookup=_lookup())
	result = run_analysis_compute_metrics_stage(
		dataset_index=0,
		dataset_id="ds-fixture",
		h5_path=h5_path,
		stream_id="well000",
		mea_output_root=mea_output_root,
		output_rel_root="analysis_outputs",
		stage_config=stage_config,
		force_restart=False,
	)
	df = pd.read_parquet(result.analysis_out_dir / "tables" / "units.parquet")
	row = df.iloc[0]
	assert row["bombcell_label"] == "non_soma_good"
	assert int(row["num_spikes"]) == 3


def test_units_parquet_handles_unit_dir_with_no_branches_json(tmp_path: Path) -> None:
	import pandas as pd

	mea_output_root, well_dir, h5_path = _make_well_dir(tmp_path)
	# Unit 1: only unit_reconstruction_summary.json, no branches/merged files.
	_make_unit_dir(
		well_dir,
		unit_id=1,
		branches=None,
		electrode_ids=None,
		unit_location_xy=None,
	)
	stage_config = _make_stage_config(well_metadata_lookup=_lookup())
	result = run_analysis_compute_metrics_stage(
		dataset_index=0,
		dataset_id="ds-fixture",
		h5_path=h5_path,
		stream_id="well000",
		mea_output_root=mea_output_root,
		output_rel_root="analysis_outputs",
		stage_config=stage_config,
		force_restart=False,
	)
	df = pd.read_parquet(result.analysis_out_dir / "tables" / "units.parquet")
	row = df.iloc[0]
	assert row["recon_status"] == "ok"
	# branch_count is NaN since branches.json is missing, so num_branches must be None.
	assert _is_nan_scalar(row["branch_count"])
	assert row["num_branches"] is None
	assert _is_nan_scalar(row["recon_density"])


def _is_nan_scalar(value: Any) -> bool:
	import math

	if value is None:
		return False
	try:
		return math.isnan(float(value))
	except (TypeError, ValueError):
		return False
