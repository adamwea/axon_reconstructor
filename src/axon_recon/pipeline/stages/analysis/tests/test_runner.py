from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir

from axon_recon.pipeline.stages.analysis.models.inputs import AnalysisInputs
from axon_recon.pipeline.stages.analysis.runner import run_analysis_stage


def _write_json(path: Path, payload: dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_run_analysis_stage_writes_summary_and_per_well_csvs(tmp_path: Path) -> None:
	output_root = tmp_path / "outputs"
	h5_path = tmp_path / "raw_data" / "Proj" / "Proj" / "250101" / "MouseA" / "AxonTracking" / "000001" / "data.raw.h5"
	h5_path.parent.mkdir(parents=True, exist_ok=True)
	h5_path.write_text("", encoding="utf-8")

	# Resolve expected well output location via the same contract used by stage code.
	well_out_dir = compute_mea_analysis_output_dir(
		output_root=output_root,
		data_file=h5_path,
		well="well001",
	)

	recon_out_dir = well_out_dir / "recon_outputs"
	branches_ok = recon_out_dir / "units" / "0094" / "branches.json"
	_write_json(
		branches_ok,
		{
			"branches": [
				{
					"branch_index": 0,
					"channels": [1, 2, 3],
					"velocity": 2.0,
					"distances": [0.0, 100.0],
					"polyline_xy": [[0.0, 0.0], [100.0, 0.0]],
				},
				{
					"branch_index": 1,
					"channels": [4, 5],
					"velocity": 1.0,
					"distances": [0.0, 50.0],
					"polyline_xy": [[0.0, 0.0], [30.0, 40.0]],
				},
			],
		},
	)
	_write_json(
		recon_out_dir / "reconstruction_summary.json",
		{
			"units": [
				{"unit_id": 94, "status": "ok", "outputs": {"branches_json": str(branches_ok)}},
				{"unit_id": 95, "status": "error", "outputs": {}},
			]
		},
	)
	_write_json(
		well_out_dir / "templates_outputs" / "templates_summary.json",
		{
			"units": [
				{"unit_id": 94, "status": "ok"},
				{"unit_id": 95, "status": "ok"},
			]
		},
	)

	inputs = AnalysisInputs(
		h5_path=h5_path,
		stream_id="well001",
		mea_output_root=output_root,
		metrics={
			"per_branch": {
				"reldir": "branch_metrics/",
				"velocity": {"write_csv": True, "csv_relpath": "branch_velocities.csv"},
				"length": {"write_csv": True, "csv_relpath": "branch_lengths.csv"},
			},
			"per_unit": {
				"reldir": "unit_metrics/",
				"branch_length_stats": {
					"write_csv": True,
					"csv_relpath": "branch_length_stats.csv",
					"source_level": "per_branch",
					"source_metric": "length",
					"columns": ["n", "total", "mean"],
				},
				"n_branches": {"write_csv": True, "csv_relpath": "n_branches.csv"},
			},
			"per_well": {
				"reldir": "well_metrics/",
				"n_units_total": {"write_csv": True, "csv_relpath": "n_units_total.csv"},
				"n_units_reconstructed": {"write_csv": True, "csv_relpath": "n_units_reconstructed.csv"},
				"n_units_with_branches": {"write_csv": True, "csv_relpath": "n_units_with_branches.csv"},
				"frac_units_with_branches": {"write_csv": True, "csv_relpath": "frac_units_with_branches.csv"},
				"branches_per_unit_stats": {
					"write_csv": True,
					"csv_relpath": "branches_per_unit_stats.csv",
					"source_level": "per_unit",
					"source_metric": "n_branches",
					"columns": ["n", "total", "mean"],
				},
			},
		},
	)

	result = run_analysis_stage(inputs)
	assert result.summary_json.exists()
	assert result.analysis_out_dir == well_out_dir / "analysis_outputs"
	assert "per_well.n_units_total" in result.outputs
	assert Path(result.outputs["per_well.n_units_total"]).exists()
	assert "per_branch.velocity" in result.outputs
	assert "per_unit.branch_length_stats" in result.outputs
	assert "per_well.branches_per_unit_stats" in result.outputs

	summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
	counts = summary["unit_counts"]
	assert counts["n_units_total"] == 2
	assert counts["n_units_reconstructed"] == 1
	assert counts["n_units_with_branches"] == 1
	assert float(counts["frac_units_with_branches"]) == 0.5

	with open(result.outputs["per_unit.branch_length_stats"], "r", encoding="utf-8") as f:
		rows = list(csv.DictReader(f))
	row_94 = next(row for row in rows if row.get("unit_id") == "94")
	assert row_94["n"] == "2"
	assert float(row_94["total"]) == 150.0
	assert float(row_94["mean"]) == 75.0

	with open(result.outputs["per_unit.n_branches"], "r", encoding="utf-8") as f:
		rows = list(csv.DictReader(f))
	row_94_n = next(row for row in rows if row.get("unit_id") == "94")
	assert row_94_n["value"] == "2"

	with open(result.outputs["per_well.branches_per_unit_stats"], "r", encoding="utf-8") as f:
		well_rows = list(csv.DictReader(f))
	assert len(well_rows) == 1
	assert well_rows[0]["n"] == "2"
	assert float(well_rows[0]["total"]) == 2.0
	assert float(well_rows[0]["mean"]) == 1.0


def test_run_analysis_stage_reads_template_outputs_summary_path(tmp_path: Path) -> None:
	output_root = tmp_path / "outputs"
	h5_path = tmp_path / "raw_data" / "Proj" / "Proj" / "250101" / "MouseA" / "AxonTracking" / "000002" / "data.raw.h5"
	h5_path.parent.mkdir(parents=True, exist_ok=True)
	h5_path.write_text("", encoding="utf-8")

	well_out_dir = compute_mea_analysis_output_dir(
		output_root=output_root,
		data_file=h5_path,
		well="well001",
	)

	recon_out_dir = well_out_dir / "recon_outputs"
	branches_path = recon_out_dir / "units" / "0001" / "branches.json"
	_write_json(branches_path, {"branches": []})
	_write_json(
		recon_out_dir / "reconstruction_summary.json",
		{
			"units": [
				{"unit_id": 1, "status": "ok", "outputs": {"branches_json": str(branches_path)}},
			]
		},
	)

	template_out_dir = well_out_dir / "template_outputs"
	template_dir = template_out_dir / "units" / "0001"
	template_dir.mkdir(parents=True, exist_ok=True)
	template_npy = template_dir / "merged_template.npy"
	locs_npy = template_dir / "merged_channel_locations.npy"
	np.save(template_npy, np.asarray([[0.0, 1.0, 0.5], [0.0, 2.0, 1.0]], dtype=float))
	np.save(locs_npy, np.asarray([[0.0, 0.0], [10.0, 20.0]], dtype=float))
	_write_json(
		template_out_dir / "templates_summary.json",
		{
			"units": [
				{
					"unit_id": 1,
					"status": "ok",
					"outputs": {
						"merged_template_npy": str(template_npy),
						"merged_template_channel_locations_npy": str(locs_npy),
					},
				},
			]
		},
	)

	inputs = AnalysisInputs(
		h5_path=h5_path,
		stream_id="well001",
		mea_output_root=output_root,
		metrics={
			"per_unit": {
				"reldir": "unit_metrics/",
				"template_area": {"write_csv": True, "csv_relpath": "template_area.csv"},
			},
			"per_well": {
				"reldir": "well_metrics/",
				"template_area_per_unit_stats": {
					"write_csv": True,
					"csv_relpath": "template_area_per_unit_stats.csv",
					"source_level": "per_unit",
					"source_metric": "template_area",
					"columns": ["n", "total", "mean"],
				},
			},
		},
	)

	result = run_analysis_stage(inputs)

	with open(result.outputs["per_unit.template_area"], "r", encoding="utf-8") as f:
		rows = list(csv.DictReader(f))
	assert len(rows) == 1
	assert rows[0]["unit_id"] == "1"
	assert float(rows[0]["value"]) == 200.0

	with open(result.outputs["per_well.template_area_per_unit_stats"], "r", encoding="utf-8") as f:
		well_rows = list(csv.DictReader(f))
	assert len(well_rows) == 1
	assert well_rows[0]["n"] == "1"
	assert float(well_rows[0]["total"]) == 200.0
	assert float(well_rows[0]["mean"]) == 200.0

	summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
	assert summary["inputs_available"]["templates_summary_json"] == str(template_out_dir / "templates_summary.json")
