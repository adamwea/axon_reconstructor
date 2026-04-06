from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from axon_recon.pipeline.stages.reconstruct.models.inputs import CircleReconConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import CircleReconDisplayConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import CircleReconOutputConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import PerUnitOutputsConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import ReconstructionGridReportsConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import ReconstructionInputs
from axon_recon.pipeline.stages.reconstruct.models.inputs import ReconstructionReportsConfig
from axon_recon.pipeline.stages.reconstruct.runner import run_reconstruct_stage
from axon_recon.pipeline.stages.templates.models.inputs import FootprintMapGridReportConfig


def test_run_reconstruct_stage_emits_summary_and_report_outputs(tmp_path: Path, monkeypatch) -> None:
	from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner

	well_out_dir = tmp_path / "well001"
	well_out_dir.mkdir(parents=True, exist_ok=True)

	def _fake_compute_mea_analysis_output_dir(*, output_root: Path, data_file: Path, well: str) -> Path:
		return well_out_dir

	def _fake_resolve_templates_dirs(_well_out_dir: Path, **kwargs) -> tuple[Path, Path, Path]:
		templates_out = tmp_path / "templates_out"
		merged = tmp_path / "templates_merged"
		full = tmp_path / "templates_full"
		templates_out.mkdir(parents=True, exist_ok=True)
		merged.mkdir(parents=True, exist_ok=True)
		full.mkdir(parents=True, exist_ok=True)
		return templates_out, merged, full

	def _fake_import_axon_velocity(*, repo_root):
		return object()

	def _fake_load_templates_for_unit(**kwargs):
		template = np.array([[-5.0, -10.0, -3.0], [-2.0, -4.0, -1.0], [-1.0, -6.0, -2.0]], dtype=float)
		locs = np.array([[0.0, 0.0], [17.5, 0.0], [0.0, 17.5]], dtype=float)
		return template, locs, template, locs, 10_000.0, "square_from_merged"

	def _fake_compute_graph_tracking(**kwargs):
		return object()

	def _fake_write_unit_amplitude_map_png(**kwargs):
		out = Path(kwargs["output_png"])
		out.parent.mkdir(parents=True, exist_ok=True)
		out.write_bytes(b"png")

	def _fake_write_unit_circle_recon_plot(**kwargs):
		if kwargs["circle_config"].output.write_png:
			out_png = Path(kwargs["output_png"])
			out_png.parent.mkdir(parents=True, exist_ok=True)
			out_png.write_bytes(b"circle_png")
		if kwargs["circle_config"].output.write_svg:
			out_svg = Path(kwargs["output_svg"])
			out_svg.parent.mkdir(parents=True, exist_ok=True)
			out_svg.write_text("<svg></svg>", encoding="utf-8")
		return {}

	def _fake_write_amplitude_map_summary_png(*, entries, output_png: Path, ncols: int, title: str = "") -> bool:
		output_png.parent.mkdir(parents=True, exist_ok=True)
		output_png.write_bytes(b"summary")
		return True

	def _fake_render_footprint_map_grid_from_assets(**kwargs):
		out: dict[str, str] = {}
		if bool(kwargs["config"].write_png):
			png_path = Path(kwargs["png_path"])
			png_path.parent.mkdir(parents=True, exist_ok=True)
			png_path.write_bytes(b"grid_png")
			out[str(kwargs["png_output_key"])] = str(png_path)
		if bool(kwargs.get("write_svg", False)) and kwargs.get("svg_path", None) is not None:
			svg_path = Path(kwargs["svg_path"])
			svg_path.parent.mkdir(parents=True, exist_ok=True)
			svg_path.write_text("<svg></svg>", encoding="utf-8")
			out[str(kwargs.get("svg_output_key", "circle_recon_grid_temp_svg"))] = str(svg_path)
		return out

	def _fake_finalize_grid_svg_output(**kwargs):
		raw_outputs = dict(kwargs["raw_outputs"])
		if bool(kwargs["write_svg"]) and kwargs["temp_svg_output_key"] in raw_outputs:
			final_svg_path = Path(kwargs["final_svg_path"])
			final_svg_path.parent.mkdir(parents=True, exist_ok=True)
			final_svg_path.write_text("<svg></svg>", encoding="utf-8")
			raw_outputs[str(kwargs["final_svg_output_key"])] = str(final_svg_path)
			if not bool(kwargs["keep_temp_svg"]):
				raw_outputs.pop(str(kwargs["temp_svg_output_key"]), None)
		return raw_outputs

	monkeypatch.setattr(reconstruct_runner, "compute_mea_analysis_output_dir", _fake_compute_mea_analysis_output_dir)
	monkeypatch.setattr(reconstruct_runner, "_resolve_templates_dirs", _fake_resolve_templates_dirs)
	monkeypatch.setattr(reconstruct_runner, "import_axon_velocity", _fake_import_axon_velocity)
	monkeypatch.setattr(reconstruct_runner, "load_templates_for_unit", _fake_load_templates_for_unit)
	monkeypatch.setattr(reconstruct_runner, "compute_graph_tracking", _fake_compute_graph_tracking)
	monkeypatch.setattr(reconstruct_runner, "write_unit_amplitude_map_png", _fake_write_unit_amplitude_map_png)
	monkeypatch.setattr(reconstruct_runner, "write_unit_circle_recon_plot", _fake_write_unit_circle_recon_plot)
	monkeypatch.setattr(reconstruct_runner, "write_amplitude_map_summary_png", _fake_write_amplitude_map_summary_png)
	monkeypatch.setattr(reconstruct_runner, "render_footprint_map_grid_from_assets", _fake_render_footprint_map_grid_from_assets)
	monkeypatch.setattr(reconstruct_runner, "finalize_grid_svg_output", _fake_finalize_grid_svg_output)

	inputs = ReconstructionInputs(
		h5_path=tmp_path / "input.raw.h5",
		stream_id="well001",
		mea_output_root=tmp_path,
		output_rel_root="recon_outputs",
		reports=ReconstructionReportsConfig(
			grids=ReconstructionGridReportsConfig(
				circle_recon_grid=FootprintMapGridReportConfig(
					write_png=True,
					write_svg=True,
					png_relpath="reports/circle_recon_grid.png",
					svg_relpath="reports/circle_recon_grid.svg",
					temp_svg_relpath="reports/circle_recon_grid__temp.svg",
				)
			)
		),
		write_summary_png=True,
		summary_png_relpath="reports/summary.png",
		summary_grid_ncols=2,
		write_report_md=True,
		report_md_relpath="reports/report.md",
		unit_ids=[1, 2],
		n_jobs=1,
		per_unit_outputs=PerUnitOutputsConfig(
			write_branches_raw_json=False,
			write_branches_json=False,
			write_heuristics_json=False,
			write_gtr_pkl=False,
			write_gtr_json=False,
			write_amplitude_map_png=True,
			amplitude_map_png_relpath="maps/amplitude_map.png",
			circle_recon=CircleReconConfig(
				display=CircleReconDisplayConfig(),
				output=CircleReconOutputConfig(
					write_png=True,
					write_svg=True,
					relpath="maps/circle_recon",
					dpi=300.0,
				),
			),
		),
	)

	result = run_reconstruct_stage(inputs)
	assert result.summary_json.exists()

	payload = json.loads(result.summary_json.read_text(encoding="utf-8"))
	outputs = payload.get("outputs", {})
	assert payload.get("units_ok") == 2
	assert payload.get("units_error") == 0
	assert "circle_recon_grid_png" in outputs
	assert "circle_recon_grid_svg" in outputs
	assert "summary_png" in outputs
	assert "report_md" in outputs
	assert Path(outputs["circle_recon_grid_png"]).exists()
	assert Path(outputs["circle_recon_grid_svg"]).exists()
	assert Path(outputs["summary_png"]).exists()
	assert Path(outputs["report_md"]).exists()

	units = payload.get("units", [])
	assert len(units) == 2
	for unit in units:
		assert unit.get("status") == "ok"
		assert "amplitude_map_png" in dict(unit.get("outputs", {}))
		assert "circle_recon_png" in dict(unit.get("outputs", {}))
		assert "circle_recon_svg" in dict(unit.get("outputs", {}))


def test_run_reconstruct_stage_sorts_circle_grid_inputs_by_max_ptp(tmp_path: Path, monkeypatch) -> None:
	from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner

	well_out_dir = tmp_path / "well001"
	well_out_dir.mkdir(parents=True, exist_ok=True)

	def _fake_compute_mea_analysis_output_dir(*, output_root: Path, data_file: Path, well: str) -> Path:
		return well_out_dir

	def _fake_resolve_templates_dirs(_well_out_dir: Path, **kwargs) -> tuple[Path, Path, Path]:
		templates_out = tmp_path / "templates_out"
		merged = tmp_path / "templates_merged"
		full = tmp_path / "templates_full"
		templates_out.mkdir(parents=True, exist_ok=True)
		merged.mkdir(parents=True, exist_ok=True)
		full.mkdir(parents=True, exist_ok=True)
		return templates_out, merged, full

	def _fake_import_axon_velocity(*, repo_root):
		return object()

	def _fake_load_templates_for_unit(**kwargs):
		unit_id = int(kwargs.get("unit_id", 0))
		scale = 2.0 if unit_id == 2 else 0.5
		template = np.array(
			[
				[-5.0, -10.0, -3.0],
				[-2.0, -4.0, -1.0],
				[-1.0, -6.0, -2.0],
			],
			dtype=float,
		) * scale
		locs = np.array([[0.0, 0.0], [17.5, 0.0], [0.0, 17.5]], dtype=float)
		return template, locs, template, locs, 10_000.0, "square_from_merged"

	def _fake_compute_graph_tracking(**kwargs):
		return object()

	def _fake_write_unit_circle_recon_plot(**kwargs):
		out_png = Path(kwargs["output_png"])
		out_png.parent.mkdir(parents=True, exist_ok=True)
		out_png.write_bytes(b"circle_png")
		return {}

	captured_circle_order: list[str] = []

	def _unit_dir_token(path_like: Path | str) -> str:
		path = Path(path_like)
		for parent in path.parents:
			name = parent.name
			if len(name) == 4 and name.isdigit():
				return name
		return path.parent.name

	def _fake_render_footprint_map_grid_from_assets(**kwargs):
		nonlocal captured_circle_order
		captured_circle_order = [_unit_dir_token(path) for path in kwargs.get("image_paths", [])]
		png_path = Path(kwargs["png_path"])
		png_path.parent.mkdir(parents=True, exist_ok=True)
		png_path.write_bytes(b"grid_png")
		return {str(kwargs.get("png_output_key", "circle_recon_grid_png")): str(png_path)}

	monkeypatch.setattr(reconstruct_runner, "compute_mea_analysis_output_dir", _fake_compute_mea_analysis_output_dir)
	monkeypatch.setattr(reconstruct_runner, "_resolve_templates_dirs", _fake_resolve_templates_dirs)
	monkeypatch.setattr(reconstruct_runner, "import_axon_velocity", _fake_import_axon_velocity)
	monkeypatch.setattr(reconstruct_runner, "load_templates_for_unit", _fake_load_templates_for_unit)
	monkeypatch.setattr(reconstruct_runner, "compute_graph_tracking", _fake_compute_graph_tracking)
	monkeypatch.setattr(reconstruct_runner, "write_unit_circle_recon_plot", _fake_write_unit_circle_recon_plot)
	monkeypatch.setattr(reconstruct_runner, "render_footprint_map_grid_from_assets", _fake_render_footprint_map_grid_from_assets)
	monkeypatch.setattr(reconstruct_runner, "finalize_grid_svg_output", lambda **kwargs: dict(kwargs["raw_outputs"]))

	inputs = ReconstructionInputs(
		h5_path=tmp_path / "input.raw.h5",
		stream_id="well001",
		mea_output_root=tmp_path,
		output_rel_root="recon_outputs",
		reports=ReconstructionReportsConfig(
			grids=ReconstructionGridReportsConfig(
				sort_by="max_ptp",
				circle_recon_grid=FootprintMapGridReportConfig(
					write_png=True,
					write_svg=False,
					png_relpath="reports/circle_recon_grid.png",
				),
			),
		),
		write_summary_png=False,
		write_report_md=False,
		unit_ids=[1, 2],
		n_jobs=1,
		per_unit_outputs=PerUnitOutputsConfig(
			write_branches_raw_json=False,
			write_branches_json=False,
			write_heuristics_json=False,
			write_gtr_pkl=False,
			write_gtr_json=False,
			write_amplitude_map_png=False,
			circle_recon=CircleReconConfig(
				display=CircleReconDisplayConfig(),
				output=CircleReconOutputConfig(write_png=True, write_svg=False, relpath="maps/circle_recon", dpi=300.0),
			),
		),
	)

	result = run_reconstruct_stage(inputs)
	assert result.summary_json.exists()
	assert captured_circle_order == ["0002", "0001"]
	payload = json.loads(result.summary_json.read_text(encoding="utf-8"))
	assert payload.get("reports_grid_sort_by") == "max_ptp"


def test_run_reconstruct_stage_force_restart_clears_output_root(tmp_path: Path, monkeypatch) -> None:
	from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner

	well_out_dir = tmp_path / "well001"
	stale_file = well_out_dir / "recon_outputs" / "stale.txt"
	stale_file.parent.mkdir(parents=True, exist_ok=True)
	stale_file.write_text("stale", encoding="utf-8")

	def _fake_compute_mea_analysis_output_dir(*, output_root: Path, data_file: Path, well: str) -> Path:
		return well_out_dir

	def _fake_resolve_templates_dirs(_well_out_dir: Path, **kwargs) -> tuple[Path, Path, Path]:
		templates_out = tmp_path / "templates_out"
		merged = tmp_path / "templates_merged"
		full = tmp_path / "templates_full"
		templates_out.mkdir(parents=True, exist_ok=True)
		merged.mkdir(parents=True, exist_ok=True)
		full.mkdir(parents=True, exist_ok=True)
		return templates_out, merged, full

	def _fake_import_axon_velocity(*, repo_root):
		return object()

	def _fake_load_templates_for_unit(**kwargs):
		template = np.array([[-5.0, -10.0, -3.0], [-2.0, -4.0, -1.0], [-1.0, -6.0, -2.0]], dtype=float)
		locs = np.array([[0.0, 0.0], [17.5, 0.0], [0.0, 17.5]], dtype=float)
		return template, locs, template, locs, 10_000.0, "square_from_merged"

	def _fake_compute_graph_tracking(**kwargs):
		return object()

	def _fake_write_unit_circle_recon_plot(**kwargs):
		out_png = Path(kwargs["output_png"])
		out_png.parent.mkdir(parents=True, exist_ok=True)
		out_png.write_bytes(b"circle_png")
		return {}

	monkeypatch.setattr(reconstruct_runner, "compute_mea_analysis_output_dir", _fake_compute_mea_analysis_output_dir)
	monkeypatch.setattr(reconstruct_runner, "_resolve_templates_dirs", _fake_resolve_templates_dirs)
	monkeypatch.setattr(reconstruct_runner, "import_axon_velocity", _fake_import_axon_velocity)
	monkeypatch.setattr(reconstruct_runner, "load_templates_for_unit", _fake_load_templates_for_unit)
	monkeypatch.setattr(reconstruct_runner, "compute_graph_tracking", _fake_compute_graph_tracking)
	monkeypatch.setattr(reconstruct_runner, "write_unit_circle_recon_plot", _fake_write_unit_circle_recon_plot)

	inputs = ReconstructionInputs(
		h5_path=tmp_path / "input.raw.h5",
		stream_id="well001",
		mea_output_root=tmp_path,
		output_rel_root="recon_outputs",
		unit_ids=[1, 2],
		force_restart=True,
		n_jobs=1,
		per_unit_outputs=PerUnitOutputsConfig(
			write_branches_raw_json=False,
			write_branches_json=False,
			write_heuristics_json=False,
			write_gtr_pkl=False,
			write_gtr_json=False,
			write_amplitude_map_png=False,
			circle_recon=CircleReconConfig(
				display=CircleReconDisplayConfig(),
				output=CircleReconOutputConfig(write_png=True, write_svg=False, relpath="maps/circle_recon", dpi=300.0),
			),
		),
	)

	result = run_reconstruct_stage(inputs)
	assert result.summary_json.exists()
	assert not stale_file.exists()


def test_run_reconstruct_stage_unit_force_restart_preserves_stage_reports_when_not_overwriting(tmp_path: Path, monkeypatch) -> None:
	from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner

	well_out_dir = tmp_path / "well001"
	reconstruction_out_dir = well_out_dir / "recon_outputs"
	report_dir = reconstruction_out_dir / "reports"
	report_dir.mkdir(parents=True, exist_ok=True)
	circle_grid_png = report_dir / "circle_recon_grid.png"
	summary_png = report_dir / "summary.png"
	report_md = report_dir / "report.md"
	circle_grid_png.write_bytes(b"existing-grid")
	summary_png.write_bytes(b"existing-summary")
	report_md.write_text("existing report", encoding="utf-8")
	stale_target_file = reconstruction_out_dir / "units" / "0001" / "stale.txt"
	stale_target_file.parent.mkdir(parents=True, exist_ok=True)
	stale_target_file.write_text("stale", encoding="utf-8")
	other_unit_file = reconstruction_out_dir / "units" / "0002" / "keep.txt"
	other_unit_file.parent.mkdir(parents=True, exist_ok=True)
	other_unit_file.write_text("keep", encoding="utf-8")

	def _fake_compute_mea_analysis_output_dir(*, output_root: Path, data_file: Path, well: str) -> Path:
		return well_out_dir

	def _fake_resolve_templates_dirs(_well_out_dir: Path, **kwargs) -> tuple[Path, Path, Path]:
		templates_out = tmp_path / "templates_out"
		merged = tmp_path / "templates_merged"
		full = tmp_path / "templates_full"
		templates_out.mkdir(parents=True, exist_ok=True)
		merged.mkdir(parents=True, exist_ok=True)
		full.mkdir(parents=True, exist_ok=True)
		return templates_out, merged, full

	def _fake_import_axon_velocity(*, repo_root):
		return object()

	def _fake_load_templates_for_unit(**kwargs):
		template = np.array([[-5.0, -10.0, -3.0], [-2.0, -4.0, -1.0]], dtype=float)
		locs = np.array([[0.0, 0.0], [17.5, 0.0]], dtype=float)
		return template, locs, template, locs, 10_000.0, "square_from_merged"

	def _fake_compute_graph_tracking(**kwargs):
		return object()

	def _fake_write_unit_circle_recon_plot(**kwargs):
		out_png = Path(kwargs["output_png"])
		out_png.parent.mkdir(parents=True, exist_ok=True)
		out_png.write_bytes(b"circle_png")
		return {}

	def _raise_unexpected(*args, **kwargs):
		raise AssertionError("stage report generation should have been skipped")

	monkeypatch.setattr(reconstruct_runner, "compute_mea_analysis_output_dir", _fake_compute_mea_analysis_output_dir)
	monkeypatch.setattr(reconstruct_runner, "_resolve_templates_dirs", _fake_resolve_templates_dirs)
	monkeypatch.setattr(reconstruct_runner, "import_axon_velocity", _fake_import_axon_velocity)
	monkeypatch.setattr(reconstruct_runner, "load_templates_for_unit", _fake_load_templates_for_unit)
	monkeypatch.setattr(reconstruct_runner, "compute_graph_tracking", _fake_compute_graph_tracking)
	monkeypatch.setattr(reconstruct_runner, "write_unit_circle_recon_plot", _fake_write_unit_circle_recon_plot)
	monkeypatch.setattr(reconstruct_runner, "render_footprint_map_grid_from_assets", _raise_unexpected)
	monkeypatch.setattr(reconstruct_runner, "write_amplitude_map_summary_png", _raise_unexpected)
	monkeypatch.setattr(reconstruct_runner, "write_reconstruct_report_markdown", _raise_unexpected)

	inputs = ReconstructionInputs(
		h5_path=tmp_path / "input.raw.h5",
		stream_id="well001",
		mea_output_root=tmp_path,
		output_rel_root="recon_outputs",
		reports=ReconstructionReportsConfig(
			overwrite_on_unit_rerun=False,
			grids=ReconstructionGridReportsConfig(
				circle_recon_grid=FootprintMapGridReportConfig(write_png=True, write_svg=False, png_relpath="reports/circle_recon_grid.png")
			),
		),
		write_summary_png=True,
		summary_png_relpath="reports/summary.png",
		write_report_md=True,
		report_md_relpath="reports/report.md",
		unit_ids=[1],
		force_restart=True,
		n_jobs=1,
		per_unit_outputs=PerUnitOutputsConfig(
			write_branches_raw_json=False,
			write_branches_json=False,
			write_heuristics_json=False,
			write_gtr_pkl=False,
			write_gtr_json=False,
			write_amplitude_map_png=False,
			circle_recon=CircleReconConfig(
				display=CircleReconDisplayConfig(),
				output=CircleReconOutputConfig(write_png=True, write_svg=False, relpath="maps/circle_recon", dpi=300.0),
			),
		),
	)

	result = run_reconstruct_stage(inputs)
	payload = json.loads(result.summary_json.read_text(encoding="utf-8"))
	assert payload["reports_overwrite_skipped"] is True
	assert payload["outputs"]["circle_recon_grid_png"] == str(circle_grid_png)
	assert payload["outputs"]["summary_png"] == str(summary_png)
	assert payload["outputs"]["report_md"] == str(report_md)
	assert circle_grid_png.read_bytes() == b"existing-grid"
	assert summary_png.read_bytes() == b"existing-summary"
	assert report_md.read_text(encoding="utf-8") == "existing report"
	assert not stale_target_file.exists()
	assert other_unit_file.exists()


def test_run_reconstruct_stage_cleans_failed_unit_outputs_and_writes_failed_units_summary(tmp_path: Path, monkeypatch) -> None:
	from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner

	well_out_dir = tmp_path / "well001"
	well_out_dir.mkdir(parents=True, exist_ok=True)

	def _fake_compute_mea_analysis_output_dir(*, output_root: Path, data_file: Path, well: str) -> Path:
		return well_out_dir

	def _fake_resolve_templates_dirs(_well_out_dir: Path, **kwargs) -> tuple[Path, Path, Path]:
		templates_out = tmp_path / "templates_out"
		merged = tmp_path / "templates_merged"
		full = tmp_path / "templates_full"
		templates_out.mkdir(parents=True, exist_ok=True)
		merged.mkdir(parents=True, exist_ok=True)
		full.mkdir(parents=True, exist_ok=True)
		return templates_out, merged, full

	def _fake_import_axon_velocity(*, repo_root):
		return object()

	def _fake_load_templates_for_unit(**kwargs):
		template = np.array([[-5.0, -10.0, -3.0], [-2.0, -4.0, -1.0]], dtype=float)
		locs = np.array([[0.0, 0.0], [17.5, 0.0]], dtype=float)
		return template, locs, template, locs, 10_000.0, "square_from_merged"

	def _fake_compute_graph_tracking(**kwargs):
		return object()

	def _fake_write_unit_circle_recon_plot(**kwargs):
		out_png = Path(kwargs["output_png"])
		out_png.parent.mkdir(parents=True, exist_ok=True)
		out_png.write_bytes(b"circle_png")
		if int(kwargs["unit_id"]) == 2:
			raise RuntimeError("No branches found")
		return {}

	monkeypatch.setattr(reconstruct_runner, "compute_mea_analysis_output_dir", _fake_compute_mea_analysis_output_dir)
	monkeypatch.setattr(reconstruct_runner, "_resolve_templates_dirs", _fake_resolve_templates_dirs)
	monkeypatch.setattr(reconstruct_runner, "import_axon_velocity", _fake_import_axon_velocity)
	monkeypatch.setattr(reconstruct_runner, "load_templates_for_unit", _fake_load_templates_for_unit)
	monkeypatch.setattr(reconstruct_runner, "compute_graph_tracking", _fake_compute_graph_tracking)
	monkeypatch.setattr(reconstruct_runner, "write_unit_circle_recon_plot", _fake_write_unit_circle_recon_plot)

	inputs = ReconstructionInputs(
		h5_path=tmp_path / "input.raw.h5",
		stream_id="well001",
		mea_output_root=tmp_path,
		output_rel_root="recon_outputs",
		cleanup_failed_unit_outputs=True,
		failed_units_summary_relpath="reports/failed_units.json",
		write_summary_png=False,
		write_report_md=False,
		unit_ids=[1, 2],
		n_jobs=1,
		per_unit_outputs=PerUnitOutputsConfig(
			write_branches_raw_json=False,
			write_branches_json=False,
			write_heuristics_json=False,
			write_gtr_pkl=False,
			write_gtr_json=False,
			write_amplitude_map_png=False,
			circle_recon=CircleReconConfig(
				display=CircleReconDisplayConfig(),
				output=CircleReconOutputConfig(write_png=True, write_svg=False, relpath="maps/circle_recon", dpi=300.0),
			),
		),
	)

	result = run_reconstruct_stage(inputs)
	payload = json.loads(result.summary_json.read_text(encoding="utf-8"))
	failed_summary_path = Path(payload["failed_units_summary_json"])
	assert payload["cleanup_failed_unit_outputs"] is True
	assert failed_summary_path.exists()
	failed_payload = json.loads(failed_summary_path.read_text(encoding="utf-8"))
	assert failed_payload["failed_unit_count"] == 1
	assert failed_payload["units"][0]["unit_id"] == 2
	assert failed_payload["units"][0]["cleanup_failed_outputs_applied"] is True
	removed_paths = failed_payload["units"][0]["removed_output_paths"]
	assert any(path.endswith("maps/circle_recon.png") for path in removed_paths)
	failing_unit_png = well_out_dir / "recon_outputs" / "units" / "0002" / "maps" / "circle_recon.png"
	failing_unit_summary = well_out_dir / "recon_outputs" / "units" / "0002" / "unit_reconstruction_summary.json"
	assert not failing_unit_png.exists()
	assert failing_unit_summary.exists()
	unit_rows = {int(unit["unit_id"]): unit for unit in payload["units"]}
	assert unit_rows[1]["status"] == "ok"
	assert unit_rows[2]["status"] == "error"
	assert unit_rows[2]["outputs"] == {}
	assert result.units[1].unit_id == 2
	assert result.units[1].outputs == {}


def test_run_reconstruct_stage_errors_when_requested_source_fails_and_fallback_is_disabled(tmp_path: Path, monkeypatch, caplog) -> None:
	from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner

	well_out_dir = tmp_path / "well001"
	well_out_dir.mkdir(parents=True, exist_ok=True)

	def _fake_compute_mea_analysis_output_dir(*, output_root: Path, data_file: Path, well: str) -> Path:
		return well_out_dir

	def _fake_resolve_templates_dirs(_well_out_dir: Path, **kwargs) -> tuple[Path, Path, Path]:
		templates_out = tmp_path / "templates_out"
		merged = tmp_path / "templates_merged"
		full = tmp_path / "templates_full"
		templates_out.mkdir(parents=True, exist_ok=True)
		merged.mkdir(parents=True, exist_ok=True)
		full.mkdir(parents=True, exist_ok=True)
		return templates_out, merged, full

	def _fake_import_axon_velocity(*, repo_root):
		return object()

	def _fake_load_templates_for_unit(**kwargs):
		plot_template = np.array([[-5.0, -10.0, -3.0], [-2.0, -4.0, -1.0]], dtype=float)
		plot_locs = np.array([[0.0, 0.0], [17.5, 0.0]], dtype=float)
		primary_template = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)
		primary_locs = np.array([[0.0, 0.0], [17.5, 0.0]], dtype=float)
		return plot_template, plot_locs, primary_template, primary_locs, 10_000.0, "square_from_merged"

	graph_calls: list[str] = []

	def _fake_compute_graph_tracking(**kwargs):
		tpl = np.asarray(kwargs["template_ch_by_t"], dtype=float)
		if float(np.max(np.abs(tpl))) <= 0.0:
			graph_calls.append("primary")
			raise ValueError("zero-size array to reduction operation maximum which has no identity")
		graph_calls.append("secondary")
		return object()

	def _fake_write_unit_amplitude_map_png(**kwargs):
		out = Path(kwargs["output_png"])
		out.parent.mkdir(parents=True, exist_ok=True)
		out.write_bytes(b"png")

	def _fake_write_unit_circle_recon_plot(**kwargs):
		if kwargs["circle_config"].output.write_png:
			out_png = Path(kwargs["output_png"])
			out_png.parent.mkdir(parents=True, exist_ok=True)
			out_png.write_bytes(b"circle_png")
		if kwargs["circle_config"].output.write_svg:
			out_svg = Path(kwargs["output_svg"])
			out_svg.parent.mkdir(parents=True, exist_ok=True)
			out_svg.write_text("<svg></svg>", encoding="utf-8")
		return {}

	monkeypatch.setattr(reconstruct_runner, "compute_mea_analysis_output_dir", _fake_compute_mea_analysis_output_dir)
	monkeypatch.setattr(reconstruct_runner, "_resolve_templates_dirs", _fake_resolve_templates_dirs)
	monkeypatch.setattr(reconstruct_runner, "import_axon_velocity", _fake_import_axon_velocity)
	monkeypatch.setattr(reconstruct_runner, "load_templates_for_unit", _fake_load_templates_for_unit)
	monkeypatch.setattr(reconstruct_runner, "compute_graph_tracking", _fake_compute_graph_tracking)
	monkeypatch.setattr(reconstruct_runner, "write_unit_amplitude_map_png", _fake_write_unit_amplitude_map_png)
	monkeypatch.setattr(reconstruct_runner, "write_unit_circle_recon_plot", _fake_write_unit_circle_recon_plot)
	caplog.set_level("WARNING", logger="axon_recon.reconstruct")

	inputs = ReconstructionInputs(
		h5_path=tmp_path / "input.raw.h5",
		stream_id="well001",
		mea_output_root=tmp_path,
		output_rel_root="recon_outputs",
		write_summary_png=False,
		write_report_md=False,
		unit_ids=[94],
		n_jobs=1,
		per_unit_outputs=PerUnitOutputsConfig(
			write_branches_raw_json=False,
			write_branches_json=False,
			write_heuristics_json=False,
			write_gtr_pkl=False,
			write_gtr_json=False,
			write_amplitude_map_png=True,
			amplitude_map_png_relpath="maps/amplitude_map.png",
			circle_recon=CircleReconConfig(
				display=CircleReconDisplayConfig(),
				output=CircleReconOutputConfig(
					write_png=True,
					write_svg=True,
					relpath="maps/circle_recon",
					dpi=300.0,
				),
			),
		),
	)

	result = run_reconstruct_stage(inputs)
	assert len(result.units) == 1
	assert result.units[0].status == "error"
	assert graph_calls == ["primary"]
	assert "Fallback to merged template source is disabled" in str(result.units[0].error)

	payload = json.loads(result.summary_json.read_text(encoding="utf-8"))
	assert payload.get("units_ok") == 0
	assert payload.get("units_error") == 1
	units = payload.get("units", [])
	assert len(units) == 1
	assert units[0].get("status") == "error"
	assert "Fallback to merged template source is disabled" in str(units[0].get("error"))
	assert "Reconstruct unit 94 failed:" in caplog.text
	assert "Traceback" not in caplog.text


def test_run_reconstruct_stage_circle_recon_uses_gtr_template_space(tmp_path: Path, monkeypatch) -> None:
	from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner

	well_out_dir = tmp_path / "well001"
	well_out_dir.mkdir(parents=True, exist_ok=True)

	plot_template = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
	plot_locs = np.array([[0.0, 0.0], [17.5, 0.0]], dtype=float)
	gtr_template = np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], dtype=float)
	gtr_locs = np.array([[0.0, 0.0], [17.5, 0.0], [35.0, 0.0]], dtype=float)

	def _fake_compute_mea_analysis_output_dir(*, output_root: Path, data_file: Path, well: str) -> Path:
		return well_out_dir

	def _fake_resolve_templates_dirs(_well_out_dir: Path, **kwargs) -> tuple[Path, Path, Path]:
		templates_out = tmp_path / "templates_out"
		merged = tmp_path / "templates_merged"
		full = tmp_path / "templates_full"
		templates_out.mkdir(parents=True, exist_ok=True)
		merged.mkdir(parents=True, exist_ok=True)
		full.mkdir(parents=True, exist_ok=True)
		return templates_out, merged, full

	def _fake_import_axon_velocity(*, repo_root):
		return object()

	def _fake_load_templates_for_unit(**kwargs):
		return plot_template, plot_locs, gtr_template, gtr_locs, 10_000.0, "square_from_merged"

	def _fake_compute_graph_tracking(**kwargs):
		return object()

	captured: dict[str, np.ndarray] = {}

	def _fake_write_unit_circle_recon_plot(**kwargs):
		captured["template"] = np.asarray(kwargs["template_ch_by_t"], dtype=float)
		captured["locs"] = np.asarray(kwargs["locs_xy"], dtype=float)
		if kwargs["circle_config"].output.write_png:
			out_png = Path(kwargs["output_png"])
			out_png.parent.mkdir(parents=True, exist_ok=True)
			out_png.write_bytes(b"circle_png")
		if kwargs["circle_config"].output.write_svg:
			out_svg = Path(kwargs["output_svg"])
			out_svg.parent.mkdir(parents=True, exist_ok=True)
			out_svg.write_text("<svg></svg>", encoding="utf-8")
		return {}

	monkeypatch.setattr(reconstruct_runner, "compute_mea_analysis_output_dir", _fake_compute_mea_analysis_output_dir)
	monkeypatch.setattr(reconstruct_runner, "_resolve_templates_dirs", _fake_resolve_templates_dirs)
	monkeypatch.setattr(reconstruct_runner, "import_axon_velocity", _fake_import_axon_velocity)
	monkeypatch.setattr(reconstruct_runner, "load_templates_for_unit", _fake_load_templates_for_unit)
	monkeypatch.setattr(reconstruct_runner, "compute_graph_tracking", _fake_compute_graph_tracking)
	monkeypatch.setattr(reconstruct_runner, "write_unit_circle_recon_plot", _fake_write_unit_circle_recon_plot)

	inputs = ReconstructionInputs(
		h5_path=tmp_path / "input.raw.h5",
		stream_id="well001",
		mea_output_root=tmp_path,
		output_rel_root="recon_outputs",
		write_summary_png=False,
		write_report_md=False,
		unit_ids=[7],
		n_jobs=1,
		per_unit_outputs=PerUnitOutputsConfig(
			write_branches_raw_json=False,
			write_branches_json=False,
			write_heuristics_json=False,
			write_gtr_pkl=False,
			write_gtr_json=False,
			write_amplitude_map_png=False,
			circle_recon=CircleReconConfig(
				display=CircleReconDisplayConfig(),
				output=CircleReconOutputConfig(
					write_png=True,
					write_svg=False,
					relpath="maps/circle_recon",
					dpi=300.0,
				),
			),
		),
	)

	result = run_reconstruct_stage(inputs)
	assert len(result.units) == 1
	assert result.units[0].status == "ok"

	np.testing.assert_allclose(captured["template"], gtr_template)
	np.testing.assert_allclose(captured["locs"], gtr_locs)


def test_run_reconstruct_stage_json_payloads_use_gtr_location_space(tmp_path: Path, monkeypatch) -> None:
	from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner

	well_out_dir = tmp_path / "well001"
	well_out_dir.mkdir(parents=True, exist_ok=True)

	plot_template = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
	plot_locs = np.array([[0.0, 0.0], [17.5, 0.0]], dtype=float)
	gtr_template = np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], dtype=float)
	gtr_locs = np.array([[0.0, 0.0], [17.5, 0.0], [35.0, 0.0]], dtype=float)

	def _fake_compute_mea_analysis_output_dir(*, output_root: Path, data_file: Path, well: str) -> Path:
		return well_out_dir

	def _fake_resolve_templates_dirs(_well_out_dir: Path, **kwargs) -> tuple[Path, Path, Path]:
		templates_out = tmp_path / "templates_out"
		merged = tmp_path / "templates_merged"
		full = tmp_path / "templates_full"
		templates_out.mkdir(parents=True, exist_ok=True)
		merged.mkdir(parents=True, exist_ok=True)
		full.mkdir(parents=True, exist_ok=True)
		return templates_out, merged, full

	def _fake_import_axon_velocity(*, repo_root):
		return object()

	def _fake_load_templates_for_unit(**kwargs):
		return plot_template, plot_locs, gtr_template, gtr_locs, 10_000.0, "square_from_merged"

	def _fake_compute_graph_tracking(**kwargs):
		return object()

	captured: dict[str, np.ndarray] = {}

	def _fake_compute_branches_with_polyline(**kwargs):
		captured["branches_locs"] = np.asarray(kwargs["locs_xy"], dtype=float)
		return {"unit_id": kwargs["unit_id"], "branches": []}

	def _fake_compute_heuristics_payload(**kwargs):
		captured["heuristics_locs"] = np.asarray(kwargs["locs_xy"], dtype=float)
		return {"unit_id": kwargs["unit_id"], "heuristics": {}}

	def _fake_compute_gtr_json_payload(**kwargs):
		captured["gtr_json_locs"] = np.asarray(kwargs["locs_xy"], dtype=float)
		return {"schema_version": 1, "unit_id": kwargs["unit_id"], "branches": []}

	monkeypatch.setattr(reconstruct_runner, "compute_mea_analysis_output_dir", _fake_compute_mea_analysis_output_dir)
	monkeypatch.setattr(reconstruct_runner, "_resolve_templates_dirs", _fake_resolve_templates_dirs)
	monkeypatch.setattr(reconstruct_runner, "import_axon_velocity", _fake_import_axon_velocity)
	monkeypatch.setattr(reconstruct_runner, "load_templates_for_unit", _fake_load_templates_for_unit)
	monkeypatch.setattr(reconstruct_runner, "compute_graph_tracking", _fake_compute_graph_tracking)
	monkeypatch.setattr(reconstruct_runner, "compute_branches_with_polyline", _fake_compute_branches_with_polyline)
	monkeypatch.setattr(reconstruct_runner, "compute_heuristics_payload", _fake_compute_heuristics_payload)
	monkeypatch.setattr(reconstruct_runner, "compute_gtr_json_payload", _fake_compute_gtr_json_payload)

	inputs = ReconstructionInputs(
		h5_path=tmp_path / "input.raw.h5",
		stream_id="well001",
		mea_output_root=tmp_path,
		output_rel_root="recon_outputs",
		write_summary_png=False,
		write_report_md=False,
		unit_ids=[7],
		n_jobs=1,
		per_unit_outputs=PerUnitOutputsConfig(
			write_branches_raw_json=False,
			write_branches_json=True,
			write_heuristics_json=True,
			write_gtr_pkl=False,
			write_gtr_json=True,
			write_amplitude_map_png=False,
			circle_recon=CircleReconConfig(
				display=CircleReconDisplayConfig(),
				output=CircleReconOutputConfig(
					write_png=False,
					write_svg=False,
					relpath="maps/circle_recon",
					dpi=300.0,
				),
			),
		),
	)

	result = run_reconstruct_stage(inputs)
	assert len(result.units) == 1
	assert result.units[0].status == "ok"

	np.testing.assert_allclose(captured["branches_locs"], gtr_locs)
	np.testing.assert_allclose(captured["heuristics_locs"], gtr_locs)
	np.testing.assert_allclose(captured["gtr_json_locs"], gtr_locs)
