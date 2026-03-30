from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from axon_recon.pipeline.stages.reconstruct.models.inputs import CircleReconConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import CircleReconDisplayConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import CircleReconOutputConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import PerUnitOutputsConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import ReconstructionInputs
from axon_recon.pipeline.stages.reconstruct.runner import run_reconstruct_stage


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

	monkeypatch.setattr(reconstruct_runner, "compute_mea_analysis_output_dir", _fake_compute_mea_analysis_output_dir)
	monkeypatch.setattr(reconstruct_runner, "_resolve_templates_dirs", _fake_resolve_templates_dirs)
	monkeypatch.setattr(reconstruct_runner, "import_axon_velocity", _fake_import_axon_velocity)
	monkeypatch.setattr(reconstruct_runner, "load_templates_for_unit", _fake_load_templates_for_unit)
	monkeypatch.setattr(reconstruct_runner, "compute_graph_tracking", _fake_compute_graph_tracking)
	monkeypatch.setattr(reconstruct_runner, "write_unit_amplitude_map_png", _fake_write_unit_amplitude_map_png)
	monkeypatch.setattr(reconstruct_runner, "write_unit_circle_recon_plot", _fake_write_unit_circle_recon_plot)
	monkeypatch.setattr(reconstruct_runner, "write_amplitude_map_summary_png", _fake_write_amplitude_map_summary_png)

	inputs = ReconstructionInputs(
		h5_path=tmp_path / "input.raw.h5",
		stream_id="well001",
		mea_output_root=tmp_path,
		output_rel_root="recon_outputs",
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
	assert "summary_png" in outputs
	assert "report_md" in outputs
	assert Path(outputs["summary_png"]).exists()
	assert Path(outputs["report_md"]).exists()

	units = payload.get("units", [])
	assert len(units) == 2
	for unit in units:
		assert unit.get("status") == "ok"
		assert "amplitude_map_png" in dict(unit.get("outputs", {}))
		assert "circle_recon_png" in dict(unit.get("outputs", {}))
		assert "circle_recon_svg" in dict(unit.get("outputs", {}))


def test_run_reconstruct_stage_errors_when_requested_source_fails_and_fallback_is_disabled(tmp_path: Path, monkeypatch) -> None:
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
	units = payload.get("units", [])
	assert len(units) == 1
	assert units[0].get("status") == "error"
	assert "Fallback to merged template source is disabled" in str(units[0].get("error"))


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
