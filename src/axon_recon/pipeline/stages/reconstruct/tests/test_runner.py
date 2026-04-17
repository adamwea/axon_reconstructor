from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
import pickle

import numpy as np
import pytest

from axon_recon.pipeline.stages.reconstruct.io import resolve_branch_phase_output_paths
from axon_recon.pipeline.stages.reconstruct.io import format_unit_reldir
from axon_recon.pipeline.stages.reconstruct.io import resolve_full_chip_layout_output_paths
from axon_recon.pipeline.stages.reconstruct.io import resolve_unit_summary_phase_output_paths
from axon_recon.pipeline.stages.reconstruct.io import resolve_unit_output_paths
from axon_recon.pipeline.stages.reconstruct.models.inputs import (
	ReconstructionBranchPlotOutputConfig,
	ReconstructionBranchPropagationDisplayConfig,
	ReconstructionBranchVelocityDisplayConfig,
	PerUnitOutputsConfig,
	ReconstructionAvReconsConfig,
	ReconstructionDiagnosticFigureConfig,
	ReconstructionFullChipLayoutColorConfig,
	ReconstructionFullChipLayoutDisplayConfig,
	ReconstructionFullChipLayoutOutputConfig,
	ReconstructionGenerateGtrsPhaseConfig,
	ReconstructionInputs,
	ReconstructionPhasesConfig,
	ReconstructionPlotBranchPropagationsPhaseConfig,
	ReconstructionPlotBranchVelocitiesPhaseConfig,
	ReconstructionPlotUnitSummaryPhaseConfig,
	ReconstructionReportFullChipLayoutPhaseConfig,
	ReconstructionReportReconsPhaseConfig,
	ReconstructionUnitSummaryDisplayConfig,
	ReconstructionUnitSummaryOutputConfig,
)
from axon_recon.pipeline.stages.reconstruct.runner import (
	run_reconstruct_generate_gtrs_phase,
	run_reconstruct_plot_branch_propagations_phase,
	run_reconstruct_plot_branch_velocities_phase,
	run_reconstruct_plot_unit_summary_phase,
	run_reconstruct_report_full_chip_layout_phase,
	run_reconstruct_report_recons_phase,
)
from axon_recon.pipeline.stages.templates.models.inputs import ProbeGeometryConfig


def test_format_unit_reldir() -> None:
	p = format_unit_reldir("units/{unit_id:04d}/", 94)
	assert str(p) == "units/0094"


def test_resolve_unit_output_paths_includes_amplitude_map() -> None:
	paths = resolve_unit_output_paths(
		reconstruction_out_dir=Path("/tmp/recon"),
		unit_id=1,
		per_unit_outputs=PerUnitOutputsConfig(
			write_amplitude_map_png=True,
			amplitude_map_png_relpath="maps/amplitude_map.png",
			write_detection_filter_json=True,
			detection_filter_relpath="filters/detection.json",
		),
	)
	assert paths["amplitude_map_png"] == Path("/tmp/recon") / "units/0001" / "maps/amplitude_map.png"
	assert paths["detection_filter_json"] == Path("/tmp/recon") / "units/0001" / "filters/detection.json"
	assert paths["channel_selection_figure_png"] == Path("/tmp/recon") / "units/0001" / "diagnostic_figs/channel_selection.png"
	assert paths["channel_selection_figure_svg"] == Path("/tmp/recon") / "units/0001" / "diagnostic_figs/channel_selection.svg"
	assert paths["axon_reconstruction_figure_png"] == Path("/tmp/recon") / "units/0001" / "diagnostic_figs/axon_reconstruction.png"
	assert paths["axon_reconstruction_figure_svg"] == Path("/tmp/recon") / "units/0001" / "diagnostic_figs/axon_reconstruction.svg"
	assert paths["circle_recon_png"] == Path("/tmp/recon") / "units/0001" / "circle_recon.png"
	assert paths["circle_recon_svg"] == Path("/tmp/recon") / "units/0001" / "circle_recon.svg"


def test_run_reconstruct_generate_gtrs_phase_writes_summary(monkeypatch, tmp_path: Path) -> None:
	from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner

	well_out_dir = tmp_path / "well001"
	well_out_dir.mkdir(parents=True, exist_ok=True)
	templates_out = tmp_path / "templates_out"
	merged = tmp_path / "templates_merged"
	full = tmp_path / "templates_full"
	templates_out.mkdir(parents=True, exist_ok=True)
	merged.mkdir(parents=True, exist_ok=True)
	full.mkdir(parents=True, exist_ok=True)

	monkeypatch.setattr(
		reconstruct_runner,
		"compute_mea_analysis_output_dir",
		lambda *, output_root, data_file, well: well_out_dir,
	)
	monkeypatch.setattr(
		reconstruct_runner,
		"_resolve_templates_dirs",
		lambda _well_out_dir, **kwargs: (templates_out, merged, full),
	)
	monkeypatch.setattr(reconstruct_runner, "import_axon_velocity", lambda *, repo_root: object())
	monkeypatch.setattr(
		reconstruct_runner,
		"load_templates_for_unit",
		lambda **kwargs: (
			np.asarray([[-5.0, -10.0, -3.0], [-2.0, -4.0, -1.0]], dtype=float),
			np.asarray([[0.0, 0.0], [17.5, 0.0]], dtype=float),
			np.asarray([[-5.0, -10.0, -3.0], [-2.0, -4.0, -1.0]], dtype=float),
			np.asarray([[0.0, 0.0], [17.5, 0.0]], dtype=float),
			10_000.0,
			"square_from_merged",
		),
	)
	monkeypatch.setattr(reconstruct_runner, "compute_graph_tracking", lambda **kwargs: {"gtr": True})

	inputs = ReconstructionInputs(
		h5_path=tmp_path / "input.raw.h5",
		stream_id="well001",
		mea_output_root=tmp_path,
		output_rel_root="recon_outputs",
		unit_ids=[1],
		n_jobs=1,
		per_unit_outputs=PerUnitOutputsConfig(
			write_branches_raw_json=False,
			write_branches_json=False,
			write_heuristics_json=False,
			write_gtr_pkl=True,
			write_gtr_json=False,
			write_amplitude_map_png=False,
		),
	)

	summary = run_reconstruct_generate_gtrs_phase(inputs)
	assert summary["phase"] == "generate_gtrs"
	assert summary["units_ok"] == 1
	summary_path = Path(str(summary["summary_json"]))
	assert summary_path.exists()
	payload = json.loads(summary_path.read_text(encoding="utf-8"))
	assert payload["phase"] == "generate_gtrs"
	assert payload["units_ok"] == 1
	unit_outputs = payload["units"][0]["outputs"]
	assert "gtr_pkl" in unit_outputs
	assert Path(unit_outputs["gtr_pkl"]).exists()


def test_run_reconstruct_generate_gtrs_phase_logs_gtr_persistence_even_when_legacy_flag_is_false(
	monkeypatch,
	tmp_path: Path,
	caplog,
) -> None:
	from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner

	well_out_dir = tmp_path / "well001"
	well_out_dir.mkdir(parents=True, exist_ok=True)
	templates_out = tmp_path / "templates_out"
	merged = tmp_path / "templates_merged"
	full = tmp_path / "templates_full"
	templates_out.mkdir(parents=True, exist_ok=True)
	merged.mkdir(parents=True, exist_ok=True)
	full.mkdir(parents=True, exist_ok=True)

	monkeypatch.setattr(
		reconstruct_runner,
		"compute_mea_analysis_output_dir",
		lambda *, output_root, data_file, well: well_out_dir,
	)
	monkeypatch.setattr(
		reconstruct_runner,
		"_resolve_templates_dirs",
		lambda _well_out_dir, **kwargs: (templates_out, merged, full),
	)
	monkeypatch.setattr(reconstruct_runner, "import_axon_velocity", lambda *, repo_root: object())
	monkeypatch.setattr(
		reconstruct_runner,
		"load_templates_for_unit",
		lambda **kwargs: (
			np.asarray([[-5.0, -10.0, -3.0], [-2.0, -4.0, -1.0]], dtype=float),
			np.asarray([[0.0, 0.0], [17.5, 0.0]], dtype=float),
			np.asarray([[-5.0, -10.0, -3.0], [-2.0, -4.0, -1.0]], dtype=float),
			np.asarray([[0.0, 0.0], [17.5, 0.0]], dtype=float),
			10_000.0,
			"square_from_merged",
		),
	)
	monkeypatch.setattr(reconstruct_runner, "compute_graph_tracking", lambda **kwargs: {"gtr": True})

	inputs = ReconstructionInputs(
		h5_path=tmp_path / "input.raw.h5",
		stream_id="well001",
		mea_output_root=tmp_path,
		output_rel_root="recon_outputs",
		unit_ids=[7],
		n_jobs=1,
		per_unit_outputs=PerUnitOutputsConfig(
			write_branches_raw_json=False,
			write_branches_json=False,
			write_heuristics_json=False,
			write_gtr_pkl=False,
			write_gtr_json=False,
			write_amplitude_map_png=False,
		),
	)

	with caplog.at_level(logging.INFO, logger="axon_recon.reconstruct"):
		summary = run_reconstruct_generate_gtrs_phase(inputs)

	unit_outputs = summary["units"][0]["outputs"]
	assert "gtr_pkl" in unit_outputs
	assert Path(unit_outputs["gtr_pkl"]).exists()
	assert "forcing gtr.pkl persistence for phase contract" in caplog.text
	assert "reconstruct.generate_gtrs unit 7 wrote gtr_pkl=" in caplog.text
	assert "reconstruct.generate_gtrs wrote summary output:" in caplog.text


def test_run_reconstruct_generate_gtrs_phase_writes_filter_selection_jsons(monkeypatch, tmp_path: Path) -> None:
	from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner

	well_out_dir = tmp_path / "well001"
	well_out_dir.mkdir(parents=True, exist_ok=True)
	templates_out = tmp_path / "templates_out"
	merged = tmp_path / "templates_merged"
	full = tmp_path / "templates_full"
	templates_out.mkdir(parents=True, exist_ok=True)
	merged.mkdir(parents=True, exist_ok=True)
	full.mkdir(parents=True, exist_ok=True)

	monkeypatch.setattr(
		reconstruct_runner,
		"compute_mea_analysis_output_dir",
		lambda *, output_root, data_file, well: well_out_dir,
	)
	monkeypatch.setattr(
		reconstruct_runner,
		"_resolve_templates_dirs",
		lambda _well_out_dir, **kwargs: (templates_out, merged, full),
	)
	monkeypatch.setattr(reconstruct_runner, "import_axon_velocity", lambda *, repo_root: object())
	monkeypatch.setattr(
		reconstruct_runner,
		"load_templates_for_unit",
		lambda **kwargs: (
			np.asarray([[-5.0, -10.0, -3.0], [-2.0, -4.0, -1.0]], dtype=float),
			np.asarray([[0.0, 0.0], [17.5, 0.0]], dtype=float),
			np.asarray([[-5.0, -10.0, -3.0], [-2.0, -4.0, -1.0]], dtype=float),
			np.asarray([[0.0, 0.0], [17.5, 0.0]], dtype=float),
			10_000.0,
			"merged_per_unit_output",
		),
	)
	monkeypatch.setattr(
		reconstruct_runner,
		"compute_graph_tracking",
		lambda **kwargs: SimpleNamespace(
			branches=[],
			init_channel=0,
			selected_channels=np.asarray([0, 1], dtype=int),
			amplitudes=np.asarray([10.0, 4.0], dtype=float),
			peak_times=np.asarray([2.0, 5.0], dtype=float),
			fs=10_000.0,
			_selected_channels_detect={0, 1},
			_selected_channels_kurt={1},
			_selected_channels_peakstd={0, 1},
			_selected_channels_init={1},
			_detect_threshold=0.5,
			_detection_type="relative",
			_kurt_threshold=0.3,
			_peak_std_threhsold=1.25,
			_peak_std_distance=30.0,
			_init_delay=0.2,
			_init_frames=2,
			_remove_isolated=True,
			_min_selected_points=1,
			compute_velocity=True,
		),
	)

	inputs = ReconstructionInputs(
		h5_path=tmp_path / "input.raw.h5",
		stream_id="well001",
		mea_output_root=tmp_path,
		output_rel_root="recon_outputs",
		unit_ids=[1],
		n_jobs=1,
		per_unit_outputs=PerUnitOutputsConfig(
			write_branches_raw_json=False,
			write_branches_json=False,
			write_detection_filter_json=True,
			write_kurtosis_filter_json=True,
			write_peak_std_filter_json=True,
			write_delay_filter_json=True,
			write_all_filters_json=True,
			write_heuristics_json=False,
			write_gtr_pkl=True,
			write_gtr_json=False,
			write_amplitude_map_png=False,
		),
	)

	summary = run_reconstruct_generate_gtrs_phase(inputs)
	unit_outputs = summary["units"][0]["outputs"]
	assert Path(unit_outputs["detection_filter_json"]).exists()
	assert Path(unit_outputs["kurtosis_filter_json"]).exists()
	assert Path(unit_outputs["peak_std_filter_json"]).exists()
	assert Path(unit_outputs["delay_filter_json"]).exists()
	assert Path(unit_outputs["all_filters_json"]).exists()

	detection_payload = json.loads(Path(unit_outputs["detection_filter_json"]).read_text(encoding="utf-8"))
	assert detection_payload["selected_channels"] == [0, 1]
	assert float(detection_payload["filter_params"]["effective_amplitude_threshold"]) == 5.0

	all_filters_payload = json.loads(Path(unit_outputs["all_filters_json"]).read_text(encoding="utf-8"))
	assert all_filters_payload["selected_channels"] == [0, 1]
	assert all_filters_payload["filters"]["delay"]["selected_channels"] == [1]


def test_run_reconstruct_generate_gtrs_phase_writes_diagnostic_figures(monkeypatch, tmp_path: Path) -> None:
	from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner

	well_out_dir = tmp_path / "well001"
	well_out_dir.mkdir(parents=True, exist_ok=True)
	templates_out = tmp_path / "templates_out"
	merged = tmp_path / "templates_merged"
	full = tmp_path / "templates_full"
	templates_out.mkdir(parents=True, exist_ok=True)
	merged.mkdir(parents=True, exist_ok=True)
	full.mkdir(parents=True, exist_ok=True)

	monkeypatch.setattr(
		reconstruct_runner,
		"compute_mea_analysis_output_dir",
		lambda *, output_root, data_file, well: well_out_dir,
	)
	monkeypatch.setattr(
		reconstruct_runner,
		"_resolve_templates_dirs",
		lambda _well_out_dir, **kwargs: (templates_out, merged, full),
	)
	monkeypatch.setattr(reconstruct_runner, "import_axon_velocity", lambda *, repo_root: object())
	monkeypatch.setattr(
		reconstruct_runner,
		"load_templates_for_unit",
		lambda **kwargs: (
			np.asarray([[-5.0, -10.0, -3.0], [-2.0, -4.0, -1.0]], dtype=float),
			np.asarray([[0.0, 0.0], [17.5, 0.0]], dtype=float),
			np.asarray([[-5.0, -10.0, -3.0], [-2.0, -4.0, -1.0]], dtype=float),
			np.asarray([[0.0, 0.0], [17.5, 0.0]], dtype=float),
			10_000.0,
			"merged_per_unit_output",
		),
	)
	monkeypatch.setattr(reconstruct_runner, "compute_graph_tracking", lambda **kwargs: SimpleNamespace())

	def _write_channel_selection_figure(**kwargs):
		output_png = kwargs.get("output_png")
		output_svg = kwargs.get("output_svg")
		if output_png is not None:
			Path(output_png).parent.mkdir(parents=True, exist_ok=True)
			Path(output_png).write_bytes(b"png")
		if output_svg is not None:
			Path(output_svg).parent.mkdir(parents=True, exist_ok=True)
			Path(output_svg).write_text("<svg></svg>", encoding="utf-8")

	def _write_axon_reconstruction_figure(**kwargs):
		output_png = kwargs.get("output_png")
		output_svg = kwargs.get("output_svg")
		if output_png is not None:
			Path(output_png).parent.mkdir(parents=True, exist_ok=True)
			Path(output_png).write_bytes(b"png")
		if output_svg is not None:
			Path(output_svg).parent.mkdir(parents=True, exist_ok=True)
			Path(output_svg).write_text("<svg></svg>", encoding="utf-8")

	monkeypatch.setattr(
		reconstruct_runner,
		"write_unit_channel_selection_diagnostic_figure",
		_write_channel_selection_figure,
	)
	monkeypatch.setattr(
		reconstruct_runner,
		"write_unit_axon_reconstruction_diagnostic_figure",
		_write_axon_reconstruction_figure,
	)

	inputs = ReconstructionInputs(
		h5_path=tmp_path / "input.raw.h5",
		stream_id="well001",
		mea_output_root=tmp_path,
		output_rel_root="recon_outputs",
		unit_ids=[1],
		n_jobs=1,
		per_unit_outputs=PerUnitOutputsConfig(
			write_branches_raw_json=False,
			write_branches_json=False,
			write_heuristics_json=False,
			write_gtr_pkl=True,
			write_gtr_json=False,
			write_amplitude_map_png=False,
			channel_selection_figure=ReconstructionDiagnosticFigureConfig(
				write_png=True,
				write_svg=True,
				relpath="diagnostics/channel_selection",
				dpi=200.0,
			),
			axon_reconstruction_figure=ReconstructionDiagnosticFigureConfig(
				write_png=True,
				write_svg=False,
				relpath="diagnostics/axon_reconstruction",
				dpi=200.0,
			),
		),
	)

	summary = run_reconstruct_generate_gtrs_phase(inputs)
	unit_outputs = summary["units"][0]["outputs"]
	assert Path(unit_outputs["channel_selection_figure_png"]).exists()
	assert Path(unit_outputs["channel_selection_figure_svg"]).exists()
	assert Path(unit_outputs["axon_reconstruction_figure_png"]).exists()
	assert "axon_reconstruction_figure_svg" not in unit_outputs


def test_resolve_generate_gtrs_execution_plan_prefers_fewer_processes() -> None:
	from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner

	inputs = ReconstructionInputs(
		h5_path=Path("/tmp/dataset.h5"),
		stream_id="well000",
		mea_output_root=Path("/tmp/out"),
		n_jobs=24,
	)

	derived_unit_workers, unit_procs, unit_batch_size, batches = reconstruct_runner._resolve_generate_gtrs_execution_plan(
		inputs=inputs,
		unit_ids=list(range(12)),
	)

	assert derived_unit_workers == 24
	assert unit_procs == 6
	assert unit_batch_size == 2
	assert batches == [
		[0, 1],
		[2, 3],
		[4, 5],
		[6, 7],
		[8, 9],
		[10, 11],
	]


def test_resolve_generate_gtrs_execution_plan_honors_unit_procs_override() -> None:
	from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner

	inputs = ReconstructionInputs(
		h5_path=Path("/tmp/dataset.h5"),
		stream_id="well000",
		mea_output_root=Path("/tmp/out"),
		phases=ReconstructionPhasesConfig(
			generate_gtrs=ReconstructionGenerateGtrsPhaseConfig(unit_procs=4)
		),
		n_jobs=24,
	)

	derived_unit_workers, unit_procs, unit_batch_size, batches = reconstruct_runner._resolve_generate_gtrs_execution_plan(
		inputs=inputs,
		unit_ids=list(range(12)),
	)

	assert derived_unit_workers == 24
	assert unit_procs == 4
	assert unit_batch_size == 3
	assert batches == [
		[0, 1, 2],
		[3, 4, 5],
		[6, 7, 8],
		[9, 10, 11],
	]


def test_run_reconstruct_generate_gtrs_batches_logs_unified_progress(tmp_path: Path, monkeypatch, caplog) -> None:
	from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner

	class _FakeFuture:
		def __init__(self, result: list[object]) -> None:
			self._result = result

		def result(self) -> list[object]:
			return self._result

	class _FakeProcessPoolExecutor:
		def __init__(self, max_workers: int) -> None:
			self.max_workers = int(max_workers)

		def __enter__(self):
			return self

		def __exit__(self, exc_type, exc, tb) -> bool:
			return False

		def submit(self, fn, batch_inputs):
			return _FakeFuture(fn(batch_inputs))

	def _fake_as_completed(futures):
		return list(futures.keys())

	def _fake_run_generate_gtrs_batch(batch_inputs):
		assert batch_inputs.inputs.n_jobs == 1
		return [
			reconstruct_runner.UnitReconstructionResult(
				unit_id=unit_id,
				status="ok",
				outputs={"gtr_pkl": str(tmp_path / f"unit_{int(unit_id):04d}" / "gtr.pkl")},
				error=None,
			)
			for unit_id in list(batch_inputs.inputs.unit_ids or [])
		]

	monkeypatch.setattr(
		reconstruct_runner.concurrent.futures,
		"ProcessPoolExecutor",
		_FakeProcessPoolExecutor,
	)
	monkeypatch.setattr(
		reconstruct_runner.concurrent.futures,
		"as_completed",
		_fake_as_completed,
	)
	monkeypatch.setattr(
		reconstruct_runner,
		"_run_generate_gtrs_batch",
		_fake_run_generate_gtrs_batch,
	)

	inputs = ReconstructionInputs(
		h5_path=tmp_path / "dataset.h5",
		stream_id="well000",
		mea_output_root=tmp_path / "outputs",
		phases=ReconstructionPhasesConfig(
			generate_gtrs=ReconstructionGenerateGtrsPhaseConfig(unit_procs=2, unit_batch_size=3)
		),
		n_jobs=24,
	)
	env = reconstruct_runner._ReconstructPhaseEnvironment(
		well_out_dir=tmp_path / "well",
		reconstruction_out_dir=tmp_path / "recon",
		merged_units_dir=tmp_path / "merged",
		full_channels_templates_dir=tmp_path / "full",
		unit_ids=[10, 11, 12, 13, 14, 15],
		preserve_stage_reports=False,
		existing_stage_outputs={},
	)

	with caplog.at_level(logging.INFO, logger="axon_recon.reconstruct"):
		result = reconstruct_runner._run_reconstruct_generate_gtrs_batches(inputs=inputs, env=env)

	messages = [record.getMessage() for record in caplog.records]
	assert any(
		"reconstruct.generate_gtrs execution plan: requested_units=6 derived_unit_workers=24 unit_procs=2 unit_batch_size=3 unit_batches=2"
		in message
		for message in messages
	)
	assert any(
		"reconstruct.generate_gtrs unified progress: 3/6 units completed (1/2 batches)" in message
		for message in messages
	)
	assert any(
		"reconstruct.generate_gtrs unified progress: 6/6 units completed (2/2 batches)" in message
		for message in messages
	)
	assert [item.unit_id for item in result] == [10, 11, 12, 13, 14, 15]


def test_run_reconstruct_report_recons_phase_writes_av_recons_pdf(monkeypatch, tmp_path: Path) -> None:
	from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner

	well_out_dir = tmp_path / "well001"
	well_out_dir.mkdir(parents=True, exist_ok=True)
	templates_out = tmp_path / "templates_out"
	merged = tmp_path / "templates_merged"
	full = tmp_path / "templates_full"
	templates_out.mkdir(parents=True, exist_ok=True)
	merged.mkdir(parents=True, exist_ok=True)
	full.mkdir(parents=True, exist_ok=True)

	monkeypatch.setattr(
		reconstruct_runner,
		"compute_mea_analysis_output_dir",
		lambda *, output_root, data_file, well: well_out_dir,
	)
	monkeypatch.setattr(
		reconstruct_runner,
		"_resolve_templates_dirs",
		lambda _well_out_dir, **kwargs: (templates_out, merged, full),
	)

	reconstruction_out_dir = well_out_dir / "recon_outputs"
	unit_dir = reconstruction_out_dir / "units" / "0001"
	unit_dir.mkdir(parents=True, exist_ok=True)
	circle_png = unit_dir / "circle_recon.png"
	circle_png.write_bytes(b"png")
	(unit_dir / "unit_reconstruction_summary.json").write_text(
		json.dumps({"status": "ok", "outputs": {"circle_recon_png": str(circle_png)}}),
		encoding="utf-8",
	)

	def _fake_render_template_report_pdf(**kwargs):
		pdf_path = Path(kwargs["pdf_path"])
		pdf_path.parent.mkdir(parents=True, exist_ok=True)
		pdf_path.write_bytes(b"pdf")
		return {str(kwargs.get("output_key", "av_recons_pdf")): str(pdf_path)}

	monkeypatch.setattr(reconstruct_runner, "render_template_report_pdf", _fake_render_template_report_pdf)

	inputs = ReconstructionInputs(
		h5_path=tmp_path / "input.raw.h5",
		stream_id="well001",
		mea_output_root=tmp_path,
		output_rel_root="recon_outputs",
		unit_ids=[1],
		phases=ReconstructionPhasesConfig(
			report_recons=ReconstructionReportReconsPhaseConfig(
				enabled=True,
				av_recons=ReconstructionAvReconsConfig(write_pdf=True, pdf_relpath="av_recons.pdf"),
			)
		),
	)

	summary = run_reconstruct_report_recons_phase(inputs)
	assert summary["phase"] == "report_recons"
	assert summary["outputs"]["av_recons_pdf"] == str(reconstruction_out_dir / "av_recons.pdf")
	assert Path(summary["outputs"]["av_recons_pdf"]).exists()


def test_run_reconstruct_report_recons_phase_requires_circle_recon_assets_for_av_recons_pdf(monkeypatch, tmp_path: Path) -> None:
	from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner

	well_out_dir = tmp_path / "well001"
	well_out_dir.mkdir(parents=True, exist_ok=True)
	templates_out = tmp_path / "templates_out"
	merged = tmp_path / "templates_merged"
	full = tmp_path / "templates_full"
	templates_out.mkdir(parents=True, exist_ok=True)
	merged.mkdir(parents=True, exist_ok=True)
	full.mkdir(parents=True, exist_ok=True)

	monkeypatch.setattr(
		reconstruct_runner,
		"compute_mea_analysis_output_dir",
		lambda *, output_root, data_file, well: well_out_dir,
	)
	monkeypatch.setattr(
		reconstruct_runner,
		"_resolve_templates_dirs",
		lambda _well_out_dir, **kwargs: (templates_out, merged, full),
	)

	reconstruction_out_dir = well_out_dir / "recon_outputs"
	unit_dir = reconstruction_out_dir / "units" / "0001"
	unit_dir.mkdir(parents=True, exist_ok=True)
	(unit_dir / "unit_reconstruction_summary.json").write_text(
		json.dumps({"status": "ok", "outputs": {}}),
		encoding="utf-8",
	)

	inputs = ReconstructionInputs(
		h5_path=tmp_path / "input.raw.h5",
		stream_id="well001",
		mea_output_root=tmp_path,
		output_rel_root="recon_outputs",
		unit_ids=[1],
		phases=ReconstructionPhasesConfig(
			report_recons=ReconstructionReportReconsPhaseConfig(
				enabled=True,
				av_recons=ReconstructionAvReconsConfig(write_pdf=True, pdf_relpath="av_recons.pdf"),
			)
		),
	)

	with pytest.raises(FileNotFoundError, match="run reconstruct.plot_recons first"):
		run_reconstruct_report_recons_phase(inputs)


def test_resolve_branch_phase_output_paths_includes_scope_and_manifest() -> None:
	paths = resolve_branch_phase_output_paths(
		reconstruction_out_dir=Path("/tmp/recon"),
		unit_id=9,
		per_unit_outputs=PerUnitOutputsConfig(unit_reldir="units/{unit_id:04d}/"),
		phase_output=ReconstructionBranchPlotOutputConfig(
			write_png=True,
			relpath="branch_qc/propagations",
			manifest_relpath="reports/branch_propagations_manifest.json",
		),
		branch_scope="raw",
	)
	assert paths["phase_root_dir"] == Path("/tmp/recon") / "units/0009" / "branch_qc/propagations"
	assert paths["output_dir"] == Path("/tmp/recon") / "units/0009" / "branch_qc/propagations" / "raw"
	assert paths["manifest_json"] == Path("/tmp/recon") / "units/0009" / "reports/branch_propagations_manifest.json"


def test_resolve_unit_summary_phase_output_paths_includes_png_and_svg() -> None:
	paths = resolve_unit_summary_phase_output_paths(
		reconstruction_out_dir=Path("/tmp/recon"),
		unit_id=9,
		per_unit_outputs=PerUnitOutputsConfig(unit_reldir="units/{unit_id:04d}/"),
		phase_output=ReconstructionUnitSummaryOutputConfig(
			write_png=True,
			write_svg=True,
			relpath="reports/unit_summary",
			dpi=220.0,
		),
	)
	assert paths["png_path"] == Path("/tmp/recon") / "units/0009" / "reports/unit_summary.png"
	assert paths["svg_path"] == Path("/tmp/recon") / "units/0009" / "reports/unit_summary.svg"


def test_run_reconstruct_plot_branch_propagations_phase_writes_manifest(monkeypatch, tmp_path: Path) -> None:
	from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner

	well_out_dir = tmp_path / "well001"
	well_out_dir.mkdir(parents=True, exist_ok=True)
	templates_out = tmp_path / "templates_out"
	merged = tmp_path / "templates_merged"
	full = tmp_path / "templates_full"
	templates_out.mkdir(parents=True, exist_ok=True)
	merged.mkdir(parents=True, exist_ok=True)
	full.mkdir(parents=True, exist_ok=True)

	monkeypatch.setattr(
		reconstruct_runner,
		"compute_mea_analysis_output_dir",
		lambda *, output_root, data_file, well: well_out_dir,
	)
	monkeypatch.setattr(
		reconstruct_runner,
		"_resolve_templates_dirs",
		lambda _well_out_dir, **kwargs: (templates_out, merged, full),
	)
	monkeypatch.setattr(
		reconstruct_runner,
		"load_templates_for_unit",
		lambda **kwargs: (
			np.asarray([[-5.0, -10.0, -3.0], [-2.0, -4.0, -1.0], [-1.0, -6.0, -2.0]], dtype=float),
			np.asarray([[0.0, 0.0], [17.5, 0.0], [35.0, 0.0]], dtype=float),
			np.asarray([[-5.0, -10.0, -3.0], [-2.0, -4.0, -1.0], [-1.0, -6.0, -2.0]], dtype=float),
			np.asarray([[0.0, 0.0], [17.5, 0.0], [35.0, 0.0]], dtype=float),
			10_000.0,
			"merged",
		),
	)

	seen = {"count": 0, "branch_records": ()}

	def _write_branch_propagation_plot(**kwargs):
		seen["count"] += 1
		seen["branch_records"] = tuple(kwargs.get("branch_records", ()))
		output_png = Path(kwargs["output_png"])
		output_png.parent.mkdir(parents=True, exist_ok=True)
		output_png.write_bytes(b"png")
		return {"png_path": str(output_png)}

	monkeypatch.setattr(reconstruct_runner, "write_unit_branch_propagation_plot", _write_branch_propagation_plot)

	inputs = ReconstructionInputs(
		h5_path=tmp_path / "input.raw.h5",
		stream_id="well001",
		mea_output_root=tmp_path,
		output_rel_root="recon_outputs",
		unit_ids=[1],
		phases=ReconstructionPhasesConfig(
			plot_branch_propagations=ReconstructionPlotBranchPropagationsPhaseConfig(
				enabled=True,
				display=ReconstructionBranchPropagationDisplayConfig(figsize=(7.0, 4.0)),
				output=ReconstructionBranchPlotOutputConfig(
					write_png=True,
					relpath="branch_qc/propagations",
					manifest_relpath="reports/branch_propagations_manifest.json",
				),
			),
		),
	)

	paths = resolve_unit_output_paths(
		reconstruction_out_dir=well_out_dir / "recon_outputs",
		unit_id=1,
		per_unit_outputs=inputs.per_unit_outputs,
	)
	paths["unit_dir"].mkdir(parents=True, exist_ok=True)
	paths["unit_summary_json"].write_text(
		json.dumps({"unit_id": 1, "status": "error", "error": "prior failure", "outputs": {}}),
		encoding="utf-8",
	)
	legacy_branch_png = paths["unit_dir"] / "branch_qc/propagations/raw/branch_0000.png"
	legacy_branch_png.parent.mkdir(parents=True, exist_ok=True)
	legacy_branch_png.write_bytes(b"legacy")
	with open(paths["gtr_pkl"], "wb") as handle:
		pickle.dump(SimpleNamespace(_paths_raw=[[2, 1, 0], [0, 1, 2]], branches=[], _paths_clean=[]), handle)

	summary = run_reconstruct_plot_branch_propagations_phase(inputs)
	assert summary["phase"] == "plot_branch_propagations"
	assert seen["count"] == 1
	assert len(seen["branch_records"]) == 2
	assert summary["units"][0]["status"] == "ok"
	unit_outputs = summary["units"][0]["outputs"]
	assert Path(unit_outputs["branch_propagations_manifest_json"]).exists()
	assert Path(unit_outputs["branch_propagations_png"]).exists()
	unit_summary_payload = json.loads(paths["unit_summary_json"].read_text(encoding="utf-8"))
	assert unit_summary_payload["status"] == "ok"
	assert unit_summary_payload["error"] is None
	manifest = json.loads(Path(unit_outputs["branch_propagations_manifest_json"]).read_text(encoding="utf-8"))
	assert manifest["branches_ok"] == 2
	assert Path(manifest["png_path"]).exists()
	assert manifest["branches"][0]["status"] == "ok"
	assert manifest["branches"][0]["column_index"] == 0
	assert manifest["branches"][1]["column_index"] == 1
	assert manifest["branches"][0]["png_path"] == manifest["branches"][1]["png_path"]
	assert Path(manifest["branches"][0]["png_path"]).exists()
	assert legacy_branch_png.exists() is False


def test_run_reconstruct_plot_branch_velocities_phase_writes_manifest(monkeypatch, tmp_path: Path) -> None:
	from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner

	well_out_dir = tmp_path / "well001"
	well_out_dir.mkdir(parents=True, exist_ok=True)
	templates_out = tmp_path / "templates_out"
	merged = tmp_path / "templates_merged"
	full = tmp_path / "templates_full"
	templates_out.mkdir(parents=True, exist_ok=True)
	merged.mkdir(parents=True, exist_ok=True)
	full.mkdir(parents=True, exist_ok=True)

	monkeypatch.setattr(
		reconstruct_runner,
		"compute_mea_analysis_output_dir",
		lambda *, output_root, data_file, well: well_out_dir,
	)
	monkeypatch.setattr(
		reconstruct_runner,
		"_resolve_templates_dirs",
		lambda _well_out_dir, **kwargs: (templates_out, merged, full),
	)
	monkeypatch.setattr(
		reconstruct_runner,
		"load_templates_for_unit",
		lambda **kwargs: (
			np.asarray([[-5.0, -10.0, -3.0], [-2.0, -4.0, -1.0], [-1.0, -6.0, -2.0]], dtype=float),
			np.asarray([[0.0, 0.0], [17.5, 0.0], [35.0, 0.0]], dtype=float),
			np.asarray([[-5.0, -10.0, -3.0], [-2.0, -4.0, -1.0], [-1.0, -6.0, -2.0]], dtype=float),
			np.asarray([[0.0, 0.0], [17.5, 0.0], [35.0, 0.0]], dtype=float),
			10_000.0,
			"merged",
		),
	)

	seen = {"count": 0, "branch_records": (), "fit_payloads": ()}

	def _write_branch_velocity_plot(**kwargs):
		seen["count"] += 1
		seen["branch_records"] = tuple(kwargs.get("branch_records", ()))
		seen["fit_payloads"] = tuple(kwargs.get("fit_payloads", ()))
		output_png = Path(kwargs["output_png"])
		output_png.parent.mkdir(parents=True, exist_ok=True)
		output_png.write_bytes(b"png")
		return {"png_path": str(output_png)}

	monkeypatch.setattr(reconstruct_runner, "write_unit_branch_velocity_plot", _write_branch_velocity_plot)

	inputs = ReconstructionInputs(
		h5_path=tmp_path / "input.raw.h5",
		stream_id="well001",
		mea_output_root=tmp_path,
		output_rel_root="recon_outputs",
		unit_ids=[1],
		phases=ReconstructionPhasesConfig(
			plot_branch_velocities=ReconstructionPlotBranchVelocitiesPhaseConfig(
				enabled=True,
				branch_scope="clean",
				display=ReconstructionBranchVelocityDisplayConfig(figsize=(6.0, 4.0), show_legend=False),
				output=ReconstructionBranchPlotOutputConfig(
					write_png=True,
					relpath="branch_qc/velocities",
					manifest_relpath="reports/branch_velocities_manifest.json",
				),
			),
		),
	)

	paths = resolve_unit_output_paths(
		reconstruction_out_dir=well_out_dir / "recon_outputs",
		unit_id=1,
		per_unit_outputs=inputs.per_unit_outputs,
	)
	paths["unit_dir"].mkdir(parents=True, exist_ok=True)
	paths["unit_summary_json"].write_text(
		json.dumps({"unit_id": 1, "status": "error", "error": "prior failure", "outputs": {}}),
		encoding="utf-8",
	)
	legacy_branch_png = paths["unit_dir"] / "branch_qc/velocities/clean/branch_0004.png"
	legacy_branch_png.parent.mkdir(parents=True, exist_ok=True)
	legacy_branch_png.write_bytes(b"legacy")
	with open(paths["gtr_pkl"], "wb") as handle:
		pickle.dump(
			SimpleNamespace(
				_paths_raw=[[2, 1, 0], [0, 2, 1]],
				_paths_clean=[[0, 1, 2]],
				branches=[
					{
						"branch_index": 4,
						"channels": [0, 1, 2],
						"velocity": 1.7,
						"offset": 0.2,
						"r2": 0.91,
						"peak_times": [0.8, 1.4, 2.1],
						"distances": [12.0, 24.0, 41.0],
					},
					{
						"branch_index": 5,
						"channels": [0, 2, 1],
						"velocity": 2.4,
						"offset": 0.1,
						"r2": 0.81,
						"peak_times": [0.7, 1.1, 1.9],
						"distances": [11.0, 19.0, 38.0],
					}
				],
			),
			handle,
		)

	summary = run_reconstruct_plot_branch_velocities_phase(inputs)
	assert summary["phase"] == "plot_branch_velocities"
	assert summary["units"][0]["status"] == "ok"
	assert seen["count"] == 1
	assert len(seen["branch_records"]) == 2
	assert len(seen["fit_payloads"]) == 2
	unit_outputs = summary["units"][0]["outputs"]
	assert Path(unit_outputs["branch_velocities_manifest_json"]).exists()
	assert Path(unit_outputs["branch_velocities_png"]).exists()
	unit_summary_payload = json.loads(paths["unit_summary_json"].read_text(encoding="utf-8"))
	assert unit_summary_payload["status"] == "ok"
	assert unit_summary_payload["error"] is None
	manifest = json.loads(Path(unit_outputs["branch_velocities_manifest_json"]).read_text(encoding="utf-8"))
	assert manifest["branches_ok"] == 2
	assert Path(manifest["png_path"]).exists()
	assert manifest["branches"][0]["status"] == "ok"
	assert float(manifest["branches"][0]["velocity"]) == pytest.approx(1.7)
	assert "r" not in manifest["branches"][0]
	assert float(manifest["branches"][1]["velocity"]) == pytest.approx(2.4)
	assert manifest["branches"][0]["png_path"] == manifest["branches"][1]["png_path"]
	assert Path(manifest["branches"][0]["png_path"]).exists()
	assert legacy_branch_png.exists() is False


def test_run_reconstruct_plot_unit_summary_phase_writes_outputs(monkeypatch, tmp_path: Path) -> None:
	from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner

	well_out_dir = tmp_path / "well001"
	well_out_dir.mkdir(parents=True, exist_ok=True)
	templates_out = tmp_path / "templates_out"
	merged = tmp_path / "templates_merged"
	full = tmp_path / "templates_full"
	templates_out.mkdir(parents=True, exist_ok=True)
	merged.mkdir(parents=True, exist_ok=True)
	full.mkdir(parents=True, exist_ok=True)

	monkeypatch.setattr(
		reconstruct_runner,
		"compute_mea_analysis_output_dir",
		lambda *, output_root, data_file, well: well_out_dir,
	)
	monkeypatch.setattr(
		reconstruct_runner,
		"_resolve_templates_dirs",
		lambda _well_out_dir, **kwargs: (templates_out, merged, full),
	)
	monkeypatch.setattr(
		reconstruct_runner,
		"load_templates_for_unit",
		lambda **kwargs: (
			np.asarray([[-5.0, -10.0, -3.0], [-2.0, -4.0, -1.0], [-1.0, -6.0, -2.0]], dtype=float),
			np.asarray([[0.0, 0.0], [17.5, 0.0], [35.0, 0.0]], dtype=float),
			np.asarray([[-5.0, -10.0, -3.0], [-2.0, -4.0, -1.0], [-1.0, -6.0, -2.0]], dtype=float),
			np.asarray([[0.0, 0.0], [17.5, 0.0], [35.0, 0.0]], dtype=float),
			10_000.0,
			"merged",
		),
	)

	seen = {"count": 0}

	def _write_unit_summary_plot(**kwargs):
		seen["count"] += 1
		output_png = Path(kwargs["output_png"])
		output_png.parent.mkdir(parents=True, exist_ok=True)
		output_png.write_bytes(b"png")
		return {"png_path": str(output_png)}

	monkeypatch.setattr(reconstruct_runner, "write_unit_summary_plot", _write_unit_summary_plot)

	inputs = ReconstructionInputs(
		h5_path=tmp_path / "input.raw.h5",
		stream_id="well001",
		mea_output_root=tmp_path,
		output_rel_root="recon_outputs",
		unit_ids=[1],
		phases=ReconstructionPhasesConfig(
			plot_branch_propagations=ReconstructionPlotBranchPropagationsPhaseConfig(
				enabled=True,
				branch_scope="raw",
				display=ReconstructionBranchPropagationDisplayConfig(figsize=(2.75, 6.0)),
				output=ReconstructionBranchPlotOutputConfig(
					write_png=True,
					relpath="branch_qc/propagations",
					manifest_relpath="reports/branch_propagations_manifest.json",
				),
			),
			plot_branch_velocities=ReconstructionPlotBranchVelocitiesPhaseConfig(
				enabled=True,
				branch_scope="raw",
				display=ReconstructionBranchVelocityDisplayConfig(figsize=(4.5, 6.5), show_title=True, show_legend=False),
				output=ReconstructionBranchPlotOutputConfig(
					write_png=True,
					relpath="branch_qc/velocities",
					manifest_relpath="reports/branch_velocities_manifest.json",
				),
			),
			plot_unit_summary=ReconstructionPlotUnitSummaryPhaseConfig(
				enabled=True,
				display=ReconstructionUnitSummaryDisplayConfig(show_title=True, show_velocity_legend=False),
				output=ReconstructionUnitSummaryOutputConfig(
					write_png=True,
					write_svg=False,
					relpath="reports/unit_summary",
					dpi=180.0,
				),
			),
		),
	)

	paths = resolve_unit_output_paths(
		reconstruction_out_dir=well_out_dir / "recon_outputs",
		unit_id=1,
		per_unit_outputs=inputs.per_unit_outputs,
	)
	paths["unit_dir"].mkdir(parents=True, exist_ok=True)
	paths["unit_summary_json"].write_text(
		json.dumps({"unit_id": 1, "status": "error", "error": "prior failure", "outputs": {}}),
		encoding="utf-8",
	)
	with open(paths["gtr_pkl"], "wb") as handle:
		pickle.dump(SimpleNamespace(branches=[{"branch_index": 0, "channels": [0, 1, 2]}]), handle)

	summary = run_reconstruct_plot_unit_summary_phase(inputs)
	assert summary["phase"] == "plot_unit_summary"
	assert summary["units"][0]["status"] == "ok"
	assert seen["count"] == 1
	unit_outputs = summary["units"][0]["outputs"]
	assert Path(unit_outputs["plot_unit_summary_png"]).exists()
	unit_summary_payload = json.loads(paths["unit_summary_json"].read_text(encoding="utf-8"))
	assert unit_summary_payload["status"] == "ok"
	assert unit_summary_payload["error"] is None


def test_resolve_full_chip_layout_output_paths_includes_manifest() -> None:
	paths = resolve_full_chip_layout_output_paths(
		reconstruction_out_dir=Path("/tmp/recon"),
		phase_output=ReconstructionFullChipLayoutOutputConfig(
			write_png=True,
			write_svg=True,
			relpath="reports/full_chip_layout",
			manifest_relpath="reports/full_chip_layout_manifest.json",
		),
	)
	assert paths["png_path"] == Path("/tmp/recon") / "reports/full_chip_layout.png"
	assert paths["svg_path"] == Path("/tmp/recon") / "reports/full_chip_layout.svg"
	assert paths["manifest_json"] == Path("/tmp/recon") / "reports/full_chip_layout_manifest.json"


def test_run_reconstruct_report_full_chip_layout_phase_writes_outputs(monkeypatch, tmp_path: Path, caplog) -> None:
	from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner

	well_out_dir = tmp_path / "well001"
	well_out_dir.mkdir(parents=True, exist_ok=True)
	templates_out = tmp_path / "templates_out"
	merged = tmp_path / "templates_merged"
	full = tmp_path / "templates_full"
	templates_out.mkdir(parents=True, exist_ok=True)
	merged.mkdir(parents=True, exist_ok=True)
	full.mkdir(parents=True, exist_ok=True)
	(merged / "unit_1").mkdir(parents=True, exist_ok=True)
	(merged / "unit_2").mkdir(parents=True, exist_ok=True)

	monkeypatch.setattr(
		reconstruct_runner,
		"compute_mea_analysis_output_dir",
		lambda *, output_root, data_file, well: well_out_dir,
	)
	monkeypatch.setattr(
		reconstruct_runner,
		"_resolve_templates_dirs",
		lambda _well_out_dir, **kwargs: (templates_out, merged, full),
	)
	monkeypatch.setattr(
		reconstruct_runner,
		"load_templates_for_unit",
		lambda **kwargs: (
			np.asarray([[-5.0, -10.0, -3.0], [-2.0, -4.0, -1.0], [-1.0, -6.0, -2.0]], dtype=float),
			np.asarray([[0.0, 0.0], [17.5, 0.0], [35.0, 0.0]], dtype=float),
			np.asarray([[-5.0, -10.0, -3.0], [-2.0, -4.0, -1.0], [-1.0, -6.0, -2.0]], dtype=float),
			np.asarray([[0.0, 0.0], [17.5, 0.0], [35.0, 0.0]], dtype=float),
			10_000.0,
			"merged",
		),
	)

	def _write_full_chip_plot(**kwargs):
		output_png = Path(kwargs["output_png"])
		output_png.parent.mkdir(parents=True, exist_ok=True)
		output_png.write_bytes(b"png")
		return {"full_chip_layout_png": str(output_png)}

	monkeypatch.setattr(reconstruct_runner, "write_full_chip_layout_plot", _write_full_chip_plot)

	inputs = ReconstructionInputs(
		h5_path=tmp_path / "input.raw.h5",
		stream_id="well001",
		mea_output_root=tmp_path,
		output_rel_root="recon_outputs",
		unit_ids=[1],
		force_replot=True,
		probe_geometry=ProbeGeometryConfig(active_area_um_x=100.0, active_area_um_y=80.0, pitch_um=17.5),
		phases=ReconstructionPhasesConfig(
			report_full_chip_layout=ReconstructionReportFullChipLayoutPhaseConfig(
				enabled=True,
				branch_scope="raw",
				unit_colors=ReconstructionFullChipLayoutColorConfig(strategy="colormap", color_scheme="tab20"),
				display=ReconstructionFullChipLayoutDisplayConfig(figsize=(10.0, 6.0), show_title=False),
				output=ReconstructionFullChipLayoutOutputConfig(
					write_png=True,
					write_svg=False,
					relpath="reports/full_chip_layout",
					manifest_relpath="reports/full_chip_layout_manifest.json",
				),
			),
		),
	)

	paths_ok = resolve_unit_output_paths(
		reconstruction_out_dir=well_out_dir / "recon_outputs",
		unit_id=1,
		per_unit_outputs=inputs.per_unit_outputs,
	)
	paths_ok["unit_dir"].mkdir(parents=True, exist_ok=True)
	paths_ok["unit_summary_json"].write_text(
		json.dumps({"unit_id": 1, "status": "ok", "outputs": {}}),
		encoding="utf-8",
	)
	with open(paths_ok["gtr_pkl"], "wb") as handle:
		pickle.dump(SimpleNamespace(_paths_raw=[[2, 1, 0]], branches=[], _paths_clean=[]), handle)

	paths_err = resolve_unit_output_paths(
		reconstruction_out_dir=well_out_dir / "recon_outputs",
		unit_id=2,
		per_unit_outputs=inputs.per_unit_outputs,
	)
	paths_err["unit_dir"].mkdir(parents=True, exist_ok=True)
	paths_err["unit_summary_json"].write_text(
		json.dumps({"unit_id": 2, "status": "ok", "outputs": {}}),
		encoding="utf-8",
	)
	with open(paths_err["gtr_pkl"], "wb") as handle:
		pickle.dump(SimpleNamespace(_paths_raw=[[2, 1, 0]], branches=[], _paths_clean=[]), handle)

	with caplog.at_level(logging.INFO, logger="axon_recon.reconstruct"):
		summary = run_reconstruct_report_full_chip_layout_phase(inputs)
	assert summary["phase"] == "report_full_chip_layout"
	assert Path(summary["outputs"]["full_chip_layout_png"]).exists()
	manifest_path = Path(summary["outputs"]["full_chip_layout_manifest_json"])
	assert manifest_path.exists()
	manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
	assert manifest["branch_scope"] == "raw"
	assert manifest["units_successful"] == 2
	assert manifest["units_plotted"] == 2
	assert manifest["branches_total"] == 2
	assert summary["reports_overwrite_skipped"] is False
	assert "reconstruct.report_full_chip_layout overwrite policy: action=rewrite" in caplog.text
	assert "reconstruct.report_full_chip_layout rewriting outputs branch_scope=raw" in caplog.text
	assert "reconstruct.report_full_chip_layout wrote outputs manifest=" in caplog.text
