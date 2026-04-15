from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from axon_recon.pipeline.stages.reconstruct.io import format_unit_reldir
from axon_recon.pipeline.stages.reconstruct.io import resolve_unit_output_paths
from axon_recon.pipeline.stages.reconstruct.models.inputs import (
	PerUnitOutputsConfig,
	ReconstructionAvReconsConfig,
	ReconstructionGenerateGtrsPhaseConfig,
	ReconstructionInputs,
	ReconstructionPhasesConfig,
	ReconstructionReportReconsPhaseConfig,
)
from axon_recon.pipeline.stages.reconstruct.runner import (
	run_reconstruct_generate_gtrs_phase,
	run_reconstruct_report_recons_phase,
)


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


def test_run_reconstruct_report_recons_phase_rejects_av_recons_pdf(monkeypatch, tmp_path: Path) -> None:
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

	with pytest.raises(NotImplementedError, match="av_recons.write_pdf"):
		run_reconstruct_report_recons_phase(inputs)
