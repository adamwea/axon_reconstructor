from __future__ import annotations

import json
from pathlib import Path

import pytest

from axon_recon.pipeline.stages.preprocess.models.inputs import PreprocessInputs
from axon_recon.pipeline.stages.preprocess.runner import run_preprocess_stage


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_run_preprocess_stage_writes_observability_artifacts(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.preprocess import runner as preprocess_runner

    well_out_dir = tmp_path / "well001"
    fake_log = well_out_dir / "RUN001_well001_pipeline.log"

    def _fake_compute_mea_analysis_output_dir(*, output_root: Path, data_file: Path, well: str) -> Path:
        return well_out_dir

    def _fake_compute_pipeline_log_file(*, well_out_dir: Path, data_file: Path, stream_id: str) -> Path:
        fake_log.parent.mkdir(parents=True, exist_ok=True)
        fake_log.write_text("pipeline log\n", encoding="utf-8")
        return fake_log

    def _fake_run_legacy_preprocess_stage(**kwargs):
        legacy_out = well_out_dir / "preprocess_outputs"
        (legacy_out / "preprocessed_recording").mkdir(parents=True, exist_ok=True)
        (legacy_out / "common_electrodes.npy").write_bytes(b"npy")
        per_segment_manifest = legacy_out / "per_segment_preprocessed" / "manifest.json"
        per_segment_manifest.parent.mkdir(parents=True, exist_ok=True)
        per_segment_manifest.write_text('{"segments": []}\n', encoding="utf-8")
        return object(), [11, 22, 33]

    monkeypatch.setattr(preprocess_runner, "compute_mea_analysis_output_dir", _fake_compute_mea_analysis_output_dir)
    monkeypatch.setattr(preprocess_runner, "compute_pipeline_log_file", _fake_compute_pipeline_log_file)
    monkeypatch.setattr(preprocess_runner, "run_legacy_preprocess_stage", _fake_run_legacy_preprocess_stage)

    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="preprocess_outputs_v2",
        observability_mode="detailed",
        observability_output_subdir="run_metadata",
        observability_save_run_manifest=True,
        observability_save_event_timeline=True,
        observability_save_environment=True,
        observability_save_artifact_inventory=True,
        observability_save_stage_log=True,
        observability_stage_log_relpath="logs/preprocess_pipeline.log",
    )

    result = run_preprocess_stage(inputs)
    summary = _read_json(result.summary_json)

    assert summary["n_common_electrodes"] == 3
    outputs = dict(summary.get("outputs", {}))
    assert "pipeline_log" in outputs
    assert outputs["pipeline_log"] == str(fake_log)
    assert "observability.run_manifest_json" in outputs
    assert "observability.event_timeline_jsonl" in outputs
    assert "observability.environment_json" in outputs
    assert "observability.artifact_inventory_json" in outputs
    assert "observability.stage_log" in outputs

    run_manifest = _read_json(Path(outputs["observability.run_manifest_json"]))
    assert run_manifest["status"] == "ok"
    assert run_manifest["mode"] == "detailed"
    assert run_manifest["common_electrodes"]["count"] == 3

    event_timeline = Path(outputs["observability.event_timeline_jsonl"])
    assert event_timeline.exists()
    assert event_timeline.read_text(encoding="utf-8").strip() != ""

    captured_log = Path(outputs["observability.stage_log"])
    assert captured_log.exists()


def test_run_preprocess_stage_writes_failure_observability_manifest(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.preprocess import runner as preprocess_runner

    well_out_dir = tmp_path / "well001"
    fake_log = well_out_dir / "RUN001_well001_pipeline.log"

    def _fake_compute_mea_analysis_output_dir(*, output_root: Path, data_file: Path, well: str) -> Path:
        return well_out_dir

    def _fake_compute_pipeline_log_file(*, well_out_dir: Path, data_file: Path, stream_id: str) -> Path:
        fake_log.parent.mkdir(parents=True, exist_ok=True)
        fake_log.write_text("pipeline log\n", encoding="utf-8")
        return fake_log

    def _fake_run_legacy_preprocess_stage(**kwargs):
        raise RuntimeError("preprocess exploded")

    monkeypatch.setattr(preprocess_runner, "compute_mea_analysis_output_dir", _fake_compute_mea_analysis_output_dir)
    monkeypatch.setattr(preprocess_runner, "compute_pipeline_log_file", _fake_compute_pipeline_log_file)
    monkeypatch.setattr(preprocess_runner, "run_legacy_preprocess_stage", _fake_run_legacy_preprocess_stage)

    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="preprocess_outputs_v2",
        observability_mode="detailed",
        observability_output_subdir="run_metadata",
        observability_save_run_manifest=True,
        observability_save_event_timeline=True,
        observability_save_environment=True,
        observability_save_artifact_inventory=True,
        observability_save_stage_log=True,
    )

    with pytest.raises(RuntimeError, match="preprocess exploded"):
        run_preprocess_stage(inputs)

    canonical_out_dir = well_out_dir / "preprocess_outputs"
    failure_summary = canonical_out_dir / "preprocess_failure_summary.json"
    run_manifest = canonical_out_dir / "run_metadata" / "run_manifest.json"

    assert failure_summary.exists()
    assert run_manifest.exists()

    payload = _read_json(run_manifest)
    assert payload["status"] == "error"
    assert "error" in payload
    assert "preprocess exploded" in str(payload["error"].get("message", ""))


def test_run_preprocess_stage_treats_non_positive_trace_max_points_as_uncapped(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.preprocess import runner as preprocess_runner

    well_out_dir = tmp_path / "well001"
    captured_legacy_kwargs: dict = {}

    def _fake_compute_mea_analysis_output_dir(*, output_root: Path, data_file: Path, well: str) -> Path:
        return well_out_dir

    def _fake_compute_pipeline_log_file(*, well_out_dir: Path, data_file: Path, stream_id: str) -> Path:
        return well_out_dir / "RUN001_well001_pipeline.log"

    def _fake_run_legacy_preprocess_stage(**kwargs):
        captured_legacy_kwargs.update(kwargs)
        legacy_out = well_out_dir / "preprocess_outputs"
        (legacy_out / "preprocessed_recording").mkdir(parents=True, exist_ok=True)
        (legacy_out / "common_electrodes.npy").write_bytes(b"npy")
        per_segment_manifest = legacy_out / "per_segment_preprocessed" / "manifest.json"
        per_segment_manifest.parent.mkdir(parents=True, exist_ok=True)
        per_segment_manifest.write_text('{"segments": []}\n', encoding="utf-8")
        return object(), [1, 2, 3]

    monkeypatch.setattr(preprocess_runner, "compute_mea_analysis_output_dir", _fake_compute_mea_analysis_output_dir)
    monkeypatch.setattr(preprocess_runner, "compute_pipeline_log_file", _fake_compute_pipeline_log_file)
    monkeypatch.setattr(preprocess_runner, "run_legacy_preprocess_stage", _fake_run_legacy_preprocess_stage)

    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="preprocess_outputs_v2",
        logging_enabled=False,
        logging_verbose=False,
        logging_file_relpath="logs/custom_preprocess.log",
        debug_limit_segments_per_well=2,
        n_representative_channels=9,
        concat_trace_n_reps=3,
        segment_trace_n_reps=6,
        plot_n_jobs=3,
        trace_max_points=-1,
    )

    result = run_preprocess_stage(inputs)
    summary = _read_json(result.summary_json)

    assert captured_legacy_kwargs.get("trace_max_points") == -1
    assert captured_legacy_kwargs.get("limit_segments_per_well") == 2
    assert captured_legacy_kwargs.get("log_enabled") is False
    assert captured_legacy_kwargs.get("log_verbose") is False
    assert captured_legacy_kwargs.get("log_file_override") == "logs/custom_preprocess.log"
    assert captured_legacy_kwargs.get("suppress_h5_plugin_messages") is False
    assert captured_legacy_kwargs.get("phase_dividers") is True
    assert captured_legacy_kwargs.get("plot_concat_trace") is True
    assert captured_legacy_kwargs.get("n_representative_channels") == 9
    assert captured_legacy_kwargs.get("concat_trace_n_reps") == 3
    assert captured_legacy_kwargs.get("segment_trace_n_reps") == 6
    assert captured_legacy_kwargs.get("plot_n_jobs") == 3
    assert captured_legacy_kwargs.get("save_concat_recording") is True
    assert captured_legacy_kwargs.get("save_segment_recordings") is True
    assert captured_legacy_kwargs.get("save_chunk_duration") == "1s"
    assert captured_legacy_kwargs.get("save_progress_bar") is False
    assert captured_legacy_kwargs.get("concat_save_n_jobs") is None
    assert captured_legacy_kwargs.get("segment_save_n_jobs") is None
    assert captured_legacy_kwargs.get("print_n_jobs_used") is False
    assert summary.get("inputs", {}).get("trace_max_points") == -1
    assert summary.get("inputs", {}).get("debug_limit_segments_per_well") == 2
    assert summary.get("inputs", {}).get("logging_enabled") is False
    assert summary.get("inputs", {}).get("logging_verbose") is False
    assert summary.get("inputs", {}).get("logging_file_relpath") == "logs/custom_preprocess.log"
    assert summary.get("inputs", {}).get("logging_suppress_h5_plugin_messages") is False
    assert summary.get("inputs", {}).get("logging_phase_dividers") is True
    assert summary.get("inputs", {}).get("plot_concat_trace") is True
    assert summary.get("inputs", {}).get("n_representative_channels") == 9
    assert summary.get("inputs", {}).get("concat_trace_n_reps") == 3
    assert summary.get("inputs", {}).get("segment_trace_n_reps") == 6
    assert summary.get("inputs", {}).get("plot_n_jobs") == 3
    assert summary.get("inputs", {}).get("print_n_jobs_used") is False


def test_run_preprocess_stage_recovers_from_self_referential_log_symlink(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.preprocess import runner as preprocess_runner

    well_out_dir = tmp_path / "well001"
    bad_log = well_out_dir / "logs" / "preprocess_pipeline.log"
    bad_log.parent.mkdir(parents=True, exist_ok=True)
    bad_log.symlink_to(bad_log)

    def _fake_compute_mea_analysis_output_dir(*, output_root: Path, data_file: Path, well: str) -> Path:
        return well_out_dir

    def _fake_compute_pipeline_log_file(*, well_out_dir: Path, data_file: Path, stream_id: str) -> Path:
        return bad_log

    def _fake_run_legacy_preprocess_stage(**kwargs):
        legacy_out = well_out_dir / "preprocess_outputs"
        (legacy_out / "preprocessed_recording").mkdir(parents=True, exist_ok=True)
        (legacy_out / "common_electrodes.npy").write_bytes(b"npy")
        per_segment_manifest = legacy_out / "per_segment_preprocessed" / "manifest.json"
        per_segment_manifest.parent.mkdir(parents=True, exist_ok=True)
        per_segment_manifest.write_text('{"segments": []}\n', encoding="utf-8")
        return object(), [1, 2, 3]

    monkeypatch.setattr(preprocess_runner, "compute_mea_analysis_output_dir", _fake_compute_mea_analysis_output_dir)
    monkeypatch.setattr(preprocess_runner, "compute_pipeline_log_file", _fake_compute_pipeline_log_file)
    monkeypatch.setattr(preprocess_runner, "run_legacy_preprocess_stage", _fake_run_legacy_preprocess_stage)

    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="preprocess_outputs_v2",
        observability_mode="detailed",
        observability_output_subdir="run_metadata",
        observability_save_stage_log=True,
        observability_stage_log_relpath="logs/preprocess_pipeline.log",
    )

    result = run_preprocess_stage(inputs)

    assert result.summary_json.exists()
    assert not bad_log.is_symlink()
