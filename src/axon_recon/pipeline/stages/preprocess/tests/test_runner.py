from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from axon_recon.pipeline.stages.preprocess.models.inputs import (
    PreprocessConcatSegmentsPhaseConfig,
    PreprocessInputs,
    PreprocessPhasesConfig,
    PreprocessSaveRecMetadataPhaseConfig,
    PreprocessWipeSrcScratchPhaseConfig,
)
from axon_recon.pipeline.stages.preprocess.runner import (
    run_preprocess_concat_segments_phase,
    run_preprocess_save_rec_metadata_phase,
    run_preprocess_stage,
    run_preprocess_wipe_src_scratch_phase,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _full_stage_phases() -> PreprocessPhasesConfig:
    return PreprocessPhasesConfig(
        save_rec_metadata=PreprocessSaveRecMetadataPhaseConfig(enabled=True),
    )


def _install_success_fakes(
    monkeypatch,
    tmp_path: Path,
    *,
    common_electrodes: list[int] | None = None,
    captured_phase_kwargs: dict[str, dict] | None = None,
    phase_call_order: list[str] | None = None,
    log_path: Path | None = None,
) -> tuple[Path, Path]:
    from axon_recon.pipeline.stages.preprocess import runner as preprocess_runner

    well_out_dir = tmp_path / "well001"
    fake_log = log_path or (well_out_dir / "preprocess_outputs" / "RUN001_well001_pipeline.log")
    common = list(common_electrodes or [11, 22, 33])

    def _capture(name: str, kwargs: dict) -> None:
        if phase_call_order is not None:
            phase_call_order.append(name)
        if captured_phase_kwargs is not None:
            captured_phase_kwargs[name] = dict(kwargs)
            captured_phase_kwargs[name]["logger_is_none"] = kwargs.get("logger") is None

    def _fake_compute_mea_analysis_output_dir(*, output_root: Path, data_file: Path, well: str) -> Path:
        _ = output_root, data_file, well
        return well_out_dir

    def _fake_compute_pipeline_log_file(*, well_out_dir: Path, data_file: Path, stream_id: str) -> Path:
        _ = well_out_dir, data_file, stream_id
        return fake_log

    def _fake_setup_pipeline_logger(*, log_file: Path, logger_name: str, verbose: bool):
        _ = verbose
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("pipeline log\n", encoding="utf-8")
        return logging.getLogger(logger_name)

    def _fake_run_save_rec_metadata_core(**kwargs):
        _capture("save_rec_metadata", kwargs)
        segment_epochs_path = Path(str(kwargs["segment_epochs_path"]))
        contiguous_epochs_path = Path(str(kwargs["contiguous_epochs_path"]))
        sampling_metadata_path = Path(str(kwargs["sampling_metadata_path"]))
        assay_stats_path = Path(str(kwargs["assay_stats_path"]))
        common_electrodes_path = Path(str(kwargs["common_electrodes_path"]))
        for path in (segment_epochs_path, contiguous_epochs_path, sampling_metadata_path, assay_stats_path, common_electrodes_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        segment_epochs_path.write_text(
            json.dumps(
                {
                    "segment_count": 2,
                    "segments": [
                        {"rec_name": "seg000", "segment_index": 0},
                        {"rec_name": "seg001", "segment_index": 1},
                    ],
                }
            ),
            encoding="utf-8",
        )
        contiguous_epochs_path.write_text(json.dumps({"contiguous_epoch_count": 2, "epochs": []}), encoding="utf-8")
        sampling_metadata_path.write_text(
            json.dumps(
                {
                    "sampling_summary": {"stream_sampling_frequency_hz": 10_000.0},
                    "segments": [
                        {"rec_name": "seg000", "sampling_frequency_hz": 10_000.0},
                        {"rec_name": "seg001", "sampling_frequency_hz": 10_000.0},
                    ],
                }
            ),
            encoding="utf-8",
        )
        assay_stats_path.write_text("assay stats\n", encoding="utf-8")
        common_electrodes_path.write_bytes(b"npy")
        return {
            "phase": "save_rec_metadata",
            "source_h5_path": str(kwargs["source_h5_path"]),
            "resolved_h5_path": str(kwargs["h5_path"]),
            "segment_count": 2,
            "contiguous_epoch_count": 2,
            "recording_info": {"sampling_frequency_hz": 10_000.0, "num_channels": 4},
            "common_electrode_count": len(common),
            "common_electrodes_preview": list(common),
            "verbose": bool(kwargs["verbose"]),
        }

    def _fake_run_preprocess_segments_core(**kwargs):
        _capture("preprocess_segments", kwargs)
        output_dir = Path(str(kwargs["output_dir"]))
        manifest_path = Path(str(kwargs["manifest_path"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                [
                    {"name": "seg000", "folder": str(output_dir / "seg000")},
                    {"name": "seg001", "folder": str(output_dir / "seg001")},
                ]
            ),
            encoding="utf-8",
        )
        return {
            "phase": "preprocess_segments",
            "segment_count": 2,
            "rec_names": ["seg000", "seg001"],
            "manifest_path": str(manifest_path),
            "output_dir": str(output_dir),
            "phase_timing_s": {"preprocess_segments": 0.2},
        }

    def _fake_run_plot_segment_traces_core(**kwargs):
        _capture("plot_segment_traces", kwargs)
        plot_output_dir = Path(str(kwargs["plot_output_dir"]))
        plot_output_dir.mkdir(parents=True, exist_ok=True)
        (plot_output_dir / "segment_traces.png").write_text("plot\n", encoding="utf-8")
        return {
            "phase": "plot_segment_traces",
            "plot_output_dir": str(plot_output_dir),
            "segment_plot_count": 2,
        }

    def _fake_run_concat_segments_core(**kwargs):
        _capture("concat_segments", kwargs)
        recording_dir = Path(str(kwargs["recording_dir"]))
        concat_manifest_path = Path(str(kwargs["concat_manifest_path"]))
        recording_dir.mkdir(parents=True, exist_ok=True)
        concat_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        concat_manifest_path.write_text(
            json.dumps(
                {
                    "segment_count": 2,
                    "segment_source": "preprocessed",
                    "stitch_frames": [100],
                }
            ),
            encoding="utf-8",
        )
        return {
            "phase": "concat_segments",
            "recording_dir": str(recording_dir),
            "concat_manifest_path": str(concat_manifest_path),
            "segment_source": "preprocessed",
            "source_segment_count": 2,
            "concatenate_preprocessed_recordings": True,
        }

    def _fake_run_plot_concat_traces_core(**kwargs):
        _capture("plot_concat_traces", kwargs)
        plot_output_dir = Path(str(kwargs["plot_output_dir"]))
        plot_output_dir.mkdir(parents=True, exist_ok=True)
        (plot_output_dir / "concat_trace.png").write_text("plot\n", encoding="utf-8")
        return {
            "phase": "plot_concat_traces",
            "plot_output_dir": str(plot_output_dir),
            "concat_trace_plot_path": str(plot_output_dir / "concat_trace.png"),
        }

    monkeypatch.setattr(preprocess_runner, "compute_mea_analysis_output_dir", _fake_compute_mea_analysis_output_dir)
    monkeypatch.setattr(preprocess_runner, "compute_pipeline_log_file", _fake_compute_pipeline_log_file)
    monkeypatch.setattr(preprocess_runner, "setup_pipeline_logger", _fake_setup_pipeline_logger)
    monkeypatch.setattr(preprocess_runner, "run_save_rec_metadata_core", _fake_run_save_rec_metadata_core)
    monkeypatch.setattr(preprocess_runner, "run_preprocess_segments_core", _fake_run_preprocess_segments_core)
    monkeypatch.setattr(preprocess_runner, "run_plot_segment_traces_core", _fake_run_plot_segment_traces_core)
    monkeypatch.setattr(preprocess_runner, "run_concat_segments_core", _fake_run_concat_segments_core)
    monkeypatch.setattr(preprocess_runner, "run_plot_concat_traces_core", _fake_run_plot_concat_traces_core)
    return well_out_dir, fake_log


def test_run_preprocess_stage_writes_observability_artifacts(tmp_path: Path, monkeypatch) -> None:
    well_out_dir, fake_log = _install_success_fakes(monkeypatch, tmp_path)
    canonical_out_dir = well_out_dir / "preprocess_outputs"

    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="preprocess_outputs_v2",
        phases=_full_stage_phases(),
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

    assert summary["n_common_electrodes"] == 0
    outputs = dict(summary.get("outputs", {}))
    assert outputs["pipeline_log"] == str(fake_log)
    assert str(fake_log).startswith(str(canonical_out_dir))
    assert "preprocess_segments_summary_json" in outputs
    assert "plot_segment_traces_summary_json" in outputs
    assert "concat_segments_summary_json" in outputs
    assert "plot_concat_traces_summary_json" in outputs
    assert "observability.run_manifest_json" in outputs
    assert "observability.event_timeline_jsonl" in outputs
    assert "observability.environment_json" in outputs
    assert "observability.artifact_inventory_json" in outputs
    assert "observability.stage_log" in outputs

    run_manifest = _read_json(Path(outputs["observability.run_manifest_json"]))
    assert run_manifest["status"] == "ok"
    assert run_manifest["mode"] == "detailed"
    assert run_manifest["common_electrodes"]["count"] == 0

    event_timeline = Path(outputs["observability.event_timeline_jsonl"])
    assert event_timeline.exists()
    assert event_timeline.read_text(encoding="utf-8").strip() != ""

    captured_log = Path(outputs["observability.stage_log"])
    assert captured_log.exists()
    assert canonical_out_dir.joinpath("context", "segment_recordings_summary.json").exists()


def test_run_preprocess_stage_writes_failure_observability_manifest(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.preprocess import runner as preprocess_runner

    well_out_dir, _fake_log = _install_success_fakes(monkeypatch, tmp_path)

    def _explode(**kwargs):
        _ = kwargs
        raise RuntimeError("preprocess exploded")

    monkeypatch.setattr(preprocess_runner, "run_preprocess_segments_core", _explode)

    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        phases=_full_stage_phases(),
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


def test_run_preprocess_stage_passes_plot_and_segment_controls_to_phase_cores(tmp_path: Path, monkeypatch) -> None:
    captured_phase_kwargs: dict[str, dict] = {}
    _install_success_fakes(monkeypatch, tmp_path, captured_phase_kwargs=captured_phase_kwargs)
    canonical_out_dir = tmp_path / "well001" / "preprocess_outputs"

    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        phases=_full_stage_phases(),
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

    assert captured_phase_kwargs["preprocess_segments"]["limit_segments_per_well"] == 2
    assert captured_phase_kwargs["preprocess_segments"]["logger_is_none"] is True
    assert Path(str(captured_phase_kwargs["save_rec_metadata"]["segment_epochs_path"])) == canonical_out_dir / "segment_epochs.json"
    assert Path(str(captured_phase_kwargs["save_rec_metadata"]["assay_stats_path"])) == canonical_out_dir / "assay_stats_well001.txt"
    assert Path(str(captured_phase_kwargs["plot_segment_traces"]["plot_output_dir"])) == canonical_out_dir
    assert Path(str(captured_phase_kwargs["plot_concat_traces"]["plot_output_dir"])) == canonical_out_dir
    assert captured_phase_kwargs["plot_segment_traces"]["segment_trace_n_reps"] == 6
    assert captured_phase_kwargs["plot_segment_traces"]["plot_n_jobs"] == 3
    assert captured_phase_kwargs["plot_segment_traces"]["trace_max_points"] == -1
    assert captured_phase_kwargs["plot_concat_traces"]["concat_trace_n_reps"] == 3
    assert captured_phase_kwargs["plot_concat_traces"]["plot_n_jobs"] == 3
    assert captured_phase_kwargs["plot_concat_traces"]["trace_max_points"] == -1
    assert summary.get("inputs", {}).get("trace_max_points") == -1
    assert summary.get("inputs", {}).get("debug_limit_segments_per_well") == 2
    assert summary.get("inputs", {}).get("logging_enabled") is False
    assert summary.get("inputs", {}).get("logging_verbose") is False
    assert summary.get("inputs", {}).get("logging_file_relpath") == "logs/custom_preprocess.log"
    assert summary.get("inputs", {}).get("n_representative_channels") == 9
    assert summary.get("inputs", {}).get("concat_trace_n_reps") == 3
    assert summary.get("inputs", {}).get("segment_trace_n_reps") == 6
    assert summary.get("inputs", {}).get("plot_n_jobs") == 3


def test_run_preprocess_stage_normalizes_preprocess_root_prefixed_relative_paths(tmp_path: Path, monkeypatch) -> None:
    captured_phase_kwargs: dict[str, dict] = {}
    _install_success_fakes(monkeypatch, tmp_path, captured_phase_kwargs=captured_phase_kwargs)
    canonical_out_dir = tmp_path / "well001" / "preprocess_outputs"

    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        logging_file_relpath="preprocess_outputs/logs/custom_preprocess.log",
        plot_output_dir="preprocess_outputs/plots",
        phases=PreprocessPhasesConfig(
            save_rec_metadata=PreprocessSaveRecMetadataPhaseConfig(
                enabled=True,
                segment_epochs_relpath="preprocess_outputs/meta/segment_epochs.json",
                contiguous_epochs_relpath="preprocess_outputs/meta/continuous_epochs.json",
                sampling_metadata_relpath="preprocess_outputs/meta/sampling_rate_metadata.json",
                common_electrodes_relpath="preprocess_outputs/meta/common_electrodes.npy",
            )
        ),
    )

    result = run_preprocess_stage(inputs)
    summary = _read_json(result.summary_json)
    outputs = dict(summary.get("outputs", {}))

    assert outputs["pipeline_log"] == str(canonical_out_dir / "logs" / "custom_preprocess.log")
    assert Path(str(captured_phase_kwargs["save_rec_metadata"]["segment_epochs_path"])) == canonical_out_dir / "meta" / "segment_epochs.json"
    assert Path(str(captured_phase_kwargs["save_rec_metadata"]["contiguous_epochs_path"])) == canonical_out_dir / "meta" / "continuous_epochs.json"
    assert Path(str(captured_phase_kwargs["save_rec_metadata"]["sampling_metadata_path"])) == canonical_out_dir / "meta" / "sampling_rate_metadata.json"
    assert Path(str(captured_phase_kwargs["save_rec_metadata"]["common_electrodes_path"])) == canonical_out_dir / "meta" / "common_electrodes.npy"
    assert Path(str(captured_phase_kwargs["plot_segment_traces"]["plot_output_dir"])) == canonical_out_dir / "plots"


def test_run_preprocess_stage_force_restart_clears_outputs_and_reruns_enabled_phases_in_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    phase_call_order: list[str] = []
    well_out_dir, _fake_log = _install_success_fakes(
        monkeypatch,
        tmp_path,
        phase_call_order=phase_call_order,
    )
    canonical_out_dir = well_out_dir / "preprocess_outputs"
    stale_path = canonical_out_dir / "stale" / "obsolete.txt"
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    stale_path.write_text("obsolete\n", encoding="utf-8")

    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        force_restart=True,
        logging_file_relpath="logs/preprocess_pipeline.log",
        phases=_full_stage_phases(),
    )

    result = run_preprocess_stage(inputs)
    summary = _read_json(result.summary_json)
    outputs = dict(summary.get("outputs", {}))

    assert not stale_path.exists()
    assert phase_call_order == [
        "save_rec_metadata",
        "preprocess_segments",
        "plot_segment_traces",
        "concat_segments",
        "plot_concat_traces",
    ]
    assert summary.get("inputs", {}).get("force_restart") is True
    assert outputs["pipeline_log"] == str(canonical_out_dir / "logs" / "preprocess_pipeline.log")
    assert Path(outputs["pipeline_log"]).exists()


def test_run_preprocess_stage_recovers_from_self_referential_log_symlink(tmp_path: Path, monkeypatch) -> None:
    well_out_dir = tmp_path / "well001"
    bad_log = well_out_dir / "preprocess_outputs" / "logs" / "preprocess_pipeline.log"
    bad_log.parent.mkdir(parents=True, exist_ok=True)
    bad_log.symlink_to(bad_log)
    _install_success_fakes(monkeypatch, tmp_path, log_path=bad_log)

    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        phases=_full_stage_phases(),
        observability_mode="detailed",
        observability_output_subdir="run_metadata",
        observability_save_stage_log=True,
        observability_stage_log_relpath="logs/preprocess_pipeline.log",
    )

    result = run_preprocess_stage(inputs)

    assert result.summary_json.exists()
    assert not bad_log.is_symlink()


def test_run_preprocess_concat_segments_phase_writes_targeted_summary(tmp_path: Path, monkeypatch) -> None:
    _install_success_fakes(monkeypatch, tmp_path)

    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
    )

    payload = run_preprocess_concat_segments_phase(inputs)

    assert payload["phase"] == "concat_segments"
    assert Path(str(payload["summary_json"])).exists()
    assert payload["outputs"]["concatenated_recording_dir"].endswith("concatenated_recording")
    assert payload["segment_source"] == "preprocessed"
    assert payload["concatenate_preprocessed_recordings"] is True


def test_run_preprocess_concat_segments_phase_uses_targeted_summary_when_raw_concat_toggle_disabled(tmp_path: Path, monkeypatch) -> None:
    _install_success_fakes(monkeypatch, tmp_path)

    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        phases=PreprocessPhasesConfig(
            concat_segments=PreprocessConcatSegmentsPhaseConfig(
                enabled=True,
                concatenate_preprocessed_recordings=False,
            )
        ),
    )

    payload = run_preprocess_concat_segments_phase(inputs)

    assert payload["phase"] == "concat_segments"
    assert payload["segment_source"] == "preprocessed"
    assert payload["source_segment_count"] == 2


def test_run_preprocess_save_rec_metadata_phase_writes_targeted_summary(tmp_path: Path, monkeypatch) -> None:
    well_out_dir, _fake_log = _install_success_fakes(monkeypatch, tmp_path)
    canonical_out_dir = well_out_dir / "preprocess_outputs"

    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        source_h5_path=tmp_path / "input.raw.h5",
        phases=PreprocessPhasesConfig(
            save_rec_metadata=PreprocessSaveRecMetadataPhaseConfig(
                enabled=True,
                verbose=True,
                segment_epochs_relpath="metadata/segment_epochs.json",
                contiguous_epochs_relpath="metadata/continuous_epochs.json",
                sampling_metadata_relpath="metadata/sampling_rate_metadata.json",
            )
        ),
    )

    payload = run_preprocess_save_rec_metadata_phase(inputs)

    assert payload["phase"] == "save_rec_metadata"
    assert Path(str(payload["summary_json"])).exists()
    assert payload["verbose"] is True
    assert payload["recording_info"]["sampling_frequency_hz"] == 10_000.0
    assert payload["outputs"]["segment_epochs_json"] == str(canonical_out_dir / "metadata" / "segment_epochs.json")
    assert payload["outputs"]["contiguous_epochs_json"] == str(canonical_out_dir / "metadata" / "continuous_epochs.json")
    assert payload["outputs"]["sampling_metadata_json"] == str(canonical_out_dir / "metadata" / "sampling_rate_metadata.json")
    assert payload["outputs"]["assay_stats_txt"] == str(canonical_out_dir / "assay_stats_well001.txt")


def test_run_preprocess_wipe_src_scratch_phase_removes_scratch_input_files(tmp_path: Path, monkeypatch) -> None:
    _install_success_fakes(monkeypatch, tmp_path)

    source_h5_path = tmp_path / "raw_data" / "input.raw.h5"
    source_h5_path.parent.mkdir(parents=True, exist_ok=True)
    source_h5_path.write_bytes(b"source")

    scratch_h5_path = tmp_path / "scratch_inputs" / "input.raw.h5"
    scratch_h5_path.parent.mkdir(parents=True, exist_ok=True)
    scratch_h5_path.write_bytes(b"scratch")
    scratch_cfg_path = scratch_h5_path.parent / "input.cfg"
    scratch_cfg_path.write_text("foo=1\n", encoding="utf-8")

    inputs = PreprocessInputs(
        h5_path=scratch_h5_path,
        stream_id="well001",
        mea_output_root=tmp_path,
        source_h5_path=source_h5_path,
        copied_to_scratch=True,
    )

    payload = run_preprocess_wipe_src_scratch_phase(inputs)

    assert payload["phase"] == "wipe_src_scratch"
    assert payload["dry_run"] is False
    assert payload["status"] == "ok"
    assert Path(str(payload["summary_json"])).exists()
    assert str(scratch_h5_path) in list(payload["removed_paths"])
    assert list(payload["would_remove_paths"]) == []
    assert str(scratch_cfg_path) in list(payload["removed_paths"])
    assert not scratch_h5_path.exists()
    assert not scratch_cfg_path.exists()


def test_run_preprocess_wipe_src_scratch_phase_dry_run_reports_paths_without_deleting(tmp_path: Path, monkeypatch) -> None:
    _install_success_fakes(monkeypatch, tmp_path)

    source_h5_path = tmp_path / "raw_data" / "input.raw.h5"
    source_h5_path.parent.mkdir(parents=True, exist_ok=True)
    source_h5_path.write_bytes(b"source")

    scratch_h5_path = tmp_path / "scratch_inputs" / "input.raw.h5"
    scratch_h5_path.parent.mkdir(parents=True, exist_ok=True)
    scratch_h5_path.write_bytes(b"scratch")
    scratch_cfg_path = scratch_h5_path.parent / "input.cfg"
    scratch_cfg_path.write_text("foo=1\n", encoding="utf-8")

    inputs = PreprocessInputs(
        h5_path=scratch_h5_path,
        stream_id="well001",
        mea_output_root=tmp_path,
        source_h5_path=source_h5_path,
        copied_to_scratch=True,
        phases=PreprocessPhasesConfig(
            wipe_src_scratch=PreprocessWipeSrcScratchPhaseConfig(
                enabled=False,
                dry_run=True,
            )
        ),
    )

    payload = run_preprocess_wipe_src_scratch_phase(inputs)

    assert payload["phase"] == "wipe_src_scratch"
    assert payload["dry_run"] is True
    assert payload["status"] == "dry_run"
    assert Path(str(payload["summary_json"])).exists()
    assert list(payload["removed_paths"]) == []
    assert str(scratch_h5_path) in list(payload["would_remove_paths"])
    assert str(scratch_cfg_path) in list(payload["would_remove_paths"])
    assert scratch_h5_path.exists()
    assert scratch_cfg_path.exists()
