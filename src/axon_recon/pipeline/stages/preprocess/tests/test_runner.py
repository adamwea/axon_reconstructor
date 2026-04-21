from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from axon_recon.pipeline.stages.preprocess.models.inputs import (
    PreprocessConcatenateRecordingsPhaseConfig,
    PreprocessInputs,
    PreprocessPhaseConfig,
    PreprocessPhasesConfig,
    PreprocessSaveRecMetadataPhaseConfig,
    PreprocessWipeSrcScratchPhaseConfig,
)
from axon_recon.pipeline.stages.preprocess.runner import (
    run_preprocess_concatenate_recordings_phase,
    run_preprocess_save_rec_metadata_phase,
    run_preprocess_stage,
    run_preprocess_wipe_src_scratch_phase,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _install_success_fakes(
    monkeypatch,
    tmp_path: Path,
    *,
    common_electrodes: list[int] | None = None,
    captured_build_kwargs: dict | None = None,
    log_path: Path | None = None,
) -> tuple[Path, Path]:
    from axon_recon.pipeline.stages.preprocess import runner as preprocess_runner

    well_out_dir = tmp_path / "well001"
    fake_log = log_path or (well_out_dir / "RUN001_well001_pipeline.log")

    def _fake_compute_mea_analysis_output_dir(*, output_root: Path, data_file: Path, well: str) -> Path:
        return well_out_dir

    def _fake_compute_pipeline_log_file(*, well_out_dir: Path, data_file: Path, stream_id: str) -> Path:
        return fake_log

    def _fake_setup_pipeline_logger(*, log_file: Path, logger_name: str, verbose: bool):
        _ = verbose
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("pipeline log\n", encoding="utf-8")
        return logging.getLogger(logger_name)

    def _fake_build_preprocess_plan(*, h5_path: Path, stream_id: str):
        return SimpleNamespace(
            h5_path=h5_path,
            stream_id=stream_id,
            cfg_discovery_summary={"n_cfg_files": 0},
        )

    def _fake_run_build_preprocessed_recording_core(**kwargs):
        if captured_build_kwargs is not None:
            captured_build_kwargs.update(kwargs)
            captured_build_kwargs["logger_is_none"] = kwargs.get("logger") is None
        return (
            object(),
            list(common_electrodes or [1, 2, 3]),
            {
                "rec_names": ["seg000", "seg001"],
                "segment_recordings_raw": [object(), object()],
                "segment_recordings_raw_concat": [object(), object()],
                "segment_recordings_preprocessed": [object(), object()],
                "segment_recordings_preprocessed_concat": [object(), object()],
                "segment_stats": [
                    {"fs": 10_000.0, "n_samples": 100, "n_channels": 4},
                    {"fs": 10_000.0, "n_samples": 120, "n_channels": 4},
                ],
                "phase_timing_s": {"build_total": 1.25},
            },
        )

    def _fake_run_save_concatenated_recording_core(**kwargs):
        recording_dir = Path(str(kwargs["recording_dir"]))
        recording_dir.mkdir(parents=True, exist_ok=True)
        return {
            "recording_dir": str(recording_dir),
            "saved": True,
            "reused_existing": False,
        }

    def _fake_run_save_segment_recordings_core(**kwargs):
        output_dir = Path(str(kwargs["output_dir"]))
        manifest_path = Path(str(kwargs["manifest_path"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text('{"segments": []}\n', encoding="utf-8")
        return {
            "output_dir": str(output_dir),
            "manifest_path": str(manifest_path),
            "saved": True,
            "reused_existing": False,
            "segment_count": 2,
        }

    def _fake_run_save_common_electrodes_core(**kwargs):
        output_path = Path(str(kwargs["output_path"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"npy")
        return {
            "common_electrodes_path": str(output_path),
            "electrode_count": int(len(list(kwargs["common_electrodes"]))),
        }

    monkeypatch.setattr(preprocess_runner, "compute_mea_analysis_output_dir", _fake_compute_mea_analysis_output_dir)
    monkeypatch.setattr(preprocess_runner, "compute_pipeline_log_file", _fake_compute_pipeline_log_file)
    monkeypatch.setattr(preprocess_runner, "setup_pipeline_logger", _fake_setup_pipeline_logger)
    monkeypatch.setattr(preprocess_runner, "build_preprocess_plan", _fake_build_preprocess_plan)
    monkeypatch.setattr(preprocess_runner, "run_build_preprocessed_recording_core", _fake_run_build_preprocessed_recording_core)
    monkeypatch.setattr(preprocess_runner, "run_save_concatenated_recording_core", _fake_run_save_concatenated_recording_core)
    monkeypatch.setattr(preprocess_runner, "run_save_segment_recordings_core", _fake_run_save_segment_recordings_core)
    monkeypatch.setattr(preprocess_runner, "run_save_common_electrodes_core", _fake_run_save_common_electrodes_core)
    return well_out_dir, fake_log


def test_run_preprocess_stage_writes_observability_artifacts(tmp_path: Path, monkeypatch) -> None:
    well_out_dir, fake_log = _install_success_fakes(monkeypatch, tmp_path, common_electrodes=[11, 22, 33])

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
    assert outputs["pipeline_log"] == str(fake_log)
    assert "preprocess_segments_summary_json" in outputs
    assert "concatenate_recordings_summary_json" in outputs
    assert "save_common_electrodes_summary_json" in outputs
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
    assert well_out_dir.joinpath("preprocess_outputs", "context", "segment_recordings_summary.json").exists()


def test_run_preprocess_stage_writes_failure_observability_manifest(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.preprocess import runner as preprocess_runner

    well_out_dir = tmp_path / "well001"
    fake_log = well_out_dir / "RUN001_well001_pipeline.log"

    def _fake_compute_mea_analysis_output_dir(*, output_root: Path, data_file: Path, well: str) -> Path:
        return well_out_dir

    def _fake_compute_pipeline_log_file(*, well_out_dir: Path, data_file: Path, stream_id: str) -> Path:
        return fake_log

    def _fake_setup_pipeline_logger(*, log_file: Path, logger_name: str, verbose: bool):
        _ = logger_name, verbose
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("pipeline log\n", encoding="utf-8")
        return logging.getLogger("test.preprocess.failure")

    def _fake_build_preprocess_plan(*, h5_path: Path, stream_id: str):
        return SimpleNamespace(h5_path=h5_path, stream_id=stream_id, cfg_discovery_summary={})

    def _fake_run_build_preprocessed_recording_core(**kwargs):
        _ = kwargs
        raise RuntimeError("preprocess exploded")

    monkeypatch.setattr(preprocess_runner, "compute_mea_analysis_output_dir", _fake_compute_mea_analysis_output_dir)
    monkeypatch.setattr(preprocess_runner, "compute_pipeline_log_file", _fake_compute_pipeline_log_file)
    monkeypatch.setattr(preprocess_runner, "setup_pipeline_logger", _fake_setup_pipeline_logger)
    monkeypatch.setattr(preprocess_runner, "build_preprocess_plan", _fake_build_preprocess_plan)
    monkeypatch.setattr(preprocess_runner, "run_build_preprocessed_recording_core", _fake_run_build_preprocessed_recording_core)

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
    captured_build_kwargs: dict = {}
    _install_success_fakes(monkeypatch, tmp_path, captured_build_kwargs=captured_build_kwargs)

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

    assert captured_build_kwargs.get("trace_max_points") == -1
    assert captured_build_kwargs.get("limit_segments_per_well") == 2
    assert captured_build_kwargs.get("logger_is_none") is True
    assert captured_build_kwargs.get("plot_concat_trace") is True
    assert captured_build_kwargs.get("n_representative_channels") == 9
    assert captured_build_kwargs.get("concat_trace_n_reps") == 3
    assert captured_build_kwargs.get("segment_trace_n_reps") == 6
    assert captured_build_kwargs.get("plot_n_jobs") == 3
    assert captured_build_kwargs.get("saved_assay_stats_path") == tmp_path / "well001" / "assay_stats_well001.txt"
    assert captured_build_kwargs.get("require_saved_assay_stats") is True
    assert captured_build_kwargs.get("emit_phase_dividers_to_stdout") is True
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
    well_out_dir = tmp_path / "well001"
    bad_log = well_out_dir / "logs" / "preprocess_pipeline.log"
    bad_log.parent.mkdir(parents=True, exist_ok=True)
    bad_log.symlink_to(bad_log)
    _install_success_fakes(monkeypatch, tmp_path, log_path=bad_log)

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


def test_run_preprocess_concatenate_recordings_phase_writes_targeted_summary(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.preprocess import runner as preprocess_runner

    _install_success_fakes(monkeypatch, tmp_path)
    segment_calls: list[str] = []
    common_calls: list[str] = []

    def _unexpected_segment_save(**kwargs):
        _ = kwargs
        segment_calls.append("segment")
        raise AssertionError("segment save should not run for concat-only phase")

    def _track_common_save(**kwargs):
        _ = kwargs
        common_calls.append("common")
        return {
            "common_electrodes_path": str(tmp_path / "common_electrodes.npy"),
            "electrode_count": 3,
        }

    monkeypatch.setattr(preprocess_runner, "run_save_segment_recordings_core", _unexpected_segment_save)
    monkeypatch.setattr(preprocess_runner, "run_save_common_electrodes_core", _track_common_save)

    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
    )

    payload = run_preprocess_concatenate_recordings_phase(inputs)

    assert payload["phase"] == "concatenate_recordings"
    assert Path(str(payload["summary_json"])).exists()
    assert payload["outputs"]["concatenated_recording_dir"].endswith("preprocessed_recording")
    assert payload["segment_source"] == "preprocessed"
    assert payload["concatenate_preprocessed_recordings"] is True
    assert segment_calls == []
    assert common_calls == ["common"]


def test_run_preprocess_concatenate_recordings_phase_can_source_raw_segments(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.preprocess import runner as preprocess_runner

    _install_success_fakes(monkeypatch, tmp_path)
    captured_concat_segments: list[object] = []
    sentinel_concat_recording = object()
    captured_save_multirecording: list[object] = []

    def _fake_concatenate_segment_recordings(segment_recordings: list[object]) -> object:
        captured_concat_segments.extend(segment_recordings)
        return sentinel_concat_recording

    def _fake_save_concatenated_recording_core(**kwargs):
        captured_save_multirecording.append(kwargs["multirecording"])
        recording_dir = Path(str(kwargs["recording_dir"]))
        recording_dir.mkdir(parents=True, exist_ok=True)
        return {
            "recording_dir": str(recording_dir),
            "saved": True,
            "reused_existing": False,
        }

    monkeypatch.setattr(preprocess_runner, "_concatenate_segment_recordings", _fake_concatenate_segment_recordings)
    monkeypatch.setattr(preprocess_runner, "run_save_concatenated_recording_core", _fake_save_concatenated_recording_core)

    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        phases=PreprocessPhasesConfig(
            concatenate_recordings=PreprocessConcatenateRecordingsPhaseConfig(
                enabled=True,
                concatenate_preprocessed_recordings=False,
                save_common_electrodes=PreprocessPhaseConfig(enabled=False),
            )
        ),
    )

    payload = run_preprocess_concatenate_recordings_phase(inputs)

    assert payload["phase"] == "concatenate_recordings"
    assert payload["segment_source"] == "raw"
    assert payload["concatenate_preprocessed_recordings"] is False
    assert payload["source_segment_count"] == 2
    assert captured_save_multirecording == [sentinel_concat_recording]
    assert len(captured_concat_segments) == 2


def test_run_preprocess_save_rec_metadata_phase_writes_targeted_summary(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.preprocess import runner as preprocess_runner

    _install_success_fakes(monkeypatch, tmp_path)

    monkeypatch.setattr(
        preprocess_runner,
        "_build_recording_metadata_phase_payload",
        lambda inputs: {
            "source_h5_path": str(inputs.source_h5_path or inputs.h5_path),
            "resolved_h5_path": str(inputs.h5_path),
            "copied_to_scratch": bool(inputs.copied_to_scratch),
            "recording_info": {
                "sampling_frequency_hz": 10000.0,
                "num_channels": 4,
            },
        },
    )

    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        source_h5_path=tmp_path / "input.raw.h5",
    )

    payload = run_preprocess_save_rec_metadata_phase(inputs)

    assert payload["phase"] == "save_rec_metadata"
    assert Path(str(payload["summary_json"])).exists()
    assert payload["recording_info"]["sampling_frequency_hz"] == 10000.0
    assert payload["outputs"]["resolved_h5_path"].endswith("input.raw.h5")


def test_run_preprocess_save_rec_metadata_phase_writes_configured_metadata_artifacts(tmp_path: Path, monkeypatch) -> None:
    h5py = pytest.importorskip("h5py")
    import numpy as np

    from axon_recon.pipeline.stages.preprocess import runner as preprocess_runner

    well_out_dir, _fake_log = _install_success_fakes(monkeypatch, tmp_path)

    h5_path = tmp_path / "input.raw.h5"
    with h5py.File(str(h5_path), "w") as h5:
        wells = h5.create_group("wells")
        well = wells.create_group("well001")

        rec0 = well.create_group("rec0000")
        rec0.create_dataset("start_time", data=np.asarray([1_700_000_000_000], dtype=np.int64))
        rec0.create_dataset("stop_time", data=np.asarray([1_700_000_001_000], dtype=np.int64))
        routed0 = rec0.create_group("groups").create_group("routed")
        routed0.create_dataset("frame_nos", data=np.asarray([0, 1, 2, 10, 11], dtype=np.int64))

        rec1 = well.create_group("rec0001")
        rec1.create_dataset("start_time", data=np.asarray([1_700_000_010_000], dtype=np.int64))
        rec1.create_dataset("stop_time", data=np.asarray([1_700_000_011_200], dtype=np.int64))
        routed1 = rec1.create_group("groups").create_group("routed")
        routed1.create_dataset("frame_nos", data=np.asarray([20, 21, 22, 23], dtype=np.int64))

        data_store = h5.create_group("data_store")
        data0 = data_store.create_group("data0000")
        data0.create_dataset("well_id", data=np.asarray([1], dtype=np.int32))
        settings0 = data0.create_group("settings")
        settings0.create_dataset("sampling", data=np.asarray([10_000.0], dtype=np.float64))

    monkeypatch.setattr(
        preprocess_runner,
        "_try_get_spikeinterface_recording_info",
        lambda **kwargs: (
            {
                "sampling_frequency_hz": 10_000.0,
                "num_channels": 4,
                "num_segments": 2,
            },
            None,
        ),
    )
    monkeypatch.setattr(
        preprocess_runner,
        "_try_get_spikeinterface_segment_infos",
        lambda **kwargs: (
            [
                {
                    "segment_index": 0,
                    "rec_name": "rec0000",
                    "sampling_frequency_hz": 10_000.0,
                    "num_channels": 4,
                    "num_samples": 5,
                    "source": "spikeinterface",
                },
                {
                    "segment_index": 1,
                    "rec_name": "rec0001",
                    "sampling_frequency_hz": 10_000.0,
                    "num_channels": 4,
                    "num_samples": 4,
                    "source": "spikeinterface",
                },
            ],
            [],
        ),
    )

    inputs = PreprocessInputs(
        h5_path=h5_path,
        stream_id="well001",
        mea_output_root=tmp_path,
        phases=PreprocessPhasesConfig(
            save_rec_metadata=PreprocessSaveRecMetadataPhaseConfig(
                enabled=True,
                verbose=True,
                summary_json_relpath="context/recording_metadata_summary.json",
                segment_epochs_relpath="metadata/segment_epochs.json",
                contiguous_epochs_relpath="metadata/continuous_epochs.json",
                sampling_metadata_relpath="metadata/sampling_rate_metadata.json",
            )
        ),
    )

    payload = run_preprocess_save_rec_metadata_phase(inputs)

    assert payload["phase"] == "save_rec_metadata"
    assert payload["verbose"] is True
    assert payload["segment_count"] == 2
    assert payload["contiguous_epoch_count"] == 3
    assert payload["outputs"]["segment_epochs_json"] == str(well_out_dir / "metadata" / "segment_epochs.json")
    assert payload["outputs"]["contiguous_epochs_json"] == str(well_out_dir / "metadata" / "continuous_epochs.json")
    assert payload["outputs"]["sampling_metadata_json"] == str(well_out_dir / "metadata" / "sampling_rate_metadata.json")
    assert payload["outputs"]["assay_stats_txt"] == str(well_out_dir / "assay_stats_well001.txt")

    segment_epochs_payload = _read_json(well_out_dir / "metadata" / "segment_epochs.json")
    assert segment_epochs_payload["segment_count"] == 2
    assert segment_epochs_payload["segments"][0]["rec_name"] == "rec0000"
    assert segment_epochs_payload["segments"][0]["timestamp_unit"] == "ms_since_epoch"
    assert segment_epochs_payload["segments"][0]["duration_wall_clock_s"] == pytest.approx(1.0)

    contiguous_epochs_payload = _read_json(well_out_dir / "metadata" / "continuous_epochs.json")
    assert contiguous_epochs_payload["contiguous_epoch_count"] == 3
    assert contiguous_epochs_payload["epochs"][0]["rec_name"] == "rec0000"
    assert contiguous_epochs_payload["epochs"][0]["n_samples"] == 3

    sampling_payload = _read_json(well_out_dir / "metadata" / "sampling_rate_metadata.json")
    assert sampling_payload["sampling_summary"]["stream_sampling_frequency_hz"] == pytest.approx(10_000.0)
    assert sampling_payload["sampling_summary"]["all_segments_match"] is True
    assert sampling_payload["segments"][1]["rec_name"] == "rec0001"
    assert sampling_payload["segments"][1]["sampling_frequency_hz"] == pytest.approx(10_000.0)

    assay_stats_text = (well_out_dir / "assay_stats_well001.txt").read_text(encoding="utf-8")
    assert "assay_stats context" in assay_stats_text
    assert str(h5_path) in assay_stats_text


def test_run_preprocess_stage_build_uses_precreated_save_rec_metadata_assay_stats(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.preprocess import runner as preprocess_runner

    well_out_dir, _fake_log = _install_success_fakes(monkeypatch, tmp_path)

    captured_build_kwargs: dict = {}

    def _fake_recording_metadata_payload(inputs):
        assay_stats_path = well_out_dir / "assay_stats_well001.txt"
        assay_stats_path.parent.mkdir(parents=True, exist_ok=True)
        assay_stats_path.write_text("saved metadata\n", encoding="utf-8")
        return {
            "phase": "save_rec_metadata",
            "source_h5_path": str(inputs.source_h5_path or inputs.h5_path),
            "resolved_h5_path": str(inputs.h5_path),
            "copied_to_scratch": bool(inputs.copied_to_scratch),
            "segment_epochs_json": str(well_out_dir / "segment_epochs.json"),
            "contiguous_epochs_json": str(well_out_dir / "continuous_epochs.json"),
            "sampling_metadata_json": str(well_out_dir / "sampling_rate_metadata.json"),
            "assay_stats_txt": str(assay_stats_path),
        }

    def _fake_run_build_preprocessed_recording_core(**kwargs):
        captured_build_kwargs.update(kwargs)
        assert Path(str(kwargs["saved_assay_stats_path"])).exists()
        return (
            object(),
            [1, 2, 3],
            {
                "rec_names": ["seg000", "seg001"],
                "segment_recordings_raw": [object(), object()],
                "segment_recordings_raw_concat": [object(), object()],
                "segment_recordings_preprocessed": [object(), object()],
                "segment_recordings_preprocessed_concat": [object(), object()],
                "segment_stats": [
                    {"fs": 10_000.0, "n_samples": 100, "n_channels": 4},
                    {"fs": 10_000.0, "n_samples": 120, "n_channels": 4},
                ],
                "phase_timing_s": {"build_total": 1.25},
            },
        )

    monkeypatch.setattr(preprocess_runner, "_build_recording_metadata_phase_payload", _fake_recording_metadata_payload)
    monkeypatch.setattr(
        preprocess_runner,
        "run_build_preprocessed_recording_core",
        _fake_run_build_preprocessed_recording_core,
    )

    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        phases=PreprocessPhasesConfig(
            save_rec_metadata=PreprocessSaveRecMetadataPhaseConfig(
                enabled=True,
            )
        ),
    )

    run_preprocess_stage(inputs)

    assert captured_build_kwargs.get("saved_assay_stats_path") == well_out_dir / "assay_stats_well001.txt"
    assert captured_build_kwargs.get("require_saved_assay_stats") is True


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


def test_run_preprocess_stage_can_disable_subphase_dividers_to_stdout(tmp_path: Path, monkeypatch) -> None:
    captured_build_kwargs: dict = {}
    _install_success_fakes(monkeypatch, tmp_path, captured_build_kwargs=captured_build_kwargs)

    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        logging_subphase_dividers_to_stdout=False,
    )

    run_preprocess_stage(inputs)

    assert captured_build_kwargs.get("emit_phase_dividers_to_stdout") is False
