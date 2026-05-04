from __future__ import annotations

import base64
import contextlib
import json
import logging
import sys
import threading
import types
from pathlib import Path

import pytest

from axon_recon.pipeline.stages.preprocess.models.inputs import (
    PreprocessConcatSegmentsPhaseConfig,
    PreprocessInputs,
    PreprocessPhasesConfig,
    PreprocessPlotRasterThresholdPhaseConfig,
    PreprocessPlotConcatTracesPhaseConfig,
    PreprocessPlotSegmentChannelLayoutsPhaseConfig,
    PreprocessPlotSegmentTracesPhaseConfig,
    PreprocessPrepareRawBinariesPhaseConfig,
    PreprocessSaveRecMetadataPhaseConfig,
    PreprocessSegmentsPhaseConfig,
    PreprocessWipeSrcScratchPhaseConfig,
)
from axon_recon.pipeline.stages.preprocess.runner import (
    _run_preprocess_selected_phase,
    run_preprocess_concat_segments_phase,
    run_preprocess_plot_raster_threshold_phase,
    run_preprocess_plot_segment_channel_layouts_phase,
    run_preprocess_prepare_raw_binaries_phase,
    run_preprocess_save_rec_metadata_phase,
    run_preprocess_stage,
    run_preprocess_wipe_src_scratch_phase,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_test_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAF/gL+2m30GQAAAABJRU5ErkJggg=="
        )
    )


def _full_stage_phases() -> PreprocessPhasesConfig:
    return PreprocessPhasesConfig(
        prepare_raw_binaries=PreprocessPrepareRawBinariesPhaseConfig(enabled=True),
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
        import numpy as np

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
                    "segment_count": 2,
                    "segments": [
                        {"rec_name": "seg000", "sampling_frequency_hz": 10_000.0},
                        {"rec_name": "seg001", "sampling_frequency_hz": 10_000.0},
                    ],
                }
            ),
            encoding="utf-8",
        )
        assay_stats_path.write_text("assay stats\n", encoding="utf-8")
        np.save(common_electrodes_path, np.asarray(common, dtype=np.int64))
        return {
            "phase": "save_rec_metadata",
            "source_h5_path": str(kwargs["source_h5_path"]),
            "resolved_h5_path": str(kwargs["h5_path"]),
            "requested_metadata_source": str(kwargs.get("requested_metadata_source", "source_h5")),
            "metadata_source": str(kwargs.get("metadata_source", "source_h5")),
            "segment_count": 2,
            "contiguous_epoch_count": 2,
            "recording_info": {"sampling_frequency_hz": 10_000.0, "num_channels": 4},
            "common_electrode_count": len(common),
            "common_electrodes_preview": list(common),
            "segment_epochs_json": str(segment_epochs_path),
            "contiguous_epochs_json": str(contiguous_epochs_path),
            "sampling_metadata_json": str(sampling_metadata_path),
            "assay_stats_txt": str(assay_stats_path),
            "common_electrodes_path": str(common_electrodes_path),
            "verbose": bool(kwargs["verbose"]),
        }

    def _fake_run_prepare_raw_binaries_core(**kwargs):
        _capture("prepare_raw_binaries", kwargs)
        recording_dir = Path(str(kwargs["recording_dir"]))
        manifest_path = Path(str(kwargs["manifest_path"]))
        recording_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        (recording_dir / "recording.marker").write_text("ok\n", encoding="utf-8")
        manifest_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "stream_id": str(kwargs["stream_id"]),
                    "recording_dir": str(recording_dir),
                    "segment_count": 2,
                    "num_channels": 4,
                    "sampling_frequency_hz": 10_000.0,
                    "num_frames_by_segment": [100, 100],
                }
            ),
            encoding="utf-8",
        )
        return {
            "phase": "prepare_raw_binaries",
            "recording_dir": str(recording_dir),
            "manifest_path": str(manifest_path),
            "raw_binary_recording_dir": str(recording_dir),
            "raw_binary_manifest_path": str(manifest_path),
            "segment_count": 2,
            "num_channels": 4,
            "sampling_frequency_hz": 10_000.0,
            "num_frames_by_segment": [100, 100],
        }

    def _fake_run_preprocess_segments_core(**kwargs):
        _capture("preprocess_segments", kwargs)
        output_dir = Path(str(kwargs["output_dir"]))
        manifest_path = Path(str(kwargs["manifest_path"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        segment_entries = []
        output_mode = str(kwargs.get("output_mode", "binary"))
        for segment_index, rec_name in enumerate(("seg000", "seg001")):
            entry = {
                "segment_index": segment_index,
                "rec_name": rec_name,
                "fs_hz": 10_000.0,
                "n_samples": 100,
                "n_channels": 4,
            }
            if output_mode == "lazy":
                provenance_path = output_dir / f"{segment_index:03d}_{rec_name}.json"
                provenance_path.write_text(json.dumps({"path": str(provenance_path)}), encoding="utf-8")
                entry["provenance_path"] = str(provenance_path)
            else:
                seg_dir = output_dir / f"{segment_index:03d}_{rec_name}"
                seg_dir.mkdir(parents=True, exist_ok=True)
                (seg_dir / "recording.marker").write_text("ok\n", encoding="utf-8")
                entry["folder"] = str(seg_dir)
            segment_entries.append(entry)
        manifest_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "output_mode": output_mode,
                    "segments": segment_entries,
                }
            ),
            encoding="utf-8",
        )
        return {
            "phase": "preprocess_segments",
            "output_mode": output_mode,
            "segment_count": 2,
            "rec_names": ["seg000", "seg001"],
            "manifest_path": str(manifest_path),
            "output_dir": str(output_dir),
            "phase_timing_s": {"preprocess_segments": 0.2},
        }

    def _fake_run_plot_segment_traces_core(**kwargs):
        capture_name = (
            "plot_segment_channel_layouts"
            if bool(kwargs["plot_layouts"]) and not bool(kwargs["plot_segment_traces"])
            else "plot_segment_traces"
        )
        _capture(capture_name, kwargs)
        plot_output_dir = Path(str(kwargs["plot_output_dir"]))
        plot_output_dir.mkdir(parents=True, exist_ok=True)
        layout_plot_paths: list[str] = []
        if bool(kwargs["plot_layouts"]):
            layout_path = plot_output_dir / str(kwargs["channel_layouts_subdir"]) / f"common_channel_layout_{kwargs['stream_id']}.png"
            _write_test_png(layout_path)
            layout_plot_paths.append(str(layout_path))
        segment_trace_paths: list[str] = []
        if bool(kwargs["plot_segment_traces"]):
            for rec_name in ("seg000", "seg001"):
                trace_path = plot_output_dir / str(kwargs["segment_traces_subdir"]) / f"segment_trace_{kwargs['stream_id']}_{rec_name}.png"
                _write_test_png(trace_path)
                segment_trace_paths.append(str(trace_path))
        return {
            "phase": "plot_segment_traces",
            "layout_plot_paths": layout_plot_paths,
            "segment_trace_paths": segment_trace_paths,
            "segment_count": 2,
        }

    def _fake_run_plot_concat_channel_layout_core(**kwargs):
        _capture("plot_concat_channel_layout", kwargs)
        plot_output_dir = Path(str(kwargs["plot_output_dir"]))
        plot_output_dir.mkdir(parents=True, exist_ok=True)
        layout_path = plot_output_dir / str(kwargs["channel_layouts_subdir"]) / f"concat_channel_layout_{kwargs['stream_id']}.png"
        _write_test_png(layout_path)
        return {
            "phase": "plot_concat_channel_layout",
            "layout_plot_paths": [str(layout_path)],
            "representative_channel_count": 3,
            "representative_channel_ids": [11, 22, 33],
        }

    def _fake_run_plot_raster_threshold_core(**kwargs):
        _capture("plot_raster_threshold", kwargs)
        raster_output_dir = Path(str(kwargs["raster_output_dir"]))
        raster_output_dir.mkdir(parents=True, exist_ok=True)
        raster_plot_path = raster_output_dir / f"threshold_raster_{kwargs['stream_id']}.png"
        _write_test_png(raster_plot_path)
        return {
            "phase": "plot_raster_threshold",
            "segment_count": 2,
            "electrode_count": 3,
            "electrode_ids": [11, 22, 33],
            "raster_output_dir": str(raster_output_dir),
            "raster_plot_path": str(raster_plot_path),
            "total_event_count": 7,
        }

    def _fake_run_concat_segments_core(**kwargs):
        _capture("concat_segments", kwargs)
        recording_dir = Path(str(kwargs["recording_dir"]))
        concat_manifest_path = Path(str(kwargs["concat_manifest_path"]))
        recording_dir.mkdir(parents=True, exist_ok=True)
        concat_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        (recording_dir / "recording.marker").write_text("ok\n", encoding="utf-8")
        concat_manifest_path.write_text(
            json.dumps(
                {
                    "segment_count": 2,
                    "segment_source": "preprocessed",
                    "segment_entries": [
                        {"segment_index": 0, "rec_name": "seg000", "folder": str(Path(str(kwargs["segment_manifest_path"])).parent / "000_seg000")},
                        {"segment_index": 1, "rec_name": "seg001", "folder": str(Path(str(kwargs["segment_manifest_path"])).parent / "001_seg001")},
                    ],
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
            "stitch_frame_count": 1,
        }

    def _fake_run_plot_concat_traces_core(**kwargs):
        _capture("plot_concat_traces", kwargs)
        plot_output_dir = Path(str(kwargs["plot_output_dir"]))
        plot_output_dir.mkdir(parents=True, exist_ok=True)
        trace_plot_path = plot_output_dir / str(kwargs["concat_trace_relpath"])
        _write_test_png(trace_plot_path)
        return {
            "phase": "plot_concat_traces",
            "trace_plot_path": str(trace_plot_path),
            "segment_count": 2,
            "stitch_frame_count": 1,
        }

    monkeypatch.setattr(preprocess_runner, "compute_mea_analysis_output_dir", _fake_compute_mea_analysis_output_dir)
    monkeypatch.setattr(preprocess_runner, "compute_pipeline_log_file", _fake_compute_pipeline_log_file)
    monkeypatch.setattr(preprocess_runner, "setup_pipeline_logger", _fake_setup_pipeline_logger)
    monkeypatch.setattr(preprocess_runner, "run_save_rec_metadata_core", _fake_run_save_rec_metadata_core)
    monkeypatch.setattr(preprocess_runner, "run_prepare_raw_binaries_core", _fake_run_prepare_raw_binaries_core)
    monkeypatch.setattr(preprocess_runner, "run_preprocess_segments_core", _fake_run_preprocess_segments_core)
    monkeypatch.setattr(preprocess_runner, "run_plot_segment_traces_core", _fake_run_plot_segment_traces_core)
    monkeypatch.setattr(preprocess_runner, "run_concat_segments_core", _fake_run_concat_segments_core)
    monkeypatch.setattr(preprocess_runner, "run_plot_concat_traces_core", _fake_run_plot_concat_traces_core)
    monkeypatch.setattr(preprocess_runner, "run_plot_concat_channel_layout_core", _fake_run_plot_concat_channel_layout_core)
    monkeypatch.setattr(preprocess_runner, "run_plot_raster_threshold_core", _fake_run_plot_raster_threshold_core)
    return well_out_dir, fake_log


def test_run_preprocess_stage_writes_observability_artifacts(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.preprocess import runner as preprocess_runner

    well_out_dir, fake_log = _install_success_fakes(monkeypatch, tmp_path)
    canonical_out_dir = well_out_dir / "preprocess_outputs"
    monkeypatch.setattr(
        preprocess_runner.getpass,
        "getuser",
        lambda: (_ for _ in ()).throw(KeyError("missing uid")),
    )
    monkeypatch.setenv("USER", "container-user")

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

    assert summary["n_common_electrodes"] == 3
    outputs = dict(summary.get("outputs", {}))
    assert outputs["pipeline_log"] == str(fake_log)
    assert str(fake_log).startswith(str(canonical_out_dir))
    assert "prepare_raw_binaries_summary_json" in outputs
    assert "preprocess_segments_summary_json" in outputs
    assert "plot_segment_traces_summary_json" in outputs
    assert "plot_segment_channel_layouts_summary_json" in outputs
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
    assert run_manifest["common_electrodes"]["count"] == 3

    environment = _read_json(Path(outputs["observability.environment_json"]))
    assert environment["user"] == "container-user"

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
        logging_suppress_h5_plugin_messages=True,
        debug_limit_segments_per_well=2,
        n_representative_channels=9,
        concat_trace_n_reps=3,
        segment_trace_n_reps=6,
        plot_n_jobs=3,
        trace_max_points=-1,
    )

    result = run_preprocess_stage(inputs)
    summary = _read_json(result.summary_json)
    preprocess_phase_summary = _read_json(Path(str(summary["phase_summaries"]["preprocess_segments"])))

    assert captured_phase_kwargs["preprocess_segments"]["limit_segments_per_well"] == 2
    assert preprocess_phase_summary["applied_debug_limits"]["limit_segments_per_well"] == 2
    assert captured_phase_kwargs["preprocess_segments"]["logger_is_none"] is True
    assert captured_phase_kwargs["preprocess_segments"]["suppress_h5_plugin_messages"] is True
    assert Path(str(captured_phase_kwargs["save_rec_metadata"]["segment_epochs_path"])) == canonical_out_dir / "segment_epochs.json"
    assert Path(str(captured_phase_kwargs["save_rec_metadata"]["assay_stats_path"])) == canonical_out_dir / "assay_stats_well001.txt"
    assert Path(str(captured_phase_kwargs["prepare_raw_binaries"]["recording_dir"])) == canonical_out_dir / "raw_binary_recording"
    assert Path(str(captured_phase_kwargs["prepare_raw_binaries"]["manifest_path"])) == canonical_out_dir / "context" / "raw_binary_manifest.json"
    assert captured_phase_kwargs["prepare_raw_binaries"]["suppress_h5_plugin_messages"] is True
    assert Path(str(captured_phase_kwargs["plot_segment_traces"]["plot_output_dir"])) == canonical_out_dir
    assert Path(str(captured_phase_kwargs["plot_segment_channel_layouts"]["plot_output_dir"])) == canonical_out_dir
    assert Path(str(captured_phase_kwargs["plot_concat_traces"]["plot_output_dir"])) == canonical_out_dir
    assert captured_phase_kwargs["plot_segment_traces"]["plot_layouts"] is False
    assert captured_phase_kwargs["plot_segment_channel_layouts"]["plot_layouts"] is True
    assert captured_phase_kwargs["plot_segment_channel_layouts"]["plot_segment_traces"] is False
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
    assert summary.get("inputs", {}).get("logging_suppress_h5_plugin_messages") is True
    assert summary.get("inputs", {}).get("n_representative_channels") == 9
    assert summary.get("inputs", {}).get("concat_trace_n_reps") == 3
    assert summary.get("inputs", {}).get("segment_trace_n_reps") == 6
    assert summary.get("inputs", {}).get("plot_n_jobs") == 3


def test_run_preprocess_stage_uses_source_h5_for_preprocess_segments_when_lazy_source_src(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_phase_kwargs: dict[str, dict] = {}
    _install_success_fakes(monkeypatch, tmp_path, captured_phase_kwargs=captured_phase_kwargs)

    source_h5_path = tmp_path / "source" / "input.raw.h5"
    source_h5_path.parent.mkdir(parents=True, exist_ok=True)
    source_h5_path.write_text("source\n", encoding="utf-8")
    scratch_h5_path = tmp_path / "scratch" / "input.raw.h5"
    scratch_h5_path.parent.mkdir(parents=True, exist_ok=True)
    scratch_h5_path.write_text("scratch\n", encoding="utf-8")

    inputs = PreprocessInputs(
        h5_path=scratch_h5_path,
        source_h5_path=source_h5_path,
        copied_to_scratch=True,
        stream_id="well001",
        mea_output_root=tmp_path,
        phases=PreprocessPhasesConfig(
            prepare_raw_binaries=PreprocessPrepareRawBinariesPhaseConfig(enabled=False),
            save_rec_metadata=PreprocessSaveRecMetadataPhaseConfig(enabled=True),
            preprocess_segments=PreprocessSegmentsPhaseConfig(enabled=True, lazy_source="src"),
        ),
    )

    result = run_preprocess_stage(inputs)
    summary = _read_json(result.summary_json)

    assert Path(str(captured_phase_kwargs["preprocess_segments"]["h5_path"])) == source_h5_path
    assert summary.get("inputs", {}).get("preprocess_segments_lazy_source") == "src"


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


def test_run_preprocess_save_rec_metadata_phase_prefers_requested_metadata_source_when_available(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_phase_kwargs: dict[str, dict] = {}
    _install_success_fakes(monkeypatch, tmp_path, captured_phase_kwargs=captured_phase_kwargs)

    source_h5_path = tmp_path / "source" / "input.raw.h5"
    source_h5_path.parent.mkdir(parents=True, exist_ok=True)
    source_h5_path.write_text("source\n", encoding="utf-8")
    scratch_h5_path = tmp_path / "scratch" / "input.raw.h5"
    scratch_h5_path.parent.mkdir(parents=True, exist_ok=True)
    scratch_h5_path.write_text("scratch\n", encoding="utf-8")

    inputs = PreprocessInputs(
        h5_path=scratch_h5_path,
        source_h5_path=source_h5_path,
        copied_to_scratch=True,
        stream_id="well001",
        mea_output_root=tmp_path,
        phases=PreprocessPhasesConfig(
            save_rec_metadata=PreprocessSaveRecMetadataPhaseConfig(
                enabled=True,
                metadata_source="scratch_copy",
            )
        ),
    )

    payload = run_preprocess_save_rec_metadata_phase(inputs)

    assert payload["phase"] == "save_rec_metadata"
    assert Path(str(captured_phase_kwargs["save_rec_metadata"]["h5_path"])) == scratch_h5_path
    assert Path(str(captured_phase_kwargs["save_rec_metadata"]["source_h5_path"])) == source_h5_path
    assert captured_phase_kwargs["save_rec_metadata"]["requested_metadata_source"] == "scratch_copy"
    assert captured_phase_kwargs["save_rec_metadata"]["metadata_source"] == "scratch_copy"


def test_run_preprocess_save_rec_metadata_phase_passes_step_timer_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_phase_kwargs: dict[str, dict] = {}
    _install_success_fakes(monkeypatch, tmp_path, captured_phase_kwargs=captured_phase_kwargs)

    source_h5_path = tmp_path / "source" / "input.raw.h5"
    source_h5_path.parent.mkdir(parents=True, exist_ok=True)
    source_h5_path.write_text("source\n", encoding="utf-8")

    inputs = PreprocessInputs(
        h5_path=source_h5_path,
        source_h5_path=source_h5_path,
        copied_to_scratch=False,
        stream_id="well001",
        mea_output_root=tmp_path,
        phases=PreprocessPhasesConfig(
            save_rec_metadata=PreprocessSaveRecMetadataPhaseConfig(
                enabled=True,
                report_step_timers=True,
            )
        ),
    )

    run_preprocess_save_rec_metadata_phase(inputs)

    assert captured_phase_kwargs["save_rec_metadata"]["report_step_timers"] is True


def test_run_preprocess_save_rec_metadata_phase_falls_back_to_source_when_scratch_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_phase_kwargs: dict[str, dict] = {}
    _install_success_fakes(monkeypatch, tmp_path, captured_phase_kwargs=captured_phase_kwargs)

    source_h5_path = tmp_path / "source" / "input.raw.h5"
    source_h5_path.parent.mkdir(parents=True, exist_ok=True)
    source_h5_path.write_text("source\n", encoding="utf-8")
    scratch_h5_path = tmp_path / "scratch" / "input.raw.h5"

    inputs = PreprocessInputs(
        h5_path=scratch_h5_path,
        source_h5_path=source_h5_path,
        copied_to_scratch=True,
        stream_id="well001",
        mea_output_root=tmp_path,
        phases=PreprocessPhasesConfig(
            save_rec_metadata=PreprocessSaveRecMetadataPhaseConfig(
                enabled=True,
                metadata_source="scratch_copy",
            )
        ),
    )

    payload = run_preprocess_save_rec_metadata_phase(inputs)

    assert payload["phase"] == "save_rec_metadata"
    assert Path(str(captured_phase_kwargs["save_rec_metadata"]["h5_path"])) == source_h5_path
    assert Path(str(captured_phase_kwargs["save_rec_metadata"]["source_h5_path"])) == source_h5_path
    assert captured_phase_kwargs["save_rec_metadata"]["requested_metadata_source"] == "scratch_copy"
    assert captured_phase_kwargs["save_rec_metadata"]["metadata_source"] == "source_h5"


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
        "prepare_raw_binaries",
        "preprocess_segments",
        "plot_segment_traces",
        "plot_segment_channel_layouts",
        "concat_segments",
        "plot_concat_traces",
    ]
    assert summary.get("inputs", {}).get("force_restart") is True
    assert outputs["pipeline_log"] == str(canonical_out_dir / "logs" / "preprocess_pipeline.log")
    assert Path(outputs["pipeline_log"]).exists()


def test_run_preprocess_stage_uses_configured_phase_sequence_and_summarizes_disabled_phases_as_skipped(
    tmp_path: Path,
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    phase_call_order: list[str] = []
    _install_success_fakes(
        monkeypatch,
        tmp_path,
        phase_call_order=phase_call_order,
    )

    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        phases=_full_stage_phases(),
        phase_sequence=("save_rec_metadata", "plot_raster_threshold", "preprocess_segments"),
    )

    with caplog.at_level(logging.INFO):
        result = run_preprocess_stage(inputs)
    summary = _read_json(result.summary_json)
    phase_summary_paths = {str(name): Path(str(path)) for name, path in dict(summary.get("phase_summaries", {})).items()}
    skipped_summary = _read_json(phase_summary_paths["plot_raster_threshold"])

    assert phase_call_order == ["save_rec_metadata", "preprocess_segments"]
    assert set(phase_summary_paths) == {"save_rec_metadata", "plot_raster_threshold", "preprocess_segments"}
    assert summary.get("phase_statuses", {}) == {
        "save_rec_metadata": "success",
        "plot_raster_threshold": "skipped",
        "preprocess_segments": "success",
    }
    assert skipped_summary["phase"] == "plot_raster_threshold"
    assert skipped_summary["status"] == "skipped"
    assert skipped_summary["enabled"] is False
    assert skipped_summary["skip_reason"] == "phase disabled in preprocess config"
    assert any(getattr(record, "event", None) == "phase_skipped" for record in caplog.records)


def test_run_preprocess_selected_phase_accepts_stage_qualified_phase_name(
    tmp_path: Path,
    monkeypatch,
) -> None:
    phase_call_order: list[str] = []
    _install_success_fakes(
        monkeypatch,
        tmp_path,
        phase_call_order=phase_call_order,
    )

    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        phases=_full_stage_phases(),
    )

    payload = _run_preprocess_selected_phase(inputs, selected_phase="preprocess.save_rec_metadata")

    assert phase_call_order == ["save_rec_metadata"]
    assert payload["phase"] == "save_rec_metadata"
    assert payload["status"] == "success"


def test_run_preprocess_stage_logs_phase_start_per_well(tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture) -> None:
    _install_success_fakes(monkeypatch, tmp_path)

    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        logging_enabled=True,
        logging_verbose=True,
        n_jobs=12,
        runtime_stage_workers=24,
        runtime_well_workers=2,
        runtime_n_jobs_source="derived",
        phases=_full_stage_phases(),
    )

    with caplog.at_level(logging.INFO):
        run_preprocess_stage(inputs)

    messages = [record.getMessage() for record in caplog.records]
    assert any("Starting preprocess work for well=well001 phase_count=7 selected_phase=all" in message for message in messages)
    assert any(
        "Preprocess phase worker allocation stage=preprocess phase=preprocess_segments well=well001 stage_workers=24 well_workers=2 n_jobs=12 n_jobs_source=derived phase_n_jobs=12"
        in message
        for message in messages
    )
    assert any("Starting preprocess phase 1/7 for well=well001 phase=save_rec_metadata" in message for message in messages)
    assert any("Starting preprocess phase 6/7 for well=well001 phase=concat_segments" in message for message in messages)
    assert not any("unit_workers" in message for message in messages if "worker allocation" in message)


def test_run_preprocess_stage_logs_phase_resource_usage_and_resource_class(
    tmp_path: Path,
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from dataclasses import replace

    from axon_recon.pipeline.logging import configure_pipeline_logging, finalize_pipeline_logging
    import axon_recon.pipeline.resource_usage as resource_usage
    from axon_recon.pipeline.stages.preprocess import runner as preprocess_runner

    class _FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = int(pid)

        def children(self, recursive: bool = True):
            _ = recursive
            return []

        def memory_info(self):
            return types.SimpleNamespace(rss=1024)

        def num_threads(self) -> int:
            return 53

        def cpu_times(self):
            return types.SimpleNamespace(user=0.0, system=0.0)

    monkeypatch.setattr(
        resource_usage,
        "psutil",
        types.SimpleNamespace(Process=lambda pid: _FakeProcess(int(pid))),
    )

    _install_success_fakes(monkeypatch, tmp_path)

    data_path = tmp_path / "data.yml"
    data_path.write_text(
        f"output_root: {tmp_path / 'outputs'}\n"
        "use_scratch_root: false\n"
        "datasets: []\n",
        encoding="utf-8",
    )
    runtime_path = tmp_path / "runtime.yml"
    runtime_path.write_text(
        f"data: {data_path}\n"
        "logging:\n"
        "  enabled: true\n"
        "  level: INFO\n"
        "  console:\n"
        "    enabled: false\n"
        "    blank_line_after_phase: true\n"
        "  structured:\n"
        "    enabled: false\n"
        "  run_log:\n"
        "    enabled: false\n"
        "  error_log:\n"
        "    enabled: false\n"
        "  summary:\n"
        "    enabled: false\n"
        "  resource_usage:\n"
        "    enabled: true\n"
        "    level: INFO\n"
        "    include_children: true\n"
        "    sample_interval_s: 0.05\n"
        "    include_gpu: false\n"
        "    include_disk_io: false\n"
        "    write_to_phase_summary: true\n",
        encoding="utf-8",
    )
    configure_pipeline_logging(config_path=runtime_path)
    blank_line_calls: list[str] = []
    monkeypatch.setattr(preprocess_runner, "emit_pipeline_console_blank_line", lambda: blank_line_calls.append("blank"))

    phases = _full_stage_phases()
    phases = replace(
        phases,
        save_rec_metadata=replace(phases.save_rec_metadata, resource_class="metadata_io"),
    )
    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        logging_enabled=True,
        logging_verbose=True,
        phase_sequence=("save_rec_metadata",),
        phases=phases,
    )

    try:
        with caplog.at_level(logging.INFO):
            result = run_preprocess_stage(inputs)
    finally:
        finalize_pipeline_logging(status="ok")

    stage_summary = _read_json(result.summary_json)
    phase_summary = _read_json(Path(str(stage_summary["phase_summaries"]["save_rec_metadata"])))
    started_record = next(record for record in caplog.records if getattr(record, "event", None) == "phase_started")
    usage_record = next(record for record in caplog.records if getattr(record, "event", None) == "phase_resource_usage")

    assert getattr(started_record, "resource_class", None) == "metadata_io"
    assert getattr(usage_record, "resource_class", None) == "metadata_io"
    assert phase_summary["resource_class"] == "metadata_io"
    assert phase_summary["resource_usage"]["wall_time_s"] is not None
    assert phase_summary["resource_usage"]["total_peak_rss_gb"] is not None
    assert phase_summary["resource_usage"]["max_threads"] == 1
    assert phase_summary["resource_usage"]["observed_process_max_threads"] == 53
    assert started_record.getMessage().startswith("Starting phase: preprocess.save_rec_metadata")
    assert usage_record.getMessage().startswith("Phase resource usage: preprocess.save_rec_metadata")
    assert getattr(usage_record, "status", None) == "success"
    assert getattr(usage_record, "resource_usage", None)["max_threads"] == 1
    assert getattr(usage_record, "resource_usage", None)["observed_process_max_threads"] == 53
    assert blank_line_calls == ["blank"]


def test_run_preprocess_stage_resumes_complete_phase_artifacts_without_force_restart(
    tmp_path: Path,
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from axon_recon.pipeline.stages.preprocess import runner as preprocess_runner

    _install_success_fakes(monkeypatch, tmp_path)

    def _fake_load_saved_recording(path: Path):
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        return {"path": str(resolved)}

    monkeypatch.setattr(preprocess_runner, "load_saved_recording", _fake_load_saved_recording)

    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        logging_enabled=True,
        logging_verbose=True,
        phases=_full_stage_phases(),
    )

    with caplog.at_level(logging.INFO):
        first_result = run_preprocess_stage(inputs)
    assert first_result.summary_json.exists()
    caplog.clear()

    def _explode(**_kwargs):
        raise AssertionError("phase core should not run when resume artifacts are complete")

    monkeypatch.setattr(preprocess_runner, "run_save_rec_metadata_core", _explode)
    monkeypatch.setattr(preprocess_runner, "run_prepare_raw_binaries_core", _explode)
    monkeypatch.setattr(preprocess_runner, "run_preprocess_segments_core", _explode)
    monkeypatch.setattr(preprocess_runner, "run_plot_segment_traces_core", _explode)
    monkeypatch.setattr(preprocess_runner, "run_concat_segments_core", _explode)
    monkeypatch.setattr(preprocess_runner, "run_plot_concat_traces_core", _explode)

    with caplog.at_level(logging.INFO):
        second_result = run_preprocess_stage(inputs)
    stage_summary = _read_json(second_result.summary_json)
    phase_summary_paths = {str(name): Path(str(path)) for name, path in dict(stage_summary.get("phase_summaries", {})).items()}
    messages = [record.getMessage() for record in caplog.records]

    for phase_name in (
        "save_rec_metadata",
        "prepare_raw_binaries",
        "preprocess_segments",
        "plot_segment_traces",
        "plot_segment_channel_layouts",
        "concat_segments",
        "plot_concat_traces",
    ):
        phase_summary = _read_json(phase_summary_paths[phase_name])
        assert phase_summary["status"] == "skipped"
        assert phase_summary["reused_existing_artifacts"] is True

    assert any(
        "Resuming preprocess phase for well=well001 phase=preprocess_segments; found existing complete artifacts:" in message
        and "manifest_path=" in message
        and "output_dir=" in message
        for message in messages
    )
    assert any(
        "Resuming preprocess phase for well=well001 phase=concat_segments; found existing complete artifacts:" in message
        and "recording_dir=" in message
        and "concat_manifest_path=" in message
        for message in messages
    )


def test_run_preprocess_stage_reruns_phase_when_resume_artifact_is_incomplete(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.preprocess import runner as preprocess_runner

    _install_success_fakes(monkeypatch, tmp_path)

    def _fake_load_saved_recording(path: Path):
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        return {"path": str(resolved)}

    monkeypatch.setattr(preprocess_runner, "load_saved_recording", _fake_load_saved_recording)

    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        phases=_full_stage_phases(),
    )

    first_result = run_preprocess_stage(inputs)
    assert first_result.summary_json.exists()

    first_stage_summary = _read_json(first_result.summary_json)
    plot_segment_summary_path = Path(str(first_stage_summary["phase_summaries"]["plot_segment_traces"]))
    plot_segment_summary = _read_json(plot_segment_summary_path)
    incomplete_plot = Path(str(plot_segment_summary["segment_trace_paths"][0]))
    incomplete_plot.unlink()

    rerun_calls: list[str] = []
    original_plot_segment = preprocess_runner.run_plot_segment_traces_core

    def _tracked_plot_segment_traces_core(**kwargs):
        rerun_calls.append("plot_segment_traces")
        return original_plot_segment(**kwargs)

    def _explode(**_kwargs):
        raise AssertionError("unexpected phase rerun")

    monkeypatch.setattr(preprocess_runner, "run_save_rec_metadata_core", _explode)
    monkeypatch.setattr(preprocess_runner, "run_prepare_raw_binaries_core", _explode)
    monkeypatch.setattr(preprocess_runner, "run_preprocess_segments_core", _explode)
    monkeypatch.setattr(preprocess_runner, "run_plot_segment_traces_core", _tracked_plot_segment_traces_core)
    monkeypatch.setattr(preprocess_runner, "run_concat_segments_core", _explode)
    monkeypatch.setattr(preprocess_runner, "run_plot_concat_traces_core", _explode)

    second_result = run_preprocess_stage(inputs)
    assert rerun_calls == ["plot_segment_traces"]

    phase_summary_path = Path(str(_read_json(second_result.summary_json)["phase_summaries"]["plot_segment_traces"]))
    phase_summary = _read_json(phase_summary_path)
    assert phase_summary.get("reused_existing_artifacts") is not True
    assert incomplete_plot.exists()


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


def test_run_preprocess_segments_core_logs_progress_per_well(monkeypatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from axon_recon.pipeline.stages.preprocess.core import preprocess_segments as preprocess_segments_core

    class _FakeRecording:
        def get_num_channels(self) -> int:
            return 4

    monkeypatch.setattr(
        preprocess_segments_core,
        "load_recording_metadata",
        lambda **_kwargs: (
            {
                "segments": [
                    {"rec_name": "seg000"},
                    {"rec_name": "seg001"},
                    {"rec_name": "seg002"},
                ]
            },
            {"epochs": []},
            {"segments": []},
        ),
    )
    monkeypatch.setattr(preprocess_segments_core, "load_common_electrodes", lambda _path: [11, 22])
    monkeypatch.setattr(
        preprocess_segments_core,
        "_load_centered_segment_with_electrode_channel_ids",
        lambda **kwargs: (_FakeRecording(), {"fs": 10_000.0, "n_samples": 100, "rec_name": kwargs["rec_name"]}),
    )
    monkeypatch.setattr(preprocess_segments_core, "_select_common_electrode_channels", lambda **kwargs: kwargs["recording"])
    monkeypatch.setattr(preprocess_segments_core, "apply_standard_preprocessing", lambda **kwargs: kwargs["recording"])

    saved_payloads: list[dict[str, object]] = []

    def _fake_save(**kwargs):
        saved_payloads.append(dict(kwargs))
        return {"output_dir": str(kwargs["output_dir"]), "manifest_path": str(kwargs["manifest_path"]), "saved": True}

    with caplog.at_level(logging.INFO):
        payload = preprocess_segments_core.run_preprocess_segments_core(
            h5_path=tmp_path / "input.raw.h5",
            source_h5_path=tmp_path / "input.raw.h5",
            stream_id="well001",
            output_mode="binary",
            lazy_source="scratch",
            n_jobs=1,
            segment_epochs_path=tmp_path / "segment_epochs.json",
            contiguous_epochs_path=tmp_path / "contiguous_epochs.json",
            sampling_metadata_path=tmp_path / "sampling_metadata.json",
            common_electrodes_path=tmp_path / "common_electrodes.npy",
            output_dir=tmp_path / "segments",
            manifest_path=tmp_path / "segments_manifest.json",
            overwrite_saved_recording=False,
            save_n_jobs=1,
            chunk_duration="1s",
            progress_bar=False,
            limit_segments_per_well=None,
            logger=logging.getLogger("test.preprocess_segments.progress"),
            run_save_segment_recordings_core=_fake_save,
        )

    messages = [record.getMessage() for record in caplog.records]
    assert payload["segment_count"] == 3
    assert len(saved_payloads) == 1
    assert any(
        "Starting preprocess_segments for well=well001 segment_count=3 common_electrodes=2 workers=1 output_mode=binary"
        in message
        for message in messages
    )
    assert any("preprocess_segments progress well=well001 completed=1/3 rec_name=seg000" in message for message in messages)
    assert any("preprocess_segments progress well=well001 completed=3/3 rec_name=seg002" in message for message in messages)


def test_load_centered_segment_suppresses_maxwell_plugin_output(monkeypatch, tmp_path: Path, capsys) -> None:
    from axon_recon.pipeline.stages.preprocess.core import preprocess_segments as preprocess_segments_core

    class _FakeRecording:
        def __init__(self) -> None:
            self._channel_ids = [11, 22]

        def get_sampling_frequency(self) -> float:
            return 10_000.0

        def get_num_samples(self) -> int:
            return 200

        def get_num_channels(self) -> int:
            return 2

        def get_property(self, name: str):
            assert name == "contact_vector"
            return {"electrode": [11, 22]}

        def rename_channels(self, channel_ids):
            self._channel_ids = list(channel_ids)
            return self

        def get_channel_ids(self):
            return list(self._channel_ids)

    read_kwargs: list[dict[str, object]] = []

    def _fake_read_maxwell(*args, **kwargs):
        _ = args
        read_kwargs.append(dict(kwargs))
        print("The h5 compression library for Maxwell is already located somewhere!")
        return _FakeRecording()

    fake_spikeinterface = types.ModuleType("spikeinterface")
    fake_extractors = types.ModuleType("spikeinterface.extractors")
    fake_extractors.read_maxwell = _fake_read_maxwell
    fake_full = types.ModuleType("spikeinterface.full")
    fake_full.center = lambda recording, chunk_size: recording
    fake_spikeinterface.extractors = fake_extractors
    fake_spikeinterface.full = fake_full
    monkeypatch.setitem(sys.modules, "spikeinterface", fake_spikeinterface)
    monkeypatch.setitem(sys.modules, "spikeinterface.extractors", fake_extractors)
    monkeypatch.setitem(sys.modules, "spikeinterface.full", fake_full)
    monkeypatch.setattr(preprocess_segments_core, "_ensure_maxwell_hdf5_plugin_path", lambda **kwargs: None)

    recording, stats = preprocess_segments_core._load_centered_segment_with_electrode_channel_ids(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        rec_name="rec0000",
        center_chunk_size=10_000,
        suppress_h5_plugin_messages=True,
    )

    captured = capsys.readouterr()
    assert "The h5 compression library for Maxwell" not in captured.out
    assert "The h5 compression library for Maxwell" not in captured.err
    assert read_kwargs[0]["install_maxwell_plugin"] is False
    assert recording.get_channel_ids() == [11, 22]
    assert stats["rec_name"] == "rec0000"


def test_load_centered_segment_avoids_process_stdio_redirect_in_worker_thread(monkeypatch, tmp_path: Path) -> None:
    from axon_recon.pipeline.stages.preprocess.core import preprocess_segments as preprocess_segments_core

    class _FakeRecording:
        def __init__(self) -> None:
            self._channel_ids = [11, 22]

        def get_sampling_frequency(self) -> float:
            return 10_000.0

        def get_num_samples(self) -> int:
            return 200

        def get_num_channels(self) -> int:
            return 2

        def get_property(self, name: str):
            assert name == "contact_vector"
            return {"electrode": [11, 22]}

        def rename_channels(self, channel_ids):
            self._channel_ids = list(channel_ids)
            return self

        def get_channel_ids(self):
            return list(self._channel_ids)

    read_kwargs: list[dict[str, object]] = []
    redirect_calls: list[str] = []

    def _fake_read_maxwell(*args, **kwargs):
        _ = args
        read_kwargs.append(dict(kwargs))
        return _FakeRecording()

    @contextlib.contextmanager
    def _fake_redirect_stdout(_stream):
        redirect_calls.append("stdout")
        yield

    @contextlib.contextmanager
    def _fake_redirect_stderr(_stream):
        redirect_calls.append("stderr")
        yield

    fake_spikeinterface = types.ModuleType("spikeinterface")
    fake_extractors = types.ModuleType("spikeinterface.extractors")
    fake_extractors.read_maxwell = _fake_read_maxwell
    fake_full = types.ModuleType("spikeinterface.full")
    fake_full.center = lambda recording, chunk_size: recording
    fake_spikeinterface.extractors = fake_extractors
    fake_spikeinterface.full = fake_full
    monkeypatch.setitem(sys.modules, "spikeinterface", fake_spikeinterface)
    monkeypatch.setitem(sys.modules, "spikeinterface.extractors", fake_extractors)
    monkeypatch.setitem(sys.modules, "spikeinterface.full", fake_full)
    monkeypatch.setattr(preprocess_segments_core, "_ensure_maxwell_hdf5_plugin_path", lambda **kwargs: None)
    monkeypatch.setattr(preprocess_segments_core.contextlib, "redirect_stdout", _fake_redirect_stdout)
    monkeypatch.setattr(preprocess_segments_core.contextlib, "redirect_stderr", _fake_redirect_stderr)

    result_holder: dict[str, object] = {}

    def _run() -> None:
        recording, stats = preprocess_segments_core._load_centered_segment_with_electrode_channel_ids(
            h5_path=tmp_path / "input.raw.h5",
            stream_id="well001",
            rec_name="rec0000",
            center_chunk_size=10_000,
            suppress_h5_plugin_messages=True,
        )
        result_holder["recording"] = recording
        result_holder["stats"] = stats

    worker = threading.Thread(target=_run, name="test-preprocess-worker")
    worker.start()
    worker.join()

    assert redirect_calls == []
    assert read_kwargs[0]["install_maxwell_plugin"] is False
    assert result_holder["recording"].get_channel_ids() == [11, 22]
    assert result_holder["stats"]["rec_name"] == "rec0000"


def test_run_preprocess_segments_core_lazy_mode_writes_provenance_manifest_without_saving(monkeypatch, tmp_path: Path) -> None:
    from axon_recon.pipeline.stages.preprocess.core import preprocess_segments as preprocess_segments_core

    class _FakeRecording:
        def get_num_channels(self) -> int:
            return 4

        def dump_to_json(self, file_path) -> None:
            Path(file_path).write_text(
                json.dumps(
                    {
                        "class": "fake._FakeRecording",
                        "annotations": {},
                        "properties": {},
                        "kwargs": {},
                        "version": "test",
                    }
                ),
                encoding="utf-8",
            )

    monkeypatch.setattr(
        preprocess_segments_core,
        "load_recording_metadata",
        lambda **_kwargs: (
            {
                "segments": [
                    {"rec_name": "seg000"},
                    {"rec_name": "seg001"},
                ]
            },
            {"epochs": []},
            {"segments": []},
        ),
    )
    monkeypatch.setattr(preprocess_segments_core, "load_common_electrodes", lambda _path: [11, 22])
    monkeypatch.setattr(
        preprocess_segments_core,
        "_load_centered_segment_with_electrode_channel_ids",
        lambda **kwargs: (_FakeRecording(), {"fs": 10_000.0, "n_samples": 100, "rec_name": kwargs["rec_name"]}),
    )
    monkeypatch.setattr(preprocess_segments_core, "_select_common_electrode_channels", lambda **kwargs: kwargs["recording"])
    monkeypatch.setattr(preprocess_segments_core, "apply_standard_preprocessing", lambda **kwargs: kwargs["recording"])

    def _explode_save(**_kwargs):
        raise AssertionError("lazy preprocess output mode should not save segment binaries")

    payload = preprocess_segments_core.run_preprocess_segments_core(
        h5_path=tmp_path / "scratch.raw.h5",
        source_h5_path=tmp_path / "source.raw.h5",
        stream_id="well001",
        output_mode="lazy",
        lazy_source="src",
        n_jobs=1,
        segment_epochs_path=tmp_path / "segment_epochs.json",
        contiguous_epochs_path=tmp_path / "contiguous_epochs.json",
        sampling_metadata_path=tmp_path / "sampling_metadata.json",
        common_electrodes_path=tmp_path / "common_electrodes.npy",
        output_dir=tmp_path / "segments",
        manifest_path=tmp_path / "segments_manifest.json",
        overwrite_saved_recording=True,
        save_n_jobs=1,
        chunk_duration="1s",
        progress_bar=False,
        limit_segments_per_well=None,
        logger=None,
        run_save_segment_recordings_core=_explode_save,
    )

    manifest_payload = _read_json(tmp_path / "segments_manifest.json")

    assert payload["segment_count"] == 2
    assert payload["output_mode"] == "lazy"
    assert payload["saved"] is False
    assert payload["materialized_segments"] is False
    assert manifest_payload["output_mode"] == "lazy"
    assert [str(item["rec_name"]) for item in manifest_payload["segments"]] == ["seg000", "seg001"]
    assert all("folder" not in item for item in manifest_payload["segments"])
    assert all("provenance_path" in item for item in manifest_payload["segments"])
    assert all(Path(str(item["provenance_path"])).is_file() for item in manifest_payload["segments"])


def test_run_save_concatenated_recording_core_lazy_mode_writes_cached_json_without_binary_save(tmp_path: Path) -> None:
    from axon_recon.pipeline.stages.preprocess.core import save_concatenated_recording as save_concat_core

    class _FakeRecording:
        def dump_to_json(self, file_path, relative_to=None) -> None:
            Path(file_path).write_text(
                json.dumps(
                    {
                        "class": "fake._FakeRecording",
                        "annotations": {},
                        "properties": {},
                        "kwargs": {},
                        "version": "test",
                    }
                ),
                encoding="utf-8",
            )

        def save(self, **_kwargs) -> None:
            raise AssertionError("lazy concat output mode should not materialize a binary recording")

    recording_dir = tmp_path / "concatenated_recording"
    payload = save_concat_core.run_save_concatenated_recording_core(
        multirecording=_FakeRecording(),
        recording_dir=recording_dir,
        overwrite_saved_recording=True,
        output_mode="lazy",
        n_jobs=1,
        chunk_duration="1s",
        progress_bar=False,
        logger=None,
    )

    assert payload["saved"] is True
    assert payload["output_mode"] == "lazy"
    assert payload["materialized_recording"] is False
    assert payload["recording_json_path"] == str(recording_dir / "cached.json")
    assert (recording_dir / "cached.json").is_file()


def test_run_concat_segments_core_records_lazy_output_mode(monkeypatch, tmp_path: Path) -> None:
    from axon_recon.pipeline.stages.preprocess.core import concat_segments as concat_segments_core

    class _FakeRecording:
        pass

    fake_spikeinterface = types.ModuleType("spikeinterface")
    fake_spikeinterface_full = types.ModuleType("spikeinterface.full")
    fake_spikeinterface_full.concatenate_recordings = lambda recordings: {"concatenated": len(recordings)}
    fake_spikeinterface.full = fake_spikeinterface_full
    monkeypatch.setitem(sys.modules, "spikeinterface", fake_spikeinterface)
    monkeypatch.setitem(sys.modules, "spikeinterface.full", fake_spikeinterface_full)

    written_manifests: list[tuple[Path, dict[str, object]]] = []
    monkeypatch.setattr(
        concat_segments_core,
        "load_segment_manifest",
        lambda _path: [
            {"segment_index": 0, "rec_name": "seg000", "provenance_path": str(tmp_path / "seg000.json"), "n_samples": 100},
            {"segment_index": 1, "rec_name": "seg001", "provenance_path": str(tmp_path / "seg001.json"), "n_samples": 120},
        ],
    )
    monkeypatch.setattr(concat_segments_core, "load_segment_recording_from_entry", lambda _entry: _FakeRecording())
    monkeypatch.setattr(concat_segments_core, "build_stitch_frames_from_segment_manifest", lambda _entries: [100])
    monkeypatch.setattr(
        concat_segments_core,
        "write_json",
        lambda path, payload: written_manifests.append((Path(path), dict(payload))),
    )

    saved_payloads: list[dict[str, object]] = []

    def _fake_save(**kwargs):
        saved_payloads.append(dict(kwargs))
        return {
            "recording_dir": str(kwargs["recording_dir"]),
            "recording_json_path": str(Path(str(kwargs["recording_dir"])) / "cached.json"),
            "output_mode": str(kwargs["output_mode"]),
            "materialized_recording": False,
            "saved": True,
        }

    payload = concat_segments_core.run_concat_segments_core(
        stream_id="well001",
        segment_manifest_path=tmp_path / "segments_manifest.json",
        recording_dir=tmp_path / "concatenated_recording",
        concat_manifest_path=tmp_path / "concat_manifest.json",
        overwrite_saved_recording=True,
        output_mode="lazy",
        n_jobs=1,
        chunk_duration="1s",
        progress_bar=False,
        logger=None,
        run_save_concatenated_recording_core=_fake_save,
    )

    assert payload["output_mode"] == "lazy"
    assert payload["materialized_recording"] is False
    assert len(saved_payloads) == 1
    assert saved_payloads[0]["output_mode"] == "lazy"
    assert len(written_manifests) == 1
    assert written_manifests[0][1]["output_mode"] == "lazy"


def test_run_concat_segments_core_limits_segments_before_concatenating(monkeypatch, tmp_path: Path) -> None:
    from axon_recon.pipeline.stages.preprocess.core import concat_segments as concat_segments_core

    class _FakeRecording:
        def __init__(self, name: str) -> None:
            self.name = name

    fake_spikeinterface = types.ModuleType("spikeinterface")
    fake_spikeinterface_full = types.ModuleType("spikeinterface.full")

    def _fake_concatenate(recordings):
        return {"concatenated": [recording.name for recording in recordings]}

    fake_spikeinterface_full.concatenate_recordings = _fake_concatenate
    fake_spikeinterface.full = fake_spikeinterface_full
    monkeypatch.setitem(sys.modules, "spikeinterface", fake_spikeinterface)
    monkeypatch.setitem(sys.modules, "spikeinterface.full", fake_spikeinterface_full)

    manifest_entries = [
        {"segment_index": 0, "rec_name": "seg000", "provenance_path": str(tmp_path / "seg000.json"), "n_samples": 100},
        {"segment_index": 1, "rec_name": "seg001", "provenance_path": str(tmp_path / "seg001.json"), "n_samples": 120},
        {"segment_index": 2, "rec_name": "seg002", "provenance_path": str(tmp_path / "seg002.json"), "n_samples": 140},
    ]
    monkeypatch.setattr(concat_segments_core, "load_segment_manifest", lambda _path: list(manifest_entries))
    monkeypatch.setattr(
        concat_segments_core,
        "load_segment_recording_from_entry",
        lambda entry: _FakeRecording(str(entry["rec_name"])),
    )

    stitch_inputs: list[list[str]] = []

    def _fake_stitch_frames(entries):
        stitch_inputs.append([str(item["rec_name"]) for item in entries])
        return [100, 220]

    monkeypatch.setattr(concat_segments_core, "build_stitch_frames_from_segment_manifest", _fake_stitch_frames)
    written_manifests: list[tuple[Path, dict[str, object]]] = []
    monkeypatch.setattr(
        concat_segments_core,
        "write_json",
        lambda path, payload: written_manifests.append((Path(path), dict(payload))),
    )

    saved_payloads: list[dict[str, object]] = []

    def _fake_save(**kwargs):
        saved_payloads.append(dict(kwargs))
        return {
            "recording_dir": str(kwargs["recording_dir"]),
            "output_mode": str(kwargs["output_mode"]),
            "materialized_recording": True,
            "saved": True,
        }

    payload = concat_segments_core.run_concat_segments_core(
        stream_id="well001",
        segment_manifest_path=tmp_path / "segments_manifest.json",
        recording_dir=tmp_path / "concatenated_recording",
        concat_manifest_path=tmp_path / "concat_manifest.json",
        overwrite_saved_recording=True,
        output_mode="binary",
        n_jobs=1,
        chunk_duration="1s",
        progress_bar=False,
        logger=None,
        run_save_concatenated_recording_core=_fake_save,
        limit_segments_per_well=2,
    )

    assert payload["segment_count"] == 2
    assert payload["source_segment_count"] == 3
    assert payload["limit_segments_per_well"] == 2
    assert stitch_inputs == [["seg000", "seg001"]]
    assert saved_payloads[0]["multirecording"] == {"concatenated": ["seg000", "seg001"]}
    assert written_manifests[0][1]["segment_count"] == 2
    assert written_manifests[0][1]["source_segment_count"] == 3
    assert [item["rec_name"] for item in written_manifests[0][1]["segment_entries"]] == ["seg000", "seg001"]


def test_run_preprocess_stage_uses_lazy_output_mode_for_preprocess_segments_when_no_downstream_consumers_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_phase_kwargs: dict[str, dict] = {}
    _install_success_fakes(monkeypatch, tmp_path, captured_phase_kwargs=captured_phase_kwargs)

    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        phases=PreprocessPhasesConfig(
            prepare_raw_binaries=PreprocessPrepareRawBinariesPhaseConfig(enabled=False),
            save_rec_metadata=PreprocessSaveRecMetadataPhaseConfig(enabled=True),
            preprocess_segments=PreprocessSegmentsPhaseConfig(enabled=True, output_mode="lazy"),
            plot_segment_traces=PreprocessPlotSegmentTracesPhaseConfig(enabled=False),
            plot_segment_channel_layouts=PreprocessPlotSegmentChannelLayoutsPhaseConfig(enabled=False),
            concat_segments=PreprocessConcatSegmentsPhaseConfig(enabled=False),
        ),
    )

    run_preprocess_stage(inputs)

    assert captured_phase_kwargs["preprocess_segments"]["output_mode"] == "lazy"


def test_run_preprocess_stage_keeps_lazy_preprocess_segments_when_downstream_consumers_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_phase_kwargs: dict[str, dict] = {}
    _install_success_fakes(monkeypatch, tmp_path, captured_phase_kwargs=captured_phase_kwargs)

    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        phases=PreprocessPhasesConfig(
            prepare_raw_binaries=PreprocessPrepareRawBinariesPhaseConfig(enabled=False),
            save_rec_metadata=PreprocessSaveRecMetadataPhaseConfig(enabled=True),
            preprocess_segments=PreprocessSegmentsPhaseConfig(enabled=True, output_mode="lazy"),
            plot_segment_traces=PreprocessPlotSegmentTracesPhaseConfig(enabled=True),
            plot_segment_channel_layouts=PreprocessPlotSegmentChannelLayoutsPhaseConfig(enabled=False),
            concat_segments=PreprocessConcatSegmentsPhaseConfig(enabled=False),
        ),
    )

    run_preprocess_stage(inputs)

    assert captured_phase_kwargs["preprocess_segments"]["output_mode"] == "lazy"


def test_run_preprocess_stage_resumes_lazy_preprocess_segments_artifacts_without_force_restart(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from axon_recon.pipeline.stages.preprocess import runner as preprocess_runner

    _install_success_fakes(monkeypatch, tmp_path)

    def _fake_load_saved_recording(path: Path):
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        return {"path": str(resolved)}

    monkeypatch.setattr(preprocess_runner, "load_saved_recording", _fake_load_saved_recording)

    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        phases=PreprocessPhasesConfig(
            prepare_raw_binaries=PreprocessPrepareRawBinariesPhaseConfig(enabled=False),
            save_rec_metadata=PreprocessSaveRecMetadataPhaseConfig(enabled=True),
            preprocess_segments=PreprocessSegmentsPhaseConfig(enabled=True, output_mode="lazy"),
            plot_segment_traces=PreprocessPlotSegmentTracesPhaseConfig(enabled=False),
            plot_segment_channel_layouts=PreprocessPlotSegmentChannelLayoutsPhaseConfig(enabled=False),
            concat_segments=PreprocessConcatSegmentsPhaseConfig(enabled=False),
            plot_concat_traces=PreprocessPlotConcatTracesPhaseConfig(enabled=False),
        ),
    )

    first_result = run_preprocess_stage(inputs)
    assert first_result.summary_json.exists()

    def _explode(**_kwargs):
        raise AssertionError("phase core should not run when lazy preprocess artifacts are complete")

    monkeypatch.setattr(preprocess_runner, "run_save_rec_metadata_core", _explode)
    monkeypatch.setattr(preprocess_runner, "run_preprocess_segments_core", _explode)

    second_result = run_preprocess_stage(inputs)
    stage_summary = _read_json(second_result.summary_json)
    phase_summary_paths = {str(name): Path(str(path)) for name, path in dict(stage_summary.get("phase_summaries", {})).items()}
    preprocess_phase_summary = _read_json(phase_summary_paths["preprocess_segments"])

    assert preprocess_phase_summary["status"] == "skipped"
    assert preprocess_phase_summary["reused_existing_artifacts"] is True
    assert preprocess_phase_summary["output_mode"] == "lazy"


def test_run_concat_segments_core_logs_progress_per_well(monkeypatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from axon_recon.pipeline.stages.preprocess.core import concat_segments as concat_segments_core

    class _FakeRecording:
        pass

    fake_spikeinterface = types.ModuleType("spikeinterface")
    fake_spikeinterface_full = types.ModuleType("spikeinterface.full")
    fake_spikeinterface_full.concatenate_recordings = lambda recordings: {"concatenated": len(recordings)}
    fake_spikeinterface.full = fake_spikeinterface_full
    monkeypatch.setitem(sys.modules, "spikeinterface", fake_spikeinterface)
    monkeypatch.setitem(sys.modules, "spikeinterface.full", fake_spikeinterface_full)

    monkeypatch.setattr(
        concat_segments_core,
        "load_segment_manifest",
        lambda _path: [
            {"segment_index": 0, "rec_name": "seg000", "folder": str(tmp_path / "seg000")},
            {"segment_index": 1, "rec_name": "seg001", "folder": str(tmp_path / "seg001")},
        ],
    )
    monkeypatch.setattr(concat_segments_core, "load_segment_recording_from_entry", lambda _entry: _FakeRecording())
    monkeypatch.setattr(concat_segments_core, "build_stitch_frames_from_segment_manifest", lambda _entries: [100])
    written_manifests: list[tuple[Path, dict[str, object]]] = []
    monkeypatch.setattr(
        concat_segments_core,
        "write_json",
        lambda path, payload: written_manifests.append((Path(path), dict(payload))),
    )

    saved_payloads: list[dict[str, object]] = []

    def _fake_save(**kwargs):
        saved_payloads.append(dict(kwargs))
        return {"recording_dir": str(kwargs["recording_dir"]), "saved": True}

    with caplog.at_level(logging.INFO):
        payload = concat_segments_core.run_concat_segments_core(
            stream_id="well001",
            segment_manifest_path=tmp_path / "segments_manifest.json",
            recording_dir=tmp_path / "concatenated_recording",
            concat_manifest_path=tmp_path / "concat_manifest.json",
            overwrite_saved_recording=False,
            output_mode="binary",
            n_jobs=1,
            chunk_duration="1s",
            progress_bar=False,
            logger=logging.getLogger("test.concat_segments.progress"),
            run_save_concatenated_recording_core=_fake_save,
        )

    messages = [record.getMessage() for record in caplog.records]
    assert payload["segment_count"] == 2
    assert len(saved_payloads) == 1
    assert len(written_manifests) == 1
    assert any("Starting concat_segments for well=well001 segment_count=2" in message for message in messages)
    assert any("concat_segments progress well=well001 loaded=1/2 rec_name=seg000" in message for message in messages)
    assert any("concat_segments progress well=well001 loaded=2/2 rec_name=seg001" in message for message in messages)


def test_run_plot_segment_traces_core_loads_lazy_provenance_entries(monkeypatch, tmp_path: Path) -> None:
    import numpy as np

    from axon_recon.pipeline.stages.preprocess.core import plot_segment_traces as plot_segment_traces_core

    class _FakeRecording:
        def set_times(self, _times) -> None:
            return None

    provenance_paths = [tmp_path / "segments" / "000_seg000.json", tmp_path / "segments" / "001_seg001.json"]
    load_calls: list[str] = []
    rendered_paths: list[str] = []

    monkeypatch.setattr(
        plot_segment_traces_core,
        "load_segment_manifest",
        lambda _path: [
            {"segment_index": 0, "rec_name": "seg000", "provenance_path": str(provenance_paths[0])},
            {"segment_index": 1, "rec_name": "seg001", "provenance_path": str(provenance_paths[1])},
        ],
    )
    monkeypatch.setattr(
        plot_segment_traces_core,
        "load_recording_metadata",
        lambda **_kwargs: ({"segments": []}, {"epochs": []}, {"segments": []}),
    )
    monkeypatch.setattr(
        plot_segment_traces_core,
        "load_segment_recording_from_entry",
        lambda entry: load_calls.append(str(entry["provenance_path"])) or _FakeRecording(),
    )
    monkeypatch.setattr(plot_segment_traces_core, "_resolve_representative_channels", lambda **_kwargs: [11, 22])
    monkeypatch.setattr(plot_segment_traces_core, "build_segment_time_vector", lambda **_kwargs: np.arange(20, dtype=float))
    monkeypatch.setattr(plot_segment_traces_core, "_plot_channel_layout", lambda **_kwargs: None)
    monkeypatch.setattr(
        plot_segment_traces_core,
        "_plot_concat_cluster_traces",
        lambda **kwargs: rendered_paths.append(str(kwargs["out_path"])),
    )

    payload = plot_segment_traces_core.run_plot_segment_traces_core(
        stream_id="well001",
        segment_manifest_path=tmp_path / "segments_manifest.json",
        segment_epochs_path=tmp_path / "segment_epochs.json",
        contiguous_epochs_path=tmp_path / "contiguous_epochs.json",
        sampling_metadata_path=tmp_path / "sampling_metadata.json",
        plot_output_dir=tmp_path / "plots",
        channel_layouts_subdir="channel_layouts",
        segment_traces_subdir="segment_traces",
        plot_layouts=False,
        plot_segment_traces=True,
        segment_trace_n_reps=2,
        plot_n_jobs=1,
        trace_downsample_hz=None,
        trace_max_points=100,
        logger=None,
    )

    assert payload["segment_count"] == 2
    assert load_calls == [str(provenance_paths[0]), str(provenance_paths[0]), str(provenance_paths[1])]
    assert len(rendered_paths) == 2


def test_run_plot_raster_threshold_core_loads_lazy_provenance_entries(monkeypatch, tmp_path: Path) -> None:
    import numpy as np

    from axon_recon.pipeline.stages.preprocess.core import plot_raster_threshold as plot_raster_threshold_core

    class _FakeRecording:
        def __init__(self, channel_ids: list[int]) -> None:
            self._channel_ids = channel_ids

        def get_channel_ids(self):
            return list(self._channel_ids)

        def get_num_samples(self) -> int:
            return 20

        def get_sampling_frequency(self) -> float:
            return 10_000.0

    provenance_paths = [tmp_path / "segments" / "000_seg000.json", tmp_path / "segments" / "001_seg001.json"]
    load_calls: list[str] = []
    written: dict[str, object] = {}

    monkeypatch.setattr(
        plot_raster_threshold_core,
        "load_segment_manifest",
        lambda _path: [
            {"segment_index": 0, "rec_name": "seg000", "provenance_path": str(provenance_paths[0])},
            {"segment_index": 1, "rec_name": "seg001", "provenance_path": str(provenance_paths[1])},
        ],
    )
    monkeypatch.setattr(
        plot_raster_threshold_core,
        "load_recording_metadata",
        lambda **_kwargs: (
            {
                "segments": [
                    {
                        "rec_name": "seg000",
                        "n_samples": 20,
                        "sampling_frequency_hz": 10_000.0,
                        "start_time_seconds_since_epoch": 100.0,
                        "stop_time_seconds_since_epoch": 100.002,
                    },
                    {
                        "rec_name": "seg001",
                        "n_samples": 20,
                        "sampling_frequency_hz": 10_000.0,
                        "start_time_seconds_since_epoch": 101.0,
                        "stop_time_seconds_since_epoch": 101.002,
                    },
                ]
            },
            {"epochs": []},
            {"segments": []},
        ),
    )
    monkeypatch.setattr(
        plot_raster_threshold_core,
        "load_segment_recording_from_entry",
        lambda entry: load_calls.append(str(entry["provenance_path"])) or _FakeRecording([11, 22]),
    )
    monkeypatch.setattr(
        plot_raster_threshold_core,
        "build_segment_time_vector",
        lambda **kwargs: np.linspace(
            0.0 if kwargs["rec_name"] == "seg000" else 1.0,
            0.0019 if kwargs["rec_name"] == "seg000" else 1.0019,
            20,
            dtype=float,
        ),
    )
    monkeypatch.setattr(
        plot_raster_threshold_core,
        "_estimate_channel_thresholds",
        lambda **_kwargs: np.asarray([5.0, 5.0], dtype=float),
    )
    monkeypatch.setattr(
        plot_raster_threshold_core,
        "_collect_threshold_crossings",
        lambda **kwargs: (
            np.asarray([float(np.asarray(kwargs["time_vector"])[3]), float(np.asarray(kwargs["time_vector"])[7])], dtype=float),
            np.asarray([11, 22], dtype=int),
        ),
    )
    monkeypatch.setattr(
        plot_raster_threshold_core,
        "_write_threshold_raster_plot",
        lambda **kwargs: written.update(kwargs),
    )

    payload = plot_raster_threshold_core.run_plot_raster_threshold_core(
        stream_id="well001",
        segment_manifest_path=tmp_path / "segments_manifest.json",
        segment_epochs_path=tmp_path / "segment_epochs.json",
        contiguous_epochs_path=tmp_path / "contiguous_epochs.json",
        sampling_metadata_path=tmp_path / "sampling_metadata.json",
        raster_output_dir=tmp_path / "raster_threshold",
        logger=None,
    )

    assert payload["segment_count"] == 2
    assert payload["electrode_ids"] == [11, 22]
    assert payload["total_event_count"] == 4
    assert load_calls == [str(provenance_paths[0]), str(provenance_paths[1])]
    assert Path(str(payload["raster_plot_path"])).name == "threshold_raster_well001.png"
    assert written["unique_electrodes"] == [11, 22]


def test_run_concat_segments_core_loads_lazy_provenance_entries(monkeypatch, tmp_path: Path) -> None:
    from axon_recon.pipeline.stages.preprocess.core import concat_segments as concat_segments_core

    class _FakeRecording:
        pass

    fake_spikeinterface = types.ModuleType("spikeinterface")
    fake_spikeinterface_full = types.ModuleType("spikeinterface.full")
    fake_spikeinterface_full.concatenate_recordings = lambda recordings: {"concatenated": len(recordings)}
    fake_spikeinterface.full = fake_spikeinterface_full
    monkeypatch.setitem(sys.modules, "spikeinterface", fake_spikeinterface)
    monkeypatch.setitem(sys.modules, "spikeinterface.full", fake_spikeinterface_full)

    provenance_paths = [tmp_path / "segments" / "000_seg000.json", tmp_path / "segments" / "001_seg001.json"]
    load_calls: list[str] = []
    monkeypatch.setattr(
        concat_segments_core,
        "load_segment_manifest",
        lambda _path: [
            {"segment_index": 0, "rec_name": "seg000", "provenance_path": str(provenance_paths[0]), "n_samples": 100},
            {"segment_index": 1, "rec_name": "seg001", "provenance_path": str(provenance_paths[1]), "n_samples": 120},
        ],
    )
    monkeypatch.setattr(
        concat_segments_core,
        "load_segment_recording_from_entry",
        lambda entry: load_calls.append(str(entry["provenance_path"])) or _FakeRecording(),
    )
    monkeypatch.setattr(concat_segments_core, "build_stitch_frames_from_segment_manifest", lambda _entries: [100])
    monkeypatch.setattr(concat_segments_core, "write_json", lambda *_args, **_kwargs: None)

    saved_payloads: list[dict[str, object]] = []

    def _fake_save(**kwargs):
        saved_payloads.append(dict(kwargs))
        return {"recording_dir": str(kwargs["recording_dir"]), "saved": True}

    payload = concat_segments_core.run_concat_segments_core(
        stream_id="well001",
        segment_manifest_path=tmp_path / "segments_manifest.json",
        recording_dir=tmp_path / "concatenated_recording",
        concat_manifest_path=tmp_path / "concat_manifest.json",
        overwrite_saved_recording=False,
        output_mode="binary",
        n_jobs=1,
        chunk_duration="1s",
        progress_bar=False,
        logger=None,
        run_save_concatenated_recording_core=_fake_save,
    )

    assert payload["segment_count"] == 2
    assert load_calls == [str(provenance_paths[0]), str(provenance_paths[1])]
    assert len(saved_payloads) == 1


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


def test_run_preprocess_prepare_raw_binaries_phase_writes_targeted_summary(tmp_path: Path, monkeypatch) -> None:
    _install_success_fakes(monkeypatch, tmp_path)

    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
    )

    payload = run_preprocess_prepare_raw_binaries_phase(inputs)

    assert payload["phase"] == "prepare_raw_binaries"
    assert Path(str(payload["summary_json"])).exists()
    assert payload["outputs"]["raw_binary_recording_dir"].endswith("raw_binary_recording")
    assert payload["outputs"]["raw_binary_manifest_json"].endswith("raw_binary_manifest.json")


def test_run_preprocess_plot_segment_channel_layouts_phase_writes_targeted_summary(tmp_path: Path, monkeypatch) -> None:
    _install_success_fakes(monkeypatch, tmp_path)

    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
    )

    payload = run_preprocess_plot_segment_channel_layouts_phase(inputs)

    assert payload["phase"] == "plot_segment_channel_layouts"
    assert Path(str(payload["summary_json"])).exists()
    assert payload["outputs"]["plot_output_dir"].endswith("preprocess_outputs")
    assert len(list(payload.get("layout_plot_paths", []))) == 1


def test_run_preprocess_plot_raster_threshold_phase_writes_targeted_summary(tmp_path: Path, monkeypatch) -> None:
    captured_phase_kwargs: dict[str, dict] = {}
    well_out_dir, _fake_log = _install_success_fakes(monkeypatch, tmp_path, captured_phase_kwargs=captured_phase_kwargs)
    canonical_out_dir = well_out_dir / "preprocess_outputs"

    inputs = PreprocessInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        phases=PreprocessPhasesConfig(
            save_rec_metadata=PreprocessSaveRecMetadataPhaseConfig(enabled=True),
            prepare_raw_binaries=PreprocessPrepareRawBinariesPhaseConfig(enabled=False),
            preprocess_segments=PreprocessSegmentsPhaseConfig(enabled=True),
            plot_segment_traces=PreprocessPlotSegmentTracesPhaseConfig(enabled=False),
            plot_segment_channel_layouts=PreprocessPlotSegmentChannelLayoutsPhaseConfig(enabled=False),
            concat_segments=PreprocessConcatSegmentsPhaseConfig(enabled=False),
            plot_concat_traces=PreprocessPlotConcatTracesPhaseConfig(enabled=False),
            plot_raster_threshold=PreprocessPlotRasterThresholdPhaseConfig(
                enabled=True,
                debug_mode_enabled=True,
                report_step_timers=True,
                rel_output_root="raster_threshold_outputs",
            ),
            wipe_src_scratch=PreprocessWipeSrcScratchPhaseConfig(enabled=False),
        ),
    )

    payload = run_preprocess_plot_raster_threshold_phase(inputs)

    assert payload["phase"] == "plot_raster_threshold"
    assert payload["outputs"]["raster_output_dir"] == str(canonical_out_dir / "raster_threshold_outputs")
    assert Path(str(payload["raster_plot_path"])).exists()
    assert Path(str(captured_phase_kwargs["plot_raster_threshold"]["raster_output_dir"])) == canonical_out_dir / "raster_threshold_outputs"
    assert captured_phase_kwargs["plot_raster_threshold"]["report_step_timers"] is True


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
