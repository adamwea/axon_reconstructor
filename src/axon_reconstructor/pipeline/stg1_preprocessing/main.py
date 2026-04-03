"""Raw preprocessing public API.

The heavy concatenation implementation lives in .runner to keep this module orchestration-only.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from ..checkpointing import (
    ProcessingStage,
    compute_checkpoint_file,
    exception_to_error_dict,
    load_checkpoint,
    save_checkpoint,
)
from ..output_paths import compute_mea_analysis_output_dir
from ..pipeline_logging import compute_pipeline_log_file, setup_pipeline_logger
from .constants import PREPROCESS_OUTPUTS_DIRNAME

from .concatenation import find_common_electrodes_from_segments
from .planning import (
    RawPreprocessPlan,
    build_preprocess_plan,
    discover_cfg_files,
    format_cfg_discovery_summary,
    parse_cfg_channel_locations,
)
from .runner import build_concatenated_recording


def run_preprocess_stage(
    *,
    h5_path: Path,
    stream_id: str,
    mea_output_root: Optional[Path],
    force_restart: bool,
    limit_segments_per_well: Optional[int] = None,
    log_enabled: bool = True,
    log_verbose: bool = True,
    log_file_override: Optional[Path | str] = None,
    suppress_h5_plugin_messages: bool = False,
    phase_dividers: bool = True,
    enable_checkpointing: bool = True,
    n_jobs: int = 8,
    plot_layouts: bool = True,
    plot_concat_trace: bool = True,
    plot_segment_traces: bool = True,
    plot_output_dir_override: Optional[Path | str] = None,
    epoch_markers_output_dir_override: Optional[Path | str] = None,
    assay_stats_relpath: str = "assay_stats_{stream_id}.txt",
    channel_layouts_subdir: str = "channel_layouts",
    segment_traces_subdir: str = "segment_traces",
    concat_trace_relpath: str = "concat_cluster_reps_{stream_id}.png",
    n_representative_channels: int = 4,
    concat_trace_n_reps: int | None = None,
    segment_trace_n_reps: int | None = None,
    plot_n_jobs: int = 1,
    trace_downsample_hz: Optional[float] = None,
    trace_max_points: int = 150_000,
    temporal_resample_factor: Optional[int] = None,
    temporal_resample_rate_hz: Optional[int] = None,
    temporal_resample_margin_ms: float = 100.0,
    temporal_resample_dtype: Optional[str] = None,
    save_recording: bool = True,
    overwrite_saved_recording: bool = True,
    save_concat_recording: bool = True,
    save_segment_recordings: bool = True,
    save_chunk_duration: str = "1s",
    save_progress_bar: bool = False,
    concat_save_n_jobs: Optional[int] = None,
    segment_save_n_jobs: Optional[int] = None,
    print_n_jobs_used: bool = False,
    logger: Optional[logging.Logger] = None,
) -> tuple[object, list[int]]:
    plan = build_preprocess_plan(h5_path=h5_path, stream_id=stream_id)

    logger = logger or logging.getLogger("axon_reconstructor.preprocess")
    save_recording = bool(save_recording)
    save_concat_recording = bool(save_concat_recording)
    save_segment_recordings = bool(save_segment_recordings)
    plot_layouts = bool(plot_layouts)
    plot_concat_trace = bool(plot_concat_trace)
    plot_segment_traces = bool(plot_segment_traces)
    save_progress_bar = bool(save_progress_bar)
    print_n_jobs_used = bool(print_n_jobs_used)
    suppress_h5_plugin_messages = bool(suppress_h5_plugin_messages)
    phase_dividers = bool(phase_dividers)

    if limit_segments_per_well is not None:
        try:
            parsed_limit_segments = int(limit_segments_per_well)
            limit_segments_per_well = (parsed_limit_segments if parsed_limit_segments > 0 else None)
        except Exception:
            limit_segments_per_well = None

    def _emit_info(message: str) -> None:
        logger.info(message)
        print(f"[axon_reconstructor] {message}", flush=True)

    def _emit_phase_divider(title: str) -> None:
        if not bool(phase_dividers):
            return
        _emit_info(f"========== {title} ==========")

    if save_recording and (not save_concat_recording) and (not save_segment_recordings):
        logger.warning(
            "save_recording=True but both save_concat_recording and save_segment_recordings are False; "
            "no recording artifacts will be written"
        )

    well_out_dir = None
    if mea_output_root is not None and (plot_layouts or plot_concat_trace or plot_segment_traces or save_recording or enable_checkpointing):
        well_out_dir = compute_mea_analysis_output_dir(
            output_root=mea_output_root,
            data_file=h5_path,
            well=stream_id,
        )
        if bool(log_enabled):
            try:
                if log_file_override is None:
                    log_file = compute_pipeline_log_file(
                        well_out_dir=well_out_dir,
                        data_file=h5_path,
                        stream_id=stream_id,
                    )
                else:
                    log_file = Path(str(log_file_override)).expanduser()
                    if not log_file.is_absolute():
                        log_file = (well_out_dir / log_file).resolve()
                    else:
                        log_file = log_file.resolve()
                logger = setup_pipeline_logger(
                    log_file=log_file,
                    logger_name=f"axon_reconstructor.{log_file.stem}",
                    verbose=bool(log_verbose),
                )
            except Exception:
                pass

    cfg_summary = plan.cfg_discovery_summary if isinstance(plan.cfg_discovery_summary, dict) else {}
    if plan.cfg_files:
        logger.info(
            "Discovered %d cfg files next to %s (%s)",
            len(plan.cfg_files),
            plan.h5_path,
            format_cfg_discovery_summary(cfg_summary),
        )
    else:
        logger.info("No .cfg files discovered next to %s; using contact_vector electrodes", plan.h5_path)

    def _format_stream_token(raw: Optional[Path | str]) -> str | None:
        if raw is None:
            return None
        text = str(raw).strip()
        if not text:
            return None
        try:
            return text.format(stream_id=str(stream_id))
        except Exception:
            return text

    def _resolve_output_path(raw: Optional[Path | str], *, base_dir: Path) -> Path | None:
        rendered = _format_stream_token(raw)
        if rendered is None:
            return None
        path = Path(rendered).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (base_dir / path).resolve()

    plot_dir = None
    plotting_enabled = bool(plot_layouts) or bool(plot_concat_trace) or bool(plot_segment_traces)
    if plotting_enabled:
        if well_out_dir is None:
            if plot_output_dir_override is not None:
                plot_dir = _resolve_output_path(plot_output_dir_override, base_dir=Path(h5_path).parent)
            else:
                logger.warning("plot outputs enabled but mea_output_root is not set and no plot_output_dir_override provided; skipping plots")
        else:
            plot_dir = _resolve_output_path(plot_output_dir_override, base_dir=well_out_dir)
            if plot_dir is None:
                plot_dir = well_out_dir / PREPROCESS_OUTPUTS_DIRNAME
        if plot_dir is not None:
            plot_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Preprocess diagnostics output: %s", plot_dir)

    if save_recording and well_out_dir is None:
        logger.warning("save_recording=True but mea_output_root is not set; skipping recording save")

    checkpoint_file = None
    checkpoint_state = None
    preprocess_dir = None
    recording_dir = None
    common_el_path = None
    epoch_maxwell_path = None
    epoch_concat_path = None
    per_segment_preprocessed_dir = None
    per_segment_manifest_path = None

    if enable_checkpointing and well_out_dir is not None:
        checkpoint_file = compute_checkpoint_file(
            output_dir=well_out_dir,
            file_path=h5_path,
            stream_id=stream_id,
        )
        checkpoint_state = load_checkpoint(
            checkpoint_file=checkpoint_file,
            force_restart=force_restart,
            output_dir=well_out_dir,
            file_path=h5_path,
            stream_id=stream_id,
        )

    if well_out_dir is not None:
        preprocess_dir = well_out_dir / PREPROCESS_OUTPUTS_DIRNAME
        recording_dir = preprocess_dir / "preprocessed_recording"
        common_el_path = preprocess_dir / "common_electrodes.npy"
        epoch_maxwell_path = preprocess_dir / f"maxwell_contiguous_epochs_{stream_id}.json"
        epoch_concat_path = preprocess_dir / f"concatenation_stitch_epochs_{stream_id}.json"
        per_segment_preprocessed_dir = preprocess_dir / "per_segment_preprocessed"
        per_segment_manifest_path = per_segment_preprocessed_dir / "manifest.json"

    epoch_markers_output_dir = None
    if epoch_markers_output_dir_override is not None:
        base_dir = (well_out_dir if well_out_dir is not None else Path(h5_path).parent)
        epoch_markers_output_dir = _resolve_output_path(epoch_markers_output_dir_override, base_dir=base_dir)
    elif preprocess_dir is not None:
        epoch_markers_output_dir = preprocess_dir
    else:
        epoch_markers_output_dir = plot_dir

    preprocess_cfg_path = (preprocess_dir / "preprocess_config.json") if preprocess_dir is not None else None
    requested_cfg = {
        "temporal_resample_factor": (int(temporal_resample_factor) if temporal_resample_factor is not None else None),
        "temporal_resample_rate_hz": (int(temporal_resample_rate_hz) if temporal_resample_rate_hz is not None else None),
        "temporal_resample_margin_ms": float(temporal_resample_margin_ms),
        "temporal_resample_dtype": (str(temporal_resample_dtype) if temporal_resample_dtype is not None else None),
        "limit_segments_per_well": (int(limit_segments_per_well) if limit_segments_per_well is not None else None),
        "plot_layouts": bool(plot_layouts),
        "plot_concat_trace": bool(plot_concat_trace),
        "plot_segment_traces": bool(plot_segment_traces),
        "n_representative_channels": int(n_representative_channels),
        "concat_trace_n_reps": (int(concat_trace_n_reps) if concat_trace_n_reps is not None else None),
        "segment_trace_n_reps": (int(segment_trace_n_reps) if segment_trace_n_reps is not None else None),
        "plot_n_jobs": max(1, int(plot_n_jobs)),
    }

    resume_requires_segment_manifest = bool(save_segment_recordings)
    if (
        save_recording
        and save_concat_recording
        and not overwrite_saved_recording
        and checkpoint_state is not None
        and checkpoint_state.stage >= ProcessingStage.PREPROCESSING_COMPLETE.value
        and recording_dir is not None
        and common_el_path is not None
        and recording_dir.exists()
        and common_el_path.exists()
        and (
            (not resume_requires_segment_manifest)
            or (
                per_segment_manifest_path is not None
                and per_segment_manifest_path.exists()
            )
        )
    ):
        try:
            import numpy as np  # type: ignore[import-not-found]
            import spikeinterface.full as si  # type: ignore[import-not-found]

            if preprocess_cfg_path is not None and preprocess_cfg_path.exists():
                import json

                saved_cfg = json.loads(preprocess_cfg_path.read_text(errors="replace"))
                if isinstance(saved_cfg, dict) and saved_cfg.get("requested_cfg") != requested_cfg:
                    raise RuntimeError(
                        f"Saved preprocess_config.json does not match requested options; re-running preprocessing. "
                        f"(saved at {preprocess_cfg_path})"
                    )
            elif temporal_resample_rate_hz is not None or temporal_resample_factor is not None:
                raise RuntimeError("Temporal resampling requested but no preprocess_config.json found; re-running")

            try:
                multirec = si.load(recording_dir)
            except Exception:
                multirec = si.load_extractor(recording_dir)

            common_el = np.load(common_el_path).tolist()
            _emit_info(f"Resuming from saved concatenated preprocessed recording: {recording_dir}")
            return multirec, common_el
        except Exception as e:
            logger.warning("Failed to resume from saved preprocessed recording (%s); re-running", e)

    if checkpoint_file is not None and checkpoint_state is not None:
        checkpoint_state = save_checkpoint(
            checkpoint_file=checkpoint_file,
            state=checkpoint_state,
            stage=ProcessingStage.PREPROCESSING,
            failed_stage=None,
            error=None,
            extra_fields={
                "stg1_preprocess_outputs_dir": str(preprocess_dir) if preprocess_dir else None,
            },
        )

    try:
        _emit_phase_divider("Build Preprocessed Recording")
        _emit_info(
            "Starting preprocess build "
            f"(stream={stream_id}, n_jobs={int(n_jobs)}, plot_layouts={bool(plot_layouts)}, "
            f"plot_concat_trace={bool(plot_concat_trace)}, plot_segment_traces={bool(plot_segment_traces)}, "
            f"n_representative_channels={int(n_representative_channels)}, "
            f"concat_trace_n_reps={int(concat_trace_n_reps) if concat_trace_n_reps is not None else 'auto'}, "
            f"segment_trace_n_reps={int(segment_trace_n_reps) if segment_trace_n_reps is not None else 'auto'}, "
            f"plot_n_jobs={max(1, int(plot_n_jobs))}, "
            f"limit_segments_per_well={int(limit_segments_per_well) if limit_segments_per_well is not None else 'none'}, "
            f"save_concat_recording={bool(save_concat_recording)}, "
            f"save_segment_recordings={bool(save_segment_recordings)})"
        )
        build_result = build_concatenated_recording(
            h5_path=plan.h5_path,
            stream_id=plan.stream_id,
            n_jobs=n_jobs,
            plot_output_dir=plot_dir,
            plot_layouts=bool(plot_layouts),
            plot_concat_trace=bool(plot_concat_trace),
            plot_segment_traces=bool(plot_segment_traces),
            epoch_markers_output_dir=epoch_markers_output_dir,
            assay_stats_relpath=str(assay_stats_relpath),
            channel_layouts_subdir=str(channel_layouts_subdir),
            segment_traces_subdir=str(segment_traces_subdir),
            concat_trace_relpath=str(concat_trace_relpath),
            n_representative_channels=int(n_representative_channels),
            n_representative_channels_concat=(int(concat_trace_n_reps) if concat_trace_n_reps is not None else None),
            n_representative_channels_segment=(int(segment_trace_n_reps) if segment_trace_n_reps is not None else None),
            plot_n_jobs=max(1, int(plot_n_jobs)),
            trace_downsample_hz=trace_downsample_hz,
            trace_max_points=(-1 if int(trace_max_points) <= 0 else max(1000, int(trace_max_points))),
            limit_segments_per_well=(int(limit_segments_per_well) if limit_segments_per_well is not None else None),
            temporal_resample_factor=(int(temporal_resample_factor) if temporal_resample_factor is not None else None),
            temporal_resample_rate_hz=(int(temporal_resample_rate_hz) if temporal_resample_rate_hz is not None else None),
            temporal_resample_margin_ms=float(temporal_resample_margin_ms),
            temporal_resample_dtype=(str(temporal_resample_dtype) if temporal_resample_dtype is not None else None),
            logger=logger,
            phase_dividers=bool(phase_dividers),
            suppress_h5_plugin_messages=bool(suppress_h5_plugin_messages),
            return_artifacts=True,
        )
        if len(build_result) == 3:
            multirec, common_el, preprocess_artifacts = build_result
        else:
            multirec, common_el = build_result
            preprocess_artifacts = {}
        _emit_info(f"Preprocessed recording built; common electrodes={len(common_el)}")
        if epoch_maxwell_path is not None and epoch_concat_path is not None:
            if epoch_maxwell_path.exists():
                logger.info("Epoch markers (Maxwell): %s", epoch_maxwell_path)
            if epoch_concat_path.exists():
                logger.info("Epoch markers (Concat stitches): %s", epoch_concat_path)
    except Exception as e:
        if checkpoint_file is not None and checkpoint_state is not None:
            save_checkpoint(
                checkpoint_file=checkpoint_file,
                state=checkpoint_state,
                stage=ProcessingStage.NOT_STARTED,
                failed_stage=ProcessingStage.PREPROCESSING.name,
                error=exception_to_error_dict(e),
            )
        raise

    def _resolve_jobs(raw: Optional[int], *, fallback: int) -> int:
        try:
            parsed = int(raw) if raw is not None else int(fallback)
        except Exception:
            parsed = int(fallback)
        return max(1, int(parsed))

    save_chunk_duration_token = str(save_chunk_duration).strip() or "1s"
    concat_jobs = _resolve_jobs(concat_save_n_jobs, fallback=int(n_jobs))
    segment_jobs = _resolve_jobs(segment_save_n_jobs, fallback=int(n_jobs))

    phase_timing_s_raw = preprocess_artifacts.get("phase_timing_s", {}) if isinstance(preprocess_artifacts, dict) else {}
    phase_timing_s: dict[str, float] = {}
    if isinstance(phase_timing_s_raw, dict):
        for key, value in phase_timing_s_raw.items():
            try:
                phase_timing_s[str(key)] = float(value)
            except Exception:
                continue

    if well_out_dir is not None and preprocess_dir is not None:
        preprocess_dir.mkdir(parents=True, exist_ok=True)

    if preprocess_cfg_path is not None:
        try:
            import datetime as dt
            import json

            preprocess_cfg_path.write_text(
                json.dumps(
                    {
                        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "h5_path": str(plan.h5_path),
                        "stream_id": str(stream_id),
                        "cfg_discovery_summary": cfg_summary,
                        "logging_cfg": {
                            "enabled": bool(log_enabled),
                            "verbose": bool(log_verbose),
                            "file_relpath": (str(log_file_override) if log_file_override is not None else None),
                            "suppress_h5_plugin_messages": bool(suppress_h5_plugin_messages),
                            "phase_dividers": bool(phase_dividers),
                        },
                        "requested_cfg": requested_cfg,
                        "plot_cfg": {
                            "plot_layouts": bool(plot_layouts),
                            "plot_concat_trace": bool(plot_concat_trace),
                            "plot_segment_traces": bool(plot_segment_traces),
                            "limit_segments_per_well": (int(limit_segments_per_well) if limit_segments_per_well is not None else None),
                            "n_representative_channels": int(n_representative_channels),
                            "concat_trace_n_reps": (int(concat_trace_n_reps) if concat_trace_n_reps is not None else None),
                            "segment_trace_n_reps": (int(segment_trace_n_reps) if segment_trace_n_reps is not None else None),
                            "plot_n_jobs": max(1, int(plot_n_jobs)),
                        },
                        "save_cfg": {
                            "save_recording": bool(save_recording),
                            "overwrite_saved_recording": bool(overwrite_saved_recording),
                            "save_concat_recording": bool(save_concat_recording),
                            "save_segment_recordings": bool(save_segment_recordings),
                            "save_chunk_duration": str(save_chunk_duration_token),
                            "save_progress_bar": bool(save_progress_bar),
                            "concat_save_n_jobs": int(concat_jobs),
                            "segment_save_n_jobs": int(segment_jobs),
                            "print_n_jobs_used": bool(print_n_jobs_used),
                        },
                        "phase_timing_s": {str(k): float(v) for k, v in phase_timing_s.items()},
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        except Exception as e:
            logger.warning("Failed to write preprocess_config.json: %s", e)

    if save_recording and well_out_dir is not None:
        assert preprocess_dir is not None
        assert recording_dir is not None
        assert common_el_path is not None

        if print_n_jobs_used:
            _emit_info(
                "Effective preprocess worker usage: "
                f"preprocess_n_jobs={int(n_jobs)}, "
                f"concat_save_n_jobs={int(concat_jobs)}, "
                f"segment_save_n_jobs={int(segment_jobs)}"
            )

        _emit_phase_divider("Save Recording Artifacts")
        try:
            import numpy as np  # type: ignore[import-not-found]
            import shutil

            _emit_info(
                "Preprocess save settings: "
                f"concat={bool(save_concat_recording)}, "
                f"segments={bool(save_segment_recordings)}, "
                f"overwrite={bool(overwrite_saved_recording)}, "
                f"concat_n_jobs={int(concat_jobs)}, "
                f"segment_n_jobs={int(segment_jobs)}, "
                f"chunk_duration={save_chunk_duration_token}, "
                f"progress_bar={bool(save_progress_bar)}"
            )

            _emit_phase_divider("Save Concatenated Recording")
            if save_concat_recording:
                if recording_dir.exists() and overwrite_saved_recording:
                    shutil.rmtree(recording_dir)

                if (not recording_dir.exists()) or overwrite_saved_recording:
                    t_concat_save = time.perf_counter()
                    _emit_info(
                        "Saving concatenated preprocessed recording to "
                        f"{recording_dir} (n_jobs={int(concat_jobs)}, chunk_duration={save_chunk_duration_token}, "
                        f"progress_bar={bool(save_progress_bar)})"
                    )
                    multirec.save(
                        folder=recording_dir,
                        format="binary",
                        overwrite=True,
                        n_jobs=int(concat_jobs),
                        chunk_duration=save_chunk_duration_token,
                        progress_bar=bool(save_progress_bar),
                    )
                    _emit_info(
                        "Saved concatenated preprocessed recording to "
                        f"{recording_dir} in {time.perf_counter() - t_concat_save:.2f}s"
                    )
                else:
                    _emit_info(f"Concatenated preprocessed recording already exists at {recording_dir}; not overwriting")
            else:
                _emit_info("Skipping concatenated preprocessed recording save (save_concat_recording=False)")
                if recording_dir.exists() and overwrite_saved_recording:
                    shutil.rmtree(recording_dir)
                    _emit_info(
                        "Removed existing concatenated preprocessed recording directory because "
                        f"save_concat_recording=False: {recording_dir}"
                    )

            # Persist per-segment preprocessed recordings for downstream segment registration.
            _emit_phase_divider("Save Segment Recordings")
            segment_recordings = list(preprocess_artifacts.get("segment_recordings_preprocessed", []) or [])
            segment_names = [str(x) for x in list(preprocess_artifacts.get("rec_names", []) or [])]
            segment_stats = list(preprocess_artifacts.get("segment_stats", []) or [])
            if per_segment_preprocessed_dir is not None and per_segment_manifest_path is not None:
                if save_segment_recordings:
                    if per_segment_preprocessed_dir.exists() and overwrite_saved_recording:
                        shutil.rmtree(per_segment_preprocessed_dir)
                    per_segment_preprocessed_dir.mkdir(parents=True, exist_ok=True)

                    total_segments = int(len(segment_recordings))
                    _emit_info(
                        "Saving "
                        f"{total_segments} preprocessed segment recording(s) under {per_segment_preprocessed_dir} "
                        f"(n_jobs={int(segment_jobs)}, chunk_duration={save_chunk_duration_token}, "
                        f"progress_bar={bool(save_progress_bar)})"
                    )

                    t_segments_save = time.perf_counter()
                    manifest_segments: list[dict] = []
                    segment_total_for_display = max(1, total_segments)
                    for seg_idx, seg_rec in enumerate(segment_recordings):
                        rec_name = segment_names[seg_idx] if seg_idx < len(segment_names) else f"segment_{seg_idx:03d}"
                        seg_token = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(rec_name)).strip("_")
                        if not seg_token:
                            seg_token = f"segment_{seg_idx:03d}"
                        seg_dir = per_segment_preprocessed_dir / f"{seg_idx:03d}_{seg_token}"

                        t_seg_save = time.perf_counter()
                        _emit_info(
                            "Saving segment "
                            f"{seg_idx + 1}/{segment_total_for_display} ({rec_name}) to {seg_dir}"
                        )
                        seg_rec.save(
                            folder=seg_dir,
                            format="binary",
                            overwrite=True,
                            n_jobs=int(segment_jobs),
                            chunk_duration=save_chunk_duration_token,
                            progress_bar=bool(save_progress_bar),
                        )
                        _emit_info(
                            "Saved segment "
                            f"{seg_idx + 1}/{segment_total_for_display} ({rec_name}) in "
                            f"{time.perf_counter() - t_seg_save:.2f}s"
                        )

                        seg_entry = {
                            "segment_index": int(seg_idx),
                            "rec_name": str(rec_name),
                            "folder": str(seg_dir),
                        }
                        if seg_idx < len(segment_stats) and isinstance(segment_stats[seg_idx], dict):
                            seg_entry.update({
                                "fs_hz": float(segment_stats[seg_idx].get("fs", 0.0) or 0.0),
                                "n_samples": int(segment_stats[seg_idx].get("n_samples", 0) or 0),
                                "n_channels": int(segment_stats[seg_idx].get("n_channels", 0) or 0),
                            })
                        manifest_segments.append(seg_entry)

                    per_segment_manifest_path.write_text(
                        json.dumps(
                            {
                                "version": 1,
                                "stream_id": str(stream_id),
                                "segments": manifest_segments,
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    _emit_info(
                        "Saved "
                        f"{len(manifest_segments)} preprocessed segment recording(s) and manifest to "
                        f"{per_segment_manifest_path} in {time.perf_counter() - t_segments_save:.2f}s"
                    )
                else:
                    _emit_info("Skipping preprocessed segment recording saves (save_segment_recordings=False)")
                    if per_segment_preprocessed_dir.exists() and overwrite_saved_recording:
                        shutil.rmtree(per_segment_preprocessed_dir)
                        _emit_info(
                            "Removed existing per-segment preprocessed recording directory because "
                            f"save_segment_recordings=False: {per_segment_preprocessed_dir}"
                        )

            _emit_phase_divider("Save Common Electrodes")
            np.save(common_el_path, np.asarray(common_el, dtype=np.int64))
            _emit_info(f"Saved common electrodes ({len(common_el)}) to {common_el_path}")
        except Exception as e:
            logger.error("Failed to save preprocessed recording artifacts: %s", e)
            if checkpoint_file is not None and checkpoint_state is not None:
                save_checkpoint(
                    checkpoint_file=checkpoint_file,
                    state=checkpoint_state,
                    stage=ProcessingStage.NOT_STARTED,
                    failed_stage=ProcessingStage.PREPROCESSING.name,
                    error={
                        "type": type(e).__name__,
                        "message": str(e),
                    },
                )
            raise RuntimeError(f"Failed to save preprocessed recording artifacts: {e}") from e

    if checkpoint_file is not None and checkpoint_state is not None:
        preprocessed_recording_dir = str(recording_dir) if recording_dir and recording_dir.exists() else None
        common_electrodes_path = str(common_el_path) if common_el_path and common_el_path.exists() else None
        maxwell_epochs_path = str(epoch_maxwell_path) if epoch_maxwell_path and epoch_maxwell_path.exists() else None
        concat_epochs_path = str(epoch_concat_path) if epoch_concat_path and epoch_concat_path.exists() else None
        per_segment_preprocessed_dir_str = (
            str(per_segment_preprocessed_dir)
            if per_segment_preprocessed_dir is not None and per_segment_preprocessed_dir.exists()
            else None
        )
        per_segment_manifest_path_str = (
            str(per_segment_manifest_path)
            if per_segment_manifest_path is not None and per_segment_manifest_path.exists()
            else None
        )
        per_segment_count = 0
        if per_segment_manifest_path is not None and per_segment_manifest_path.exists():
            try:
                payload = json.loads(per_segment_manifest_path.read_text(encoding="utf-8"))
                segs = payload.get("segments") if isinstance(payload, dict) else []
                per_segment_count = int(len(segs)) if isinstance(segs, list) else 0
            except Exception:
                per_segment_count = 0
        _ = save_checkpoint(
            checkpoint_file=checkpoint_file,
            state=checkpoint_state,
            stage=ProcessingStage.PREPROCESSING_COMPLETE,
            failed_stage=None,
            error=None,
            extra_fields={
                "preprocessed_recording_dir": preprocessed_recording_dir,
                "common_electrodes_path": common_electrodes_path,
                "n_common_electrodes": len(common_el),
                "maxwell_epochs_path": maxwell_epochs_path,
                "concat_epochs_path": concat_epochs_path,
                "per_segment_preprocessed_dir": per_segment_preprocessed_dir_str,
                "per_segment_manifest_path": per_segment_manifest_path_str,
                "n_preprocessed_segments": int(per_segment_count),
                "save_recording": bool(save_recording),
                "save_concat_recording": bool(save_concat_recording),
                "save_segment_recordings": bool(save_segment_recordings),
                "cfg_discovery_summary": cfg_summary,
                "phase_timing_s": {str(k): float(v) for k, v in phase_timing_s.items()},
            },
        )

    return multirec, common_el

__all__ = [
    "RawPreprocessPlan",
    "discover_cfg_files",
    "parse_cfg_channel_locations",
    "build_preprocess_plan",
    "find_common_electrodes_from_segments",
    "build_concatenated_recording",
    "run_preprocess_stage",
]
