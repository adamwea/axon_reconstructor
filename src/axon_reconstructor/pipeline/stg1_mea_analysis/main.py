"""Raw preprocessing public API.

The heavy concatenation implementation lives in .runner to keep this module orchestration-only.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

from ..output_paths import compute_mea_analysis_output_dir
from ..pipeline_logging import compute_pipeline_log_file, setup_pipeline_logger
from .constants import PREPROCESS_OUTPUTS_DIRNAME

from .concatenation import find_common_electrodes_from_segments
from .h5_helpers import _read_well_rec_frame_nos_and_trigger_settings
from .planning import RawPreprocessPlan, build_preprocess_plan, discover_cfg_files, parse_cfg_channel_locations
from .runner import build_concatenated_recording
from .utils import _ensure_maxwell_hdf5_plugin_path


class _LogTee(io.TextIOBase):
    """Mirror a stream to terminal and pipeline logger line-by-line."""

    def __init__(
        self,
        *,
        terminal_stream: io.TextIOBase,
        logger: logging.Logger,
        level: int,
        label: str,
        emit_to_terminal: bool,
        emit_to_logger: bool,
    ) -> None:
        super().__init__()
        self._terminal_stream = terminal_stream
        self._logger = logger
        self._level = int(level)
        self._label = str(label)
        self._emit_to_terminal = bool(emit_to_terminal)
        self._emit_to_logger = bool(emit_to_logger)
        self._buffer = ""

    def _emit_line(self, line: str) -> None:
        if not self._emit_to_logger:
            return
        if self._emit_to_terminal:
            self._terminal_stream.write(line + "\n")
        self._logger.log(self._level, "[%s] %s", self._label, line)

    def write(self, s: str) -> int:  # type: ignore[override]
        text = "" if s is None else str(s)
        if text == "":
            return 0
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip() != "":
                self._emit_line(line)
        return len(text)

    def flush(self) -> None:  # type: ignore[override]
        if self._buffer.strip() != "":
            self._emit_line(self._buffer)
        self._buffer = ""
        try:
            self._terminal_stream.flush()
        except Exception:
            pass


@contextlib.contextmanager
def _capture_external_output(*, logger: logging.Logger):
    emit_to_terminal = _is_console_debug_enabled(logger=logger)
    emit_to_logger = _is_any_debug_enabled(logger=logger)
    out_tee = _LogTee(
        terminal_stream=sys.__stdout__,
        logger=logger,
        level=logging.DEBUG,
        label="stdout",
        emit_to_terminal=emit_to_terminal,
        emit_to_logger=emit_to_logger,
    )
    err_tee = _LogTee(
        terminal_stream=sys.__stderr__,
        logger=logger,
        level=logging.DEBUG,
        label="stderr",
        emit_to_terminal=emit_to_terminal,
        emit_to_logger=emit_to_logger,
    )
    with contextlib.redirect_stdout(out_tee), contextlib.redirect_stderr(err_tee):
        yield
    out_tee.flush()
    err_tee.flush()


def _is_console_debug_enabled(*, logger: logging.Logger) -> bool:
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            continue
        if isinstance(handler, logging.StreamHandler):
            return int(handler.level) <= int(logging.DEBUG)
    return False


def _is_any_debug_enabled(*, logger: logging.Logger) -> bool:
    for handler in logger.handlers:
        if int(handler.level) <= int(logging.DEBUG):
            return True
    return False


def _resolve_relative_artifact_path(*, well_out_dir: Path, override_rel: Optional[str], default_rel: str) -> Path:
    candidate = Path(str(override_rel).strip()) if override_rel is not None else Path(default_rel)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Invalid artifact override path '{candidate}'. Use a relative path under the well output dir.")
    return Path(well_out_dir) / candidate


def _remove_path_if_exists(*, path: Optional[Path], logger: logging.Logger) -> None:
    if path is None:
        return
    p = Path(path)
    if not p.exists():
        return
    try:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        logger.debug("Removed existing artifact for replot: %s", p)
    except Exception as e:
        logger.warning("Failed to remove artifact for replot (%s): %s", p, e)


def _save_recording_artifact(
    *,
    recording,
    out_dir: Path,
    n_jobs: int,
    chunk_duration: str,
    overwrite: bool,
    logger: logging.Logger,
    label: str,
) -> None:
    if out_dir.exists() and bool(overwrite):
        shutil.rmtree(out_dir)
    if out_dir.exists() and not bool(overwrite):
        logger.info("%s already exists at %s; not overwriting", label, out_dir)
        return
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving %s to %s", label, out_dir)
    with _capture_external_output(logger=logger):
        recording.save(
            folder=out_dir,
            format="binary",
            overwrite=True,
            n_jobs=int(n_jobs),
            chunk_duration=str(chunk_duration),
            progress_bar=False,
        )


def _write_assay_stats_diagnostic(*, h5_path: Path, stream_id: str, out_path: Path, logger: logging.Logger) -> None:
    try:
        import datetime as dt
        mod = importlib.import_module("MEA_Analysis.IPNAnalysis.h5_utils")
        list_stream_recording_names = getattr(mod, "list_stream_recording_names")
    except Exception as e:
        logger.warning("Skipping assay_stats diagnostic generation: %s", e)
        return

    try:
        rec_names = list_stream_recording_names(h5_path=h5_path, stream_id=stream_id)
        rows: list[dict[str, float | int | str]] = []
        for rn in rec_names:
            info = _read_well_rec_frame_nos_and_trigger_settings(h5_path=h5_path, stream_id=stream_id, rec_name=rn)
            frame_nos = info["frame_nos"]
            start_ms = int(info["start_ms"])
            stop_ms = int(info["stop_ms"])
            n_frames = int(frame_nos.size)
            frame_span = int(frame_nos[-1] - frame_nos[0] + 1) if n_frames > 0 else 0
            rows.append(
                {
                    "rec_name": str(rn),
                    "start_ms": int(start_ms),
                    "stop_ms": int(stop_ms),
                    "sampling_hz": float(info.get("sampling_hz", 0.0)),
                    "n_frames": int(n_frames),
                    "frame_no_start": int(frame_nos[0]) if n_frames > 0 else -1,
                    "frame_no_end": int(frame_nos[-1]) if n_frames > 0 else -1,
                    "frame_span": int(frame_span),
                    "duration_clock_s": float(max(0, stop_ms - start_ms)) / 1000.0,
                }
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        lines.append(f"assay_stats file: {out_path}")
        lines.append(f"generated_utc: {dt.datetime.now(dt.timezone.utc).isoformat()}")
        lines.append(f"h5_path: {h5_path}")
        lines.append(f"stream_id: {stream_id}")
        lines.append(f"n_segments: {len(rows)}")
        lines.append("")

        prev_stop_ms: int | None = None
        for i, r in enumerate(rows):
            lines.append(
                "segment[{i}] rec_name={rec} start_ms={start} stop_ms={stop} fs_hz={fs:.6f} n_frames={nf} "
                "frame_no_start={f0} frame_no_end={f1} frame_span={span} duration_clock_s={dur:.6f}".format(
                    i=i,
                    rec=r["rec_name"],
                    start=int(r["start_ms"]),
                    stop=int(r["stop_ms"]),
                    fs=float(r["sampling_hz"]),
                    nf=int(r["n_frames"]),
                    f0=int(r["frame_no_start"]),
                    f1=int(r["frame_no_end"]),
                    span=int(r["frame_span"]),
                    dur=float(r["duration_clock_s"]),
                )
            )
            if prev_stop_ms is not None:
                gap_s = float(int(r["start_ms"]) - int(prev_stop_ms)) / 1000.0
                lines.append(f"gap(prev->segment[{i}])_s={gap_s:.6f}")
            prev_stop_ms = int(r["stop_ms"])

        if rows:
            min_start = min(int(r["start_ms"]) for r in rows)
            max_stop = max(int(r["stop_ms"]) for r in rows)
            total_elapsed_s = float(max_stop - min_start) / 1000.0
            total_clock_s = sum(float(r["duration_clock_s"]) for r in rows)
            lines.append("")
            lines.append(f"overall_start_ms={min_start}")
            lines.append(f"overall_stop_ms={max_stop}")
            lines.append(f"overall_elapsed_s={total_elapsed_s:.6f}")
            lines.append(f"sum_segment_clock_s={total_clock_s:.6f}")
            lines.append(f"inter_segment_gap_s={max(0.0, total_elapsed_s - total_clock_s):.6f}")

        out_path.write_text("\n".join(lines) + "\n")
        logger.info("Wrote assay stats diagnostic: %s", out_path)
    except Exception as e:
        logger.warning("Failed to write assay_stats diagnostic: %s", e)


def _write_channels_metadata(
    *,
    h5_path: Path,
    stream_id: str,
    common_electrodes: list[int],
    out_path: Path,
    logger: logging.Logger,
    rec_names: Optional[list[str]] = None,
) -> None:
    try:
        import datetime as dt
        import json

        import numpy as np  # type: ignore[import-not-found]
        import spikeinterface.extractors as se  # type: ignore[import-not-found]
    except Exception as e:
        logger.warning("Skipping channels metadata generation: %s", e)
        return

    try:
        if not rec_names:
            mod = importlib.import_module("MEA_Analysis.IPNAnalysis.h5_utils")
            list_stream_recording_names = getattr(mod, "list_stream_recording_names")
            rec_names = list_stream_recording_names(h5_path=h5_path, stream_id=stream_id)
    except Exception as e:
        logger.warning("Failed to discover recording names for channels metadata: %s", e)
        return

    _ensure_maxwell_hdf5_plugin_path()
    common_set = {int(el) for el in common_electrodes}
    electrode_xy: dict[int, tuple[float, float]] = {}
    electrode_presence: dict[int, set[str]] = {}
    segments_payload: list[dict[str, object]] = []

    for rec_name in rec_names:
        rec = se.read_maxwell(file_path=str(h5_path), stream_id=stream_id, rec_name=str(rec_name))
        cv = rec.get_property("contact_vector")
        electrodes = np.asarray(cv["electrode"], dtype=int)
        x_um = np.asarray(cv["x"], dtype=float)
        y_um = np.asarray(cv["y"], dtype=float)

        if electrodes.size == 0:
            segments_payload.append(
                {
                    "rec_name": str(rec_name),
                    "n_channels": 0,
                    "common_electrodes": [],
                    "unique_electrodes": [],
                }
            )
            continue

        sort_idx = np.lexsort((x_um, y_um))
        seg_common: list[int] = []
        seg_unique: list[int] = []

        for idx in sort_idx.tolist():
            el = int(electrodes[int(idx)])
            x = float(x_um[int(idx)])
            y = float(y_um[int(idx)])
            if el not in electrode_xy:
                electrode_xy[el] = (x, y)
            electrode_presence.setdefault(el, set()).add(str(rec_name))
            if el in common_set:
                seg_common.append(el)
            else:
                seg_unique.append(el)

        segments_payload.append(
            {
                "rec_name": str(rec_name),
                "n_channels": int(electrodes.size),
                "n_common": int(len(seg_common)),
                "n_unique": int(len(seg_unique)),
                "common_electrodes": seg_common,
                "unique_electrodes": seg_unique,
            }
        )

    sorted_electrodes = sorted(
        electrode_xy.items(),
        key=lambda kv: (float(kv[1][1]), float(kv[1][0]), int(kv[0])),
    )

    channels_payload: list[dict[str, object]] = []
    common_sorted: list[int] = []
    for el, (x, y) in sorted_electrodes:
        is_common = int(el) in common_set
        if is_common:
            common_sorted.append(int(el))
        channels_payload.append(
            {
                "electrode": int(el),
                "x_um": float(x),
                "y_um": float(y),
                "is_common": bool(is_common),
                "segments": sorted(electrode_presence.get(int(el), set())),
            }
        )

    payload = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "h5_path": str(h5_path),
        "stream_id": str(stream_id),
        "n_segments": int(len(rec_names)),
        "segment_names": [str(rn) for rn in rec_names],
        "n_common_electrodes": int(len(common_sorted)),
        "common_electrodes": common_sorted,
        "n_total_electrodes": int(len(channels_payload)),
        "sort_order": "y_um_then_x_um",
        "segments": segments_payload,
        "channels": channels_payload,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    logger.info("Wrote channels metadata: %s", out_path)


def _mea_checkpoint_api():
    mod = importlib.import_module("MEA_Analysis.IPNAnalysis.multiseg_utils.preprocess_multiseg_h5.checkpoint")
    return {
        "ProcessingStage": getattr(mod, "ProcessingStage"),
        "compute_checkpoint_file": getattr(mod, "compute_checkpoint_file"),
        "exception_to_error_dict": getattr(mod, "exception_to_error_dict"),
        "load_checkpoint": getattr(mod, "load_checkpoint"),
        "save_checkpoint": getattr(mod, "save_checkpoint"),
    }


def run_preprocess_stage(
    *,
    h5_path: Path,
    stream_id: str,
    mea_output_root: Optional[Path],
    force_restart: bool,
    enable_checkpointing: bool = True,
    n_jobs: int = 8,
    plot_layouts: bool = True,
    temporal_resample_factor: Optional[int] = None,
    temporal_resample_rate_hz: Optional[int] = None,
    temporal_resample_margin_ms: float = 100.0,
    temporal_resample_dtype: Optional[str] = None,
    hdmea_geometry: Optional[dict[str, object]] = None,
    save_recording: bool = True,
    overwrite_saved_recording: bool = True,
    force_replot: bool = False,
    preprocess_root_relpath: Optional[str] = None,
    data_in_root: bool = True,
    centered_in_root: bool = True,
    preprocessed_in_root: bool = True,
    reports_in_root: bool = False,
    plot_centered_traces: bool = True,
    plot_preprocessed_traces: bool = True,
    plot_channel_layouts: bool = True,
    channel_layout_heatmap_enabled: bool = False,
    enable_centered_outputs: bool = True,
    enable_centered_recordings: bool = True,
    enable_preprocessed_recordings: bool = True,
    enable_centered_plots: bool = True,
    n_trace_channels: int = 24,
    preprocessed_recording_relpath: Optional[str] = None,
    concat_recording_relpath: Optional[str] = None,
    preprocessed_concat_relpath: Optional[str] = None,
    common_electrodes_relpath: Optional[str] = None,
    maxwell_epochs_relpath: Optional[str] = None,
    concat_epochs_relpath: Optional[str] = None,
    channels_relpath: Optional[str] = None,
    preprocess_config_relpath: Optional[str] = None,
    multiseg_preprocess_config_relpath: Optional[str] = None,
    assay_stats_relpath: Optional[str] = None,
    multiseg_preprocess_outputs_relpath: Optional[str] = None,
    centered_segments_relpath: Optional[str] = None,
    preprocessed_segments_relpath: Optional[str] = None,
    channel_layouts_relpath: Optional[str] = None,
    channel_layout_heatmap_relpath: Optional[str] = None,
    segment_traces_relpath: Optional[str] = None,
    preprocessed_segment_traces_relpath: Optional[str] = None,
    concat_cluster_reps_relpath: Optional[str] = None,
    preprocessed_concat_cluster_reps_relpath: Optional[str] = None,
    console_log_level: int | str | None = None,
    file_log_level: int | str | None = None,
    logger: Optional[logging.Logger] = None,
) -> tuple[object, list[int]]:
    ckpt_api = _mea_checkpoint_api()
    ProcessingStage = ckpt_api["ProcessingStage"]
    compute_checkpoint_file = ckpt_api["compute_checkpoint_file"]
    exception_to_error_dict = ckpt_api["exception_to_error_dict"]
    load_checkpoint = ckpt_api["load_checkpoint"]
    save_checkpoint = ckpt_api["save_checkpoint"]

    plan = build_preprocess_plan(h5_path=h5_path, stream_id=stream_id)

    logger = logger or logging.getLogger("axon_reconstructor.preprocess")
    well_out_dir = None
    if mea_output_root is not None and (plot_layouts or save_recording or enable_checkpointing):
        well_out_dir = compute_mea_analysis_output_dir(
            output_root=mea_output_root,
            data_file=h5_path,
            well=stream_id,
        )
        try:
            log_file = compute_pipeline_log_file(
                well_out_dir=well_out_dir,
                data_file=h5_path,
                stream_id=stream_id,
            )
            logger = setup_pipeline_logger(
                log_file=log_file,
                logger_name=f"{__name__}[stream={stream_id}][phase=preprocess]",
                verbose=True,
                console_level=console_log_level,
                file_level=file_log_level,
                stream=sys.__stdout__,
            )
        except Exception:
            pass

    logger.debug("Debug mode active: module-path logs enabled; external stdout/stderr capture enabled.")
    logger.info(
        "Preprocess execution flags: force_restart=%s force_replot=%s save_recording=%s",
        bool(force_restart),
        bool(force_replot),
        bool(save_recording),
    )
    effective_overwrite_saved_recording = bool(overwrite_saved_recording)
    if bool(force_replot) and not bool(force_restart) and bool(save_recording):
        # Replot requests should refresh diagnostics without forcing recording rewrite.
        effective_overwrite_saved_recording = False
        logger.info("force_replot=True: preserving existing preprocessed recording on disk")

    if plan.cfg_files:
        logger.info("Discovered %d cfg files next to %s", len(plan.cfg_files), plan.h5_path)
    else:
        logger.info("No .cfg files discovered next to %s; using contact_vector electrodes", plan.h5_path)

    plot_dir = None
    if plot_layouts:
        if well_out_dir is None:
            logger.warning("plot_layouts=True but mea_output_root is not set; skipping plots")

    if save_recording and well_out_dir is None:
        logger.warning("save_recording=True but mea_output_root is not set; skipping recording save")
    elif not save_recording:
        logger.info("save_recording=False; preprocessing artifacts will not be persisted to disk")

    checkpoint_file = None
    checkpoint_state = None
    preprocess_dir = None
    recording_dir = None
    concat_recording_dir = None
    preprocessed_concat_dir = None
    common_el_path = None
    epoch_maxwell_path = None
    epoch_concat_path = None
    channel_layouts_path = None
    channel_layout_heatmap_path = None
    segment_traces_path = None
    preprocessed_segment_traces_path = None
    concat_cluster_reps_path = None
    preprocessed_concat_cluster_reps_path = None
    multiseg_preprocess_outputs_dir = None
    centered_segments_dir = None
    preprocessed_segments_dir = None
    assay_stats_path = None
    channels_path = None
    preprocess_root_rel = str(preprocess_root_relpath).strip() if preprocess_root_relpath is not None else PREPROCESS_OUTPUTS_DIRNAME
    data_base_rel = preprocess_root_rel if bool(data_in_root) else "."
    centered_base_rel = f"{preprocess_root_rel}/centered" if bool(centered_in_root) else "centered"
    preprocessed_base_rel = f"{preprocess_root_rel}/preprocessed" if bool(preprocessed_in_root) else "preprocessed"
    reports_base_rel = f"{preprocess_root_rel}/reports" if bool(reports_in_root) else "reports"

    def _group_override_rel(override_rel: Optional[str], *, in_root: bool) -> Optional[str]:
        if override_rel is None:
            return None
        rel = str(override_rel).strip()
        if rel == "":
            return None
        if not bool(in_root):
            return rel
        candidate = Path(rel)
        if candidate.is_absolute() or ".." in candidate.parts:
            return rel
        root_parts = Path(preprocess_root_rel).parts
        cand_parts = candidate.parts
        if len(root_parts) > 0 and tuple(cand_parts[: len(root_parts)]) == tuple(root_parts):
            return rel
        return str(Path(preprocess_root_rel) / candidate)

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
        preprocess_dir = _resolve_relative_artifact_path(
            well_out_dir=well_out_dir,
            override_rel=preprocess_root_relpath,
            default_rel=PREPROCESS_OUTPUTS_DIRNAME,
        )
        if plot_layouts:
            plot_dir = preprocess_dir
            plot_dir.mkdir(parents=True, exist_ok=True)
        multiseg_preprocess_outputs_dir = _resolve_relative_artifact_path(
            well_out_dir=well_out_dir,
            override_rel=multiseg_preprocess_outputs_relpath,
            default_rel=f"{preprocess_root_rel}",
        )
        concat_recording_dir = _resolve_relative_artifact_path(
            well_out_dir=well_out_dir,
            override_rel=_group_override_rel(concat_recording_relpath, in_root=bool(centered_in_root)),
            default_rel=f"{centered_base_rel}/concat_recording",
        )
        preprocessed_concat_dir = _resolve_relative_artifact_path(
            well_out_dir=well_out_dir,
            override_rel=_group_override_rel(preprocessed_concat_relpath, in_root=bool(preprocessed_in_root)),
            default_rel=f"{preprocessed_base_rel}/concat_recording",
        )
        centered_segments_dir = _resolve_relative_artifact_path(
            well_out_dir=well_out_dir,
            override_rel=_group_override_rel(centered_segments_relpath, in_root=bool(centered_in_root)),
            default_rel=f"{centered_base_rel}/segments",
        )
        preprocessed_segments_dir = _resolve_relative_artifact_path(
            well_out_dir=well_out_dir,
            override_rel=_group_override_rel(preprocessed_segments_relpath, in_root=bool(preprocessed_in_root)),
            default_rel=f"{preprocessed_base_rel}/segments",
        )
        recording_dir = _resolve_relative_artifact_path(
            well_out_dir=well_out_dir,
            override_rel=_group_override_rel(
                (preprocessed_concat_relpath if preprocessed_concat_relpath is not None else preprocessed_recording_relpath),
                in_root=bool(preprocessed_in_root),
            ),
            default_rel=f"{preprocessed_base_rel}/concat_recording",
        )
        if preprocessed_concat_relpath is None:
            preprocessed_concat_dir = recording_dir
        common_el_path = _resolve_relative_artifact_path(
            well_out_dir=well_out_dir,
            override_rel=_group_override_rel(common_electrodes_relpath, in_root=bool(data_in_root)),
            default_rel=("common_electrodes.npy" if data_base_rel == "." else f"{data_base_rel}/common_electrodes.npy"),
        )
        epoch_maxwell_path = _resolve_relative_artifact_path(
            well_out_dir=well_out_dir,
            override_rel=_group_override_rel(maxwell_epochs_relpath, in_root=bool(reports_in_root)),
            default_rel=f"{reports_base_rel}/maxwell_contiguous_epochs.json",
        )
        epoch_concat_path = _resolve_relative_artifact_path(
            well_out_dir=well_out_dir,
            override_rel=_group_override_rel(concat_epochs_relpath, in_root=bool(reports_in_root)),
            default_rel=f"{reports_base_rel}/concatenation_stitch_epochs.json",
        )
        channel_layouts_path = _resolve_relative_artifact_path(
            well_out_dir=well_out_dir,
            override_rel=_group_override_rel(channel_layouts_relpath, in_root=bool(centered_in_root)),
            default_rel=f"{centered_base_rel}/channel_layouts",
        )
        channel_layout_heatmap_path = _resolve_relative_artifact_path(
            well_out_dir=well_out_dir,
            override_rel=_group_override_rel(channel_layout_heatmap_relpath, in_root=bool(reports_in_root)),
            default_rel=f"{reports_base_rel}/channel_layout_heatmap.png",
        )
        segment_traces_path = _resolve_relative_artifact_path(
            well_out_dir=well_out_dir,
            override_rel=_group_override_rel(segment_traces_relpath, in_root=bool(centered_in_root)),
            default_rel=f"{centered_base_rel}/segment_traces",
        )
        preprocessed_segment_traces_path = _resolve_relative_artifact_path(
            well_out_dir=well_out_dir,
            override_rel=_group_override_rel(preprocessed_segment_traces_relpath, in_root=bool(preprocessed_in_root)),
            default_rel=f"{preprocessed_base_rel}/segment_traces",
        )
        concat_cluster_reps_path = _resolve_relative_artifact_path(
            well_out_dir=well_out_dir,
            override_rel=_group_override_rel(concat_cluster_reps_relpath, in_root=bool(centered_in_root)),
            default_rel=f"{centered_base_rel}/concat_cluster_reps.png",
        )
        preprocessed_concat_cluster_reps_path = _resolve_relative_artifact_path(
            well_out_dir=well_out_dir,
            override_rel=_group_override_rel(
                preprocessed_concat_cluster_reps_relpath,
                in_root=bool(preprocessed_in_root),
            ),
            default_rel=f"{preprocessed_base_rel}/concat_cluster_reps.png",
        )
        assay_stats_path = _resolve_relative_artifact_path(
            well_out_dir=well_out_dir,
            override_rel=_group_override_rel(assay_stats_relpath, in_root=bool(reports_in_root)),
            default_rel=f"{reports_base_rel}/assay_stats.txt",
        )
        channels_path = _resolve_relative_artifact_path(
            well_out_dir=well_out_dir,
            override_rel=_group_override_rel(channels_relpath, in_root=bool(reports_in_root)),
            default_rel=f"{reports_base_rel}/channels.json",
        )

    preprocess_cfg_path = (
        _resolve_relative_artifact_path(
            well_out_dir=well_out_dir,
            override_rel=_group_override_rel(
                (multiseg_preprocess_config_relpath if multiseg_preprocess_config_relpath is not None else preprocess_config_relpath),
                in_root=bool(reports_in_root),
            ),
            default_rel=f"{reports_base_rel}/preprocess_config.json",
        )
        if well_out_dir is not None
        else None
    )
    if preprocess_dir is not None:
        logger.info("Preprocess diagnostics output: %s", preprocess_dir)
    if multiseg_preprocess_outputs_dir is not None:
        logger.info("Preprocess outputs root: %s", multiseg_preprocess_outputs_dir)

    centered_plots_enabled = bool(enable_centered_outputs) and bool(enable_centered_plots)
    channel_layouts_enabled = bool(centered_plots_enabled) and bool(plot_channel_layouts)
    channel_layout_heatmap_enabled = bool(centered_plots_enabled) and bool(channel_layout_heatmap_enabled)
    centered_recordings_enabled = bool(enable_centered_outputs) and bool(enable_centered_recordings)
    preprocessed_recordings_enabled = bool(enable_preprocessed_recordings)
    effective_plot_centered_traces = bool(plot_centered_traces) and centered_plots_enabled
    if not centered_recordings_enabled:
        concat_recording_dir = None
        centered_segments_dir = None
    if not centered_plots_enabled:
        channel_layouts_path = None
        segment_traces_path = None
        concat_cluster_reps_path = None
        channel_layout_heatmap_path = None
    if not channel_layouts_enabled:
        channel_layouts_path = None
    if not channel_layout_heatmap_enabled:
        channel_layout_heatmap_path = None

    if bool(force_replot):
        logger.info("force_replot=True: refreshing plotting diagnostics artifacts")
        if centered_plots_enabled:
            _remove_path_if_exists(path=channel_layouts_path, logger=logger)
            _remove_path_if_exists(path=segment_traces_path, logger=logger)
            _remove_path_if_exists(path=concat_cluster_reps_path, logger=logger)
            _remove_path_if_exists(path=channel_layout_heatmap_path, logger=logger)
        _remove_path_if_exists(path=preprocessed_segment_traces_path, logger=logger)
        _remove_path_if_exists(path=preprocessed_concat_cluster_reps_path, logger=logger)

    expected_outputs: dict[str, Optional[Path]] = {
        "preprocessed_recording": (recording_dir if preprocessed_recordings_enabled else None),
        "concat_recording": concat_recording_dir,
        "preprocessed_concat": (preprocessed_concat_dir if preprocessed_recordings_enabled else None),
        "multiseg_preprocess_outputs": multiseg_preprocess_outputs_dir,
        "centered_segments": centered_segments_dir,
        "preprocessed_segments": (preprocessed_segments_dir if preprocessed_recordings_enabled else None),
        "common_electrodes": common_el_path,
        "maxwell_epochs": epoch_maxwell_path,
        "concat_epochs": epoch_concat_path,
        "preprocess_config": preprocess_cfg_path,
    }
    if plot_dir is not None and centered_plots_enabled:
        expected_outputs.update(
            {
                "channel_layouts": channel_layouts_path,
                "channel_layout_heatmap": channel_layout_heatmap_path,
                "concat_cluster_reps": concat_cluster_reps_path,
                "segment_traces": segment_traces_path,
            }
        )
    if plot_dir is not None:
        expected_outputs.update(
            {
                "preprocessed_segment_traces": preprocessed_segment_traces_path,
                "preprocessed_concat_cluster_reps": preprocessed_concat_cluster_reps_path,
            }
        )
    if assay_stats_path is not None:
        expected_outputs["assay_stats"] = assay_stats_path
    if channels_path is not None:
        expected_outputs["channels"] = channels_path
    for label, path in expected_outputs.items():
        if path is not None:
            logger.info("Expected preprocess output [%s]: %s", label, path)

    if assay_stats_path is not None:
        _write_assay_stats_diagnostic(
            h5_path=plan.h5_path,
            stream_id=plan.stream_id,
            out_path=assay_stats_path,
            logger=logger,
        )

    requested_cfg = {
        "temporal_resample_factor": (int(temporal_resample_factor) if temporal_resample_factor is not None else None),
        "temporal_resample_rate_hz": (int(temporal_resample_rate_hz) if temporal_resample_rate_hz is not None else None),
        "temporal_resample_margin_ms": float(temporal_resample_margin_ms),
        "temporal_resample_dtype": (str(temporal_resample_dtype) if temporal_resample_dtype is not None else None),
        "hdmea_geometry": (dict(hdmea_geometry) if isinstance(hdmea_geometry, dict) else None),
    }

    if (
        save_recording
        and not bool(force_replot)
        and not effective_overwrite_saved_recording
        and checkpoint_state is not None
        and checkpoint_state.stage >= ProcessingStage.PREPROCESSING_COMPLETE.value
        and recording_dir is not None
        and common_el_path is not None
        and recording_dir.exists()
        and common_el_path.exists()
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
            if channels_path is not None:
                _write_channels_metadata(
                    h5_path=plan.h5_path,
                    stream_id=plan.stream_id,
                    common_electrodes=[int(el) for el in common_el],
                    out_path=channels_path,
                    logger=logger,
                    rec_names=None,
                )
            logger.info("Resuming: loaded preprocessed recording from %s", recording_dir)
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
        artifacts: dict[str, object] = {}
        with _capture_external_output(logger=logger):
            runtime_result = build_concatenated_recording(
                h5_path=plan.h5_path,
                stream_id=plan.stream_id,
                n_jobs=n_jobs,
                plot_output_dir=plot_dir,
                channel_layouts_output_dir=channel_layouts_path,
                channel_layout_heatmap_output_path=channel_layout_heatmap_path,
                channel_layout_heatmap_enabled=bool(channel_layout_heatmap_enabled),
                plot_channel_layouts=bool(channel_layouts_enabled),
                segment_traces_output_dir=segment_traces_path,
                preprocessed_segment_traces_output_dir=preprocessed_segment_traces_path,
                concat_cluster_reps_output_path=concat_cluster_reps_path,
                preprocessed_concat_cluster_reps_output_path=preprocessed_concat_cluster_reps_path,
                n_trace_channels=int(n_trace_channels),
                plot_centered_traces=bool(effective_plot_centered_traces),
                plot_preprocessed_traces=bool(plot_preprocessed_traces),
                epoch_markers_output_dir=(epoch_maxwell_path.parent if epoch_maxwell_path is not None else (preprocess_dir if preprocess_dir is not None else plot_dir)),
                temporal_resample_factor=(int(temporal_resample_factor) if temporal_resample_factor is not None else None),
                temporal_resample_rate_hz=(int(temporal_resample_rate_hz) if temporal_resample_rate_hz is not None else None),
                temporal_resample_margin_ms=float(temporal_resample_margin_ms),
                temporal_resample_dtype=(str(temporal_resample_dtype) if temporal_resample_dtype is not None else None),
                hdmea_geometry=(dict(hdmea_geometry) if isinstance(hdmea_geometry, dict) else None),
                return_artifacts=True,
                logger=logger,
            )
        if isinstance(runtime_result, tuple) and len(runtime_result) == 3:
            multirec, common_el, artifacts = runtime_result
        else:
            multirec, common_el = runtime_result  # type: ignore[misc]
            artifacts = {}
        logger.info("Preprocessed recording built; common electrodes=%d", len(common_el))
        if channels_path is not None:
            artifact_rec_names = artifacts.get("rec_names") if isinstance(artifacts, dict) else None
            _write_channels_metadata(
                h5_path=plan.h5_path,
                stream_id=plan.stream_id,
                common_electrodes=[int(el) for el in common_el],
                out_path=channels_path,
                logger=logger,
                rec_names=(artifact_rec_names if isinstance(artifact_rec_names, list) else None),
            )
        if epoch_maxwell_path is not None and epoch_concat_path is not None:
            generated_maxwell = epoch_maxwell_path.parent / f"maxwell_contiguous_epochs_{stream_id}.json"
            generated_concat = epoch_concat_path.parent / f"concatenation_stitch_epochs_{stream_id}.json"
            if generated_maxwell.exists() and generated_maxwell != epoch_maxwell_path:
                epoch_maxwell_path.parent.mkdir(parents=True, exist_ok=True)
                generated_maxwell.replace(epoch_maxwell_path)
            if generated_concat.exists() and generated_concat != epoch_concat_path:
                epoch_concat_path.parent.mkdir(parents=True, exist_ok=True)
                generated_concat.replace(epoch_concat_path)
            if epoch_maxwell_path.exists():
                logger.info("Epoch markers (Maxwell): %s", epoch_maxwell_path)
            if epoch_concat_path.exists():
                logger.info("Epoch markers (Concat stitches): %s", epoch_concat_path)
    except Exception as e:
        logger.exception("Preprocess runtime failed")
        if checkpoint_file is not None and checkpoint_state is not None:
            save_checkpoint(
                checkpoint_file=checkpoint_file,
                state=checkpoint_state,
                stage=ProcessingStage.NOT_STARTED,
                failed_stage=ProcessingStage.PREPROCESSING.name,
                error=exception_to_error_dict(e),
            )
        raise

    if save_recording and well_out_dir is not None:
        assert preprocess_dir is not None
        assert recording_dir is not None
        assert common_el_path is not None
        preprocess_dir.mkdir(parents=True, exist_ok=True)

        if preprocess_cfg_path is not None:
            try:
                import datetime as dt
                import json

                preprocess_cfg_path.parent.mkdir(parents=True, exist_ok=True)
                preprocess_cfg_path.write_text(
                    json.dumps(
                        {
                            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                            "requested_cfg": requested_cfg,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
            except Exception as e:
                logger.warning("Failed to write preprocess_config.json: %s", e)
            try:
                import numpy as np  # type: ignore[import-not-found]

                if preprocessed_recordings_enabled:
                    _save_recording_artifact(
                        recording=multirec,
                        out_dir=recording_dir,
                        n_jobs=n_jobs,
                        chunk_duration="1s",
                        overwrite=bool(effective_overwrite_saved_recording),
                        logger=logger,
                        label="preprocessed recording",
                    )

                    if preprocessed_concat_dir is not None and preprocessed_concat_dir != recording_dir:
                        _save_recording_artifact(
                            recording=multirec,
                            out_dir=preprocessed_concat_dir,
                            n_jobs=n_jobs,
                            chunk_duration="1s",
                            overwrite=bool(effective_overwrite_saved_recording),
                            logger=logger,
                            label="preprocessed concat recording",
                        )

                centered_concat = artifacts.get("centered_concat_recording") if isinstance(artifacts, dict) else None
                if centered_recordings_enabled and concat_recording_dir is not None and centered_concat is not None:
                    _save_recording_artifact(
                        recording=centered_concat,
                        out_dir=concat_recording_dir,
                        n_jobs=n_jobs,
                        chunk_duration="1s",
                        overwrite=bool(effective_overwrite_saved_recording),
                        logger=logger,
                        label="centered concat recording",
                    )

                rec_names = artifacts.get("rec_names") if isinstance(artifacts, dict) else None
                centered_rec_list = artifacts.get("centered_rec_list") if isinstance(artifacts, dict) else None
                preprocessed_rec_list = artifacts.get("preprocessed_rec_list") if isinstance(artifacts, dict) else None
                if (
                    centered_recordings_enabled
                    and
                    centered_segments_dir is not None
                    and isinstance(rec_names, list)
                    and isinstance(centered_rec_list, list)
                ):
                    for rec_name, rec in zip(rec_names, centered_rec_list, strict=False):
                        seg_path = centered_segments_dir / str(rec_name)
                        _save_recording_artifact(
                            recording=rec,
                            out_dir=seg_path,
                            n_jobs=n_jobs,
                            chunk_duration="1s",
                            overwrite=bool(effective_overwrite_saved_recording),
                            logger=logger,
                            label=f"centered segment {rec_name}",
                        )

                if (
                    preprocessed_recordings_enabled
                    and
                    preprocessed_segments_dir is not None
                    and isinstance(rec_names, list)
                    and isinstance(preprocessed_rec_list, list)
                ):
                    for rec_name, rec in zip(rec_names, preprocessed_rec_list, strict=False):
                        seg_path = preprocessed_segments_dir / str(rec_name)
                        _save_recording_artifact(
                            recording=rec,
                            out_dir=seg_path,
                            n_jobs=n_jobs,
                            chunk_duration="1s",
                            overwrite=bool(effective_overwrite_saved_recording),
                            logger=logger,
                            label=f"preprocessed segment {rec_name}",
                        )

                common_el_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(common_el_path, np.asarray(common_el, dtype=np.int64))
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
