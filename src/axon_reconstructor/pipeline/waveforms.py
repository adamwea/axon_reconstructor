from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .checkpointing import (
    ProcessingStage,
    compute_checkpoint_file,
    exception_to_error_dict,
    load_checkpoint,
    save_checkpoint,
)
from .pipeline_logging import compute_pipeline_log_file, setup_pipeline_logger
from .pipeline_driver import PREPROCESS_OUTPUTS_DIRNAME, _compute_mea_analysis_output_dir
from .raw_preprocessing.raw_preprocessing import _ensure_maxwell_hdf5_plugin_path


WAVEFORMS_OUTPUTS_DIRNAME = "waveforms_outputs"


def _compute_waveforms_checkpoint_file(*, well_out_dir: Path, h5_path: Path, stream_id: str) -> Path:
    """Use a dedicated checkpoint file for waveforms.

    The MEA_Analysis-style stage machine (PREPROCESSING..REPORTS_COMPLETE) doesn't
    include waveforms, so storing waveforms progress in the main checkpoint can
    accidentally *regress* stage numbers (e.g. REPORTS_COMPLETE -> ANALYZER_COMPLETE).
    """

    main_ckpt = compute_checkpoint_file(output_dir=well_out_dir, file_path=h5_path, stream_id=stream_id)
    name = main_ckpt.name
    if name.endswith("_checkpoint.json"):
        name = name[: -len("_checkpoint.json")] + "_waveforms_checkpoint.json"
    else:
        name = main_ckpt.stem + "_waveforms_checkpoint.json"
    return main_ckpt.with_name(name)


@dataclass(frozen=True)
class WaveformExtractInputs:
    h5_path: Path
    stream_id: str
    mea_output_root: Path
    sorter: str = "kilosort4"

    # Waveform window. If None, try to infer from trigger_pre/post.
    ms_before: Optional[float] = None
    ms_after: Optional[float] = None

    n_jobs: int = 8
    max_spikes_per_unit: Optional[int] = None

    # If True, also extract waveforms per concatenated segment.
    per_segment: bool = True

    # If True, per-segment waveforms are extracted only on channels that were
    # excluded during concatenation (i.e. not in the common-electrode intersection).
    # This avoids duplicating waveforms for the common channels already covered by
    # concat_waveforms.
    per_segment_only_additional_channels: bool = True

    # Resume/overwrite controls
    force_restart: bool = False

    # If True, drop spikes whose waveform window would cross Maxwell snippet boundaries.
    filter_by_maxwell_epochs: bool = True

    # Plotting
    plot_waveforms_grid_pdf: bool = True


@dataclass(frozen=True)
class WaveformExtractOutputs:
    well_out_dir: Path
    waveforms_out_dir: Path
    concat_waveforms_dir: Path
    segment_waveforms_dir: Optional[Path]
    params_json: Path
    filtering_json: Path
    waveforms_grid_pdf: Optional[Path]
    spikesorting_waveforms_grid_pdf: Optional[Path]


def _write_waveforms_grid_pdf(
    *,
    waveforms_folder: Path,
    pdf_path: Path,
    unit_ids: Optional[list[Any]] = None,
) -> None:
    """Write a multi-page PDF of per-unit waveforms.

    This is an (intentionally) very close copy of MEA_Analysis:
    MEA_Analysis/IPNAnalysis/mea_analysis_routine.py::_plot_waveforms_grid.

    We adapt it to load from a SpikeInterface `SortingAnalyzer` folder.

    For backwards compatibility, this will also try to load a legacy waveforms folder
    (WaveformExtractor) if a SortingAnalyzer cannot be loaded.
    """

    try:
        import logging

        import numpy as np  # type: ignore[import-not-found]
        import spikeinterface.full as si  # type: ignore[import-not-found]

        # Avoid extremely verbose font/debug output when the pipeline logger is in DEBUG.
        # This needs to happen *before* importing matplotlib, because matplotlib can emit
        # DEBUG logs during import.
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import matplotlib.backends.backend_pdf as pdf

        # Prefer the exact MEA_Analysis scalebar implementation.
        # In MEA_Analysis it's a local module next to mea_analysis_routine.py, so depending
        # on how MEA_Analysis is installed/imported it may not be importable as `scalebury`.
        try:
            from scalebury import add_scalebar  # type: ignore[import-not-found]
        except Exception:
            try:
                from MEA_Analysis.IPNAnalysis.scalebury import add_scalebar  # type: ignore[import-not-found]
            except Exception:
                # Fallback: inline copy of MEA_Analysis/IPNAnalysis/scalebury.py (PSF license).
                from matplotlib.offsetbox import AnchoredOffsetbox

                class AnchoredScaleBar(AnchoredOffsetbox):
                    def __init__(
                        self,
                        transform,
                        sizex=0,
                        sizey=0,
                        labelx=None,
                        labely=None,
                        loc=4,
                        pad=0.1,
                        borderpad=0.1,
                        sep=2,
                        prop=None,
                        barcolor="black",
                        barwidth=None,
                        **kwargs,
                    ):
                        from matplotlib.patches import Rectangle
                        from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, TextArea

                        bars = AuxTransformBox(transform)
                        if sizex:
                            bars.add_artist(
                                Rectangle((0, 0), sizex, 0, ec=barcolor, lw=barwidth, fc="none")
                            )
                        if sizey:
                            bars.add_artist(
                                Rectangle((0, 0), 0, sizey, ec=barcolor, lw=barwidth, fc="none")
                            )

                        if sizex and labelx:
                            self.xlabel = TextArea(labelx)
                            bars = VPacker(children=[bars, self.xlabel], align="center", pad=0, sep=sep)
                        if sizey and labely:
                            self.ylabel = TextArea(labely)
                            bars = HPacker(children=[self.ylabel, bars], align="center", pad=0, sep=sep)

                        AnchoredOffsetbox.__init__(
                            self,
                            loc,
                            pad=pad,
                            borderpad=borderpad,
                            child=bars,
                            prop=prop,
                            frameon=False,
                            **kwargs,
                        )

                def add_scalebar(ax, matchx=True, matchy=True, hidex=True, hidey=True, **kwargs):
                    def f(axis):
                        l = axis.get_majorticklocs()
                        return len(l) > 1 and (l[1] - l[0])

                    if matchx:
                        kwargs["sizex"] = f(ax.xaxis)
                        kwargs["labelx"] = str(kwargs["sizex"])
                    if matchy:
                        kwargs["sizey"] = f(ax.yaxis)
                        kwargs["labely"] = str(kwargs["sizey"])

                    sb = AnchoredScaleBar(ax.transData, **kwargs)
                    ax.add_artist(sb)

                    if hidex:
                        ax.xaxis.set_visible(False)
                    if hidey:
                        ax.yaxis.set_visible(False)
                    if hidex and hidey:
                        ax.set_frame_on(False)

                    return sb
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Plotting waveforms grid requires numpy/matplotlib/spikeinterface") from e

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    analyzer = None
    waveforms_ext = None
    legacy_we = None

    try:
        analyzer = si.load_sorting_analyzer(waveforms_folder)
        waveforms_ext = analyzer.get_extension("waveforms")
    except Exception:
        analyzer = None
        waveforms_ext = None

    if analyzer is None or waveforms_ext is None:
        # Legacy fallback: old WaveformExtractor folder.
        legacy_we = si.load_waveforms(waveforms_folder)

    if unit_ids is None:
        if analyzer is not None:
            unit_ids = list(analyzer.sorting.unit_ids)
        else:
            unit_ids = list(legacy_we.sorting.unit_ids)
    if len(unit_ids) == 0:
        return

    if analyzer is not None:
        fs = float(analyzer.sampling_frequency)
    else:
        fs = float(legacy_we.recording.get_sampling_frequency())

    with pdf.PdfPages(pdf_path) as pdf_doc:
        units_per_page = 12
        for i in range(0, len(unit_ids), units_per_page):
            batch = unit_ids[i : i + units_per_page]
            fig, axes = plt.subplots(3, 4, figsize=(12, 9))
            axes = axes.flatten()

            for ax, uid in zip(axes, batch, strict=False):
                try:
                    # Remove standard axes/ticks (MEA_Analysis-style QC panels).
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                    for spine in ax.spines.values():
                        spine.set_visible(False)

                    if analyzer is not None:
                        wf = waveforms_ext.get_waveforms_one_unit(unit_id=uid)
                        unit_channel_ids = None
                        try:
                            if analyzer.sparsity is not None:
                                unit_channel_ids = analyzer.sparsity.unit_id_to_channel_ids.get(uid)
                        except Exception:
                            unit_channel_ids = None
                    else:
                        wf = legacy_we.get_waveforms(uid)
                        unit_channel_ids = None
                        try:
                            unit_channel_ids = list(legacy_we.recording.get_channel_ids())
                        except Exception:
                            unit_channel_ids = None
                    if wf is None or wf.shape[0] == 0:
                        ax.axis("off")
                        continue

                    mean_wf = np.mean(wf, axis=0)
                    best_ch = int(np.argmin(np.min(mean_wf, axis=0)))

                    time_ms = np.arange(wf.shape[1]) / fs * 1000

                    n_spikes = int(wf.shape[0])
                    if n_spikes > 50:
                        indices = np.random.choice(n_spikes, 50, replace=False)
                        spikes_to_plot = wf[indices, :, best_ch]
                    else:
                        spikes_to_plot = wf[:, :, best_ch]

                    ax.plot(time_ms, spikes_to_plot.T, c="gray", lw=0.5, alpha=0.3)
                    ax.plot(time_ms, mean_wf[:, best_ch], c="red", lw=1.5)

                    ch_label = best_ch
                    try:
                        if unit_channel_ids is not None and best_ch < len(unit_channel_ids):
                            ch_label = int(unit_channel_ids[best_ch])
                    except Exception:
                        pass

                    ax.set_title(f"Unit {uid} | Ch {ch_label}", fontsize=10)

                    # Scale bar (1 ms x 50 uV)
                    if add_scalebar is not None:
                        try:
                            add_scalebar(
                                ax,
                                matchx=False,
                                matchy=False,
                                sizex=1.0,
                                labelx="1 ms",
                                sizey=50,
                                labely="50 µV",
                                loc=4,
                                hidex=True,
                                hidey=True,
                            )
                        except Exception:
                            # Keep plots usable even if scalebar fails.
                            pass
                except Exception:
                    ax.axis("off")

            for j in range(len(batch), len(axes)):
                axes[j].axis("off")

            pdf_doc.savefig(fig)
            plt.close(fig)


def _run_mea_analysis_style_curation(
    *,
    recording: Any,
    sorting: Any,
    output_dir: Path,
    n_jobs: int,
    ms_before: float,
    ms_after: float,
    force_restart: bool,
    logger: Any,
) -> tuple[list[Any], dict[str, Path]]:
    """Compute MEA_Analysis-like curation artifacts and return curated unit IDs.

    Writes the following files into output_dir:
    - qm_unfiltered.xlsx
    - tm_unfiltered.xlsx
    - metrics_curated.xlsx
    - rejection_log.xlsx
    - tm_curated.xlsx

    Returns (clean_units, paths).
    """

    try:
        import numpy as np  # type: ignore[import-not-found]
        import pandas as pd  # type: ignore[import-not-found]
        import spikeinterface.full as si  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Curation requires spikeinterface/numpy/pandas") from e

    output_dir.mkdir(parents=True, exist_ok=True)

    qm_unfiltered_xlsx = output_dir / "qm_unfiltered.xlsx"
    tm_unfiltered_xlsx = output_dir / "tm_unfiltered.xlsx"
    metrics_curated_xlsx = output_dir / "metrics_curated.xlsx"
    rejection_log_xlsx = output_dir / "rejection_log.xlsx"
    tm_curated_xlsx = output_dir / "tm_curated.xlsx"

    analyzer_dir = output_dir / "curation_analyzer"

    analyzer = None
    if analyzer_dir.exists() and not force_restart:
        try:
            analyzer = si.load_sorting_analyzer(analyzer_dir)
        except Exception:
            analyzer = None
    if analyzer is None:
        if analyzer_dir.exists():
            try:
                import shutil

                shutil.rmtree(analyzer_dir)
            except Exception:
                pass

        logger.info("Computing SortingAnalyzer for curation -> %s", analyzer_dir)

        sparsity = si.estimate_sparsity(
            sorting,
            recording,
            method="radius",
            radius_um=50,
            peak_sign="neg",
        )

        analyzer = si.create_sorting_analyzer(
            sorting,
            recording,
            format="binary_folder",
            folder=analyzer_dir,
            sparsity=sparsity,
            return_in_uV=True,
        )

        ext_list = [
            "random_spikes",
            "spike_amplitudes",
            "waveforms",
            "templates",
            "noise_levels",
            "quality_metrics",
            "template_metrics",
            "unit_locations",
        ]
        ext_params = {
            "waveforms": {"ms_before": float(ms_before), "ms_after": float(ms_after)},
            "unit_locations": {"method": "monopolar_triangulation"},
        }

        analyzer.compute(
            ext_list,
            extension_params=ext_params,
            verbose=False,
            n_jobs=int(n_jobs),
        )

    q_metrics = analyzer.get_extension("quality_metrics").get_data()
    t_metrics = analyzer.get_extension("template_metrics").get_data()
    locations = analyzer.get_extension("unit_locations").get_data()

    # Match MEA_Analysis: add unit locations into q_metrics.
    q_metrics = q_metrics.copy()
    q_metrics["loc_x"] = locations[:, 0]
    q_metrics["loc_y"] = locations[:, 1]

    q_metrics.to_excel(qm_unfiltered_xlsx)
    t_metrics.to_excel(tm_unfiltered_xlsx)

    # Apply the same curation logic as MEA_Analysis.
    clean_units: list[Any]
    try:
        from MEA_Analysis.IPNAnalysis.mea_analysis_routine import MEAPipeline  # type: ignore[import-not-found]

        # _apply_curation_logic does not depend on MEAPipeline instance state.
        dummy = MEAPipeline.__new__(MEAPipeline)
        clean_metrics, rejection_log = MEAPipeline._apply_curation_logic(dummy, q_metrics, None)
    except Exception:
        defaults = {
            "presence_ratio": 0.75,
            "rp_contamination": 0.15,
            "firing_rate": 0.05,
            "amplitude_median": -20,
            "amplitude_cv_median": 0.5,
        }
        keep_mask = np.ones(len(q_metrics), dtype=bool)
        rejections: list[dict[str, Any]] = []
        for idx, row in q_metrics.iterrows():
            reasons: list[str] = []
            if row.get("presence_ratio", 1) < defaults["presence_ratio"]:
                reasons.append("Low Presence")
            if row.get("rp_contamination", 0) > defaults["rp_contamination"]:
                reasons.append("High Contam")
            if row.get("firing_rate", 0) < defaults["firing_rate"]:
                reasons.append("Low FR")
            if row.get("amplitude_median", -100) > defaults["amplitude_median"]:
                reasons.append("Low Amp")
            if reasons:
                keep_mask[q_metrics.index.get_loc(row.name)] = False
                rejections.append({"unit_id": row.name, "reasons": "; ".join(reasons)})
        clean_metrics = q_metrics[keep_mask]
        rejection_log = pd.DataFrame(rejections)

    clean_units = list(clean_metrics.index.values)
    clean_metrics.to_excel(metrics_curated_xlsx)
    rejection_log.to_excel(rejection_log_xlsx)

    try:
        t_metrics.loc[clean_units].to_excel(tm_curated_xlsx)
    except Exception:
        # If something goes wrong with indexing, still emit the unfiltered TM.
        pass

    paths = {
        "qm_unfiltered.xlsx": qm_unfiltered_xlsx,
        "tm_unfiltered.xlsx": tm_unfiltered_xlsx,
        "metrics_curated.xlsx": metrics_curated_xlsx,
        "rejection_log.xlsx": rejection_log_xlsx,
        "tm_curated.xlsx": tm_curated_xlsx,
        "curation_analyzer_dir": analyzer_dir,
    }
    return clean_units, paths


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _infer_cutout_ms(*, h5_path: Path, stream_id: str, fs_hz: float) -> tuple[float, float]:
    """Infer ms_before/ms_after from trigger_pre/trigger_post when available."""

    try:
        from .raw_preprocessing.h5_helpers import _read_well_rec_frame_nos_and_trigger_settings

        # Grab settings from the first rec in the stream.
        import h5py  # type: ignore[import-not-found]

        with h5py.File(h5_path, "r") as h5:
            rec_name = list(h5["wells"][stream_id].keys())[0]

        info = _read_well_rec_frame_nos_and_trigger_settings(
            h5_path=h5_path,
            stream_id=stream_id,
            rec_name=rec_name,
        )
        pre = info.get("trigger_pre")
        post = info.get("trigger_post")
        if pre is None or post is None:
            raise ValueError("trigger_pre/post missing")

        # trigger_pre/post are in samples.
        ms_before = float(pre) / (fs_hz / 1000.0)
        ms_after = float(post) / (fs_hz / 1000.0)

        # Guard against nonsense.
        if not (0 < ms_before < 50 and 0 < ms_after < 50):
            raise ValueError(f"unexpected cutout ms: {ms_before}, {ms_after}")

        return ms_before, ms_after
    except Exception:
        # Fall back to Mandar defaults.
        return 1.0, 2.0


def _compute_waveforms_out_dir(*, output_root: Path, data_file: Path, well: str) -> Path:
    return _compute_mea_analysis_output_dir(output_root=output_root, data_file=data_file, well=well) / WAVEFORMS_OUTPUTS_DIRNAME


def _load_preprocessed_recording(*, well_out_dir: Path) -> Any:
    recording_dir = well_out_dir / PREPROCESS_OUTPUTS_DIRNAME / "preprocessed_recording"
    if not recording_dir.exists():
        raise FileNotFoundError(f"preprocessed_recording not found: {recording_dir}")

    import spikeinterface.full as si  # type: ignore[import-not-found]

    try:
        return si.load(recording_dir)
    except Exception:
        return si.load_extractor(recording_dir)


def _load_raw_segment_recording_full_channels(
    *,
    h5_path: Path,
    stream_id: str,
    rec_name: str,
    center_chunk_size: int = 10_000,
) -> Any:
    """Load a *single* raw Maxwell rec segment and keep its full channel set.

    This is the key difference vs. the concatenated recording:
    concatenation slices to the shared electrode intersection, which drops
    channels that are not present in every segment.

    In other words:
    - concat recording: fewer channels (intersection), but a single continuous time axis
    - raw segment recording: many more channels, but only one segment worth of time
    """

    try:
        import numpy as np  # type: ignore[import-not-found]
        import spikeinterface.full as si  # type: ignore[import-not-found]
        import spikeinterface.extractors as se  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("waveform extraction requires spikeinterface/numpy installed") from e

    _ensure_maxwell_hdf5_plugin_path()

    if hasattr(se, "read_maxwell"):
        rec = se.read_maxwell(file_path=str(h5_path), stream_id=stream_id, rec_name=rec_name)
    else:  # pragma: no cover
        rec = se.MaxwellRecordingExtractor(str(h5_path), stream_id=stream_id, rec_name=rec_name)

    chunk = min(center_chunk_size, int(rec.get_num_samples())) - 100
    chunk = max(int(chunk), 100)
    rec_centered = si.center(rec, chunk_size=int(chunk))

    # Rename channel_ids to electrode ids for identity stability.
    # This mirrors how preprocessing renames the *common* channels before concatenation,
    # but here we do it for the full channel set so downstream waveforms are tied to
    # physical electrode ids when possible.
    try:
        electrodes = np.asarray(rec_centered.get_property("contact_vector")["electrode"], dtype=int)
        if int(np.unique(electrodes).size) != int(electrodes.size):
            raise RuntimeError(f"Duplicate electrode ids in contact_vector for rec={rec_name}")
        rec_centered = rec_centered.rename_channels([int(e) for e in electrodes])
    except Exception:
        # Best-effort: proceed without renaming.
        pass

    return rec_centered


def _resolve_mea_sorter_output_dir(*, well_out_dir: Path) -> Path:
    # New step-style layout: spikesorting_outputs/sorter_output
    p = well_out_dir / "spikesorting_outputs" / "sorter_output"
    if p.exists():
        return p

    # Legacy fallback.
    legacy = well_out_dir / "sorter_output"
    if legacy.exists():
        return legacy

    return p


def _load_sorting_from_sorter_output_dir(*, sorter_output_dir: Path, sorter: str) -> Any:
    import spikeinterface.full as si  # type: ignore[import-not-found]

    # Prefer the generic entry point when available.
    if hasattr(si, "read_sorter_folder"):
        try:
            return si.read_sorter_folder(sorter_output_dir, sorter_name=sorter)
        except TypeError:
            # Older SpikeInterface versions use a positional arg (no keyword).
            return si.read_sorter_folder(sorter_output_dir, sorter)

    # Fallbacks: try to load as an extractor.
    try:
        return si.load_extractor(sorter_output_dir)
    except Exception:
        pass

    raise RuntimeError(
        f"Could not load sorting from sorter_output_dir={sorter_output_dir} (sorter={sorter})."
    )


def _epochs_to_intervals(epochs: list[dict]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for e in epochs:
        try:
            start = int(e["start_sample"])
            end = int(e["end_sample"])
        except Exception:
            continue
        if end > start:
            intervals.append((start, end))
    intervals.sort()
    return intervals


def _maxwell_epochs_to_segment_local_intervals(
    *,
    maxwell_epochs: list[dict],
    segment_index: int,
) -> list[tuple[int, int]]:
    """Convert maxwell epoch marker JSON to segment-local [start, end) intervals.

    The preprocessing artifact includes both concatenated coordinates and
    segment-local coordinates. For per-segment waveform extraction we want
    segment-local intervals so we can exclude waveform windows that would cross
    snippet boundaries *inside that segment*.
    """

    intervals: list[tuple[int, int]] = []
    for e in maxwell_epochs:
        try:
            if int(e.get("segment_index")) != int(segment_index):
                continue
            start = int(e["segment_start_sample"])
            end = int(e["segment_end_sample"])
        except Exception:
            continue
        if end > start:
            intervals.append((start, end))

    intervals.sort()
    return intervals


def _filter_spike_train_by_intervals(
    *,
    spike_train: list[int],
    intervals: list[tuple[int, int]],
    pre_samples: int,
    post_samples: int,
) -> tuple[list[int], int, int, list[int], list[int]]:
    """Keep spikes whose cutout window stays within some interval.

    Returns:
            (kept_spikes, removed_outside_interval, removed_window_crosses_interval_edge,
             removed_outside_spikes, removed_edge_spikes)
    """

    if not intervals:
                return spike_train, 0, 0, [], []

    kept: list[int] = []
    removed_outside = 0
    removed_edge = 0
    removed_outside_spikes: list[int] = []
    removed_edge_spikes: list[int] = []

    # Two-pointer scan because both spike_train and intervals are sorted.
    # We advance based on the spike time `t` (not t0), then classify removal
    # depending on whether the spike was in any interval vs. its window failing.
    i = 0
    for t in spike_train:
        t_int = int(t)
        t0 = t_int - int(pre_samples)
        t1 = t_int + int(post_samples)

        while i < len(intervals) and intervals[i][1] <= t_int:
            i += 1

        in_interval = False
        ok = False
        if i < len(intervals):
            start, end = intervals[i]
            if start <= t_int < end:
                in_interval = True
                if t0 >= start and t1 < end:
                    ok = True

        if ok:
            kept.append(t_int)
        else:
            if in_interval:
                removed_edge += 1
                removed_edge_spikes.append(t_int)
            else:
                removed_outside += 1
                removed_outside_spikes.append(t_int)

    return kept, removed_outside, removed_edge, removed_outside_spikes, removed_edge_spikes


def _write_wf_rejection_log_xlsx(
    *,
    wf_rejection_log_xlsx: Path,
    rows: list[dict[str, Any]],
    force_restart: bool,
    logger,
) -> None:
    """Write per-spike waveform rejection log.

    This tracks spikes removed during waveforms extraction filtering (epoch/out-of-epoch
    removals and edge/window violations). The output is designed to be joinable to the
    waveforms analyzers used by footprinting.
    """

    if wf_rejection_log_xlsx.exists() and (not force_restart):
        logger.info("wf_rejection_log.xlsx exists; not overwriting: %s", wf_rejection_log_xlsx)
        return

    import pandas as pd  # type: ignore[import-not-found]

    if not rows:
        df = pd.DataFrame(
            columns=[
                "scope",
                "source_name",
                "segment_index",
                "rec_name",
                "unit_id",
                "spike_sample_local",
                "spike_sample_concat",
                "spike_time_s",
                "reason",
                "stream_id",
                "sorter",
                "h5_path",
                "fs_hz",
                "ms_before",
                "ms_after",
                "pre_samples",
                "post_samples",
            ]
        )
    else:
        df = pd.DataFrame(rows)

    # Ensure stable column order.
    preferred_cols = [
        "scope",
        "source_name",
        "segment_index",
        "rec_name",
        "unit_id",
        "spike_sample_local",
        "spike_sample_concat",
        "spike_time_s",
        "reason",
        "stream_id",
        "sorter",
        "h5_path",
        "fs_hz",
        "ms_before",
        "ms_after",
        "pre_samples",
        "post_samples",
    ]
    cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
    df = df.loc[:, cols]

    # Summaries that are fast to read later (avoid loading giant sheets).
    summary_rows: list[dict[str, Any]] = []
    try:
        summary_rows.append({"metric": "n_rows", "value": int(len(df))})
        for key, label in [
            ("scope", "by_scope"),
            ("reason", "by_reason"),
            ("source_name", "by_source"),
        ]:
            if key in df.columns:
                vc = df[key].value_counts(dropna=False)
                for k, v in vc.items():
                    summary_rows.append({"metric": f"{label}:{k}", "value": int(v)})
    except Exception:
        pass

    summary_df = pd.DataFrame(summary_rows)

    unit_counts_df = None
    try:
        if not df.empty and {"source_name", "unit_id", "reason"}.issubset(set(df.columns)):
            unit_counts_df = (
                df.groupby(["scope", "source_name", "segment_index", "rec_name", "unit_id", "reason"], dropna=False)
                .size()
                .reset_index(name="n_rejected_spikes")
            )
    except Exception:
        unit_counts_df = None

    # Excel row limit is 1,048,576 including header.
    max_rows_per_sheet = 1_000_000
    try:
        import xlsxwriter  # type: ignore[import-not-found]

        engine = "xlsxwriter"
    except Exception:
        engine = "openpyxl"

    wf_rejection_log_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(wf_rejection_log_xlsx, engine=engine) as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        if unit_counts_df is not None:
            unit_counts_df.to_excel(writer, sheet_name="unit_counts", index=False)

        if df.empty:
            df.to_excel(writer, sheet_name="rejections_000", index=False)
        else:
            for i0 in range(0, len(df), max_rows_per_sheet):
                chunk = df.iloc[i0 : i0 + max_rows_per_sheet]
                sheet = f"rejections_{i0 // max_rows_per_sheet:03d}"
                chunk.to_excel(writer, sheet_name=sheet, index=False)

    logger.info("Wrote wf_rejection_log.xlsx -> %s (rows=%d)", wf_rejection_log_xlsx, int(len(df)))


def _load_wf_rejection_log_unit_counts(*, well_out_dir: Path, logger) -> tuple[Optional[list[dict[str, Any]]], Optional[Path]]:
    """Best-effort loader for waveforms-stage per-spike rejection counts.

    This reads the lightweight `unit_counts` sheet from:
      <well>/waveforms_outputs/wf_rejection_log.xlsx

    Returns a list of row dicts (JSON-friendly) so downstream stages can make
    pragmatic decisions without loading the huge per-spike sheets.

    NOTE: This is intentionally minimal and exists mainly to support temporary,
    downstream "monkey patches" while we design a principled way to carry
    spike-level waveform exclusions forward.
    """

    wf_rej_xlsx = well_out_dir / "waveforms_outputs" / "wf_rejection_log.xlsx"
    if not wf_rej_xlsx.exists():
        return None, None

    try:
        import pandas as pd  # type: ignore[import-not-found]

        df = pd.read_excel(wf_rej_xlsx, sheet_name="unit_counts")
        if df is None or df.empty:
            return [], wf_rej_xlsx

        # Convert to plain python scalars for JSON friendliness.
        rows: list[dict[str, Any]] = []
        for _, r in df.iterrows():
            try:
                rows.append({k: (v.item() if hasattr(v, "item") else v) for k, v in r.to_dict().items()})
            except Exception:
                continue
        return rows, wf_rej_xlsx
    except Exception as e:
        logger.warning("Failed reading wf_rejection_log.xlsx unit_counts: %s", e)
        return None, wf_rej_xlsx


def _to_numpy_sorting(*, unit_trains: dict[int, list[int]], fs_hz: float) -> Any:
    import numpy as np  # type: ignore[import-not-found]
    from spikeinterface.core import NumpySorting  # type: ignore[import-not-found]

    unit_ids = sorted(unit_trains.keys())
    all_times: list[int] = []
    all_labels: list[int] = []
    for u in unit_ids:
        times = unit_trains[u]
        all_times.extend(times)
        all_labels.extend([u] * len(times))

    if not all_times:
        # Empty sorting: still return a valid object with unit_ids.
        sorting = NumpySorting.from_unit_dict(unit_trains, sampling_frequency=float(fs_hz))
        return sorting

    times_arr = np.asarray(all_times, dtype=np.int64)
    labels_arr = np.asarray(all_labels, dtype=np.int64)

    # SpikeInterface expects sorted times within segment.
    order = np.argsort(times_arr, kind="mergesort")
    times_arr = times_arr[order]
    labels_arr = labels_arr[order]

    sorting = NumpySorting.from_times_labels(
        times_list=[times_arr],
        labels_list=[labels_arr],
        sampling_frequency=float(fs_hz),
    )
    return sorting


def extract_waveforms(
    *,
    inputs: WaveformExtractInputs,
    logger_name_prefix: str = "axon_reconstructor",
) -> WaveformExtractOutputs:
    """Extract waveforms from the preprocessed recording and sorter output.

    Produces:
      <well>/waveforms_outputs/concat_waveforms/
      <well>/waveforms_outputs/segment_waveforms/ (optional)
    <well>/waveforms_outputs/waveforms_grid_uncurated.pdf
    <well>/waveforms_outputs/waveforms_grid_curated.pdf (if curation succeeds)
    <well>/waveforms_outputs/qm_unfiltered.xlsx
    <well>/waveforms_outputs/tm_unfiltered.xlsx
    <well>/waveforms_outputs/metrics_curated.xlsx
    <well>/waveforms_outputs/tm_curated.xlsx
    <well>/waveforms_outputs/rejection_log.xlsx
    <well>/waveforms_outputs/wf_rejection_log.xlsx
    plus JSON summaries.

    Uses existing epoch marker JSONs (from preprocessing) to avoid extracting
    waveforms that cross Maxwell snippet discontinuities.
    """

    well_out_dir = _compute_mea_analysis_output_dir(
        output_root=inputs.mea_output_root,
        data_file=inputs.h5_path,
        well=inputs.stream_id,
    )

    log_file = compute_pipeline_log_file(well_out_dir=well_out_dir, data_file=inputs.h5_path, stream_id=inputs.stream_id)
    logger = setup_pipeline_logger(
        log_file=log_file,
        logger_name=f"{logger_name_prefix}.{inputs.stream_id}",
        verbose=True,
    )

    waveforms_out_dir = _compute_waveforms_out_dir(
        output_root=inputs.mea_output_root,
        data_file=inputs.h5_path,
        well=inputs.stream_id,
    )
    waveforms_out_dir.mkdir(parents=True, exist_ok=True)

    concat_waveforms_dir = waveforms_out_dir / "concat_waveforms"
    segment_waveforms_dir = waveforms_out_dir / "segment_waveforms" if inputs.per_segment else None

    params_json = waveforms_out_dir / "waveform_extraction_params.json"
    filtering_json = waveforms_out_dir / "waveform_filtering_summary.json"

    ckpt_file = _compute_waveforms_checkpoint_file(well_out_dir=well_out_dir, h5_path=inputs.h5_path, stream_id=inputs.stream_id)
    ckpt = load_checkpoint(
        checkpoint_file=ckpt_file,
        force_restart=bool(inputs.force_restart),
        output_dir=well_out_dir,
        file_path=inputs.h5_path,
        stream_id=inputs.stream_id,
    )

    # Resume shortcut: trust existing artifacts if present.
    if not inputs.force_restart and concat_waveforms_dir.exists():
        logger.info("Resuming waveforms: existing outputs found at %s", concat_waveforms_dir)

        # Best-effort: populate optional PDF fields if they exist.
        waveforms_grid_pdf = waveforms_out_dir / "waveforms_grid_uncurated.pdf"
        waveforms_grid_pdf = waveforms_grid_pdf if waveforms_grid_pdf.exists() else None
        spikesorting_waveforms_grid_pdf = None

        return WaveformExtractOutputs(
            well_out_dir=well_out_dir,
            waveforms_out_dir=waveforms_out_dir,
            concat_waveforms_dir=concat_waveforms_dir,
            segment_waveforms_dir=segment_waveforms_dir,
            params_json=params_json,
            filtering_json=filtering_json,
            waveforms_grid_pdf=waveforms_grid_pdf,
            spikesorting_waveforms_grid_pdf=spikesorting_waveforms_grid_pdf,
        )

    ckpt = save_checkpoint(
        checkpoint_file=ckpt_file,
        state=ckpt,
        stage=ProcessingStage.ANALYZER,
        failed_stage=None,
        error=None,
        extra_fields={
            "waveforms_out_dir": str(waveforms_out_dir),
        },
    )

    logger.info("Waveform extraction starting: well_out_dir=%s", well_out_dir)

    try:
        recording = _load_preprocessed_recording(well_out_dir=well_out_dir)
        fs_hz = float(recording.get_sampling_frequency())

        ms_before = float(inputs.ms_before) if inputs.ms_before is not None else None
        ms_after = float(inputs.ms_after) if inputs.ms_after is not None else None
        if ms_before is None or ms_after is None:
            inferred_before, inferred_after = _infer_cutout_ms(h5_path=inputs.h5_path, stream_id=inputs.stream_id, fs_hz=fs_hz)
            ms_before = inferred_before if ms_before is None else ms_before
            ms_after = inferred_after if ms_after is None else ms_after

        pre_samples = int(math.ceil(ms_before * fs_hz / 1000.0))
        post_samples = int(math.ceil(ms_after * fs_hz / 1000.0))

        sorter_output_dir = _resolve_mea_sorter_output_dir(well_out_dir=well_out_dir)
        sorting = _load_sorting_from_sorter_output_dir(sorter_output_dir=sorter_output_dir, sorter=inputs.sorter)

        # Load epoch markers from preprocessing.
        preprocess_dir = well_out_dir / PREPROCESS_OUTPUTS_DIRNAME
        maxwell_epochs_path = preprocess_dir / f"maxwell_contiguous_epochs_{inputs.stream_id}.json"
        concat_epochs_path = preprocess_dir / f"concatenation_stitch_epochs_{inputs.stream_id}.json"

        maxwell_epochs: list[dict] = []
        maxwell_intervals: list[tuple[int, int]] = []
        concat_epochs: list[dict] = []

        if maxwell_epochs_path.exists():
            maxwell_epochs = list(_read_json(maxwell_epochs_path))
            # These are in concatenated sample coordinates.
            maxwell_intervals = _epochs_to_intervals(maxwell_epochs)
        if concat_epochs_path.exists():
            concat_epochs = list(_read_json(concat_epochs_path))

        # Filter spikes by Maxwell epochs (avoid snippet boundary crossings).
        filtered_sorting = sorting
        filtering_summary: dict[str, Any] = {
            "filter_by_maxwell_epochs": bool(inputs.filter_by_maxwell_epochs),
            "ms_before": ms_before,
            "ms_after": ms_after,
            "pre_samples": pre_samples,
            "post_samples": post_samples,
            "maxwell_epochs_path": str(maxwell_epochs_path) if maxwell_epochs_path.exists() else None,
            "concat_epochs_path": str(concat_epochs_path) if concat_epochs_path.exists() else None,
            "removed_spikes_total": 0,
            "kept_spikes_total": 0,
            # Concat-level breakdown:
            # - removed_by_maxwell_epoch_total: spikes outside contiguous Maxwell epochs
            # - removed_by_edge_total: spikes inside an epoch but too close to an epoch edge
            #   for the requested waveform window (pre/post)
            "removed_by_maxwell_epoch_total": 0,
            "removed_by_edge_total": 0,
            # Per-segment extraction filtering summary (filled in below when enabled).
            "per_segment": {
                "enabled": bool(inputs.per_segment),
                "removed_by_maxwell_epoch_total": 0,
                "removed_by_edge_total": 0,
                "kept_spikes_total": 0,
                "segments": [],
            },
        }

        # Per-spike rejection rows for auditability / downstream alignment.
        # Written to waveforms_outputs/wf_rejection_log.xlsx at the end of the stage.
        wf_rejection_rows: list[dict[str, Any]] = []
        base_rej_fields: dict[str, Any] = {
            "stream_id": inputs.stream_id,
            "sorter": inputs.sorter,
            "h5_path": str(inputs.h5_path),
            "fs_hz": float(fs_hz),
            "ms_before": float(ms_before),
            "ms_after": float(ms_after),
            "pre_samples": int(pre_samples),
            "post_samples": int(post_samples),
        }

        if inputs.filter_by_maxwell_epochs and maxwell_intervals:
            unit_trains: dict[int, list[int]] = {}
            removed_total = 0
            removed_outside_total = 0
            removed_edge_total = 0
            kept_total = 0

            unit_ids = list(sorting.get_unit_ids())
            for u in unit_ids:
                st = sorting.get_unit_spike_train(u)
                st_list = [int(x) for x in st]
                st_list.sort()

                kept, removed_outside, removed_edge, removed_outside_spikes, removed_edge_spikes = (
                    _filter_spike_train_by_intervals(
                    spike_train=st_list,
                    intervals=maxwell_intervals,
                    pre_samples=pre_samples,
                    post_samples=post_samples,
                    )
                )
                unit_trains[int(u)] = kept
                removed_outside_total += int(removed_outside)
                removed_edge_total += int(removed_edge)
                removed_total += int(removed_outside) + int(removed_edge)
                kept_total += len(kept)

                # Record exact removed spikes for joinability.
                for t in removed_outside_spikes:
                    wf_rejection_rows.append(
                        {
                            **base_rej_fields,
                            "scope": "concat",
                            "source_name": "concat",
                            "segment_index": None,
                            "rec_name": None,
                            "unit_id": int(u),
                            "spike_sample_local": None,
                            "spike_sample_concat": int(t),
                            "spike_time_s": float(t) / float(fs_hz),
                            "reason": "outside_maxwell_epoch",
                        }
                    )
                for t in removed_edge_spikes:
                    wf_rejection_rows.append(
                        {
                            **base_rej_fields,
                            "scope": "concat",
                            "source_name": "concat",
                            "segment_index": None,
                            "rec_name": None,
                            "unit_id": int(u),
                            "spike_sample_local": None,
                            "spike_sample_concat": int(t),
                            "spike_time_s": float(t) / float(fs_hz),
                            "reason": "waveform_window_crosses_epoch_edge",
                        }
                    )

            filtered_sorting = _to_numpy_sorting(unit_trains=unit_trains, fs_hz=fs_hz)
            filtering_summary["removed_spikes_total"] = int(removed_total)
            filtering_summary["kept_spikes_total"] = int(kept_total)
            filtering_summary["removed_by_maxwell_epoch_total"] = int(removed_outside_total)
            filtering_summary["removed_by_edge_total"] = int(removed_edge_total)
            logger.info(
                "Filtered spikes by Maxwell epochs: kept=%d removed=%d (pre=%d post=%d samples)",
                kept_total,
                removed_total,
                pre_samples,
                post_samples,
            )
        else:
            # Best-effort counts.
            try:
                n_total = sum(len(sorting.get_unit_spike_train(u)) for u in sorting.get_unit_ids())
                filtering_summary["kept_spikes_total"] = int(n_total)
            except Exception:
                pass

        _write_json(params_json, {
            "h5_path": str(inputs.h5_path),
            "stream_id": inputs.stream_id,
            "sorter": inputs.sorter,
            "sorter_output_dir": str(sorter_output_dir),
            "ms_before": ms_before,
            "ms_after": ms_after,
            "n_jobs": int(inputs.n_jobs),
            "max_spikes_per_unit": inputs.max_spikes_per_unit,
            "per_segment": bool(inputs.per_segment),
            "per_segment_recording_source": "raw_maxwell_full_channels" if inputs.per_segment else None,
            "per_segment_only_additional_channels": bool(inputs.per_segment_only_additional_channels),
        })

        # Used to avoid redundant per-segment waveforms on common channels.
        try:
            common_channel_ids = set(int(x) for x in recording.get_channel_ids())
        except Exception:
            common_channel_ids = set()

        # Extract concatenated waveforms.
        import shutil
        import spikeinterface.full as si  # type: ignore[import-not-found]

        if concat_waveforms_dir.exists() and inputs.force_restart:
            shutil.rmtree(concat_waveforms_dir)

        logger.info("Extracting concat waveforms -> %s", concat_waveforms_dir)
        concat_analyzer = si.create_sorting_analyzer(
            filtered_sorting,
            recording,
            format="binary_folder",
            folder=concat_waveforms_dir,
            return_in_uV=True,
        )
        concat_analyzer.compute(
            ["random_spikes", "waveforms"],
            extension_params={
                "random_spikes": {
                    "method": "uniform",
                    "max_spikes_per_unit": int(inputs.max_spikes_per_unit),
                    "seed": 0,
                },
                "waveforms": {"ms_before": float(ms_before), "ms_after": float(ms_after)},
            },
            verbose=False,
            n_jobs=int(inputs.n_jobs),
        )

        # Optional: per-segment extraction using concatenation epochs.
        if inputs.per_segment and concat_epochs:
            assert segment_waveforms_dir is not None
            segment_waveforms_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Extracting per-segment waveforms -> %s", segment_waveforms_dir)
            for seg in concat_epochs:
                try:
                    seg_index = int(seg["segment_index"])
                    rec_name = str(seg.get("rec_name", f"seg{seg_index}"))
                    start = int(seg["start_sample"])
                    end = int(seg["end_sample"])
                except Exception:
                    continue

                seg_dir = segment_waveforms_dir / f"seg{seg_index:02d}_{rec_name}"
                if seg_dir.exists() and inputs.force_restart:
                    shutil.rmtree(seg_dir)

                # STEP 1) Load the *raw* segment recording with its full channel set.
                #
                # This is the point of the per-segment extraction: the concatenated
                # recording (used for sorting) only contains electrodes shared across
                # all segments. Any electrodes present in *some* segments but not all
                # are intentionally dropped during concatenation.
                #
                # By extracting waveforms on the raw segment, we can recover waveforms
                # on those dropped electrodes for this segment.
                seg_rec = _load_raw_segment_recording_full_channels(
                    h5_path=inputs.h5_path,
                    stream_id=inputs.stream_id,
                    rec_name=rec_name,
                    center_chunk_size=10_000,
                )

                # Optional: avoid redundant waveforms.
                # If enabled, remove the concat/common electrodes from the raw segment
                # so the per-segment waveforms contain only the *additional* channels.
                raw_channels_total: Optional[int]
                excluded_common_channels_total: Optional[int]
                kept_additional_channels_total: Optional[int]
                raw_channels_total = None
                excluded_common_channels_total = None
                kept_additional_channels_total = None

                try:
                    raw_channels_total = int(seg_rec.get_num_channels())
                except Exception:
                    pass

                if inputs.per_segment_only_additional_channels and common_channel_ids:
                    try:
                        import numpy as np  # type: ignore[import-not-found]

                        cv = seg_rec.get_property("contact_vector")
                        electrodes = np.asarray(cv["electrode"], dtype=int)
                        ch_ids = list(seg_rec.get_channel_ids())
                        if len(electrodes) == len(ch_ids):
                            keep_mask = [int(e) not in common_channel_ids for e in electrodes]
                            keep_channel_ids = [ch_ids[i] for i, keep in enumerate(keep_mask) if keep]
                            kept_electrodes = [int(electrodes[i]) for i, keep in enumerate(keep_mask) if keep]
                            seg_rec = seg_rec.select_channels(keep_channel_ids)
                            excluded_common_channels_total = int(len(ch_ids) - len(keep_channel_ids))
                            kept_additional_channels_total = int(len(keep_channel_ids))

                            # Ensure channel ids are electrode ids for stable identity.
                            # (Helpful if earlier rename failed.)
                            try:
                                if len(kept_electrodes) == int(seg_rec.get_num_channels()):
                                    if int(np.unique(np.asarray(kept_electrodes)).size) == int(len(kept_electrodes)):
                                        seg_rec = seg_rec.rename_channels([int(e) for e in kept_electrodes])
                            except Exception:
                                pass
                    except Exception:
                        # Best-effort: if we cannot compute the additional-channel set,
                        # keep full channels rather than failing.
                        pass

                if inputs.per_segment_only_additional_channels:
                    try:
                        if int(seg_rec.get_num_channels()) == 0:
                            logger.info(
                                "Segment %s: no additional channels after excluding common set; skipping per-segment waveforms.",
                                rec_name,
                            )
                            filtering_summary["per_segment"]["segments"].append(
                                {
                                    "segment_index": int(seg_index),
                                    "rec_name": str(rec_name),
                                    "spikes_in_segment_total": 0,
                                    "removed_by_maxwell_epoch": 0,
                                    "removed_by_edge": 0,
                                    "kept_spikes_total": 0,
                                    "maxwell_intervals_in_segment": None,
                                    "raw_channels_total": raw_channels_total,
                                    "excluded_common_channels_total": excluded_common_channels_total,
                                    "kept_additional_channels_total": kept_additional_channels_total,
                                    "skipped_reason": "no_additional_channels",
                                }
                            )
                            continue
                    except Exception:
                        pass

                # Sanity: concat_epochs were computed from the preprocessed segments.
                # Channel slicing does not change sample count, so this should match.
                seg_len_expected = int(end - start)
                try:
                    seg_len = int(seg_rec.get_num_samples())
                    if seg_len != seg_len_expected:
                        logger.warning(
                            "Segment length mismatch for %s: raw=%d expected=%d (start=%d end=%d). Proceeding.",
                            rec_name,
                            seg_len,
                            seg_len_expected,
                            start,
                            end,
                        )
                except Exception:
                    seg_len = seg_len_expected

                try:
                    if inputs.per_segment_only_additional_channels:
                        logger.info(
                            "Segment %s: raw channels=%s, kept additional=%s (excluded common=%s, concat/common=%d)",
                            rec_name,
                            raw_channels_total,
                            kept_additional_channels_total,
                            excluded_common_channels_total,
                            int(recording.get_num_channels()),
                        )
                    else:
                        logger.info(
                            "Segment %s: raw channels=%d (concat/common=%d)",
                            rec_name,
                            int(seg_rec.get_num_channels()),
                            int(recording.get_num_channels()),
                        )
                except Exception:
                    pass
                # Compute Maxwell contiguous-epoch intervals in *segment-local* coordinates.
                # This ensures we don't extract waveforms whose windows cross snippet gaps
                # within this segment.
                seg_maxwell_intervals: list[tuple[int, int]] = []
                if inputs.filter_by_maxwell_epochs and maxwell_epochs:
                    seg_maxwell_intervals = _maxwell_epochs_to_segment_local_intervals(
                        maxwell_epochs=maxwell_epochs,
                        segment_index=seg_index,
                    )

                # STEP 2) Build a segment-local sorting from the *concatenated* sorting.
                #
                # Conceptually:
                # - `filtered_sorting` spike times are in the concatenated time base.
                # - `seg_rec` is a single raw recording segment with its own 0..N time base.
                # So we:
                #   - select spikes that fall inside this concatenated segment window [start, end)
                #   - shift them to segment-local coordinates by subtracting `start`
                #   - drop spikes too close to segment edges so waveform windows fit.
                unit_trains_seg: dict[int, list[int]] = {}

                seg_spikes_total = 0
                seg_removed_epoch_total = 0
                seg_removed_edge_total = 0
                seg_kept_total = 0

                for u in filtered_sorting.get_unit_ids():
                    st = filtered_sorting.get_unit_spike_train(u)
                    st_list = [int(x) for x in st]

                    # Keep only spikes in the segment, shift to segment-local coordinates.
                    # Also drop spikes too close to segment edges for waveform windows.
                    local_all = [int(t - start) for t in st_list if start <= t < end]
                    local_all.sort()
                    seg_spikes_total += len(local_all)

                    # Filter by Maxwell contiguous epochs *within this segment*.
                    local_epoch_filtered = local_all
                    removed_epoch = 0
                    if inputs.filter_by_maxwell_epochs and seg_maxwell_intervals:
                        (
                            local_epoch_filtered,
                            removed_outside,
                            removed_edge_epoch,
                            removed_outside_spikes,
                            removed_edge_spikes,
                        ) = _filter_spike_train_by_intervals(
                            spike_train=local_all,
                            intervals=seg_maxwell_intervals,
                            pre_samples=pre_samples,
                            post_samples=post_samples,
                        )
                        removed_epoch = int(removed_outside) + int(removed_edge_epoch)

                        source_name = f"seg{int(seg_index):02d}_{rec_name}"
                        for t_local in removed_outside_spikes:
                            t_concat = int(t_local) + int(start)
                            wf_rejection_rows.append(
                                {
                                    **base_rej_fields,
                                    "scope": "segment",
                                    "source_name": str(source_name),
                                    "segment_index": int(seg_index),
                                    "rec_name": str(rec_name),
                                    "unit_id": int(u),
                                    "spike_sample_local": int(t_local),
                                    "spike_sample_concat": int(t_concat),
                                    "spike_time_s": float(t_concat) / float(fs_hz),
                                    "reason": "outside_maxwell_epoch",
                                }
                            )
                        for t_local in removed_edge_spikes:
                            t_concat = int(t_local) + int(start)
                            wf_rejection_rows.append(
                                {
                                    **base_rej_fields,
                                    "scope": "segment",
                                    "source_name": str(source_name),
                                    "segment_index": int(seg_index),
                                    "rec_name": str(rec_name),
                                    "unit_id": int(u),
                                    "spike_sample_local": int(t_local),
                                    "spike_sample_concat": int(t_concat),
                                    "spike_time_s": float(t_concat) / float(fs_hz),
                                    "reason": "waveform_window_crosses_epoch_edge",
                                }
                            )

                    # Extra safety: enforce segment-edge constraints even if epoch markers
                    # are missing/unexpected.
                    kept_edges: list[int] = []
                    for t_local in local_epoch_filtered:
                        if t_local - pre_samples < 0:
                            continue
                        if t_local + post_samples >= int(seg_len):
                            continue
                        kept_edges.append(int(t_local))

                    # Log spikes rejected due to segment-edge constraints.
                    try:
                        source_name = f"seg{int(seg_index):02d}_{rec_name}"
                        kept_edge_set = set(int(x) for x in kept_edges)
                        for t_local in local_epoch_filtered:
                            if int(t_local) in kept_edge_set:
                                continue
                            if (int(t_local) - int(pre_samples) < 0) or (
                                int(t_local) + int(post_samples) >= int(seg_len)
                            ):
                                t_concat = int(t_local) + int(start)
                                wf_rejection_rows.append(
                                    {
                                        **base_rej_fields,
                                        "scope": "segment",
                                        "source_name": str(source_name),
                                        "segment_index": int(seg_index),
                                        "rec_name": str(rec_name),
                                        "unit_id": int(u),
                                        "spike_sample_local": int(t_local),
                                        "spike_sample_concat": int(t_concat),
                                        "spike_time_s": float(t_concat) / float(fs_hz),
                                        "reason": "waveform_window_outside_segment_bounds",
                                    }
                                )
                    except Exception:
                        pass

                    removed_edge = int(len(local_epoch_filtered) - len(kept_edges))
                    seg_removed_epoch_total += int(removed_epoch)
                    seg_removed_edge_total += int(removed_edge)
                    seg_kept_total += int(len(kept_edges))

                    unit_trains_seg[int(u)] = kept_edges

                # Convert the segment-local spike trains into a SortingExtractor.
                # This is the bridge that lets us use the holistic concat sorting
                # to drive waveform extraction on each raw segment.
                seg_sort = _to_numpy_sorting(unit_trains=unit_trains_seg, fs_hz=fs_hz)

                # STEP 3) Register the sorting to the raw segment recording.
                #
                # Not strictly required for `si.create_sorting_analyzer(sorting=..., recording=...)`,
                # but this matches the intent/semantics of the legacy pipeline and helps
                # downstream utilities that expect the sorting to “know” its recording.
                try:
                    seg_sort.register_recording(seg_rec)
                except Exception:
                    pass

                # STEP 4) Extract waveforms on the raw segment.
                #
                # Key outcome: these waveforms include channels that are absent from the
                # concatenated recording (i.e. the non-shared electrodes), which is the
                # whole reason we do this per-segment pass.
                seg_analyzer = si.create_sorting_analyzer(
                    seg_sort,
                    seg_rec,
                    format="binary_folder",
                    folder=seg_dir,
                    return_in_uV=True,
                )
                seg_analyzer.compute(
                    ["random_spikes", "waveforms"],
                    extension_params={
                        "random_spikes": {
                            "method": "uniform",
                            "max_spikes_per_unit": int(inputs.max_spikes_per_unit),
                            "seed": 0,
                        },
                        "waveforms": {"ms_before": float(ms_before), "ms_after": float(ms_after)},
                    },
                    verbose=False,
                    n_jobs=max(1, int(inputs.n_jobs)),
                )

                # Accumulate per-segment filtering stats.
                try:
                    filtering_summary["per_segment"]["removed_by_maxwell_epoch_total"] += int(seg_removed_epoch_total)
                    filtering_summary["per_segment"]["removed_by_edge_total"] += int(seg_removed_edge_total)
                    filtering_summary["per_segment"]["kept_spikes_total"] += int(seg_kept_total)
                    filtering_summary["per_segment"]["segments"].append(
                        {
                            "segment_index": int(seg_index),
                            "rec_name": str(rec_name),
                            "spikes_in_segment_total": int(seg_spikes_total),
                            "removed_by_maxwell_epoch": int(seg_removed_epoch_total),
                            "removed_by_edge": int(seg_removed_edge_total),
                            "kept_spikes_total": int(seg_kept_total),
                            "maxwell_intervals_in_segment": int(len(seg_maxwell_intervals)),
                            "raw_channels_total": raw_channels_total,
                            "excluded_common_channels_total": excluded_common_channels_total,
                            "kept_additional_channels_total": kept_additional_channels_total,
                        }
                    )
                except Exception:
                    pass

        # Persist filtering summary after all filtering passes have contributed their counts
        # (concat-level filtering + optional per-segment filtering).
        _write_json(filtering_json, filtering_summary)

        # Persist per-spike rejection log (best effort; safe to skip if pandas engine is missing).
        try:
            wf_rejection_log_xlsx = waveforms_out_dir / "wf_rejection_log.xlsx"
            _write_wf_rejection_log_xlsx(
                wf_rejection_log_xlsx=wf_rejection_log_xlsx,
                rows=wf_rejection_rows,
                force_restart=bool(inputs.force_restart),
                logger=logger,
            )
        except Exception as e:
            logger.warning("Failed to write wf_rejection_log.xlsx: %s", e)

        waveforms_grid_pdf: Optional[Path] = None
        spikesorting_waveforms_grid_pdf: Optional[Path] = None
        if inputs.plot_waveforms_grid_pdf:
            waveforms_grid_pdf = waveforms_out_dir / "waveforms_grid_uncurated.pdf"

            # Recreate only when force_restart or missing.
            if (not waveforms_grid_pdf.exists()) or inputs.force_restart:
                logger.info("Writing waveforms grid PDF -> %s", waveforms_grid_pdf)
                _write_waveforms_grid_pdf(
                    waveforms_folder=concat_waveforms_dir,
                    pdf_path=waveforms_grid_pdf,
                )

            # Curation outputs (MEA_Analysis-style) + curated grid PDF.
            try:
                curated_units, curation_paths = _run_mea_analysis_style_curation(
                    recording=recording,
                    sorting=sorting,
                    output_dir=waveforms_out_dir,
                    n_jobs=int(inputs.n_jobs),
                    ms_before=float(ms_before),
                    ms_after=float(ms_after),
                    force_restart=bool(inputs.force_restart),
                    logger=logger,
                )

                curated_pdf = waveforms_out_dir / "waveforms_grid_curated.pdf"
                if (not curated_pdf.exists()) or inputs.force_restart:
                    logger.info("Writing curated waveforms grid PDF -> %s", curated_pdf)
                    _write_waveforms_grid_pdf(
                        waveforms_folder=concat_waveforms_dir,
                        pdf_path=curated_pdf,
                        unit_ids=list(curated_units),
                    )
            except Exception as e:
                logger.warning("Curation failed; skipping curated metrics/grid: %s", e)

        ckpt = save_checkpoint(
            checkpoint_file=ckpt_file,
            state=ckpt,
            stage=ProcessingStage.ANALYZER_COMPLETE,
            failed_stage=None,
            error=None,
            extra_fields={
                "waveforms_out_dir": str(waveforms_out_dir),
                "concat_waveforms_dir": str(concat_waveforms_dir),
                "segment_waveforms_dir": str(segment_waveforms_dir) if segment_waveforms_dir else None,
                "waveforms_params_json": str(params_json),
                "waveforms_filtering_json": str(filtering_json),
                "waveforms_grid_pdf": str(waveforms_grid_pdf) if waveforms_grid_pdf else None,
                "spikesorting_waveforms_grid_pdf": None,
            },
        )

        logger.info("Waveform extraction complete")

        return WaveformExtractOutputs(
            well_out_dir=well_out_dir,
            waveforms_out_dir=waveforms_out_dir,
            concat_waveforms_dir=concat_waveforms_dir,
            segment_waveforms_dir=segment_waveforms_dir,
            params_json=params_json,
            filtering_json=filtering_json,
            waveforms_grid_pdf=waveforms_grid_pdf,
            spikesorting_waveforms_grid_pdf=spikesorting_waveforms_grid_pdf,
        )

    except Exception as e:
        save_checkpoint(
            checkpoint_file=ckpt_file,
            state=ckpt,
            stage=ProcessingStage.ANALYZER,
            failed_stage="WAVEFORMS",
            error=exception_to_error_dict(e),
        )
        logger.exception("Waveform extraction FAILED")
        raise
