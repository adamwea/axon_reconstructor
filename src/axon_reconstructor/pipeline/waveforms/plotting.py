"""Waveforms plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def _write_waveforms_grid_pdf(
    *,
    waveforms_folder: Path,
    pdf_path: Path,
    unit_ids: Optional[list[Any]] = None,
    segment_waveforms_folders: Optional[list[Path]] = None,
    show_debug_annotation: bool = False,
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

    # Pre-load per-segment analyzers/extensions once (avoid re-loading for every unit).
    segment_analyzers: list[tuple[Path, Any, Any]] = []
    if segment_waveforms_folders:
        for seg_folder in segment_waveforms_folders:
            try:
                seg_an = si.load_sorting_analyzer(seg_folder)
                seg_wf_ext = seg_an.get_extension("waveforms")
                if seg_wf_ext is None:
                    continue
                segment_analyzers.append((seg_folder, seg_an, seg_wf_ext))
            except Exception:
                continue

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

                    # Contribution stats + plotting source collection.
                    # We'll plot *all* waveforms across concat + segments, overlaid on a single
                    # per-unit "best" electrode (chosen across the union of available channel_ids).
                    concat_n_waveforms = int(wf.shape[0])
                    concat_channel_ids: list[Any] | None = None
                    if unit_channel_ids is not None:
                        try:
                            concat_channel_ids = list(unit_channel_ids)
                        except Exception:
                            concat_channel_ids = None
                    if concat_channel_ids is None:
                        # Fallback: we only know channel *count* (no stable electrode IDs).
                        # Use positions 0..n-1 so "unique" bookkeeping still behaves.
                        concat_channel_ids = list(range(int(wf.shape[2]) if wf.ndim == 3 else 0))
                    concat_channel_set = set(concat_channel_ids)
                    concat_n_channels_unique = int(len(concat_channel_set))

                    # Collect waveform arrays and their channel_id lists.
                    # Each wf array is (n_spikes, n_samples, n_channels).
                    wf_sources: list[tuple[str, Any, list[Any]]] = [("concat", wf, concat_channel_ids)]

                    # If per-segment analyzers exist, load per-unit waveforms and include them in plotting.
                    seg_total_waveforms = 0
                    seg_channel_set: set[Any] = set()
                    seg_max_channels = 0
                    seg_overlap_set: set[Any] = set()
                    if segment_analyzers:
                        seen_seg_channels: set[Any] = set()
                        for _seg_folder, seg_an, seg_wf_ext in segment_analyzers:
                            try:
                                seg_wf = seg_wf_ext.get_waveforms_one_unit(unit_id=uid)
                                if seg_wf is None or seg_wf.shape[0] == 0:
                                    continue
                                seg_total_waveforms += int(seg_wf.shape[0])
                                if seg_wf.ndim == 3:
                                    seg_max_channels = max(seg_max_channels, int(seg_wf.shape[2]))

                                seg_unit_channel_ids: list[Any] | None = None
                                try:
                                    if seg_an.sparsity is not None:
                                        seg_unit_channel_ids = seg_an.sparsity.unit_id_to_channel_ids.get(uid)
                                except Exception:
                                    seg_unit_channel_ids = None
                                if seg_unit_channel_ids is None:
                                    try:
                                        seg_unit_channel_ids = list(seg_an.recording.get_channel_ids())
                                    except Exception:
                                        seg_unit_channel_ids = None
                                if seg_unit_channel_ids is None:
                                    seg_unit_channel_ids = list(
                                        range(int(seg_wf.shape[2]) if seg_wf.ndim == 3 else 0)
                                    )

                                seg_unit_channel_set = set(seg_unit_channel_ids)
                                seg_overlap_set |= (seen_seg_channels & seg_unit_channel_set)
                                seen_seg_channels |= seg_unit_channel_set
                                seg_channel_set |= seg_unit_channel_set

                                wf_sources.append(("seg", seg_wf, seg_unit_channel_ids))
                            except Exception:
                                continue

                    all_channel_set = concat_channel_set | seg_channel_set
                    all_n_channels_unique = int(len(all_channel_set))
                    all_n_waveforms_sum = int(concat_n_waveforms + seg_total_waveforms)

                    # Choose a global "best" channel_id across all sources.
                    # Primary: most negative mean deflection. Tie-break: present in more sources.
                    channel_presence: dict[Any, int] = {}
                    for _src_name, _src_wf, _src_ch_ids in wf_sources:
                        for _ch in set(_src_ch_ids):
                            channel_presence[_ch] = channel_presence.get(_ch, 0) + 1

                    best_channel_id: Any | None = None
                    best_score: float | None = None
                    best_presence: int = -1
                    for _src_name, _src_wf, _src_ch_ids in wf_sources:
                        try:
                            if _src_wf is None or _src_wf.shape[0] == 0:
                                continue
                            mean_wf = np.mean(_src_wf, axis=0)  # (n_samples, n_channels)
                            # Safety: ensure channel axis matches channel_id list.
                            n_ch = int(mean_wf.shape[1]) if mean_wf.ndim == 2 else 0
                            for ch_idx in range(min(n_ch, len(_src_ch_ids))):
                                ch_id = _src_ch_ids[ch_idx]
                                score = float(np.min(mean_wf[:, ch_idx]))
                                presence = int(channel_presence.get(ch_id, 0))
                                if best_score is None:
                                    best_channel_id = ch_id
                                    best_score = score
                                    best_presence = presence
                                    continue
                                if score < best_score:
                                    best_channel_id = ch_id
                                    best_score = score
                                    best_presence = presence
                                elif score == best_score and presence > best_presence:
                                    best_channel_id = ch_id
                                    best_score = score
                                    best_presence = presence
                        except Exception:
                            continue

                    # Fallback: if something went wrong, keep concat's original heuristic.
                    if best_channel_id is None:
                        mean_wf = np.mean(wf, axis=0)
                        best_ch = int(np.argmin(np.min(mean_wf, axis=0)))
                        best_channel_id = concat_channel_ids[best_ch] if best_ch < len(concat_channel_ids) else best_ch

                    # Gather waveforms for the chosen channel across all sources.
                    stacked_wfs: list[Any] = []
                    n_wf_used = 0
                    for _src_name, _src_wf, _src_ch_ids in wf_sources:
                        try:
                            ch_to_idx = {cid: idx for idx, cid in enumerate(_src_ch_ids)}
                            if best_channel_id not in ch_to_idx:
                                continue
                            ch_idx = int(ch_to_idx[best_channel_id])
                            if _src_wf is None or _src_wf.shape[0] == 0:
                                continue
                            if _src_wf.ndim != 3 or ch_idx >= _src_wf.shape[2]:
                                continue
                            w2d = _src_wf[:, :, ch_idx]  # (n_spikes, n_samples)
                            if w2d is None or w2d.shape[0] == 0:
                                continue
                            stacked_wfs.append(w2d)
                            n_wf_used += int(w2d.shape[0])
                        except Exception:
                            continue

                    if len(stacked_wfs) == 0:
                        ax.axis("off")
                        continue

                    wf2d_all = np.concatenate(stacked_wfs, axis=0)
                    mean_wf_1d = np.mean(wf2d_all, axis=0)

                    time_ms = np.arange(wf2d_all.shape[1]) / fs * 1000

                    n_spikes = int(wf2d_all.shape[0])
                    max_spikes_to_plot = 500
                    if n_spikes > max_spikes_to_plot:
                        indices = np.random.choice(n_spikes, max_spikes_to_plot, replace=False)
                        spikes_to_plot = wf2d_all[indices, :]
                    else:
                        spikes_to_plot = wf2d_all

                    ax.plot(time_ms, spikes_to_plot.T, c="gray", lw=0.5, alpha=0.3)
                    ax.plot(time_ms, mean_wf_1d, c="red", lw=1.5)

                    ch_label = best_channel_id
                    try:
                        ch_label = int(ch_label)
                    except Exception:
                        pass

                    ax.set_title(f"Unit {uid} | Ch {ch_label}", fontsize=10)

                    # Add a small per-subplot annotation describing what contributed.
                    # Keep this compact by default; optionally include a multi-line debug breakdown.
                    try:
                        if show_debug_annotation:
                            annotation_text = (
                                f"concat: nCh={concat_n_channels_unique} nWf={concat_n_waveforms}"
                                + (
                                    f"\nsegs:  nChUniq={len(seg_channel_set)} nWfSum={seg_total_waveforms}"
                                    + (f" overlap={len(seg_overlap_set)}" if len(seg_overlap_set) else "")
                                    + f" (nChMax={seg_max_channels})"
                                    if segment_analyzers
                                    else ""
                                )
                                + (
                                    f"\nused:  ch={ch_label} nWf={n_wf_used}"
                                    if segment_analyzers
                                    else ""
                                )
                            )
                        else:
                            # "All" (union) contribution summary, without an "all:" prefix.
                            annotation_text = f"nChUniq={all_n_channels_unique} \nnWfSum={all_n_waveforms_sum} \nnWF={n_wf_used}"

                        ax.text(
                            0.98,
                            0.14,
                            annotation_text,
                            transform=ax.transAxes,
                            ha="left",
                            va="bottom",
                            fontsize=8,
                            color="black",
                            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1.0},
                        )
                    except Exception:
                        pass

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


__all__ = [
    "_write_waveforms_grid_pdf",
]
