"""Waveforms plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


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


__all__ = [
    "_write_waveforms_grid_pdf",
]
