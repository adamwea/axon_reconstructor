from __future__ import annotations

import json
import logging
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
from .pipeline_driver import _compute_mea_analysis_output_dir


TEMPLATES_OUTPUTS_DIRNAME = "templates_outputs"


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _jsonable(x: Any) -> Any:
    """Convert common non-JSON-native scalars into JSON-safe Python types."""

    try:
        import numpy as np  # type: ignore[import-not-found]

        if isinstance(x, (np.integer, np.floating)):
            return x.item()
    except Exception:
        pass
    if isinstance(x, Path):
        return str(x)
    return x


def _jsonable_list(xs: Optional[list[Any]]) -> Optional[list[Any]]:
    if xs is None:
        return None
    return [_jsonable(v) for v in xs]


def _jsonable_sequence(xs: Any) -> Optional[list[Any]]:
    """Like `_jsonable_list`, but accepts list/tuple/numpy arrays (best effort)."""

    if xs is None:
        return None
    try:
        return [_jsonable(v) for v in list(xs)]
    except Exception:
        try:
            return [_jsonable(xs)]
        except Exception:
            return None


def _compute_templates_checkpoint_file(*, well_out_dir: Path, h5_path: Path, stream_id: str) -> Path:
    """Use a dedicated checkpoint file for templates."""

    main_ckpt = compute_checkpoint_file(output_dir=well_out_dir, file_path=h5_path, stream_id=stream_id)
    name = main_ckpt.name
    if name.endswith("_checkpoint.json"):
        name = name[: -len("_checkpoint.json")] + "_templates_checkpoint.json"
    else:
        name = main_ckpt.stem + "_templates_checkpoint.json"
    return main_ckpt.with_name(name)


@dataclass(frozen=True)
class TemplateExtractInputs:
    h5_path: Path
    stream_id: str
    mea_output_root: Path

    include_concat: bool = True
    include_segments: bool = True

    unit_ids: Optional[list[Any]] = None
    unit_limit: Optional[int] = None

    # Optional SpikeInterface auto-merge of units (best-effort).
    run_unit_merging: bool = True
    merge_presets: Optional[list[str]] = None
    merge_recursive: bool = False

    # Plotting
    plot_templates_grid_pdf: bool = True
    plot_multi_source_templates_pdf: bool = True

    # Template overlay plot controls
    top_channels_per_template: int = 8

    n_jobs: int = 8
    force_restart: bool = False


@dataclass(frozen=True)
class TemplateExtractOutputs:
    well_out_dir: Path
    templates_out_dir: Path
    extracted_templates_dir: Path
    merged_union_by_unit_dir: Optional[Path]

    summary_json: Path
    templates_grid_pdf: Optional[Path]
    multi_source_templates_dir: Optional[Path]


def _try_get_electrode_ids(recording) -> Optional[list[Any]]:
    """Best-effort retrieval of electrode ids from a RecordingExtractor."""

    try:
        cv = recording.get_property("contact_vector")
        if cv is None:
            return None
        electrodes = cv.get("electrode")
        if electrodes is None:
            return None
        return list(electrodes)
    except Exception:
        return None


def _infer_location_tolerance(locs) -> float:
    """Infer a tolerance for matching channel locations (in same units as locs)."""

    import numpy as np  # type: ignore[import-not-found]

    locs = np.asarray(locs)
    if locs.ndim != 2 or locs.shape[0] < 2:
        return 1e-6
    # Roughly: 1/4 of the median nearest-neighbor distance.
    try:
        from scipy.spatial import cKDTree  # type: ignore[import-not-found]

        tree = cKDTree(locs[:, :2])
        d, _ = tree.query(locs[:, :2], k=2)
        nn = d[:, 1]
        med = float(np.median(nn[nn > 0])) if np.any(nn > 0) else 0.0
        if med <= 0:
            return 1e-6
        return max(1e-6, med / 4.0)
    except Exception:
        return 1e-6


def _loc_key(xy: Any, tol: float) -> tuple[int, int]:
    x = float(xy[0])
    y = float(xy[1])
    # bucketize by tol
    return (int(round(x / tol)), int(round(y / tol)))


def _build_union_template_for_unit(*, sources_for_unit: list[dict[str, Any]], logger) -> Optional[dict[str, Any]]:
    """Build a merged-union template across sources.

    Keeps the first occurrence for overlapping channels (warns on overlaps).

    Returns a dict with fields compatible with `sources_for_unit` entries.
    """

    import numpy as np  # type: ignore[import-not-found]

    if not sources_for_unit:
        return None

    # Use first source as base for time axis (#samples).
    n_samples = int(np.asarray(sources_for_unit[0]["template"]).shape[0])

    tol = None
    try:
        tol = _infer_location_tolerance(sources_for_unit[0]["channel_locations"])
    except Exception:
        tol = 1e-6

    union_waveforms: list[np.ndarray] = []
    union_locs: list[np.ndarray] = []
    union_channel_ids: list[Any] = []
    union_electrode_ids: list[Any] = []
    union_source_names: list[str] = []

    # Map from match key -> union index.
    key_to_index: dict[Any, int] = {}

    overlap_count = 0

    for src in sources_for_unit:
        name = str(src["name"])
        tmpl = np.asarray(src["template"])
        locs = np.asarray(src["channel_locations"])
        ch_ids = src.get("channel_ids")
        el_ids = src.get("electrode_ids")

        if tmpl.ndim != 2 or tmpl.shape[0] != n_samples:
            logger.warning("Skipping %s for merged_union: bad template shape %s", name, tmpl.shape)
            continue

        for j in range(tmpl.shape[1]):
            key = None
            try:
                if el_ids is not None:
                    key = ("electrode", int(el_ids[j]))
                elif ch_ids is not None:
                    key = ("channel", str(ch_ids[j]))
                else:
                    key = ("loc", _loc_key(locs[j], float(tol)))
            except Exception:
                key = ("loc", _loc_key(locs[j], float(tol)))

            if key in key_to_index:
                overlap_count += 1
                continue

            key_to_index[key] = len(union_waveforms)
            union_waveforms.append(np.asarray(tmpl[:, j], dtype=float))
            union_locs.append(np.asarray(locs[j, :2], dtype=float))
            union_source_names.append(str(name))
            try:
                union_channel_ids.append(None if ch_ids is None else ch_ids[j])
            except Exception:
                union_channel_ids.append(None)
            try:
                union_electrode_ids.append(None if el_ids is None else el_ids[j])
            except Exception:
                union_electrode_ids.append(None)

    if not union_waveforms:
        return None

    if overlap_count:
        logger.warning("merged_union: skipped %d overlapping channels (keep-first)", overlap_count)

    merged = np.stack(union_waveforms, axis=1)
    merged_locs = np.stack(union_locs, axis=0)

    return {
        "name": "merged_union",
        "template": merged,
        "channel_locations": merged_locs,
        "channel_ids": union_channel_ids,
        "electrode_ids": union_electrode_ids,
        "channel_source_names": union_source_names,
        "stats": {"overlap_skipped": int(overlap_count), "n_channels": int(merged.shape[1])},
    }


def _apply_wf_exclusion_monkey_patch_to_merged_union(
    *,
    merged_union_src: dict[str, Any],
    unit_id: Any,
    excluded_source_names: set[str],
    logger,
) -> dict[str, Any]:
    """TEMPORARY monkey patch: drop channels from merged_union based on waveforms-stage rejections.

    Background:
    - Waveforms extraction can exclude spikes (e.g., crossing Maxwell snippet gaps).
    - Today, templates are derived from waveforms analyzers, but we do not yet
      have a principled way to carry *spike-level* exclusion decisions into
      downstream reconstruction artifacts.

    This is an intentionally blunt stop-gap:
    - If a segment-source had any waveforms-stage spike rejections for this unit,
      we drop that segment's contributed channels from `merged_union`.

    IMPORTANT:
    - This only affects the *merged_union* artifact (plotting + saved npy/meta).
    - Per-source templates remain unchanged.
    - This should be removed once proper spike-level template recomputation exists.
    """

    import numpy as np  # type: ignore[import-not-found]

    if not excluded_source_names:
        return merged_union_src

    # Never drop the concat source via this monkey patch (it is the backbone).
    excluded = {s for s in excluded_source_names if str(s) != "concat"}
    if not excluded:
        return merged_union_src

    src_names = merged_union_src.get("channel_source_names")
    if src_names is None:
        return merged_union_src

    try:
        src_names_list = [str(x) for x in list(src_names)]
    except Exception:
        return merged_union_src

    keep_mask = np.asarray([name not in excluded for name in src_names_list], dtype=bool)
    if keep_mask.size == 0:
        return merged_union_src
    if bool(np.all(keep_mask)):
        return merged_union_src

    if not bool(np.any(keep_mask)):
        logger.warning(
            "MONKEY PATCH: would drop all merged_union channels for unit %s (excluded sources=%s); keeping unmodified",
            unit_id,
            sorted(excluded),
        )
        return merged_union_src

    tmpl = np.asarray(merged_union_src.get("template"))
    locs = np.asarray(merged_union_src.get("channel_locations"))
    if tmpl.ndim != 2 or locs.ndim != 2 or tmpl.shape[1] != locs.shape[0] or keep_mask.shape[0] != tmpl.shape[1]:
        return merged_union_src

    dropped = int(np.sum(~keep_mask))
    kept = int(np.sum(keep_mask))
    logger.info(
        "MONKEY PATCH: merged_union channel curation for unit %s: dropping %d channels from sources=%s (kept=%d)",
        unit_id,
        dropped,
        sorted(excluded),
        kept,
    )

    merged_union_src = dict(merged_union_src)
    merged_union_src["template"] = tmpl[:, keep_mask]
    merged_union_src["channel_locations"] = locs[keep_mask]

    # Optional aux lists
    for key in ("channel_ids", "electrode_ids", "channel_source_names"):
        try:
            vals = merged_union_src.get(key)
            if vals is not None and len(vals) == int(keep_mask.shape[0]):
                merged_union_src[key] = [v for v, keep in zip(list(vals), keep_mask.tolist(), strict=False) if keep]
        except Exception:
            pass

    try:
        stats = dict(merged_union_src.get("stats") or {})
        stats["monkey_patch_dropped_sources"] = sorted(excluded)
        stats["monkey_patch_dropped_channels"] = dropped
        stats["n_channels"] = int(merged_union_src["template"].shape[1])
        merged_union_src["stats"] = stats
    except Exception:
        pass

    return merged_union_src


def _write_template_overlay(
    *,
    ax,
    template: Any,
    fs_hz: float,
    ms_before: Optional[float],
    ms_after: Optional[float],
    top_channels: int,
    title: str,
) -> None:
    import numpy as np  # type: ignore[import-not-found]

    tmpl = np.asarray(template)
    if tmpl.ndim != 2 or tmpl.size == 0:
        ax.set_axis_off()
        return

    n_samples = tmpl.shape[0]

    # Time axis in ms (best effort).
    if ms_before is not None and ms_after is not None:
        # Use linspace so the full window maps nicely.
        t_ms = np.linspace(-float(ms_before), float(ms_after), n_samples, endpoint=False)
    else:
        t_ms = (np.arange(n_samples, dtype=float) / float(fs_hz)) * 1000.0

    ptp = np.ptp(tmpl, axis=0)
    order = np.argsort(ptp)[::-1]
    # If top_channels <= 0, plot all channels.
    if int(top_channels) <= 0:
        sel = order
    else:
        k = int(min(max(1, int(top_channels)), len(order)))
        sel = order[:k]

    # Overlay selected channels.
    for j_idx, j in enumerate(sel):
        y = tmpl[:, int(j)]
        ax.plot(t_ms, y, lw=0.8, alpha=0.85)

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Time (ms)", fontsize=8)
    ax.set_ylabel("uV", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)


def _write_templates_grid_pdf(
    *,
    pdf_path: Path,
    unit_entries: list[dict[str, Any]],
    fs_hz: float,
    ms_before: Optional[float],
    ms_after: Optional[float],
    top_channels: int,
    logger,
) -> None:
    """Write a grid PDF of templates (one subplot per unit)."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_pdf as pdf

    ncols = 3
    nrows = 4
    per_page = ncols * nrows

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with pdf.PdfPages(pdf_path) as out:
        for i0 in range(0, len(unit_entries), per_page):
            chunk = unit_entries[i0 : i0 + per_page]
            fig, axes = plt.subplots(nrows, ncols, figsize=(11, 8.5), constrained_layout=True)
            axes = axes.ravel().tolist()

            for ax, entry in zip(axes, chunk):
                uid = entry.get("unit_id")
                title = f"unit {uid}"
                _write_template_overlay(
                    ax=ax,
                    template=entry["template"],
                    fs_hz=float(fs_hz),
                    ms_before=ms_before,
                    ms_after=ms_after,
                    top_channels=int(top_channels),
                    title=title,
                )

            for j in range(len(chunk), len(axes)):
                axes[j].set_axis_off()

            out.savefig(fig, dpi=150)
            plt.close(fig)

    logger.info("Wrote templates grid PDF -> %s", pdf_path)


def _write_unit_templates_across_sources_pdf(
    *,
    pdf_path: Path,
    unit_id: Any,
    sources_for_unit: list[dict[str, Any]],
    fs_hz: float,
    ms_before: Optional[float],
    ms_after: Optional[float],
    top_channels: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_pdf as pdf

    n = len(sources_for_unit)
    if n <= 0:
        return

    if n == 1:
        ncols, nrows = 1, 1
        figsize = (11, 8.5)
    else:
        ncols = 3
        nrows = int(math.ceil(n / ncols))
        figsize = (11, 3.0 * nrows)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with pdf.PdfPages(pdf_path) as out:
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
        import numpy as np  # type: ignore[import-not-found]

        axes_list = list(np.asarray(axes).ravel())

        for ax, src in zip(axes_list, sources_for_unit):
            # For merged_union specifically, plot *all* channels so nothing is hidden
            # by the top-N selection.
            tc = 0 if str(src.get("name")) == "merged_union" else int(top_channels)
            _write_template_overlay(
                ax=ax,
                template=src["template"],
                fs_hz=float(fs_hz),
                ms_before=ms_before,
                ms_after=ms_after,
                top_channels=int(tc),
                title=str(src["name"]),
            )

        for j in range(len(sources_for_unit), len(axes_list)):
            axes_list[j].set_axis_off()

        fig.suptitle(f"Templates overlay (unit {unit_id})", fontsize=12)
        out.savefig(fig, dpi=150)
        plt.close(fig)


def extract_and_merge_templates(*, inputs: TemplateExtractInputs, logger_name_prefix: str = "axon_reconstructor") -> TemplateExtractOutputs:
    """Extract templates from waveforms analyzers, save `.npy`, and produce QC PDFs.

    Mirrors the footprinting step's multi-source logic:
    - loads concat + per-segment waveforms analyzers
    - applies waveforms-stage unit curation (metrics_curated.xlsx) when available
    - handles missing units in some segments by skipping that source
    - builds a per-unit `merged_union` template across sources
    """

    well_out_dir = _compute_mea_analysis_output_dir(
        output_root=inputs.mea_output_root,
        data_file=inputs.h5_path,
        well=inputs.stream_id,
    )

    log_file = compute_pipeline_log_file(well_out_dir=well_out_dir, data_file=inputs.h5_path, stream_id=inputs.stream_id)
    logger = setup_pipeline_logger(
        log_file=log_file,
        logger_name=f"{logger_name_prefix}.{inputs.stream_id}.templates",
        verbose=True,
    )

    templates_out_dir = well_out_dir / TEMPLATES_OUTPUTS_DIRNAME
    extracted_templates_dir = templates_out_dir / "extracted_templates"
    merged_union_by_unit_dir = templates_out_dir / "merged_union_by_unit"

    summary_json = templates_out_dir / "templates_summary.json"
    templates_grid_pdf = templates_out_dir / "templates_grid_uncurated.pdf" if inputs.plot_templates_grid_pdf else None
    multi_source_templates_dir = templates_out_dir / "multi_source_by_unit" if inputs.plot_multi_source_templates_pdf else None

    ckpt_file = _compute_templates_checkpoint_file(well_out_dir=well_out_dir, h5_path=inputs.h5_path, stream_id=inputs.stream_id)
    ckpt = load_checkpoint(
        checkpoint_file=ckpt_file,
        force_restart=bool(inputs.force_restart),
        output_dir=well_out_dir,
        file_path=inputs.h5_path,
        stream_id=inputs.stream_id,
    )

    # Resume shortcut.
    if (
        (not inputs.force_restart)
        and extracted_templates_dir.exists()
        and summary_json.exists()
        and ((templates_grid_pdf is None) or templates_grid_pdf.exists())
    ):
        logger.info("Resuming templates: existing outputs found at %s", templates_out_dir)
        return TemplateExtractOutputs(
            well_out_dir=well_out_dir,
            templates_out_dir=templates_out_dir,
            extracted_templates_dir=extracted_templates_dir,
            merged_union_by_unit_dir=(merged_union_by_unit_dir if merged_union_by_unit_dir.exists() else None),
            summary_json=summary_json,
            templates_grid_pdf=(templates_grid_pdf if (templates_grid_pdf and templates_grid_pdf.exists()) else None),
            multi_source_templates_dir=(
                multi_source_templates_dir if (multi_source_templates_dir and multi_source_templates_dir.exists()) else None
            ),
        )

    ckpt = save_checkpoint(
        checkpoint_file=ckpt_file,
        state=ckpt,
        stage=ProcessingStage.ANALYZER,
        failed_stage=None,
        error=None,
        extra_fields={"templates_out_dir": str(templates_out_dir)},
    )

    templates_out_dir.mkdir(parents=True, exist_ok=True)
    extracted_templates_dir.mkdir(parents=True, exist_ok=True)
    merged_union_by_unit_dir.mkdir(parents=True, exist_ok=True)
    if multi_source_templates_dir is not None:
        multi_source_templates_dir.mkdir(parents=True, exist_ok=True)

    # Load waveforms analyzers (concat + segments).
    import numpy as np  # type: ignore[import-not-found]
    import spikeinterface.full as si  # type: ignore[import-not-found]

    from .footprinting import (
        _ensure_analyzer_extensions,
        _get_unit_template_from_extension,
        _load_curated_unit_ids_from_waveforms_outputs,
        _load_waveforms_analyzers,
        _normalize_id_for_compare,
    )

    # --- TEMPORARY monkey patch plumbing (waveforms-stage spike rejections) ---
    # We use waveforms-stage metadata to optionally prune segment-contributed channels
    # from merged_union templates. This is a stop-gap until we have principled,
    # spike-level propagation of waveform exclusions.
    from .waveforms import _load_wf_rejection_log_unit_counts

    wf_unit_counts_rows, wf_rej_xlsx = _load_wf_rejection_log_unit_counts(well_out_dir=well_out_dir, logger=logger)
    # Build: unit_id_norm -> set(source_name) that had any rejections.
    excluded_sources_by_unit: dict[Any, set[str]] = {}
    if wf_unit_counts_rows:
        for row in wf_unit_counts_rows:
            try:
                src_name = str(row.get("source_name"))
                reason = str(row.get("reason"))
                n_rej = int(row.get("n_rejected_spikes") or 0)
                unit_id_norm = _normalize_id_for_compare(row.get("unit_id"))
            except Exception:
                continue

            # Only consider segment-level rejections for this monkey patch.
            # (We avoid pruning concat channels, which are the reconstruction backbone.)
            if str(row.get("scope")) != "segment":
                continue
            if n_rej <= 0:
                continue

            # Restrict to the waveforms-exclusion reasons (vs. other future reasons).
            if reason not in {
                "outside_maxwell_epoch",
                "waveform_window_crosses_epoch_edge",
                "waveform_window_outside_segment_bounds",
            }:
                continue

            excluded_sources_by_unit.setdefault(unit_id_norm, set()).add(src_name)

    if excluded_sources_by_unit and wf_rej_xlsx is not None:
        logger.warning(
            "MONKEY PATCH enabled: will prune merged_union channels using waveforms-stage rejections from %s",
            wf_rej_xlsx,
        )

    analyzers = _load_waveforms_analyzers(
        well_out_dir=well_out_dir,
        include_concat=bool(inputs.include_concat),
        include_segments=bool(inputs.include_segments),
        logger=logger,
    )

    for _, an in analyzers:
        _ensure_analyzer_extensions(analyzer=an, extension_names=["templates"], logger=logger, n_jobs=int(inputs.n_jobs))

    # Determine unit list from concat if present, else from first source.
    unit_ids: list[Any]
    if inputs.unit_ids is not None:
        unit_ids = list(inputs.unit_ids)
    else:
        unit_ids = list(analyzers[0][1].sorting.unit_ids)

    # Apply waveforms-stage unit curation if available.
    curation_metrics_xlsx: Optional[Path] = None
    curated_units_norm: Optional[list[Any]] = None
    if inputs.unit_ids is None:
        curated_units_norm, curation_metrics_xlsx = _load_curated_unit_ids_from_waveforms_outputs(
            well_out_dir=well_out_dir,
            logger=logger,
        )
        if curated_units_norm is not None:
            curated_set = set(curated_units_norm)
            before = len(unit_ids)
            unit_ids = [uid for uid in unit_ids if _normalize_id_for_compare(uid) in curated_set]
            logger.info(
                "Applying waveforms curation: %d -> %d units (from %s)",
                before,
                len(unit_ids),
                curation_metrics_xlsx,
            )

    if inputs.unit_limit is not None:
        unit_ids = unit_ids[: int(inputs.unit_limit)]

    # Time window (for plotting). Best effort: read from waveforms params JSON if present.
    ms_before: Optional[float] = None
    ms_after: Optional[float] = None
    fs_hz: float
    try:
        fs_hz = float(analyzers[0][1].recording.get_sampling_frequency())
    except Exception:
        fs_hz = 10_000.0

    wf_params_json = well_out_dir / "waveforms_outputs" / "waveform_extraction_params.json"
    if wf_params_json.exists():
        try:
            params = _read_json(wf_params_json)
            ms_before = float(params.get("ms_before")) if params.get("ms_before") is not None else None
            ms_after = float(params.get("ms_after")) if params.get("ms_after") is not None else None
        except Exception:
            pass

    unit_grid_entries: list[dict[str, Any]] = []

    summary: dict[str, Any] = {
        "h5_path": str(inputs.h5_path),
        "stream_id": inputs.stream_id,
        "well_out_dir": str(well_out_dir),
        "templates_out_dir": str(templates_out_dir),
        "sources": [name for name, _ in analyzers],
        "curation": {
            "metrics_curated_xlsx": str(curation_metrics_xlsx) if curation_metrics_xlsx else None,
            "applied": bool(curated_units_norm is not None),
            "n_curated_units": int(len(curated_units_norm)) if curated_units_norm is not None else None,
        },
        "units": [],
    }

    for uid in unit_ids:
        sources_for_unit: list[dict[str, Any]] = []

        for name, an in analyzers:
            t_ext = an.get_extension("templates")
            tmpl = _get_unit_template_from_extension(analyzer=an, templates_ext=t_ext, unit_id=uid)
            if tmpl is None:
                continue
            tmpl = np.asarray(tmpl)
            if tmpl.ndim != 2 or tmpl.size == 0:
                continue

            locs = np.asarray(an.recording.get_channel_locations())
            try:
                ch_ids = list(an.recording.get_channel_ids())
            except Exception:
                ch_ids = None
            el_ids = _try_get_electrode_ids(an.recording)

            if tmpl.shape[1] != locs.shape[0]:
                continue

            sources_for_unit.append(
                {
                    "name": str(name),
                    "template": tmpl,
                    "channel_locations": locs,
                    "channel_ids": ch_ids,
                    "electrode_ids": el_ids,
                }
            )

        if not sources_for_unit:
            continue

        merged_union = _build_union_template_for_unit(sources_for_unit=sources_for_unit, logger=logger)
        if merged_union is not None:
            # TEMPORARY MONKEY PATCH:
            # Drop channels contributed by segment sources where waveforms-stage spike filtering
            # rejected spikes for this unit. This is a blunt heuristic to avoid carrying
            # problematic segment snippets into reconstruction artifacts.
            unit_norm = _normalize_id_for_compare(uid)
            excluded = excluded_sources_by_unit.get(unit_norm, set())
            merged_union = _apply_wf_exclusion_monkey_patch_to_merged_union(
                merged_union_src=merged_union,
                unit_id=uid,
                excluded_source_names=set(excluded),
                logger=logger,
            )
        if merged_union is not None:
            sources_for_unit_with_union = list(sources_for_unit) + [merged_union]
        else:
            sources_for_unit_with_union = list(sources_for_unit)

        # Save per-source templates.
        unit_entry: dict[str, Any] = {"unit_id": _jsonable(uid), "sources": []}
        for src in sources_for_unit_with_union:
            src_name = str(src["name"])
            tmpl = np.asarray(src["template"], dtype=float)
            locs = np.asarray(src["channel_locations"], dtype=float)

            if src_name == "merged_union":
                out_dir = merged_union_by_unit_dir / f"unit_{uid}"
                out_dir.mkdir(parents=True, exist_ok=True)
                npy_path = out_dir / "merged_union_template.npy"
                locs_npy = out_dir / "merged_union_channel_locations.npy"
                ch_ids_npy = out_dir / "merged_union_channel_ids.npy"
                el_ids_npy = out_dir / "merged_union_electrode_ids.npy"
                meta_path = out_dir / "merged_union_template_meta.json"
            else:
                out_dir = extracted_templates_dir / src_name
                out_dir.mkdir(parents=True, exist_ok=True)
                npy_path = out_dir / f"unit_{uid}.npy"
                meta_path = out_dir / f"unit_{uid}_meta.json"

            if (not npy_path.exists()) or inputs.force_restart:
                np.save(npy_path, tmpl)

            # For merged_union, also persist the merged channel identifiers/locations as arrays
            # (these are the primary inputs needed by downstream reconstruction).
            if src_name == "merged_union":
                try:
                    if (not locs_npy.exists()) or inputs.force_restart:
                        np.save(locs_npy, np.asarray(locs[:, :2], dtype=float))

                    ch_ids_seq = _jsonable_sequence(src.get("channel_ids"))
                    if (not ch_ids_npy.exists()) or inputs.force_restart:
                        np.save(ch_ids_npy, np.asarray(ch_ids_seq, dtype=object))

                    el_ids_seq = _jsonable_sequence(src.get("electrode_ids"))
                    if (not el_ids_npy.exists()) or inputs.force_restart:
                        np.save(el_ids_npy, np.asarray(el_ids_seq, dtype=object))
                except Exception as e:
                    logger.warning("Failed writing merged_union aux arrays for unit %s: %s", uid, e)
            if (not meta_path.exists()) or inputs.force_restart:
                meta = {
                    "unit_id": _jsonable(uid),
                    "source_name": src_name,
                    "template_npy": str(npy_path),
                    "channel_locations_npy": (str(locs_npy) if src_name == "merged_union" else None),
                    "channel_ids_npy": (str(ch_ids_npy) if src_name == "merged_union" else None),
                    "electrode_ids_npy": (str(el_ids_npy) if src_name == "merged_union" else None),
                    "sampling_frequency_hz": float(fs_hz),
                    "ms_before": ms_before,
                    "ms_after": ms_after,
                    "n_samples": int(tmpl.shape[0]),
                    "n_channels": int(tmpl.shape[1]),
                    "channel_ids": _jsonable_sequence(src.get("channel_ids")),
                    "electrode_ids": _jsonable_sequence(src.get("electrode_ids")),
                    "channel_locations": locs[:, :2].tolist(),
                }
                _write_json(meta_path, meta)

            unit_entry["sources"].append(
                {
                    "name": src_name,
                    "template_npy": str(npy_path),
                    "meta_json": str(meta_path),
                    "n_channels": int(tmpl.shape[1]),
                    "channel_locations_npy": (str(locs_npy) if src_name == "merged_union" else None),
                    "channel_ids_npy": (str(ch_ids_npy) if src_name == "merged_union" else None),
                }
            )

        summary["units"].append(unit_entry)

        # For grid PDF: take concat if present else first source.
        chosen = None
        for s in sources_for_unit:
            if str(s["name"]) == "concat":
                chosen = s
                break
        if chosen is None:
            chosen = sources_for_unit[0]

        unit_grid_entries.append({"unit_id": uid, "template": chosen["template"]})

        # Per-unit multi-source overlay PDF.
        if multi_source_templates_dir is not None:
            unit_dir = multi_source_templates_dir / f"unit_{uid}"
            unit_dir.mkdir(parents=True, exist_ok=True)
            unit_pdf = unit_dir / "templates.pdf"
            if (not unit_pdf.exists()) or inputs.force_restart:
                _write_unit_templates_across_sources_pdf(
                    pdf_path=unit_pdf,
                    unit_id=uid,
                    sources_for_unit=sources_for_unit_with_union,
                    fs_hz=float(fs_hz),
                    ms_before=ms_before,
                    ms_after=ms_after,
                    top_channels=int(inputs.top_channels_per_template),
                )

    # Write grid PDF.
    if templates_grid_pdf is not None:
        if (not templates_grid_pdf.exists()) or inputs.force_restart:
            _write_templates_grid_pdf(
                pdf_path=templates_grid_pdf,
                unit_entries=unit_grid_entries,
                fs_hz=float(fs_hz),
                ms_before=ms_before,
                ms_after=ms_after,
                top_channels=int(inputs.top_channels_per_template),
                logger=logger,
            )

    _write_json(summary_json, summary)

    ckpt = save_checkpoint(
        checkpoint_file=ckpt_file,
        state=ckpt,
        stage=ProcessingStage.ANALYZER_COMPLETE,
        failed_stage=None,
        error=None,
        extra_fields={
            "templates_out_dir": str(templates_out_dir),
            "extracted_templates_dir": str(extracted_templates_dir),
            "merged_union_by_unit_dir": str(merged_union_by_unit_dir),
            "templates_summary_json": str(summary_json),
            "templates_grid_pdf": str(templates_grid_pdf) if templates_grid_pdf else None,
        },
    )

    return TemplateExtractOutputs(
        well_out_dir=well_out_dir,
        templates_out_dir=templates_out_dir,
        extracted_templates_dir=extracted_templates_dir,
        merged_union_by_unit_dir=merged_union_by_unit_dir,
        summary_json=summary_json,
        templates_grid_pdf=templates_grid_pdf,
        multi_source_templates_dir=multi_source_templates_dir,
    )


__all__ = [
    "TemplateExtractInputs",
    "TemplateExtractOutputs",
    "extract_and_merge_templates",
]
