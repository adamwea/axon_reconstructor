from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..checkpointing import compute_checkpoint_file


def _infer_location_tolerance(locations: "Any") -> float:
    """Infer a reasonable coordinate-space tolerance for location matching.

    Channel locations are typically in µm for MEA data, but can be mm or meters.
    We use a 0.5 µm tolerance (scaled to the inferred units).
    """

    import numpy as np  # type: ignore[import-not-found]

    locs = np.asarray(locations)
    if locs.size == 0:
        return 0.0
    max_coord = float(np.nanmax(np.abs(locs)))
    # Heuristic matching the one used for electrode square sizing.
    if max_coord > 100.0:
        return 0.5
    if max_coord > 1.0:
        return 0.5 / 1000.0
    return 0.5e-6


def _try_get_recording_property(recording: Any, key: str):
    try:
        if hasattr(recording, "get_property_keys"):
            keys = set(recording.get_property_keys())
            if key not in keys:
                return None
        if hasattr(recording, "get_property"):
            return recording.get_property(key)
    except Exception:
        return None
    return None


def _try_get_electrode_ids(recording: Any):
    """Best-effort extraction of an 'electrode id' per channel.

    SpikeInterface recordings can carry various per-channel properties.
    We check a few common candidates; if none exist, returns None.
    """

    for key in (
        "electrode_id",
        "electrode",
        "contact_id",
        "contact_ids",
        "contact",
        "site_id",
        "site",
    ):
        vals = _try_get_recording_property(recording, key)
        if vals is not None:
            return vals
    return None


def _sparsity_unit_channel_indices(*, sparsity, unit_id: Any):
    """Return per-unit channel indices from a `ChannelSparsity` object.

    SpikeInterface API differs across versions:
    - Some versions expose `unit_id_to_channel_indices` as a callable.
    - Others (e.g. SI 0.103.x) expose it as a dict.
    """

    if sparsity is None:
        return None

    try:
        mapping = getattr(sparsity, "unit_id_to_channel_indices", None)
        if callable(mapping):
            return mapping(unit_id)
        if isinstance(mapping, dict):
            if unit_id in mapping:
                return mapping[unit_id]
            # Best-effort normalized match.
            try:
                from ..waveforms.exclusions import normalize_unit_id

                uid_norm = normalize_unit_id(unit_id)
                for k, v in mapping.items():
                    if normalize_unit_id(k) == uid_norm:
                        return v
            except Exception:
                pass
    except Exception:
        pass

    # Additional fallbacks across versions.
    for attr in ("get_channel_indices", "get_channel_indices_for_unit"):
        if hasattr(sparsity, attr):
            fn = getattr(sparsity, attr)
            if callable(fn):
                try:
                    return fn(unit_id)
                except Exception:
                    pass

    return None


def _get_unit_template_from_waveforms_with_exclusions(
    *,
    analyzer,
    unit_id: Any,
    excluded_spike_samples: Optional[set[int]],
    logger,
    return_result: bool = False,
):
    """Compute a unit template from waveforms, applying spike-level exclusions.

    Deprecated: the pipeline no longer persists or consumes `wf_exclusions.npz` by default.
    This helper exists for backwards-compatible plotting code paths.
    """

    try:
        from ..waveforms.exclusions import compute_unit_template_from_waveforms

        res = compute_unit_template_from_waveforms(
            analyzer=analyzer,
            unit_id=unit_id,
            excluded_spike_samples=excluded_spike_samples,
            logger=logger,
        )
        if return_result:
            return res
        return None if res is None else res.template
    except Exception:
        return None


def _normalize_id_for_compare(x: Any) -> Any:
    """Normalize ids so np scalars / floats round-trip consistently."""

    try:
        if hasattr(x, "item"):
            x = x.item()
    except Exception:
        pass

    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float) and x.is_integer():
        return int(x)
    return str(x)


def _load_curated_unit_ids_from_waveforms_outputs(*, well_out_dir: Path, logger) -> tuple[Optional[list[Any]], Optional[Path]]:
    """Load curated (kept) unit ids from waveforms outputs, if available.

    Waveforms stage writes MEA_Analysis-style curation artifacts, including:
      <well>/waveforms_outputs/metrics_curated.xlsx

    We treat the index of that spreadsheet as the curated/kept unit ids.
    """

    metrics_curated_xlsx = well_out_dir / "waveforms_outputs" / "metrics_curated.xlsx"
    if not metrics_curated_xlsx.exists():
        return None, None

    try:
        import pandas as pd  # type: ignore[import-not-found]
    except Exception:
        logger.warning("Found %s but pandas is unavailable; cannot apply unit curation", metrics_curated_xlsx)
        return None, metrics_curated_xlsx

    try:
        df = pd.read_excel(metrics_curated_xlsx, index_col=0)
        curated = [_normalize_id_for_compare(x) for x in list(df.index.values)]

        seen: set[Any] = set()
        curated_unique: list[Any] = []
        for u in curated:
            if u in seen:
                continue
            seen.add(u)
            curated_unique.append(u)
        return curated_unique, metrics_curated_xlsx
    except Exception as e:
        logger.warning("Failed reading curated unit list from %s: %s", metrics_curated_xlsx, e)
        return None, metrics_curated_xlsx


def _load_wf_rejection_log_summary(*, well_out_dir: Path, logger) -> tuple[Optional[dict[str, Any]], Optional[Path]]:
    """Load waveforms-stage per-spike rejection log summary (best effort)."""

    wf_rej_xlsx = well_out_dir / "waveforms_outputs" / "wf_rejection_log.xlsx"
    if not wf_rej_xlsx.exists():
        return None, None

    try:
        import pandas as pd  # type: ignore[import-not-found]

        df = pd.read_excel(wf_rej_xlsx, sheet_name="summary")
        summary: dict[str, Any] = {"wf_rejection_log_xlsx": str(wf_rej_xlsx)}
        if not df.empty and {"metric", "value"}.issubset(set(df.columns)):
            for _, row in df.iterrows():
                try:
                    summary[str(row["metric"])] = row["value"]
                except Exception:
                    continue
        return summary, wf_rej_xlsx
    except Exception as e:
        logger.warning("Failed to read wf_rejection_log.xlsx summary: %s", e)
        return {"wf_rejection_log_xlsx": str(wf_rej_xlsx), "error": str(e)}, wf_rej_xlsx


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _compute_footprinting_checkpoint_file(*, well_out_dir: Path, h5_path: Path, stream_id: str) -> Path:
    main_ckpt = compute_checkpoint_file(output_dir=well_out_dir, file_path=h5_path, stream_id=stream_id)
    name = main_ckpt.name
    if name.endswith("_checkpoint.json"):
        name = name[: -len("_checkpoint.json")] + "_footprinting_checkpoint.json"
    else:
        name = main_ckpt.stem + "_footprinting_checkpoint.json"
    return main_ckpt.with_name(name)


def _ensure_analyzer_extensions(*, analyzer, extension_names: list[str], logger, n_jobs: int) -> None:
    missing = [name for name in extension_names if not analyzer.has_extension(name)]
    if not missing:
        return
    logger.info("Computing extensions: %s", ", ".join(missing))
    analyzer.compute(missing, verbose=False, n_jobs=max(1, int(n_jobs)))


def _load_waveforms_analyzers(
    *,
    well_out_dir: Path,
    include_concat: bool,
    include_segments: bool,
    logger,
):
    """Load analyzers produced by the waveforms stage.

    Returns a list of (source_name, analyzer).
    """

    import spikeinterface.full as si  # type: ignore[import-not-found]

    waveforms_out_dir = well_out_dir / "waveforms_outputs"
    concat_waveforms_dir = waveforms_out_dir / "concat_waveforms"
    segment_waveforms_dir = waveforms_out_dir / "segment_waveforms"

    analyzers: list[tuple[str, Any]] = []

    if include_concat:
        if not concat_waveforms_dir.exists():
            raise FileNotFoundError(f"Missing concat waveforms analyzer at {concat_waveforms_dir}")
        logger.info("Loading concat analyzer: %s", concat_waveforms_dir)
        analyzers.append(("concat", si.load_sorting_analyzer(concat_waveforms_dir)))

    if include_segments and segment_waveforms_dir.exists():
        seg_dirs = sorted([p for p in segment_waveforms_dir.iterdir() if p.is_dir()])
        logger.info("Found %d segment analyzers", len(seg_dirs))
        for p in seg_dirs:
            try:
                analyzers.append((p.name, si.load_sorting_analyzer(p)))
            except Exception:
                logger.warning("Skipping unreadable segment analyzer: %s", p)

    if not analyzers:
        raise RuntimeError("No analyzers available for footprinting")

    return analyzers


def _get_unit_template_from_extension(*, analyzer, templates_ext, unit_id: Any):
    """Compatibility helper for SpikeInterface templates extension."""

    try:
        if hasattr(analyzer, "sorting") and hasattr(analyzer.sorting, "id_to_index"):
            analyzer.sorting.id_to_index(unit_id)
    except Exception:
        return None

    if hasattr(templates_ext, "get_unit_template"):
        try:
            return templates_ext.get_unit_template(unit_id=unit_id)
        except Exception:
            return None

    if hasattr(templates_ext, "get_templates"):
        try:
            all_templates = templates_ext.get_templates()
            unit_index = list(analyzer.sorting.unit_ids).index(unit_id)
            return all_templates[unit_index]
        except Exception:
            return None

    return None


def _build_union_source_for_unit(
    *,
    sources: list[dict[str, Any]],
    unit_id: Any,
    logger,
) -> Optional[dict[str, Any]]:
    """Union channels across per-source footprints for a unit.

    If overlaps are detected (by channel_id, electrode_id, or location within tolerance),
    we try to resolve them by averaging the underlying waveforms across the
    contributing sources and recomputing the amplitude (ptp) for that channel.

    If waveform access fails for an overlap group, we fall back to keep-first.
    """

    import numpy as np  # type: ignore[import-not-found]

    if not sources:
        return None

    all_locs = [np.asarray(s.get("channel_locations")) for s in sources if s.get("channel_locations") is not None]
    if not all_locs:
        return None
    stacked = np.concatenate(all_locs, axis=0)
    tol = float(_infer_location_tolerance(stacked))
    if tol <= 0:
        tol = 0.0

    def loc_key(x: float, y: float) -> tuple[int, int]:
        if tol <= 0:
            return (int(round(x * 1e6)), int(round(y * 1e6)))
        return (int(round(x / tol)), int(round(y / tol)))

    union_locs: list[list[float]] = []
    union_amp: list[float] = []
    union_channel_ids: list[Any] = []
    union_electrode_ids: list[Any] = []

    from axon_reconstructor.pipeline.overlap import (  # local import to avoid import cycles
        WaveformContribution,
        mean_waveform_from_contributions,
        ptp_amplitude_from_mean_waveform,
    )

    channel_id_to_index: dict[Any, int] = {}
    electrode_id_to_index: dict[Any, int] = {}
    loc_key_to_index: dict[tuple[int, int], int] = {}

    contribs_by_index: dict[int, list[WaveformContribution]] = {}
    overlap_details: list[dict[str, Any]] = []

    overlap_counts = {"channel_id": 0, "electrode_id": 0, "location": 0}

    for src in sources:
        src_name = str(src.get("name"))
        locs = np.asarray(src.get("channel_locations"))
        amp = np.asarray(src.get("amp"))
        ch_ids = src.get("channel_ids")
        el_ids = src.get("electrode_ids")
        an = src.get("_analyzer")

        if locs.size == 0 or amp.size == 0 or locs.shape[0] != amp.shape[0]:
            continue

        try:
            if ch_ids is not None:
                ch_list = list(ch_ids)
                if len(set(ch_list)) != len(ch_list):
                    logger.warning("Duplicate channel_ids within source %s (unit %s)", src.get("name"), unit_id)
        except Exception:
            pass

        for i in range(locs.shape[0]):
            x = float(locs[i, 0])
            y = float(locs[i, 1])
            lk = loc_key(x, y)

            cid = None
            if ch_ids is not None:
                try:
                    cid = ch_ids[i]
                except Exception:
                    cid = None

            eid = None
            if el_ids is not None:
                try:
                    eid = el_ids[i]
                except Exception:
                    eid = None

            # Determine whether this (source, channel) overlaps an existing union entry.
            overlap_type: Optional[str] = None
            union_index: Optional[int] = None

            if cid is not None and cid in channel_id_to_index:
                overlap_type = "channel_id"
                union_index = channel_id_to_index[cid]
            elif eid is not None and eid in electrode_id_to_index:
                overlap_type = "electrode_id"
                union_index = electrode_id_to_index[eid]
            elif lk in loc_key_to_index:
                overlap_type = "location"
                union_index = loc_key_to_index[lk]

            if union_index is not None and overlap_type is not None:
                overlap_counts[overlap_type] += 1
                try:
                    if an is not None:
                        contribs_by_index.setdefault(int(union_index), []).append(
                            WaveformContribution(
                                source_name=src_name,
                                analyzer=an,
                                unit_id=unit_id,
                                channel_ref=(cid if cid is not None else int(i)),
                            )
                        )
                except Exception:
                    pass
                continue

            # First occurrence: add new union entry and seed contributions.
            idx = len(union_locs)
            loc_key_to_index[lk] = idx
            if cid is not None:
                channel_id_to_index[cid] = idx
            if eid is not None:
                electrode_id_to_index[eid] = idx

            union_locs.append([x, y])
            union_amp.append(float(amp[i]))
            union_channel_ids.append(cid)
            union_electrode_ids.append(eid)

            try:
                if an is not None:
                    contribs_by_index.setdefault(idx, []).append(
                        WaveformContribution(
                            source_name=src_name,
                            analyzer=an,
                            unit_id=unit_id,
                            channel_ref=(cid if cid is not None else int(i)),
                        )
                    )
            except Exception:
                pass

    total_overlaps = sum(overlap_counts.values())
    overlap_resolved = 0

    # Resolve overlaps by recomputing amp from the mean waveform across contributions.
    if union_amp:
        for idx, contribs in contribs_by_index.items():
            if len(contribs) <= 1:
                continue
            try:
                mean_wf = mean_waveform_from_contributions(contributions=contribs, logger=logger)
                merged_amp = ptp_amplitude_from_mean_waveform(mean_wf)
            except Exception:
                merged_amp = None

            if merged_amp is None:
                continue

            if 0 <= int(idx) < len(union_amp):
                union_amp[int(idx)] = float(merged_amp)
                overlap_resolved += 1
                try:
                    overlap_details.append(
                        {
                            "channel_index": int(idx),
                            "sources": sorted({c.source_name for c in contribs}),
                            "n_contributions": int(len(contribs)),
                            "strategy": "mean_waveforms",
                        }
                    )
                except Exception:
                    pass

    if total_overlaps:
        logger.warning(
            "Merged footprint overlaps for unit %s: %s (resolved=%d via mean-waveforms; keep-first fallback)",
            unit_id,
            overlap_counts,
            int(overlap_resolved),
        )

    if not union_locs:
        return None

    union_locs_arr = np.asarray(union_locs)
    union_amp_arr = np.asarray(union_amp)

    return {
        "name": "merged_union",
        "channel_locations": union_locs_arr,
        "amp": union_amp_arr,
        "best_ch": int(np.argmax(union_amp_arr)) if union_amp_arr.size else 0,
        "n_channels": int(union_locs_arr.shape[0]),
        "channel_ids": union_channel_ids,
        "electrode_ids": union_electrode_ids,
        "merge": {
            "location_tolerance": tol,
            "overlap_counts": overlap_counts,
            "overlap_strategy": "mean_waveforms",
            "overlap_resolved": int(overlap_resolved),
            "overlap_details": overlap_details if overlap_details else None,
        },
    }


__all__ = [
    "_infer_location_tolerance",
    "_try_get_electrode_ids",
    "_sparsity_unit_channel_indices",
    "_get_unit_template_from_waveforms_with_exclusions",
    "_normalize_id_for_compare",
    "_load_curated_unit_ids_from_waveforms_outputs",
    "_load_wf_rejection_log_summary",
    "_write_json",
    "_compute_footprinting_checkpoint_file",
    "_ensure_analyzer_extensions",
    "_load_waveforms_analyzers",
    "_get_unit_template_from_extension",
    "_build_union_source_for_unit",
]
