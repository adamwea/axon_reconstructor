from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..checkpointing import compute_checkpoint_file


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


def _try_get_electrode_ids(recording: Any) -> Optional[list[Any]]:
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


def _infer_location_tolerance(locs: Any) -> float:
    """Infer a tolerance for matching channel locations (in same units as locs)."""

    import numpy as np  # type: ignore[import-not-found]

    locs = np.asarray(locs)
    if locs.ndim != 2 or locs.shape[0] < 2:
        return 1e-6

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
    return (int(round(x / tol)), int(round(y / tol)))


def _build_union_template_for_unit(*, sources_for_unit: list[dict[str, Any]], logger: Any) -> Optional[dict[str, Any]]:
    """Build a merged-union template across sources.

    Keeps the first occurrence for overlapping channels (warns on overlaps).

    Returns a dict with fields compatible with `sources_for_unit` entries.
    """

    import numpy as np  # type: ignore[import-not-found]

    if not sources_for_unit:
        return None

    n_samples = int(np.asarray(sources_for_unit[0]["template"]).shape[0])

    try:
        tol = _infer_location_tolerance(sources_for_unit[0]["channel_locations"])
    except Exception:
        tol = 1e-6

    union_waveforms: list[np.ndarray] = []
    union_locs: list[np.ndarray] = []
    union_channel_ids: list[Any] = []
    union_electrode_ids: list[Any] = []
    union_source_names: list[str] = []

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
    logger: Any,
) -> dict[str, Any]:
    """TEMPORARY: drop channels from merged_union based on waveforms-stage rejections."""

    import numpy as np  # type: ignore[import-not-found]

    if not excluded_source_names:
        return merged_union_src

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


__all__ = [
    "_apply_wf_exclusion_monkey_patch_to_merged_union",
    "_build_union_template_for_unit",
    "_compute_templates_checkpoint_file",
    "_infer_location_tolerance",
    "_jsonable",
    "_jsonable_list",
    "_jsonable_sequence",
    "_loc_key",
    "_read_json",
    "_try_get_electrode_ids",
    "_write_json",
]
