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


def _looks_like_maxwell_full_chip_electrode_ids(electrode_ids: Any) -> bool:
    """Heuristic: detect Maxwell full-chip electrode id scheme.

    Maxwell electrode ids are expected to be integers in [0, CHIP_ROWS*CHIP_COLS).
    """

    try:
        import numpy as np  # type: ignore[import-not-found]

        from .plotting import CHIP_COLS, CHIP_ROWS

        if electrode_ids is None:
            return False
        arr = np.asarray(list(electrode_ids), dtype=object)
        if arr.size == 0:
            return False

        ints: list[int] = []
        for v in arr.tolist():
            if v is None:
                continue
            try:
                ints.append(int(v))
            except Exception:
                return False
        if not ints:
            return False

        mn = min(ints)
        mx = max(ints)
        if mn < 0:
            return False

        n = int(CHIP_COLS) * int(CHIP_ROWS)
        if mx >= n:
            return False
        return True
    except Exception:
        return False


def _maxwell_full_chip_channel_metadata():
    """Return (locations_xy, channel_ids, electrode_ids) for the Maxwell full chip."""

    import numpy as np  # type: ignore[import-not-found]

    from .plotting import CHIP_COLS, CHIP_PITCH_UM, CHIP_ROWS

    n = int(CHIP_COLS) * int(CHIP_ROWS)
    eids = np.arange(n, dtype=int)
    rows = (eids // int(CHIP_COLS)).astype(float)
    cols = (eids % int(CHIP_COLS)).astype(float)
    x_um = cols * float(CHIP_PITCH_UM)
    y_um = rows * float(CHIP_PITCH_UM)
    locs_xy = np.column_stack([x_um, y_um]).astype(float)

    # For this pipeline, channel_ids are only used as a fallback mapping; electrode_ids are preferred.
    # Using electrode id integers as channel ids keeps everything JSON/pickle-stable.
    ch_ids = eids.astype(object)
    el_ids = eids.astype(object)
    return locs_xy, ch_ids, el_ids


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


def _build_merged_contributing_template_for_unit(
    *,
    sources_for_unit: list[dict[str, Any]],
    unit_id: Any,
    logger: Any,
) -> Optional[dict[str, Any]]:
    """Build a merged contributing-channels template across sources.

    Default strategy for overlapping channels is to compute a merged per-channel
    waveform by stacking the underlying waveforms from each contributing source
    (segment) and taking the mean. This mimics the conceptual template
    computation and avoids keep-first bias.

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

    contributing_waveforms: list[np.ndarray] = []
    contributing_locs: list[np.ndarray] = []
    contributing_channel_ids: list[Any] = []
    contributing_electrode_ids: list[Any] = []
    contributing_source_names: list[str] = []

    from axon_reconstructor.pipeline.templates.overlaps import (  # local import to avoid circular deps
        WaveformContribution,
        robust_baseline_pre_negative_peak,
        mean_waveform_from_contributions,
    )

    key_to_index: dict[Any, int] = {}
    overlap_count = 0
    overlap_details: list[dict[str, Any]] = []
    contribs_by_key: dict[Any, list[WaveformContribution]] = {}

    for src in sources_for_unit:
        name = str(src["name"])
        tmpl = np.asarray(src["template"])
        locs = np.asarray(src["channel_locations"])
        ch_ids = src.get("channel_ids")
        el_ids = src.get("electrode_ids")

        if tmpl.ndim != 2 or tmpl.shape[0] != n_samples:
            logger.warning("Skipping %s for merged_contributing: bad template shape %s", name, tmpl.shape)
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

                # Track overlap contributions for later waveform-mean merge.
                try:
                    an = src.get("_analyzer")
                    if an is not None:
                        ch_ref = None
                        try:
                            if el_ids is not None:
                                ch_ref = el_ids[j]
                        except Exception:
                            ch_ref = None
                        if ch_ref is None:
                            try:
                                if ch_ids is not None:
                                    ch_ref = ch_ids[j]
                            except Exception:
                                ch_ref = None
                        contribs_by_key.setdefault(key, []).append(
                            WaveformContribution(
                                source_name=str(name),
                                analyzer=an,
                                unit_id=unit_id,
                                channel_ref=(ch_ref if ch_ref is not None else int(j)),
                            )
                        )
                except Exception:
                    pass

                continue

            key_to_index[key] = len(contributing_waveforms)
            # Center per-channel waveform so merged_contributing is baseline-consistent across sources.
            try:
                wf = np.asarray(tmpl[:, j], dtype=float)
                wf = wf - robust_baseline_pre_negative_peak(wf)
            except Exception:
                wf = np.asarray(tmpl[:, j], dtype=float)
            contributing_waveforms.append(wf)
            contributing_locs.append(np.asarray(locs[j, :2], dtype=float))
            contributing_source_names.append(str(name))
            try:
                contributing_channel_ids.append(None if ch_ids is None else ch_ids[j])
            except Exception:
                contributing_channel_ids.append(None)
            try:
                contributing_electrode_ids.append(None if el_ids is None else el_ids[j])
            except Exception:
                contributing_electrode_ids.append(None)

            # Seed overlap contributions for this key with the first occurrence.
            try:
                an = src.get("_analyzer")
                if an is not None:
                    ch_ref = None
                    try:
                        if el_ids is not None:
                            ch_ref = el_ids[j]
                    except Exception:
                        ch_ref = None
                    if ch_ref is None:
                        try:
                            if ch_ids is not None:
                                ch_ref = ch_ids[j]
                        except Exception:
                            ch_ref = None
                    contribs_by_key.setdefault(key, []).append(
                        WaveformContribution(
                            source_name=str(name),
                            analyzer=an,
                            unit_id=unit_id,
                            channel_ref=(ch_ref if ch_ref is not None else int(j)),
                        )
                    )
            except Exception:
                pass

    if not contributing_waveforms:
        return None

    # Resolve overlaps by recomputing the per-channel waveform using the mean of
    # all contributing waveforms across sources.
    overlap_resolved = 0
    for key, contribs in contribs_by_key.items():
        if len(contribs) <= 1:
            continue
        idx = key_to_index.get(key)
        if idx is None:
            continue

        try:
            merged_wf = mean_waveform_from_contributions(contributions=contribs, logger=logger)
        except Exception:
            merged_wf = None

        if merged_wf is None:
            # Fallback: keep the first template column (existing behavior).
            continue

        if int(merged_wf.shape[0]) != int(n_samples):
            continue

        # Ensure overlap-resolved waveform is centered too.
        try:
            mw = np.asarray(merged_wf, dtype=float)
            mw = mw - robust_baseline_pre_negative_peak(mw)
            contributing_waveforms[idx] = mw
        except Exception:
            contributing_waveforms[idx] = merged_wf
        overlap_resolved += 1
        try:
            overlap_details.append(
                {
                    "key": (str(key[0]), _jsonable(key[1]) if len(key) > 1 else None),
                    "channel_index": int(idx),
                    "sources": sorted({c.source_name for c in contribs}),
                    "n_contributions": int(len(contribs)),
                    "strategy": "mean_waveforms",
                }
            )
        except Exception:
            pass

    if overlap_count:
        logger.warning(
            "merged_contributing: encountered %d overlapping channels; resolved=%d via mean-waveforms (kept first otherwise)",
            int(overlap_count),
            int(overlap_resolved),
        )

    merged = np.stack(contributing_waveforms, axis=1)
    merged_locs = np.stack(contributing_locs, axis=0)

    return {
        "name": "merged_contributing",
        "template": merged,
        "channel_locations": merged_locs,
        "channel_ids": contributing_channel_ids,
        "electrode_ids": contributing_electrode_ids,
        "channel_source_names": contributing_source_names,
        "stats": {
            "overlap_encountered": int(overlap_count),
            "overlap_resolved": int(overlap_resolved),
            "overlap_strategy": "mean_waveforms",
            "n_contributing_channels": int(merged.shape[1]),
        },
        "overlap": {"channels": overlap_details} if overlap_details else None,
    }


__all__ = [
    "_build_merged_contributing_template_for_unit",
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
