from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def _gather_template_sources_for_unit(
    *,
    uid: Any,
    analyzers: list[tuple[str, Any]],
    get_template_from_extension,
    sparsity_unit_channel_indices,
    try_get_electrode_ids,
) -> list[dict[str, Any]]:
    """Gather per-source template + channel metadata for one unit.

    Uses the analyzer's `templates` extension (preferred) and supports sparse templates by
    subsetting channel locations/ids according to the analyzer's sparsity.

    Args:
        uid: Unit id.
        analyzers: List of (source_name, SortingAnalyzer).
        get_template_from_extension: Callable compatible with templates helpers.
        sparsity_unit_channel_indices: Callable that extracts per-unit channel indices from a sparsity object.
        try_get_electrode_ids: Callable that best-effort extracts electrode ids.

    Returns:
        List of dicts with keys: name, template, channel_locations, channel_ids, electrode_ids.
    """

    import numpy as np  # type: ignore[import-not-found]

    sources_for_unit: list[dict[str, Any]] = []

    # Defensive carry-through of waveforms-stage channel policy:
    # if concat is present, per-segment sources should contribute only channels
    # outside concat/common electrodes.
    common_electrode_ids: Optional[set[int]] = None
    try:
        concat_an = None
        for src_name, src_an in analyzers:
            if str(src_name) == "concat":
                concat_an = src_an
                break
        if concat_an is not None:
            concat_el_ids = try_get_electrode_ids(concat_an.recording)
            if concat_el_ids is not None:
                electrodes = np.asarray(concat_el_ids, dtype=int)
                if electrodes.size > 0:
                    common_electrode_ids = set(int(x) for x in electrodes.tolist())
    except Exception:
        common_electrode_ids = None

    for name, an in analyzers:
        tmpl = None
        try:
            t_ext = an.get_extension("templates") if an.has_extension("templates") else None
            if t_ext is not None:
                tmpl = get_template_from_extension(analyzer=an, templates_ext=t_ext, unit_id=uid)
        except Exception:
            tmpl = None

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
        el_ids = try_get_electrode_ids(an.recording)

        # Prefer sparsity-aware channel selection when available.
        # In some SpikeInterface versions, the templates extension returns *dense*
        # templates (n_channels == recording.get_num_channels()) with zeros outside
        # the waveforms sparsity. Downstream merging should operate on the sparse
        # channel set to avoid false overlaps.
        try:
            sp = getattr(an, "sparsity", None)
            if sp is None and an.has_extension("waveforms"):
                sp = getattr(an.get_extension("waveforms"), "sparsity", None)
            if sp is not None:
                ch_inds = sparsity_unit_channel_indices(sparsity=sp, unit_id=uid)
                ch_inds = np.asarray(ch_inds, dtype=int)

                # Determine whether the template is dense.
                n_rec_ch = None
                try:
                    if ch_ids is not None:
                        n_rec_ch = int(len(ch_ids))
                except Exception:
                    n_rec_ch = None
                if n_rec_ch is None:
                    try:
                        n_rec_ch = int(locs.shape[0])
                    except Exception:
                        n_rec_ch = None

                if n_rec_ch is not None and int(tmpl.shape[1]) == int(n_rec_ch) and int(ch_inds.size) > 0 and int(ch_inds.size) < int(n_rec_ch):
                    tmpl = tmpl[:, ch_inds]
                    locs = locs[ch_inds, :]
                    if ch_ids is not None:
                        ch_ids = list(np.asarray(ch_ids, dtype=object)[ch_inds])
                    if el_ids is not None:
                        try:
                            el_ids = list(np.asarray(el_ids, dtype=object)[ch_inds])
                        except Exception:
                            pass
        except Exception:
            pass

        # Support sparse templates by subsetting locations/ids according to sparsity.
        if tmpl.shape[1] != locs.shape[0]:
            try:
                sp = getattr(an, "sparsity", None)
                if sp is None and an.has_extension("waveforms"):
                    sp = getattr(an.get_extension("waveforms"), "sparsity", None)
                if sp is not None:
                    ch_inds = sparsity_unit_channel_indices(sparsity=sp, unit_id=uid)
                    ch_inds = np.asarray(ch_inds, dtype=int)
                    if int(ch_inds.size) == int(tmpl.shape[1]):
                        locs = locs[ch_inds, :]
                        if ch_ids is not None:
                            ch_ids = list(np.asarray(ch_ids, dtype=object)[ch_inds])
                        if el_ids is not None:
                            try:
                                el_ids = list(np.asarray(el_ids, dtype=object)[ch_inds])
                            except Exception:
                                pass
            except Exception:
                pass

        if tmpl.shape[1] != locs.shape[0]:
            continue

        # Enforce "segments contribute only non-common channels" if possible.
        if common_electrode_ids and str(name) != "concat" and el_ids is not None:
            try:
                el_arr = np.asarray(el_ids, dtype=object)
                keep_mask = np.asarray([int(x) not in common_electrode_ids for x in el_arr], dtype=bool)
                if int(keep_mask.size) == int(tmpl.shape[1]) and (not bool(np.all(keep_mask))):
                    tmpl = tmpl[:, keep_mask]
                    locs = locs[keep_mask, :]
                    if ch_ids is not None:
                        ch_ids = list(np.asarray(ch_ids, dtype=object)[keep_mask])
                    el_ids = list(el_arr[keep_mask])
            except Exception:
                pass

        if int(tmpl.shape[1]) == 0:
            continue

        sources_for_unit.append(
            {
                "name": str(name),
                "template": tmpl,
                "channel_locations": locs,
                "channel_ids": ch_ids,
                "electrode_ids": el_ids,
                # Runtime-only pointer used for overlap resolution (not serialized).
                "_analyzer": an,
            }
        )

    return sources_for_unit


def _choose_grid_source_for_unit(*, sources_for_unit: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Choose a representative source for grid plotting (prefer concat)."""

    if not sources_for_unit:
        return None
    for s in sources_for_unit:
        if str(s.get("name")) == "concat":
            return s
    return sources_for_unit[0]


def _persist_unit_templates(
    *,
    uid: Any,
    sources_for_unit_with_merged: list[dict[str, Any]],
    extracted_templates_dir: Path,
    merged_units_dir: Path,
    merged_unit_full_chip_maps_dir: Path,
    axon_velocity_outputs_root_dir: Optional[Path] = None,
    full_channels_templates_dir: Optional[Path] = None,
    full_channel_locations_xy: Any = None,
    full_channel_ids: Any = None,
    full_electrode_ids: Any = None,
    fs_hz: float,
    fs_hz_native: Optional[float] = None,
    template_time_upsample_factor: int = 1,
    template_time_upsample_method: str = "sinc",
    ms_before: Optional[float],
    ms_after: Optional[float],
    all_recorded_electrode_ids: Any = None,
    make_axon_velocity_plots: bool = False,
    jsonable,
    jsonable_sequence,
    write_json,
    force_restart: bool,
    logger,
) -> dict[str, Any]:
    """Persist per-unit templates to disk and return a JSON-able unit summary entry."""

    import numpy as np  # type: ignore[import-not-found]

    up_factor = int(template_time_upsample_factor or 1)
    up_method = str(template_time_upsample_method or "sinc")
    fs_native = (
        float(fs_hz_native)
        if fs_hz_native is not None
        else (float(fs_hz) / float(up_factor) if up_factor > 1 else float(fs_hz))
    )

    unit_entry: dict[str, Any] = {
        "unit_id": jsonable(uid),
        "sources": [],
        "template_time_upsampling": {
            "enabled": bool(up_factor > 1),
            "factor": int(up_factor),
            "method": str(up_method),
            "native_sampling_frequency_hz": float(fs_native),
            "effective_sampling_frequency_hz": float(fs_hz),
        },
    }

    for src in sources_for_unit_with_merged:
        src_name = str(src["name"])
        tmpl = np.asarray(src["template"], dtype=float)
        locs = np.asarray(src["channel_locations"], dtype=float)

        if src_name == "merged_contributing":
            data_dir = merged_units_dir / f"unit_{uid}"
            data_dir.mkdir(parents=True, exist_ok=True)

            # Full-chip QC maps are organized by map type under:
            #   stg4_templates_outputs/footprints/full_chip_maps/{amplitude,peak_latency}/
            amp_maps_dir = Path(merged_unit_full_chip_maps_dir) / "amplitude"
            lat_maps_dir = Path(merged_unit_full_chip_maps_dir) / "peak_latency"
            amp_maps_dir.mkdir(parents=True, exist_ok=True)
            lat_maps_dir.mkdir(parents=True, exist_ok=True)

            npy_path = data_dir / "merged_contributing_template.npy"
            locs_npy = data_dir / "merged_contributing_channel_locations.npy"
            ch_ids_npy = data_dir / "merged_contributing_channel_ids.npy"
            el_ids_npy = data_dir / "merged_contributing_electrode_ids.npy"
            footprint_ptp_npy = data_dir / "merged_contributing_footprint_ptp.npy"
            axon_velocity_npz = data_dir / "axon_velocity_inputs.npz"

            template_amplitude_png = amp_maps_dir / f"unit_{uid}_template_amplitude_map_full_chip.png"
            template_peak_latency_png = lat_maps_dir / f"unit_{uid}_template_peak_latency_map_full_chip.png"
            axon_velocity_plots_dir = (Path(axon_velocity_outputs_root_dir) / f"unit_{uid}") if axon_velocity_outputs_root_dir is not None else None
            meta_path = data_dir / "merged_contributing_template_meta.json"
        else:
            out_dir = extracted_templates_dir / src_name
            out_dir.mkdir(parents=True, exist_ok=True)
            npy_path = out_dir / f"unit_{uid}.npy"
            meta_path = out_dir / f"unit_{uid}_meta.json"
            locs_npy = None
            ch_ids_npy = None
            el_ids_npy = None
            footprint_ptp_npy = None
            axon_velocity_npz = None
            template_amplitude_png = None
            template_peak_latency_png = None
            axon_velocity_plots_dir = None

        if (not npy_path.exists()) or force_restart:
            np.save(npy_path, tmpl)

        # For merged_contributing, also persist the merged channel identifiers/locations as arrays
        # (these are the primary inputs needed by downstream reconstruction).
        if src_name == "merged_contributing":
            try:
                assert locs_npy is not None and ch_ids_npy is not None and el_ids_npy is not None
                assert footprint_ptp_npy is not None and axon_velocity_npz is not None

                if (not locs_npy.exists()) or force_restart:
                    np.save(locs_npy, np.asarray(locs[:, :2], dtype=float))

                ch_ids_seq = jsonable_sequence(src.get("channel_ids"))
                if (not ch_ids_npy.exists()) or force_restart:
                    np.save(ch_ids_npy, np.asarray(ch_ids_seq, dtype=object))

                el_ids_seq = jsonable_sequence(src.get("electrode_ids"))
                if (not el_ids_npy.exists()) or force_restart:
                    np.save(el_ids_npy, np.asarray(el_ids_seq, dtype=object))

                # Footprint amplitude (ptp across time for each channel).
                # `tmpl` is (n_samples, n_channels).
                if (not footprint_ptp_npy.exists()) or force_restart:
                    try:
                        np.save(footprint_ptp_npy, np.ptp(tmpl, axis=0).astype(float))
                    except Exception:
                        # Keep this best-effort; reconstruction only needs template+locations.
                        pass

                # Convenience bundle for axon_velocity usage (channels x time expected).
                if (not axon_velocity_npz.exists()) or force_restart:
                    try:
                        np.savez(
                            axon_velocity_npz,
                            unit_id=jsonable(uid),
                            template_ch_by_t=np.asarray(tmpl, dtype=float).T,
                            locations_xy=np.asarray(locs[:, :2], dtype=float),
                            sampling_frequency_hz=float(fs_hz),
                            channel_ids=np.asarray(ch_ids_seq, dtype=object) if ch_ids_seq is not None else None,
                            electrode_ids=np.asarray(el_ids_seq, dtype=object) if el_ids_seq is not None else None,
                        )
                    except Exception:
                        pass

                # Best-effort template QC maps (full-chip rendering, no external axon_velocity dependency).
                try:
                    assert template_amplitude_png is not None and template_peak_latency_png is not None
                    from axon_reconstructor.pipeline.templates.plotting import (
                        _write_full_chip_template_amplitude_map_png,
                        _write_full_chip_template_peak_latency_map_png,
                    )

                    tmpl_ch_by_t = np.asarray(tmpl, dtype=float).T

                    if (not template_amplitude_png.exists()) or force_restart:
                        _write_full_chip_template_amplitude_map_png(
                            out_path=template_amplitude_png,
                            template_ch_by_t=tmpl_ch_by_t,
                            electrode_ids=el_ids_seq,
                            all_recorded_electrode_ids=all_recorded_electrode_ids,
                            title="Amplitude map",
                            cmap="viridis",
                        )

                    if (not template_peak_latency_png.exists()) or force_restart:
                        _write_full_chip_template_peak_latency_map_png(
                            out_path=template_peak_latency_png,
                            template_ch_by_t=tmpl_ch_by_t,
                            electrode_ids=el_ids_seq,
                            all_recorded_electrode_ids=all_recorded_electrode_ids,
                            sampling_frequency_hz=float(fs_hz),
                            title="Peak latency map",
                            cmap="viridis",
                        )
                except Exception:
                    pass

                # Clean up legacy naming so merged output dirs don't keep old axon_velocity-like QC maps.
                if axon_velocity_outputs_root_dir is not None:
                    try:
                        legacy_plots_dir = Path(axon_velocity_outputs_root_dir).parent / "merged_unit_plots" / f"unit_{uid}"
                        legacy_amp = legacy_plots_dir / "axon_velocity_amplitude_map.png"
                        legacy_lat = legacy_plots_dir / "axon_velocity_peak_latency_map.png"
                        legacy_std = legacy_plots_dir / "axon_velocity_peak_std_map.png"
                        if template_amplitude_png is not None and template_amplitude_png.exists() and legacy_amp.exists():
                            legacy_amp.unlink()
                        if template_peak_latency_png is not None and template_peak_latency_png.exists() and legacy_lat.exists():
                            legacy_lat.unlink()
                        if legacy_std.exists():
                            legacy_std.unlink()
                    except Exception:
                        pass

                # Optional: real axon_velocity integration outputs (separate folder).
                if bool(make_axon_velocity_plots) and axon_velocity_plots_dir is not None and axon_velocity_npz is not None:
                    try:
                        from axon_reconstructor.pipeline.templates.av_plotting import (
                            try_write_axon_velocity_plots_from_npz,
                        )

                        if force_restart or (not axon_velocity_plots_dir.exists()):
                            axon_velocity_plots_dir.mkdir(parents=True, exist_ok=True)
                        try_write_axon_velocity_plots_from_npz(
                            npz_path=axon_velocity_npz,
                            out_dir=axon_velocity_plots_dir,
                            unit_id=jsonable(uid),
                        )
                    except Exception:
                        pass

                # Optional: persist a full-channel (dense) template for reconstruction.
                # This places the merged contributing-channels template into the reference recording's channel order,
                # with zeros for channels that were not contributing for this unit.
                if full_channels_templates_dir is not None:
                    try:
                        # Persist the *all recorded electrodes* universe (across sources) once under
                        # full_channels_templates so disk-only replotting can distinguish:
                        # - quiet electrodes: not present in any recording
                        # - non-contributing electrodes: present in recording but not contributing
                        all_recorded_eids_npy = Path(full_channels_templates_dir) / "all_recorded_electrode_ids.npy"
                        all_recorded_eids_meta = Path(full_channels_templates_dir) / "all_recorded_electrode_ids_meta.json"
                        all_recorded_eids_seq = jsonable_sequence(all_recorded_electrode_ids)
                        if all_recorded_eids_seq is not None and len(all_recorded_eids_seq) > 0:
                            if (not all_recorded_eids_npy.exists()) or force_restart:
                                np.save(all_recorded_eids_npy, np.asarray(all_recorded_eids_seq, dtype=object))
                            if (not all_recorded_eids_meta.exists()) or force_restart:
                                write_json(
                                    all_recorded_eids_meta,
                                    {
                                        "all_recorded_electrode_ids_npy": str(all_recorded_eids_npy),
                                        "n_all_recorded_electrodes": int(len(all_recorded_eids_seq)),
                                        "definition": "electrode ids that appear in at least one templates-stage recording/analyzer (across sources)",
                                    },
                                )

                        unit_full_dir = Path(full_channels_templates_dir) / f"unit_{uid}"
                        unit_full_dir.mkdir(parents=True, exist_ok=True)

                        full_template_npy = unit_full_dir / "full_template.npy"
                        full_locs_npy = unit_full_dir / "full_channel_locations_xy.npy"
                        full_ch_ids_npy = unit_full_dir / "full_channel_ids.npy"
                        full_el_ids_npy = unit_full_dir / "full_electrode_ids.npy"
                        full_contrib_inds_npy = unit_full_dir / "contributing_full_channel_indices.npy"
                        full_meta_json = unit_full_dir / "full_template_meta.json"

                        full_locs = None
                        try:
                            full_locs = np.asarray(full_channel_locations_xy, dtype=float)
                        except Exception:
                            full_locs = None

                        # Persist channel metadata (shared across units but kept per-unit for convenience).
                        if full_locs is not None and full_locs.ndim == 2 and int(full_locs.shape[0]) > 0:
                            if (not full_locs_npy.exists()) or force_restart:
                                np.save(full_locs_npy, np.asarray(full_locs[:, :2], dtype=float))

                        full_ch_ids_seq = jsonable_sequence(full_channel_ids)
                        if full_ch_ids_seq is not None:
                            if (not full_ch_ids_npy.exists()) or force_restart:
                                np.save(full_ch_ids_npy, np.asarray(full_ch_ids_seq, dtype=object))

                        full_el_ids_seq = jsonable_sequence(full_electrode_ids)
                        if full_el_ids_seq is not None:
                            if (not full_el_ids_npy.exists()) or force_restart:
                                np.save(full_el_ids_npy, np.asarray(full_el_ids_seq, dtype=object))

                        # Build mapping from merged contributing-channels -> full channel indices.
                        n_full = None
                        if full_locs is not None and full_locs.ndim == 2:
                            n_full = int(full_locs.shape[0])
                        elif full_ch_ids_seq is not None:
                            n_full = int(len(full_ch_ids_seq))

                        if n_full is not None and n_full > 0:
                            # Prefer electrode id mapping (most stable across sources).
                            contrib_inds: list[int] = []
                            full_template = np.zeros((int(tmpl.shape[0]), int(n_full)), dtype=float)

                            merged_el_ids_seq = jsonable_sequence(src.get("electrode_ids"))
                            merged_ch_ids_seq = jsonable_sequence(src.get("channel_ids"))

                            el_to_index = None
                            if full_el_ids_seq is not None:
                                try:
                                    el_to_index = {int(e): int(i) for i, e in enumerate(full_el_ids_seq) if e is not None}
                                except Exception:
                                    el_to_index = None

                            ch_to_index = None
                            if el_to_index is None and full_ch_ids_seq is not None:
                                try:
                                    ch_to_index = {str(c): int(i) for i, c in enumerate(full_ch_ids_seq) if c is not None}
                                except Exception:
                                    ch_to_index = None

                            # Location fallback.
                            loc_to_index = None
                            tol = None
                            if el_to_index is None and ch_to_index is None and full_locs is not None:
                                try:
                                    from axon_reconstructor.pipeline.templates.utils import _infer_location_tolerance, _loc_key

                                    tol = float(_infer_location_tolerance(full_locs))
                                    loc_to_index = {_loc_key(xy, tol): int(i) for i, xy in enumerate(np.asarray(full_locs)[:, :2])}
                                except Exception:
                                    loc_to_index = None

                            for j in range(int(tmpl.shape[1])):
                                idx = None
                                if el_to_index is not None and merged_el_ids_seq is not None:
                                    try:
                                        e = merged_el_ids_seq[j]
                                        if e is not None:
                                            idx = el_to_index.get(int(e))
                                    except Exception:
                                        idx = None
                                if idx is None and ch_to_index is not None and merged_ch_ids_seq is not None:
                                    try:
                                        c = merged_ch_ids_seq[j]
                                        if c is not None:
                                            idx = ch_to_index.get(str(c))
                                    except Exception:
                                        idx = None
                                if idx is None and loc_to_index is not None and tol is not None:
                                    try:
                                        key = _loc_key(locs[j, :2], float(tol))
                                        idx = loc_to_index.get(key)
                                    except Exception:
                                        idx = None

                                if idx is None:
                                    continue

                                full_template[:, int(idx)] = tmpl[:, j]
                                contrib_inds.append(int(idx))

                            if (not full_template_npy.exists()) or force_restart:
                                np.save(full_template_npy, full_template)
                            if (not full_contrib_inds_npy.exists()) or force_restart:
                                np.save(full_contrib_inds_npy, np.asarray(sorted(set(contrib_inds)), dtype=int))

                            if (not full_meta_json.exists()) or force_restart:
                                write_json(
                                    full_meta_json,
                                    {
                                        "unit_id": jsonable(uid),
                                        "full_template_npy": str(full_template_npy),
                                        "full_channel_locations_xy_npy": (str(full_locs_npy) if full_locs is not None else None),
                                        "full_channel_ids_npy": (str(full_ch_ids_npy) if full_ch_ids_seq is not None else None),
                                        "full_electrode_ids_npy": (str(full_el_ids_npy) if full_el_ids_seq is not None else None),
                                        "contributing_full_channel_indices_npy": str(full_contrib_inds_npy),
                                        "all_recorded_electrode_ids_npy": (
                                            str(all_recorded_eids_npy)
                                            if all_recorded_eids_seq is not None and len(all_recorded_eids_seq) > 0
                                            else None
                                        ),
                                        "n_all_recorded_electrodes": (
                                            int(len(all_recorded_eids_seq)) if all_recorded_eids_seq is not None else None
                                        ),
                                        "n_samples": int(tmpl.shape[0]),
                                        "n_full_channels": int(n_full),
                                        "n_contributing_channels": int(len(set(contrib_inds))),
                                        "sampling_frequency_hz": float(fs_hz),
                                        "native_sampling_frequency_hz": float(fs_native),
                                        "template_time_upsample_factor": int(up_factor),
                                        "template_time_upsample_method": str(up_method),
                                        "mapping_strategy": (
                                            "electrode_ids" if el_to_index is not None else ("channel_ids" if ch_to_index is not None else "locations")
                                        ),
                                    },
                                )
                    except Exception as e:
                        logger.warning("Failed writing full_channels_templates for unit %s: %s", uid, e)
            except Exception as e:
                logger.warning("Failed writing merged_contributing aux arrays for unit %s: %s", uid, e)

        if (not meta_path.exists()) or force_restart:
            meta = {
                "unit_id": jsonable(uid),
                "source_name": src_name,
                "template_npy": str(npy_path),
                "channel_locations_npy": (
                    str(locs_npy) if src_name == "merged_contributing" and locs_npy is not None else None
                ),
                "channel_ids_npy": (
                    str(ch_ids_npy) if src_name == "merged_contributing" and ch_ids_npy is not None else None
                ),
                "electrode_ids_npy": (
                    str(el_ids_npy) if src_name == "merged_contributing" and el_ids_npy is not None else None
                ),
                "footprint_ptp_npy": (
                    str(footprint_ptp_npy)
                    if src_name == "merged_contributing" and footprint_ptp_npy is not None
                    else None
                ),
                "axon_velocity_inputs_npz": (
                    str(axon_velocity_npz) if src_name == "merged_contributing" and axon_velocity_npz is not None else None
                ),
                "template_amplitude_map_full_chip_png": (
                    str(template_amplitude_png)
                    if src_name == "merged_contributing" and template_amplitude_png is not None
                    else None
                ),
                "template_peak_latency_map_full_chip_png": (
                    str(template_peak_latency_png)
                    if src_name == "merged_contributing" and template_peak_latency_png is not None
                    else None
                ),
                "axon_velocity_outputs_dir": (
                    str(axon_velocity_plots_dir)
                    if src_name == "merged_contributing" and axon_velocity_plots_dir is not None
                    else None
                ),
                "sampling_frequency_hz": float(fs_hz),
                "native_sampling_frequency_hz": float(fs_native),
                "template_time_upsample_factor": int(up_factor),
                "template_time_upsample_method": str(up_method),
                "ms_before": ms_before,
                "ms_after": ms_after,
                "n_samples": int(tmpl.shape[0]),
                "n_channels": int(tmpl.shape[1]),
                "channel_ids": jsonable_sequence(src.get("channel_ids")),
                "electrode_ids": jsonable_sequence(src.get("electrode_ids")),
                "channel_locations": locs[:, :2].tolist(),
                # Best-effort extra diagnostics for merged_contributing.
                "stats": (src.get("stats") if src_name == "merged_contributing" else None),
                "overlap": (src.get("overlap") if src_name == "merged_contributing" else None),
            }
            write_json(meta_path, meta)

        unit_entry["sources"].append(
            {
                "name": src_name,
                "template_npy": str(npy_path),
                "meta_json": str(meta_path),
                "n_channels": int(tmpl.shape[1]),
                "channel_locations_npy": (
                    str(locs_npy) if src_name == "merged_contributing" and locs_npy is not None else None
                ),
                "channel_ids_npy": (
                    str(ch_ids_npy) if src_name == "merged_contributing" and ch_ids_npy is not None else None
                ),
            }
        )

    return unit_entry
