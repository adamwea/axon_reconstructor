"""Replot templates QC outputs from persisted template artifacts.

This module exists for debugging / iteration on plots without needing to re-run
waveform extraction or load SortingAnalyzers.

It intentionally reads only from the files produced by the templates step:
- merged templates: <well>/templates_outputs/merged_units/unit_<id>/...
- full templates (optional): <well>/templates_outputs/full_channels_templates/unit_<id>/...

Usage (module):
    python -m axon_reconstructor.pipeline.templates.replot \
        --well-out-dir /path/to/<well> \
        --unit-ids 9 10 11 \
        --force

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class PersistedMergedUnit:
    unit_id: Any
    template: Any
    channel_locations_xy: Any
    channel_ids: Any
    electrode_ids: Any
    footprint_ptp: Any


@dataclass(frozen=True)
class PersistedFullUnit:
    unit_id: Any
    full_template: Any
    full_channel_locations_xy: Any
    full_channel_ids: Any
    full_electrode_ids: Any
    contributing_full_channel_indices: Any


def _read_json(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _try_load_npy(path: Optional[Path]):
    if path is None:
        return None
    if not path.exists():
        return None
    import numpy as np  # type: ignore[import-not-found]

    try:
        return np.load(path, allow_pickle=True)
    except Exception:
        return None


def _load_all_recorded_electrode_ids(*, full_channels_templates_dir: Path):
    """Load the observed recording electrode universe (union over sources).

    This is used to distinguish:
      - quiet electrodes: not present in any recording/analyzer
      - non-contributing electrodes: present in recording but not contributing for this unit

    Source:
      <well>/templates_outputs/full_channels_templates/all_recorded_electrode_ids.npy
    """

    all_recorded_npy = Path(full_channels_templates_dir) / "all_recorded_electrode_ids.npy"
    if not all_recorded_npy.exists():
        return None

    arr = _try_load_npy(all_recorded_npy)
    if arr is None:
        return None

    try:
        return list(arr.ravel().tolist())
    except Exception:
        try:
            return list(arr.tolist())
        except Exception:
            return None


def _coerce_unit_id(s: str) -> Any:
    # Keep unit ids stable with upstream: int when possible.
    try:
        return int(s)
    except Exception:
        return s


def _iter_unit_ids_from_disk(merged_units_dir: Path) -> list[Any]:
    unit_ids: list[Any] = []
    if not merged_units_dir.exists():
        return unit_ids
    for p in sorted(merged_units_dir.glob("unit_*")):
        if not p.is_dir():
            continue
        try:
            raw = p.name.split("unit_", 1)[1]
        except Exception:
            continue
        unit_ids.append(_coerce_unit_id(raw))
    return unit_ids


def load_persisted_merged_unit(*, merged_units_dir: Path, unit_id: Any) -> Optional[PersistedMergedUnit]:
    """Load merged contributing-channels artifacts for a unit from disk."""

    unit_dir = merged_units_dir / f"unit_{unit_id}"
    meta = unit_dir / "merged_contributing_template_meta.json"
    if not meta.exists():
        return None

    meta_j = _read_json(meta)

    template = _try_load_npy(Path(meta_j.get("template_npy")) if meta_j.get("template_npy") else None)
    locs = _try_load_npy(Path(meta_j.get("channel_locations_npy")) if meta_j.get("channel_locations_npy") else None)
    ch_ids = _try_load_npy(Path(meta_j.get("channel_ids_npy")) if meta_j.get("channel_ids_npy") else None)
    el_ids = _try_load_npy(Path(meta_j.get("electrode_ids_npy")) if meta_j.get("electrode_ids_npy") else None)
    fp = _try_load_npy(Path(meta_j.get("footprint_ptp_npy")) if meta_j.get("footprint_ptp_npy") else None)

    if template is None or locs is None:
        return None

    return PersistedMergedUnit(
        unit_id=unit_id,
        template=template,
        channel_locations_xy=locs,
        channel_ids=ch_ids,
        electrode_ids=el_ids,
        footprint_ptp=fp,
    )


def load_persisted_full_unit(*, full_channels_templates_dir: Path, unit_id: Any) -> Optional[PersistedFullUnit]:
    """Load full_template artifacts for a unit from disk (optional output)."""

    unit_dir = full_channels_templates_dir / f"unit_{unit_id}"
    meta = unit_dir / "full_template_meta.json"
    if not meta.exists():
        return None

    meta_j = _read_json(meta)

    tmpl = _try_load_npy(Path(meta_j.get("full_template_npy")) if meta_j.get("full_template_npy") else None)
    locs = _try_load_npy(Path(meta_j.get("full_channel_locations_xy_npy")) if meta_j.get("full_channel_locations_xy_npy") else None)
    ch_ids = _try_load_npy(Path(meta_j.get("full_channel_ids_npy")) if meta_j.get("full_channel_ids_npy") else None)
    el_ids = _try_load_npy(Path(meta_j.get("full_electrode_ids_npy")) if meta_j.get("full_electrode_ids_npy") else None)
    contrib = _try_load_npy(
        Path(meta_j.get("contributing_full_channel_indices_npy"))
        if meta_j.get("contributing_full_channel_indices_npy")
        else None
    )

    if tmpl is None:
        return None

    return PersistedFullUnit(
        unit_id=unit_id,
        full_template=tmpl,
        full_channel_locations_xy=locs,
        full_channel_ids=ch_ids,
        full_electrode_ids=el_ids,
        contributing_full_channel_indices=contrib,
    )


def _load_plot_window_from_waveforms_outputs(*, well_out_dir: Path) -> tuple[float, Optional[float], Optional[float]]:
    """Best-effort plot window inference without analyzers."""

    fs_hz: float = 10_000.0
    ms_before: Optional[float] = None
    ms_after: Optional[float] = None

    params_json = well_out_dir / "waveforms_outputs" / "waveform_extraction_params.json"
    if params_json.exists():
        try:
            params = _read_json(params_json)
            if params.get("sampling_frequency_hz") is not None:
                fs_hz = float(params.get("sampling_frequency_hz"))
            if params.get("fs_hz") is not None:
                fs_hz = float(params.get("fs_hz"))
            if params.get("ms_before") is not None:
                ms_before = float(params.get("ms_before"))
            if params.get("ms_after") is not None:
                ms_after = float(params.get("ms_after"))
        except Exception:
            pass

    return fs_hz, ms_before, ms_after


def _maybe_rebuild_full_unit_from_merged(
    *,
    full_channels_templates_dir: Path,
    merged: PersistedMergedUnit,
    existing: Optional[PersistedFullUnit],
    unit_id: Any,
    rebuild_full_channels_templates: bool,
) -> Optional[PersistedFullUnit]:
    """Rebuild a full-chip (Maxwell) full template from merged template if needed.

    This is specifically for cases where the saved "full template" artifacts were
    built over a sparse channel universe (e.g. 268 channels) and we need a true
    26400-channel full-chip template for topo plots.
    """

    try:
        from .plotting import CHIP_COLS, CHIP_ROWS
        from .utils import _looks_like_maxwell_full_chip_electrode_ids, _maxwell_full_chip_channel_metadata

        if merged.electrode_ids is None:
            return existing
        if not _looks_like_maxwell_full_chip_electrode_ids(merged.electrode_ids):
            return existing

        n_chip = int(CHIP_COLS) * int(CHIP_ROWS)
        need = bool(rebuild_full_channels_templates)
        if not need:
            if existing is None or existing.full_template is None:
                need = True
            else:
                try:
                    need = int(existing.full_template.shape[1]) != int(n_chip)
                except Exception:
                    need = True
        if not need:
            return existing

        import json
        import numpy as np  # type: ignore[import-not-found]

        unit_dir = full_channels_templates_dir / f"unit_{unit_id}"
        unit_dir.mkdir(parents=True, exist_ok=True)

        locs_xy, ch_ids, el_ids = _maxwell_full_chip_channel_metadata()
        merged_template = np.asarray(merged.template)
        n_samples = int(merged_template.shape[0])

        full_template = np.zeros((n_samples, int(n_chip)), dtype=merged_template.dtype)
        contributing_inds: list[int] = []

        merged_eids = np.asarray(merged.electrode_ids, dtype=object).ravel().tolist()
        for j, e in enumerate(merged_eids):
            if e is None:
                continue
            try:
                idx = int(e)
            except Exception:
                continue
            if idx < 0 or idx >= int(n_chip):
                continue
            full_template[:, idx] = merged_template[:, int(j)]
            contributing_inds.append(idx)

        full_template_npy = unit_dir / "full_template.npy"
        full_locs_npy = unit_dir / "full_channel_locations_xy.npy"
        full_ch_ids_npy = unit_dir / "full_channel_ids.npy"
        full_el_ids_npy = unit_dir / "full_electrode_ids.npy"
        contrib_npy = unit_dir / "contributing_full_channel_indices.npy"
        meta_json = unit_dir / "full_template_meta.json"

        np.save(full_template_npy, full_template)
        np.save(full_locs_npy, np.asarray(locs_xy, dtype=float))
        np.save(full_ch_ids_npy, np.asarray(ch_ids, dtype=object))
        np.save(full_el_ids_npy, np.asarray(el_ids, dtype=object))
        np.save(contrib_npy, np.asarray(sorted(set(contributing_inds)), dtype=int))

        meta_payload = {
            "unit_id": unit_id,
            "full_template_npy": str(full_template_npy),
            "full_channel_locations_xy_npy": str(full_locs_npy),
            "full_channel_ids_npy": str(full_ch_ids_npy),
            "full_electrode_ids_npy": str(full_el_ids_npy),
            "contributing_full_channel_indices_npy": str(contrib_npy),
            "n_samples": int(full_template.shape[0]),
            "n_full_channels": int(full_template.shape[1]),
            "n_contributing_channels": int(len(set(contributing_inds))),
            "mapping_strategy": "electrode_ids",
            "full_chip_assumed": True,
        }

        # If available, also point to the persisted recording electrode universe so topo
        # can distinguish quiet vs non-contributing electrodes during replot.
        try:
            rec_path = Path(full_channels_templates_dir) / "all_recorded_electrode_ids.npy"
            if rec_path.exists():
                meta_payload["all_recorded_electrode_ids_npy"] = str(rec_path)
        except Exception:
            pass
        meta_json.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

        return load_persisted_full_unit(full_channels_templates_dir=full_channels_templates_dir, unit_id=unit_id)
    except Exception:
        return existing


def replot_unit_from_disk(
    *,
    well_out_dir: Path,
    unit_id: Any,
    force: bool = False,
    make_footprints: bool = True,
    make_svgs: bool = True,
    make_full_chip_maps: bool = True,
    make_topo: bool = True,
    make_propagation: bool = True,
    rebuild_full_channels_templates: bool = False,
    propagation_top_channels: int = 25,
    propagation_channels_per_panel: int = 25,
    propagation_channel_overlap: int = 5,
) -> None:
    """Regenerate QC plots for a unit using persisted artifacts only."""

    from .plotting import (
        _write_footprint_ptp_map,
        _write_full_chip_template_amplitude_map_png,
        _write_full_chip_template_peak_latency_map_png,
        _write_topo_unit_footprint_png,
        _write_unit_propagation_plots_png,
        _write_unit_template_and_footprint_svg,
    )

    templates_out_dir = well_out_dir / "templates_outputs"
    merged_units_dir = templates_out_dir / "merged_units"
    full_channels_templates_dir = templates_out_dir / "full_channels_templates"

    merged = load_persisted_merged_unit(merged_units_dir=merged_units_dir, unit_id=unit_id)
    if merged is None:
        raise FileNotFoundError(f"No merged_contributing artifacts found for unit {unit_id} under {merged_units_dir}")

    full = None
    if full_channels_templates_dir.exists():
        full = load_persisted_full_unit(full_channels_templates_dir=full_channels_templates_dir, unit_id=unit_id)

    full = _maybe_rebuild_full_unit_from_merged(
        full_channels_templates_dir=full_channels_templates_dir,
        merged=merged,
        existing=full,
        unit_id=unit_id,
        rebuild_full_channels_templates=bool(rebuild_full_channels_templates),
    )

    # Load the recording electrode universe (NOT the full-chip electrode list).
    # This is needed to distinguish:
    # - quiet electrodes (not present in recording)
    # - non-contributing electrodes (present in recording but not contributing)
    all_recorded_electrode_ids = None
    try:
        if full_channels_templates_dir.exists():
            all_recorded_electrode_ids = _load_all_recorded_electrode_ids(
                full_channels_templates_dir=full_channels_templates_dir,
            )
    except Exception:
        all_recorded_electrode_ids = None

    fs_hz, ms_before, ms_after = _load_plot_window_from_waveforms_outputs(well_out_dir=well_out_dir)

    footprints_dir = templates_out_dir / "footprints"
    svgs_dir = templates_out_dir / "svgs"
    full_chip_maps_dir = templates_out_dir / "full_chip_maps"
    topo_dir = templates_out_dir / "topo_unit_footprints"
    propagation_dir = templates_out_dir / "propagation_plots"

    if make_footprints:
        # Prefer persisted footprint if present; else compute from template.
        import numpy as np  # type: ignore[import-not-found]

        amp = merged.footprint_ptp
        if amp is None:
            amp = np.ptp(np.asarray(merged.template, dtype=float), axis=0).astype(float)

        out_lin = footprints_dir / f"unit_{unit_id}_merged_contributing_footprint_ptp_linear.png"
        out_log = footprints_dir / f"unit_{unit_id}_merged_contributing_footprint_ptp_log.png"
        if force or (not out_lin.exists()):
            _write_footprint_ptp_map(
                out_path=out_lin,
                channel_locations_xy=merged.channel_locations_xy,
                footprint_ptp=amp,
                title=f"Unit {unit_id} contributing-channels footprint (PTP)",
                log_scale=False,
                electrode_ids=merged.electrode_ids,
                all_recorded_electrode_ids=all_recorded_electrode_ids,
            )
        if force or (not out_log.exists()):
            _write_footprint_ptp_map(
                out_path=out_log,
                channel_locations_xy=merged.channel_locations_xy,
                footprint_ptp=amp,
                title=f"Unit {unit_id} contributing-channels footprint (PTP, log)",
                log_scale=True,
                electrode_ids=merged.electrode_ids,
                all_recorded_electrode_ids=all_recorded_electrode_ids,
            )

    if make_svgs:
        out_svg_lin = svgs_dir / f"unit_{unit_id}_merged_contributing_template_footprint_linear.svg"
        out_svg_log = svgs_dir / f"unit_{unit_id}_merged_contributing_template_footprint_log.svg"
        if force or (not out_svg_lin.exists()):
            _write_unit_template_and_footprint_svg(
                out_path=out_svg_lin,
                unit_id=unit_id,
                template=merged.template,
                channel_locations_xy=merged.channel_locations_xy,
                fs_hz=float(fs_hz),
                ms_before=ms_before,
                ms_after=ms_after,
                top_channels=8,
                log_footprint=False,
                electrode_ids=merged.electrode_ids,
                all_recorded_electrode_ids=all_recorded_electrode_ids,
            )
        if force or (not out_svg_log.exists()):
            _write_unit_template_and_footprint_svg(
                out_path=out_svg_log,
                unit_id=unit_id,
                template=merged.template,
                channel_locations_xy=merged.channel_locations_xy,
                fs_hz=float(fs_hz),
                ms_before=ms_before,
                ms_after=ms_after,
                top_channels=8,
                log_footprint=True,
                electrode_ids=merged.electrode_ids,
                all_recorded_electrode_ids=all_recorded_electrode_ids,
            )

    if make_full_chip_maps and merged.electrode_ids is not None:
        import numpy as np  # type: ignore[import-not-found]

        tmp_ch_by_t = np.asarray(merged.template, dtype=float).T
        out_amp = full_chip_maps_dir / f"unit_{unit_id}_template_amplitude_map_full_chip.png"
        out_lat = full_chip_maps_dir / f"unit_{unit_id}_template_peak_latency_map_full_chip.png"
        if force or (not out_amp.exists()):
            _write_full_chip_template_amplitude_map_png(
                out_path=out_amp,
                template_ch_by_t=tmp_ch_by_t,
                electrode_ids=merged.electrode_ids,
                all_recorded_electrode_ids=all_recorded_electrode_ids,
                title=f"Unit {unit_id} template amplitude (full chip)",
            )
        if force or (not out_lat.exists()):
            _write_full_chip_template_peak_latency_map_png(
                out_path=out_lat,
                template_ch_by_t=tmp_ch_by_t,
                electrode_ids=merged.electrode_ids,
                all_recorded_electrode_ids=all_recorded_electrode_ids,
                sampling_frequency_hz=float(fs_hz),
                title=f"Unit {unit_id} template peak latency (full chip)",
            )

    if make_topo and full is not None and full.full_electrode_ids is not None:
        out_topo = topo_dir / f"unit_{unit_id}.png"
        if force or (not out_topo.exists()):
            _write_topo_unit_footprint_png(
                out_path=out_topo,
                unit_id=unit_id,
                full_template=full.full_template,
                full_electrode_ids=full.full_electrode_ids,
                all_recorded_electrode_ids=all_recorded_electrode_ids,
                title=f"Unit {unit_id} topographical footprint (full template)",
            )

    if make_propagation:
        # PNG output (one file if one panel; multiple if multiple panels).
        out_png = propagation_dir / f"unit_{unit_id}.png"
        legacy_pdf = propagation_dir / f"unit_{unit_id}.pdf"
        if force and legacy_pdf.exists():
            try:
                legacy_pdf.unlink()
            except Exception:
                pass
        if force or (not out_png.exists()):
            _write_unit_propagation_plots_png(
                out_dir=propagation_dir,
                unit_id=unit_id,
                merged_contributing={
                    "template": merged.template,
                    "channel_locations": merged.channel_locations_xy,
                    "channel_ids": merged.channel_ids,
                    "electrode_ids": merged.electrode_ids,
                },
                fs_hz=float(fs_hz),
                ms_before=ms_before,
                ms_after=ms_after,
                top_channels=int(propagation_top_channels),
                n_waveforms=1,
                channels_per_panel=int(propagation_channels_per_panel),
                channel_overlap=int(propagation_channel_overlap),
                ap_timings_json_path=(merged_units_dir / f"unit_{unit_id}" / "ap_timings.json"),
                logger=None,
            )


def replot_templates_outputs_from_disk(
    *,
    well_out_dir: Path,
    unit_ids: Optional[Iterable[Any]] = None,
    force: bool = False,
    rebuild_full_channels_templates: bool = False,
    **kwargs,
) -> None:
    """Replot multiple units from disk."""

    templates_out_dir = well_out_dir / "templates_outputs"
    merged_units_dir = templates_out_dir / "merged_units"

    if unit_ids is None:
        unit_ids = _iter_unit_ids_from_disk(merged_units_dir)

    for uid in unit_ids:
        replot_unit_from_disk(
            well_out_dir=well_out_dir,
            unit_id=uid,
            force=bool(force),
            rebuild_full_channels_templates=bool(rebuild_full_channels_templates),
            **kwargs,
        )


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replot templates QC from persisted artifacts")
    p.add_argument("--well-out-dir", type=Path, required=True)
    p.add_argument("--unit-ids", type=str, nargs="*", default=None, help="Unit ids (space-separated). If omitted, plots all units found.")
    p.add_argument("--force", action="store_true", help="Overwrite existing plots")

    p.add_argument("--no-footprints", action="store_true")
    p.add_argument("--no-svgs", action="store_true")
    p.add_argument("--no-full-chip-maps", action="store_true")
    p.add_argument("--no-topo", action="store_true")
    p.add_argument("--no-propagation", action="store_true")

    p.add_argument(
        "--rebuild-full-channels-templates",
        action="store_true",
        help="Rebuild templates_outputs/full_channels_templates from merged_units using Maxwell full-chip electrode ids (fixes sparse full templates for topo plots).",
    )

    p.add_argument("--propagation-top-channels", type=int, default=25)
    p.add_argument("--propagation-channels-per-panel", type=int, default=25)
    p.add_argument("--propagation-channel-overlap", type=int, default=5)

    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    unit_ids = None
    if args.unit_ids:
        unit_ids = [_coerce_unit_id(x) for x in args.unit_ids]

    replot_templates_outputs_from_disk(
        well_out_dir=args.well_out_dir,
        unit_ids=unit_ids,
        force=bool(args.force),
        rebuild_full_channels_templates=bool(args.rebuild_full_channels_templates),
        make_footprints=not bool(args.no_footprints),
        make_svgs=not bool(args.no_svgs),
        make_full_chip_maps=not bool(args.no_full_chip_maps),
        make_topo=not bool(args.no_topo),
        make_propagation=not bool(args.no_propagation),
        propagation_top_channels=int(args.propagation_top_channels),
        propagation_channels_per_panel=int(args.propagation_channels_per_panel),
        propagation_channel_overlap=int(args.propagation_channel_overlap),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
