from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..checkpointing import (
    ProcessingStage,
    compute_checkpoint_file,
    exception_to_error_dict,
    load_checkpoint,
    save_checkpoint,
)
from ..pipeline_logging import compute_pipeline_log_file, setup_pipeline_logger
from ..pipeline_driver import _compute_mea_analysis_output_dir

from .plotting import write_all_units_overview_pdf, write_unit_reconstruction_pdfs

RECONSTRUCTION_OUTPUTS_DIRNAME = "reconstruction_outputs"


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _jsonable(x: Any) -> Any:
    try:
        import numpy as np  # type: ignore[import-not-found]

        if isinstance(x, (np.integer, np.floating)):
            return x.item()
    except Exception:
        pass
    if isinstance(x, Path):
        return str(x)
    return x


def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, (str, bytes)):
        return [x]
    try:
        import numpy as np  # type: ignore[import-not-found]

        if isinstance(x, np.ndarray):
            return x.ravel().tolist()
    except Exception:
        pass
    try:
        return list(x)
    except Exception:
        return [x]


def _as_float_list(x: Any) -> list[float]:
    out: list[float] = []
    for v in _as_list(x):
        try:
            out.append(float(v))
        except Exception:
            continue
    return out


def _as_int_list(x: Any) -> list[int]:
    out: list[int] = []
    for v in _as_list(x):
        try:
            out.append(int(v))
        except Exception:
            continue
    return out


def _compute_reconstruction_checkpoint_file(*, well_out_dir: Path, h5_path: Path, stream_id: str) -> Path:
    main_ckpt = compute_checkpoint_file(output_dir=well_out_dir, file_path=h5_path, stream_id=stream_id)
    name = main_ckpt.name
    if name.endswith("_checkpoint.json"):
        name = name[: -len("_checkpoint.json")] + "_reconstruction_checkpoint.json"
    else:
        name = main_ckpt.stem + "_reconstruction_checkpoint.json"
    return main_ckpt.with_name(name)


def _filter_kwargs_for_callable(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter a dict to only kwargs accepted by `fn` (best effort)."""

    try:
        import inspect

        sig = inspect.signature(fn)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return dict(kwargs)


@dataclass(frozen=True)
class ReconstructionInputs:
    h5_path: Path
    stream_id: str
    mea_output_root: Path

    # Units to reconstruct. If None, reconstruct all units found under templates_outputs/templates/merged.
    unit_ids: Optional[list[Any]] = None
    unit_limit: Optional[int] = None

    # Template source for axon_velocity.
    #
    # axon_velocity is most robust when given a dense full-channel template on a deterministic
    # geometry (e.g. Maxwell full chip). By default we therefore consume templates-stage outputs
    # under `<well>/templates_outputs/templates/full/`.
    #
    # Set `use_full_channels_templates=False` to fall back to the sparse merged-contributing
    # template under `<well>/templates_outputs/templates/merged/`.
    use_full_channels_templates: bool = True

    # If True, raise if full-channel templates are missing.
    require_full_channels_templates: bool = True

    # axon_velocity params (merged onto defaults). Only keys accepted by
    # axon_velocity.compute_graph_propagation_velocity are passed through.
    axon_velocity_params: Optional[dict[str, Any]] = None

    # Plotting / outputs
    write_unit_pdfs: bool = True
    write_all_units_overview_pdf: bool = True

    # If True, skip axon_velocity tracking and only (re)render per-unit summary plots
    # from already-written reconstruction/templates artifacts on disk.
    replot_summaries_only: bool = False

    # If True, run axon_velocity tracking but only (re)write branches_raw.json (raw paths)
    # and skip all other per-unit plots/outputs.
    recompute_branches_raw_only: bool = False

    # Runtime
    verbose: bool = False

    # Resume controls
    force_restart: bool = False


@dataclass(frozen=True)
class ReconstructionOutputs:
    well_out_dir: Path
    reconstruction_out_dir: Path
    summary_json: Path
    by_unit_dir: Path
    all_units_overview_pdf: Optional[Path]


def reconstruct_from_templates(*, inputs: ReconstructionInputs, logger_name_prefix: str = "axon_reconstructor") -> ReconstructionOutputs:
    """Run axon reconstruction/velocity estimation using axon_velocity.

        This stage primarily consumes the dense full-channel template artifacts from templates extraction:
            <well>/templates_outputs/templates/full/unit_<id>/full_template.npy
            <well>/templates_outputs/templates/full/unit_<id>/full_channel_locations_xy.npy

        It also reads (best-effort) sampling frequency metadata from:
            <well>/templates_outputs/templates/merged/unit_<id>/merged_contributing_template_meta.json

    And produces:
      <well>/reconstruction_outputs/
        reconstruction_summary.json
        all_units_morphology.pdf (optional)
        by_unit/unit_<id>/ ... per-unit json + pdfs

    Focused subset (requested):
    - sparse morphology per unit (branches as polylines)
    - heuristics per unit (selected channels + node heuristics)
    - velocity plots per unit/branch
    - all-units morphology overview
    """

    well_out_dir = _compute_mea_analysis_output_dir(
        output_root=inputs.mea_output_root,
        data_file=inputs.h5_path,
        well=inputs.stream_id,
    )

    log_file = compute_pipeline_log_file(well_out_dir=well_out_dir, data_file=inputs.h5_path, stream_id=inputs.stream_id)
    logger = setup_pipeline_logger(
        log_file=log_file,
        logger_name=f"{logger_name_prefix}.{inputs.stream_id}.reconstruction",
        verbose=True,
    )

    recon_out_dir = well_out_dir / RECONSTRUCTION_OUTPUTS_DIRNAME
    by_unit_dir = recon_out_dir / "by_unit"
    summary_json = recon_out_dir / "reconstruction_summary.json"
    if bool(inputs.replot_summaries_only) and bool(inputs.recompute_branches_raw_only):
        raise ValueError("replot_summaries_only and recompute_branches_raw_only are mutually exclusive")

    # In modes that skip normal plotting, avoid generating the all-units overview artifact.
    all_units_overview_pdf = (
        None
        if (bool(inputs.replot_summaries_only) or bool(inputs.recompute_branches_raw_only))
        else (recon_out_dir / "all_units_morphology.pdf" if inputs.write_all_units_overview_pdf else None)
    )

    ckpt_file = _compute_reconstruction_checkpoint_file(well_out_dir=well_out_dir, h5_path=inputs.h5_path, stream_id=inputs.stream_id)
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
        and (not bool(inputs.replot_summaries_only))
        and (not bool(inputs.recompute_branches_raw_only))
        and summary_json.exists()
        and ((all_units_overview_pdf is None) or all_units_overview_pdf.exists())
        and by_unit_dir.exists()
    ):
        logger.info("Resuming reconstruction: existing outputs found at %s", recon_out_dir)
        return ReconstructionOutputs(
            well_out_dir=well_out_dir,
            reconstruction_out_dir=recon_out_dir,
            summary_json=summary_json,
            by_unit_dir=by_unit_dir,
            all_units_overview_pdf=(all_units_overview_pdf if (all_units_overview_pdf and all_units_overview_pdf.exists()) else None),
        )

    ckpt = save_checkpoint(
        checkpoint_file=ckpt_file,
        state=ckpt,
        stage=ProcessingStage.ANALYZER,
        failed_stage=None,
        error=None,
        extra_fields={"reconstruction_out_dir": str(recon_out_dir)},
    )

    recon_out_dir.mkdir(parents=True, exist_ok=True)
    by_unit_dir.mkdir(parents=True, exist_ok=True)

    templates_out_dir = well_out_dir / "templates_outputs"
    templates_dir = templates_out_dir / "templates"

    merged_units_dir = templates_dir / "merged"
    full_channels_templates_dir = templates_dir / "full"

    # Legacy fallback.
    if not merged_units_dir.exists():
        legacy = templates_out_dir / "merged_units"
        if legacy.exists():
            merged_units_dir = legacy
    if not full_channels_templates_dir.exists():
        legacy = templates_out_dir / "full_channels_templates"
        if legacy.exists():
            full_channels_templates_dir = legacy
    if not merged_units_dir.exists():
        raise FileNotFoundError(f"Missing merged templates at {merged_units_dir}")
    if bool(inputs.use_full_channels_templates) and bool(inputs.require_full_channels_templates):
        if not full_channels_templates_dir.exists():
            raise FileNotFoundError(
                "Reconstruction is configured to require full-channel templates for axon_velocity, "
                f"but {full_channels_templates_dir} does not exist. "
                "Re-run templates with save_full_channels_templates=True, or set "
                "ReconstructionInputs(require_full_channels_templates=False)."
            )

    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Reconstruction requires numpy") from e

    try:
        import axon_velocity as av  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        # Best-effort fallback: user often has axon_velocity as an editable checkout,
        # but the active interpreter may not have it installed.
        axon_velocity_repo = Path("/home/adamm/dev/pkgs/axon_velocity")
        if (axon_velocity_repo / "axon_velocity").exists():
            sys.path.insert(0, str(axon_velocity_repo))
            try:
                import axon_velocity as av  # type: ignore[import-not-found]
            except Exception as e2:
                msg = str(e2)
                if "No module named 'sklearn'" in msg or "No module named sklearn" in msg:
                    raise RuntimeError(
                        "axon_velocity imported, but its dependency scikit-learn is missing. Install it in the active env (e.g. `pip install scikit-learn` or `conda install scikit-learn`)."
                    ) from e2
                raise RuntimeError(
                    "Reconstruction requires axon_velocity; attempted sys.path fallback to /home/adamm/dev/pkgs/axon_velocity"
                ) from e2
        else:
            raise RuntimeError(
                "Reconstruction requires axon_velocity (expected editable install at /home/adamm/dev/pkgs/axon_velocity)"
            ) from e

    # Determine unit list.
    discovered_unit_ids: list[Any] = []
    for p in sorted(merged_units_dir.glob("unit_*")):
        if not p.is_dir():
            continue
        try:
            discovered_unit_ids.append(int(p.name.split("unit_", 1)[1]))
        except Exception:
            discovered_unit_ids.append(p.name.split("unit_", 1)[1])

    if inputs.unit_ids is not None:
        unit_ids = list(inputs.unit_ids)
    else:
        unit_ids = list(discovered_unit_ids)

    if inputs.unit_limit is not None:
        unit_ids = unit_ids[: int(inputs.unit_limit)]

    params = av.get_default_graph_velocity_params()
    if inputs.axon_velocity_params:
        params.update(dict(inputs.axon_velocity_params))
    params = _filter_kwargs_for_callable(av.compute_graph_propagation_velocity, params)
    params.setdefault("verbose", bool(inputs.verbose))

    summary: dict[str, Any] = {
        "h5_path": str(inputs.h5_path),
        "stream_id": inputs.stream_id,
        "well_out_dir": str(well_out_dir),
        "templates_merged_units_dir": str(merged_units_dir),
        "templates_full_channels_templates_dir": (
            str(full_channels_templates_dir) if full_channels_templates_dir.exists() else None
        ),
        "reconstruction_out_dir": str(recon_out_dir),
        "axon_velocity_params": {k: _jsonable(v) for k, v in params.items()},
        "template_source": {
            "use_full_channels_templates": bool(inputs.use_full_channels_templates),
            "require_full_channels_templates": bool(inputs.require_full_channels_templates),
        },
        "units": [],
    }

    # For all-units overview.
    all_unit_polylines: list[dict[str, Any]] = []
    all_locations: list[Any] = []

    for uid in unit_ids:
        unit_dir = merged_units_dir / f"unit_{uid}"
        merged_tmpl_npy = unit_dir / "merged_contributing_template.npy"
        merged_locs_npy = unit_dir / "merged_contributing_channel_locations.npy"
        merged_meta_json = unit_dir / "merged_contributing_template_meta.json"

        full_unit_dir = full_channels_templates_dir / f"unit_{uid}"
        full_tmpl_npy = full_unit_dir / "full_template.npy"
        full_locs_npy = full_unit_dir / "full_channel_locations_xy.npy"
        full_meta_json = full_unit_dir / "full_template_meta.json"

        out_unit_dir = by_unit_dir / f"unit_{uid}"
        out_unit_dir.mkdir(parents=True, exist_ok=True)

        unit_summary: dict[str, Any] = {
            "unit_id": _jsonable(uid),
            "inputs": {
                "merged_contributing": {
                    "template_npy": str(merged_tmpl_npy),
                    "channel_locations_npy": str(merged_locs_npy),
                    "template_meta_json": str(merged_meta_json),
                },
                "full_channels": {
                    "full_template_npy": str(full_tmpl_npy),
                    "full_channel_locations_xy_npy": str(full_locs_npy),
                    "full_template_meta_json": str(full_meta_json),
                },
            },
            "outputs": {},
            "status": "ok",
            "error": None,
        }

        try:
            use_full = bool(inputs.use_full_channels_templates)
            if use_full:
                if not full_tmpl_npy.exists() or not full_locs_npy.exists():
                    if bool(inputs.require_full_channels_templates):
                        raise FileNotFoundError(
                            f"Missing full-channel templates for unit {uid}: expected {full_tmpl_npy} and {full_locs_npy}"
                        )
                    use_full = False

            if use_full:
                tmpl = np.load(full_tmpl_npy)
                locs = np.load(full_locs_npy)
                unit_summary["inputs"]["selected_template_source"] = "full_channels_templates"
            else:
                if not merged_tmpl_npy.exists() or not merged_locs_npy.exists():
                    raise FileNotFoundError(f"Missing merged template inputs for unit {uid}")
                tmpl = np.load(merged_tmpl_npy)
                locs = np.load(merged_locs_npy)
                unit_summary["inputs"]["selected_template_source"] = "merged_contributing"

            if tmpl.ndim != 2:
                raise ValueError(f"Unexpected template shape for unit {uid}: {tmpl.shape}")
            if locs.ndim != 2 or locs.shape[1] < 2:
                raise ValueError(f"Unexpected locations shape for unit {uid}: {locs.shape}")

            # Our saved templates are (n_samples, n_channels); axon_velocity expects (n_channels, n_timepoints).
            tmpl_ch_by_t = np.asarray(tmpl).T
            locs_xy = np.asarray(locs)[:, :2]

            # Summary-only mode: plot summaries directly from on-disk artifacts.
            if bool(inputs.replot_summaries_only):
                from .plotting import write_unit_summary_plots_from_disk

                fs_hz_for_summary: Optional[float] = None
                try:
                    meta_path = full_meta_json if (use_full and full_meta_json.exists()) else merged_meta_json
                    meta = _read_json(meta_path)
                    if meta.get("sampling_frequency_hz") is not None:
                        fs_hz_for_summary = float(meta.get("sampling_frequency_hz"))
                except Exception:
                    fs_hz_for_summary = None

                out = write_unit_summary_plots_from_disk(
                    uid=uid,
                    out_unit_dir=out_unit_dir,
                    template_ch_by_t=tmpl_ch_by_t,
                    locs_xy=locs_xy,
                    fs_hz=fs_hz_for_summary,
                    force_restart=bool(inputs.force_restart),
                    logger=logger,
                )
                unit_summary["outputs"].update(out)
                summary["units"].append(unit_summary)
                continue

            # branches_raw-only mode: run tracking, write branches_raw.json, and skip other outputs.
            if bool(inputs.recompute_branches_raw_only):
                fs_hz = None
                try:
                    meta_path = full_meta_json if (use_full and full_meta_json.exists()) else merged_meta_json
                    meta = _read_json(meta_path)
                    fs_hz = (
                        float(meta.get("sampling_frequency_hz"))
                        if meta.get("sampling_frequency_hz") is not None
                        else None
                    )
                except Exception:
                    fs_hz = None
                if fs_hz is None:
                    fs_hz = 10_000.0

                gtr = av.compute_graph_propagation_velocity(tmpl_ch_by_t, locs_xy, float(fs_hz), **params)

                raw_branches_json = out_unit_dir / "branches_raw.json"
                try:
                    from .plotting import compute_raw_branches_for_summary

                    raw_branches_out = compute_raw_branches_for_summary(uid=uid, gtr=gtr)
                    _write_json(raw_branches_json, {"unit_id": _jsonable(uid), "branches": raw_branches_out})
                    unit_summary["outputs"]["branches_raw_json"] = str(raw_branches_json)
                except Exception as e:
                    logger.warning("Failed writing branches_raw.json for unit %s: %s", uid, e)

                summary["units"].append(unit_summary)
                continue

            fs_hz = None
            try:
                meta_path = full_meta_json if (use_full and full_meta_json.exists()) else merged_meta_json
                meta = _read_json(meta_path)
                fs_hz = (
                    float(meta.get("sampling_frequency_hz"))
                    if meta.get("sampling_frequency_hz") is not None
                    else None
                )
            except Exception:
                fs_hz = None
            if fs_hz is None:
                # Fallback: try templates_summary.json (best effort)
                fs_hz = 10_000.0

            gtr = av.compute_graph_propagation_velocity(tmpl_ch_by_t, locs_xy, float(fs_hz), **params)

            # --- Save sparse morphology + velocity per branch ---
            branches_out: list[dict[str, Any]] = []
            branches_val = getattr(gtr, "branches", None)
            branches_iter = _as_list(branches_val)
            for bi, br in enumerate(branches_iter):
                try:
                    chans = _as_int_list(br.get("channels"))
                except Exception:
                    chans = []
                xy = []
                try:
                    for ch in chans:
                        xy.append([float(locs_xy[ch, 0]), float(locs_xy[ch, 1])])
                except Exception:
                    xy = []

                br_out = {
                    "branch_index": int(bi),
                    "channels": chans,
                    "polyline_xy": xy,
                    "velocity": _jsonable(br.get("velocity")),
                    "offset": _jsonable(br.get("offset")),
                    "r2": _jsonable(br.get("r2")),
                    "pval": _jsonable(br.get("pval")),
                    "distances": _as_float_list(br.get("distances")),
                    "peak_times": _as_float_list(br.get("peak_times")),
                }
                branches_out.append(br_out)

                if xy:
                    all_unit_polylines.append({"unit_id": _jsonable(uid), "branch_index": int(bi), "polyline_xy": xy})

            all_locations.append(locs_xy)

            heuristics: dict[str, Any] = {
                "init_channel": int(getattr(gtr, "init_channel", 0)),
                "n_channels_total": int(locs_xy.shape[0]),
                "n_selected_channels": int(len(_as_list(getattr(gtr, "selected_channels", None)))),
                "selected_channels": _as_int_list(getattr(gtr, "selected_channels", None)),
                "node_heuristic": None,
            }
            try:
                node_h = getattr(gtr, "_node_heuristic", None)
                if node_h is not None:
                    heuristics["node_heuristic"] = [float(x) for x in list(node_h)]
            except Exception:
                pass

            branches_json = out_unit_dir / "branches.json"
            raw_branches_json = out_unit_dir / "branches_raw.json"
            heuristics_json = out_unit_dir / "heuristics.json"
            _write_json(branches_json, {"unit_id": _jsonable(uid), "branches": branches_out})
            _write_json(heuristics_json, {"unit_id": _jsonable(uid), "heuristics": heuristics})

            # Persist raw paths (best effort) so summary plots can be regenerated without
            # recomputing axon_velocity tracking.
            try:
                from .plotting import compute_raw_branches_for_summary

                raw_branches_out = compute_raw_branches_for_summary(uid=uid, gtr=gtr)
                _write_json(raw_branches_json, {"unit_id": _jsonable(uid), "branches": raw_branches_out})
                unit_summary["outputs"]["branches_raw_json"] = str(raw_branches_json)
            except Exception:
                pass

            unit_summary["outputs"].update(
                {
                    "branches_json": str(branches_json),
                    "heuristics_json": str(heuristics_json),
                }
            )

            # --- Plots ---
            if inputs.write_unit_pdfs:
                try:
                    plot_outputs = write_unit_reconstruction_pdfs(
                        uid=uid,
                        gtr=gtr,
                        locs_xy=locs_xy,
                        out_unit_dir=out_unit_dir,
                        force_restart=bool(inputs.force_restart),
                        logger=logger,
                    )
                    unit_summary["outputs"].update(plot_outputs)
                except Exception as e:
                    logger.warning("Plotting failed for unit %s: %s", uid, e)

        except Exception as e:
            unit_summary["status"] = "error"
            unit_summary["error"] = exception_to_error_dict(e)
            logger.warning("Reconstruction failed for unit %s: %s", uid, e)

        summary["units"].append(unit_summary)

    # All-units overview morphology plot.
    if (not bool(inputs.replot_summaries_only)) and (all_units_overview_pdf is not None):
        try:
            ok = write_all_units_overview_pdf(
                all_units_overview_pdf=all_units_overview_pdf,
                all_locations=all_locations,
                all_unit_polylines=all_unit_polylines,
                stream_id=inputs.stream_id,
                force_restart=bool(inputs.force_restart),
                logger=logger,
            )
            if ok:
                summary["all_units_overview_pdf"] = str(all_units_overview_pdf)
                all_units_overview_png = all_units_overview_pdf.with_suffix(".png")
                if all_units_overview_png.exists():
                    summary["all_units_overview_png"] = str(all_units_overview_png)
        except Exception as e:
            logger.warning("Failed writing all-units overview pdf: %s", e)

    _write_json(summary_json, summary)

    ckpt = save_checkpoint(
        checkpoint_file=ckpt_file,
        state=ckpt,
        stage=ProcessingStage.ANALYZER_COMPLETE,
        failed_stage=None,
        error=None,
        extra_fields={
            "reconstruction_out_dir": str(recon_out_dir),
            "reconstruction_summary_json": str(summary_json),
            "all_units_overview_pdf": str(all_units_overview_pdf) if all_units_overview_pdf else None,
        },
    )

    return ReconstructionOutputs(
        well_out_dir=well_out_dir,
        reconstruction_out_dir=recon_out_dir,
        summary_json=summary_json,
        by_unit_dir=by_unit_dir,
        all_units_overview_pdf=all_units_overview_pdf,
    )


__all__ = [
    "ReconstructionInputs",
    "ReconstructionOutputs",
    "reconstruct_from_templates",
]
