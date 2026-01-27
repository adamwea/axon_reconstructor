from __future__ import annotations

import json
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

from .plotting import (
    _build_global_channel_layout,
    _write_merged_union_footprints_grid_pdf,
    _write_unit_footprints_across_sources_pdf,
)
from .utils import (
    _build_union_source_for_unit,
    _compute_footprinting_checkpoint_file,
    _ensure_analyzer_extensions,
    _get_unit_template_from_waveforms_with_exclusions,
    _get_unit_template_from_extension,
    _infer_location_tolerance,
    _load_curated_unit_ids_from_waveforms_outputs,
    _load_wf_rejection_log_summary,
    _load_waveforms_analyzers,
    _normalize_id_for_compare,
    _sparsity_unit_channel_indices,
    _try_get_electrode_ids,
    _write_json,
)

FOOTPRINTING_OUTPUTS_DIRNAME = "footprinting_outputs"


def _build_union_source_for_unit(
    *,
    sources: list[dict[str, Any]],
    unit_id: Any,
    logger,
) -> Optional[dict[str, Any]]:
    """Union channels across per-source footprints for a unit.

    If overlaps are detected (by channel_id, electrode_id, or location within tolerance),
    we warn and keep the first occurrence.
    """

    import numpy as np  # type: ignore[import-not-found]

    if not sources:
        return None

    # Determine location tolerance from all locations.
    all_locs = [np.asarray(s.get("channel_locations")) for s in sources if s.get("channel_locations") is not None]
    if not all_locs:
        return None
    stacked = np.concatenate(all_locs, axis=0)
    tol = float(_infer_location_tolerance(stacked))
    if tol <= 0:
        tol = 0.0


    concat_footprints_grid_pdf: Optional[Path]

    multi_source_footprints_dir: Optional[Path]
    multi_source_footprints_summary_json: Optional[Path]

    footprinting_summary_json: Path
    concat_footprints_grid_curated_pdf: Optional[Path] = None


def run_footprinting(*, inputs: FootprintingInputs, logger_name_prefix: str = "axon_reconstructor") -> FootprintingOutputs:
    """Compute/plot footprints from existing waveforms analyzers.

    This step is intentionally independent of template extraction/merging.
    It reads:
      <well>/waveforms_outputs/concat_waveforms/
      <well>/waveforms_outputs/segment_waveforms/*/

        And writes under:
            <well>/footprinting_outputs/
    """

    import numpy as np  # type: ignore[import-not-found]

    well_out_dir = _compute_mea_analysis_output_dir(
        output_root=inputs.mea_output_root,
        data_file=inputs.h5_path,
        well=inputs.stream_id,
    )

    log_file = compute_pipeline_log_file(
        well_out_dir=well_out_dir,
        data_file=inputs.h5_path,
        stream_id=inputs.stream_id,
    )
    logger = setup_pipeline_logger(
        log_file=log_file,
        logger_name=f"{logger_name_prefix}.{inputs.stream_id}.footprinting",
        verbose=True,
    )

    templates_out_dir = well_out_dir / "templates_outputs"
    footprinting_out_dir = well_out_dir / FOOTPRINTING_OUTPUTS_DIRNAME

    concat_grid_pdf = footprinting_out_dir / "footprints_grid_concat.pdf"
    concat_grid_curated_pdf = footprinting_out_dir / "footprints_grid_concat_curated.pdf"
    multi_source_dir = footprinting_out_dir / "footprints_by_source"
    multi_source_summary_json = multi_source_dir / "footprints_by_source_summary.json"
    multi_source_dir_uncurated = footprinting_out_dir / "footprints_by_source_uncurated"
    multi_source_summary_json_uncurated = multi_source_dir_uncurated / "footprints_by_source_uncurated_summary.json"
    merged_union_dir = footprinting_out_dir / "merged_union_by_unit"
    merged_union_summary_json = merged_union_dir / "merged_union_summary.json"
    merged_union_dir_uncurated = footprinting_out_dir / "merged_union_by_unit_uncurated"
    merged_union_summary_json_uncurated = merged_union_dir_uncurated / "merged_union_uncurated_summary.json"
    summary_json = footprinting_out_dir / "footprinting_summary.json"

    ckpt_file = _compute_footprinting_checkpoint_file(
        well_out_dir=well_out_dir,
        h5_path=inputs.h5_path,
        stream_id=inputs.stream_id,
    )
    ckpt = load_checkpoint(
        checkpoint_file=ckpt_file,
        force_restart=bool(inputs.force_restart),
        output_dir=well_out_dir,
        file_path=inputs.h5_path,
        stream_id=inputs.stream_id,
    )

    # When plotting is enabled, always write BOTH curated+uncurated grids.
    # If waveforms-stage curation exists, "curated" reflects that list; otherwise it's identical to uncurated.
    expect_curated_grid = bool(inputs.plot_concat_footprints_grid_pdf)

    resume_ok = summary_json.exists()
    if inputs.plot_concat_footprints_grid_pdf:
        resume_ok = resume_ok and concat_grid_pdf.exists()
        resume_ok = resume_ok and concat_grid_curated_pdf.exists()
    if inputs.plot_multi_source_footprints_pdf:
        resume_ok = resume_ok and multi_source_summary_json.exists()
        resume_ok = resume_ok and merged_union_summary_json.exists()
        resume_ok = resume_ok and multi_source_summary_json_uncurated.exists()
        resume_ok = resume_ok and merged_union_summary_json_uncurated.exists()

    if not inputs.force_restart and resume_ok:
        logger.info("Resuming footprinting: existing outputs found at %s", footprinting_out_dir)
        return FootprintingOutputs(
            well_out_dir=well_out_dir,
            templates_out_dir=templates_out_dir,
            footprinting_out_dir=footprinting_out_dir,
            concat_footprints_grid_pdf=(concat_grid_pdf if inputs.plot_concat_footprints_grid_pdf else None),
            concat_footprints_grid_curated_pdf=(
                concat_grid_curated_pdf
                if (inputs.plot_concat_footprints_grid_pdf and concat_grid_curated_pdf.exists())
                else None
            ),
            multi_source_footprints_dir=(multi_source_dir if inputs.plot_multi_source_footprints_pdf else None),
            multi_source_footprints_summary_json=(
                multi_source_summary_json if inputs.plot_multi_source_footprints_pdf else None
            ),
            footprinting_summary_json=summary_json,
        )

    ckpt = save_checkpoint(
        checkpoint_file=ckpt_file,
        state=ckpt,
        stage=ProcessingStage.ANALYZER,
        failed_stage=None,
        error=None,
        extra_fields={"footprinting_out_dir": str(footprinting_out_dir)},
    )

    try:
        analyzers = _load_waveforms_analyzers(
            well_out_dir=well_out_dir,
            include_concat=bool(inputs.include_concat),
            include_segments=bool(inputs.include_segments),
            logger=logger,
        )

        # Spike-level exclusions produced by the waveforms stage (fast .npz).
        try:
            from ..waveforms.exclusions import (
                init_wf_exclusion_report,
                load_wf_exclusions_by_source,
                normalize_unit_id,
            )

            exclusions_by_source = load_wf_exclusions_by_source(well_out_dir=well_out_dir, logger=logger)
            _norm_unit_id = normalize_unit_id
        except Exception:
            exclusions_by_source = {}
            _norm_unit_id = lambda x: x

        wf_exclusions_applied_report_json = footprinting_out_dir / "wf_exclusions_applied_report.json"
        try:
            wf_excl_report = init_wf_exclusion_report(stage="footprinting", exclusions_by_source=exclusions_by_source)
        except Exception:
            wf_excl_report = {
                "stage": "footprinting",
                "loaded_exclusions": {"n_sources": 0, "n_units": 0, "n_spikes": 0, "by_source": {}},
                "application": {"scopes": {}},
            }

        for _, an in analyzers:
            _ensure_analyzer_extensions(
                analyzer=an,
                extension_names=["random_spikes", "waveforms"],
                logger=logger,
                n_jobs=int(inputs.n_jobs),
            )

        # Determine unit list from concat if present, else from first source.
        unit_ids: list[Any]
        if inputs.unit_ids is not None:
            unit_ids = list(inputs.unit_ids)
        else:
            unit_ids = list(analyzers[0][1].sorting.unit_ids)

        # Preserve an uncurated copy for QC grid PDFs.
        unit_ids_all = list(unit_ids)

        # If waveforms-stage curation ran, it produced a curated unit list.
        # Use it here so footprinting skips rejected units without re-running curation.
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
            unit_ids_all = unit_ids_all[: int(inputs.unit_limit)]
            unit_ids = unit_ids[: int(inputs.unit_limit)]

        wf_rejection_summary: Optional[dict[str, Any]] = None
        if inputs.unit_ids is None:
            wf_rejection_summary, _ = _load_wf_rejection_log_summary(well_out_dir=well_out_dir, logger=logger)
            if wf_rejection_summary is not None:
                try:
                    logger.info(
                        "Found waveforms wf_rejection_log.xlsx (rows=%s)",
                        wf_rejection_summary.get("n_rows"),
                    )
                except Exception:
                    pass

        footprinting_out_dir.mkdir(parents=True, exist_ok=True)

        multi_source_summary: dict[str, Any] = {
            "h5_path": str(inputs.h5_path),
            "stream_id": inputs.stream_id,
            "well_out_dir": str(well_out_dir),
            "sources": [name for name, _ in analyzers],
            "curation": {
                "metrics_curated_xlsx": str(curation_metrics_xlsx) if curation_metrics_xlsx else None,
                "applied": bool(curated_units_norm is not None),
                "n_curated_units": int(len(curated_units_norm)) if curated_units_norm is not None else None,
            },
            "waveform_rejections": wf_rejection_summary,
            "units": [],
        }

        multi_source_summary_uncurated: dict[str, Any] = {
            "h5_path": str(inputs.h5_path),
            "stream_id": inputs.stream_id,
            "well_out_dir": str(well_out_dir),
            "sources": [name for name, _ in analyzers],
            "curation": {
                "metrics_curated_xlsx": str(curation_metrics_xlsx) if curation_metrics_xlsx else None,
                "applied": False,
                "n_curated_units": int(len(curated_units_norm)) if curated_units_norm is not None else None,
            },
            "waveform_rejections": wf_rejection_summary,
            "units": [],
        }

        merged_union_summary: dict[str, Any] = {
            "h5_path": str(inputs.h5_path),
            "stream_id": inputs.stream_id,
            "well_out_dir": str(well_out_dir),
            "curation": {
                "metrics_curated_xlsx": str(curation_metrics_xlsx) if curation_metrics_xlsx else None,
                "applied": bool(curated_units_norm is not None),
                "n_curated_units": int(len(curated_units_norm)) if curated_units_norm is not None else None,
            },
            "waveform_rejections": wf_rejection_summary,
            "units": [],
        }

        merged_union_summary_uncurated: dict[str, Any] = {
            "h5_path": str(inputs.h5_path),
            "stream_id": inputs.stream_id,
            "well_out_dir": str(well_out_dir),
            "curation": {
                "metrics_curated_xlsx": str(curation_metrics_xlsx) if curation_metrics_xlsx else None,
                "applied": False,
                "n_curated_units": int(len(curated_units_norm)) if curated_units_norm is not None else None,
            },
            "waveform_rejections": wf_rejection_summary,
            "units": [],
        }

        # Build a global electrode layout across all sources (lets us show non-contributing channels).
        layout_locs, layout_key_to_index, layout_tol = _build_global_channel_layout(analyzers=analyzers)

        def _gather_sources_for_unit(*, uid: Any, apply_exclusions: bool) -> list[dict[str, Any]]:
            sources_for_unit: list[dict[str, Any]] = []
            for name, an in analyzers:
                excluded = set()
                if apply_exclusions:
                    try:
                        excluded = exclusions_by_source.get(str(name), {}).get(_norm_unit_id(uid), set())
                    except Exception:
                        excluded = set()

                res = _get_unit_template_from_waveforms_with_exclusions(
                    analyzer=an,
                    unit_id=uid,
                    excluded_spike_samples=(excluded if apply_exclusions else set()),
                    logger=logger,
                    return_result=True,
                )

                try:
                    from ..waveforms.exclusions import update_wf_exclusion_report

                    update_wf_exclusion_report(
                        wf_excl_report,
                        scope=("multi_source_curated" if apply_exclusions else "multi_source_uncurated"),
                        source_name=str(name),
                        unit_id=uid,
                        excluded_spike_samples=(excluded if apply_exclusions else set()),
                        result=res,
                    )
                except Exception:
                    pass

                tmpl_src = (None if res is None else getattr(res, "template", None))
                if tmpl_src is None:
                    continue
                tmpl_src = np.asarray(tmpl_src)
                if tmpl_src.ndim != 2 or tmpl_src.size == 0:
                    continue

                locs_src = np.asarray(an.recording.get_channel_locations())[:, :2]
                ch_ids_src = None
                try:
                    ch_ids_src = np.asarray(an.recording.get_channel_ids())
                except Exception:
                    ch_ids_src = None

                el_ids_src = _try_get_electrode_ids(an.recording)
                if el_ids_src is not None:
                    try:
                        el_ids_src = np.asarray(el_ids_src)
                    except Exception:
                        el_ids_src = None

                # Support sparse templates by subsetting locations/ids according to sparsity.
                if tmpl_src.shape[1] != locs_src.shape[0]:
                    try:
                        sp = getattr(an, "sparsity", None)
                        if sp is None and an.has_extension("waveforms"):
                            sp = getattr(an.get_extension("waveforms"), "sparsity", None)
                        if sp is not None:
                            ch_inds = _sparsity_unit_channel_indices(sparsity=sp, unit_id=uid)
                            ch_inds = np.asarray(ch_inds, dtype=int)
                            if int(ch_inds.size) == int(tmpl_src.shape[1]):
                                locs_src = locs_src[ch_inds, :]
                                if ch_ids_src is not None:
                                    ch_ids_src = np.asarray(ch_ids_src)[ch_inds]
                                if el_ids_src is not None:
                                    el_ids_src = np.asarray(el_ids_src)[ch_inds]
                    except Exception:
                        pass

                if tmpl_src.shape[1] != locs_src.shape[0]:
                    continue

                amp = np.ptp(tmpl_src, axis=0)
                best_ch = int(np.argmax(amp))
                sources_for_unit.append(
                    {
                        "name": name,
                        "channel_locations": locs_src,
                        "amp": amp,
                        "best_ch": best_ch,
                        "n_channels": int(locs_src.shape[0]),
                        "channel_ids": ch_ids_src,
                        "electrode_ids": el_ids_src,
                    }
                )
            return sources_for_unit

        def _emit_multi_source_outputs(
            *,
            unit_list: list[Any],
            apply_exclusions: bool,
            out_dir: Path,
            out_summary: dict[str, Any],
            merged_dir: Path,
            merged_summary: dict[str, Any],
        ) -> None:
            processed = 0
            out_dir.mkdir(parents=True, exist_ok=True)
            merged_dir.mkdir(parents=True, exist_ok=True)

            for uid in unit_list:
                sources_for_unit = _gather_sources_for_unit(uid=uid, apply_exclusions=apply_exclusions)

                merged_union_src = _build_union_source_for_unit(sources=sources_for_unit, unit_id=uid, logger=logger)
                merged_sources_for_unit = (
                    ([merged_union_src] + sources_for_unit) if merged_union_src is not None else sources_for_unit
                )

                pdf_path = out_dir / f"unit_{uid}_footprints.pdf"

                unit_entry: dict[str, Any] = {
                    "unit_id": int(uid) if str(uid).isdigit() else str(uid),
                    "num_sources": int(len(sources_for_unit)),
                    "sources": [s["name"] for s in sources_for_unit],
                    "merged_union": (merged_union_src.get("merge") if merged_union_src is not None else None),
                    "pdf_path": str(pdf_path) if merged_sources_for_unit else None,
                    "merged_union_pdf_path": None,
                    "error": None,
                }

                # Write merged-union-only PDF per unit.
                if inputs.plot_multi_source_footprints_pdf and merged_union_src is not None:
                    merged_pdf_path = merged_dir / f"unit_{uid}_merged_union.pdf"
                    try:
                        _write_unit_footprints_across_sources_pdf(
                            sources=[merged_union_src],
                            unit_id=uid,
                            pdf_path=merged_pdf_path,
                            logger=logger,
                            layout_locs=layout_locs,
                            layout_key_to_index=layout_key_to_index,
                            layout_tol=float(layout_tol),
                            show_non_contributing_channels=True,
                        )
                        unit_entry["merged_union_pdf_path"] = str(merged_pdf_path)
                        merged_summary["units"].append(
                            {
                                "unit_id": int(uid) if str(uid).isdigit() else str(uid),
                                "pdf_path": str(merged_pdf_path),
                                "merge": merged_union_src.get("merge"),
                                "n_channels": int(merged_union_src.get("n_channels", 0)),
                            }
                        )
                    except Exception as e:
                        logger.warning("Failed writing merged_union PDF for unit %s: %s", uid, e)

                if inputs.plot_multi_source_footprints_pdf and merged_sources_for_unit:
                    try:
                        _write_unit_footprints_across_sources_pdf(
                            sources=merged_sources_for_unit,
                            unit_id=uid,
                            pdf_path=pdf_path,
                            logger=logger,
                            layout_locs=layout_locs,
                            layout_key_to_index=layout_key_to_index,
                            layout_tol=float(layout_tol),
                            show_non_contributing_channels=True,
                        )
                    except Exception as e:
                        unit_entry["error"] = str(e)
                        unit_entry["pdf_path"] = None

                out_summary["units"].append(unit_entry)

                processed += 1
                if inputs.unit_limit is not None and processed >= int(inputs.unit_limit):
                    break

        if inputs.plot_multi_source_footprints_pdf:
            # Uncurated: all units, pre-exclusion.
            _emit_multi_source_outputs(
                unit_list=unit_ids_all,
                apply_exclusions=False,
                out_dir=multi_source_dir_uncurated,
                out_summary=multi_source_summary_uncurated,
                merged_dir=merged_union_dir_uncurated,
                merged_summary=merged_union_summary_uncurated,
            )

            # Curated: curated units, exclusions applied.
            _emit_multi_source_outputs(
                unit_list=unit_ids,
                apply_exclusions=True,
                out_dir=multi_source_dir,
                out_summary=multi_source_summary,
                merged_dir=merged_union_dir,
                merged_summary=merged_union_summary,
            )

        if inputs.plot_multi_source_footprints_pdf:
            _write_json(multi_source_summary_json, multi_source_summary)
            _write_json(merged_union_summary_json, merged_union_summary)
            _write_json(multi_source_summary_json_uncurated, multi_source_summary_uncurated)
            _write_json(merged_union_summary_json_uncurated, merged_union_summary_uncurated)

        if inputs.plot_concat_footprints_grid_pdf:
            # Uncurated grid: all units, pre-exclusion, merged_union on global layout.
            _write_merged_union_footprints_grid_pdf(
                analyzers=analyzers,
                pdf_path=concat_grid_pdf,
                unit_ids=unit_ids_all,
                exclusions_by_source=exclusions_by_source,
                apply_exclusions=False,
                layout_locs=layout_locs,
                layout_key_to_index=layout_key_to_index,
                layout_tol=float(layout_tol),
                logger=logger,
                wf_excl_report=wf_excl_report,
                wf_excl_scope="concat_grid_uncurated",
            )
            # Curated grid: curated units, exclusions applied, merged_union on global layout.
            _write_merged_union_footprints_grid_pdf(
                analyzers=analyzers,
                pdf_path=concat_grid_curated_pdf,
                unit_ids=unit_ids,
                exclusions_by_source=exclusions_by_source,
                apply_exclusions=True,
                layout_locs=layout_locs,
                layout_key_to_index=layout_key_to_index,
                layout_tol=float(layout_tol),
                logger=logger,
                wf_excl_report=wf_excl_report,
                wf_excl_scope="concat_grid_curated",
            )

        # Persist a compact report so it's easy to audit whether exclusions were applied.
        try:
            _write_json(wf_exclusions_applied_report_json, wf_excl_report)
            scopes = (wf_excl_report.get("application") or {}).get("scopes") or {}
            for scope_name, scope_entry in scopes.items():
                by_source = (scope_entry or {}).get("by_source") or {}
                for src, src_entry in by_source.items():
                    n_req_units = int((src_entry or {}).get("n_units_with_exclusions_requested", 0) or 0)
                    n_matched_units = int((src_entry or {}).get("n_units_with_exclusions_matched", 0) or 0)
                    n_excl = int((src_entry or {}).get("waveforms_excluded_matched", 0) or 0)
                    if (n_req_units > 0) or (n_excl > 0):
                        logger.info(
                            "wf_exclusions applied (%s/%s): requested_units=%d matched_units=%d matched_excluded=%d",
                            str(scope_name),
                            str(src),
                            n_req_units,
                            n_matched_units,
                            n_excl,
                        )
        except Exception as e:
            logger.warning("Failed writing wf_exclusions applied report: %s", e)

        _write_json(
            summary_json,
            {
                "h5_path": str(inputs.h5_path),
                "stream_id": inputs.stream_id,
                "well_out_dir": str(well_out_dir),
                "footprinting_out_dir": str(footprinting_out_dir),
                "sources": [name for name, _ in analyzers],
                "curation": {
                    "metrics_curated_xlsx": str(curation_metrics_xlsx) if curation_metrics_xlsx else None,
                    "applied": bool(curated_units_norm is not None),
                    "n_curated_units": int(len(curated_units_norm)) if curated_units_norm is not None else None,
                },
                "concat_footprints_grid_pdf": str(concat_grid_pdf) if inputs.plot_concat_footprints_grid_pdf else None,
                "concat_footprints_grid_curated_pdf": str(concat_grid_curated_pdf)
                if inputs.plot_concat_footprints_grid_pdf
                else None,
                "multi_source_footprints_dir": str(multi_source_dir) if inputs.plot_multi_source_footprints_pdf else None,
                "multi_source_footprints_summary_json": str(multi_source_summary_json)
                if inputs.plot_multi_source_footprints_pdf
                else None,
                "multi_source_footprints_dir_uncurated": str(multi_source_dir_uncurated)
                if inputs.plot_multi_source_footprints_pdf
                else None,
                "multi_source_footprints_summary_json_uncurated": str(multi_source_summary_json_uncurated)
                if inputs.plot_multi_source_footprints_pdf
                else None,
                "merged_union_by_unit_dir": str(merged_union_dir) if inputs.plot_multi_source_footprints_pdf else None,
                "merged_union_summary_json": str(merged_union_summary_json)
                if inputs.plot_multi_source_footprints_pdf
                else None,
                "merged_union_by_unit_dir_uncurated": str(merged_union_dir_uncurated)
                if inputs.plot_multi_source_footprints_pdf
                else None,
                "merged_union_summary_json_uncurated": str(merged_union_summary_json_uncurated)
                if inputs.plot_multi_source_footprints_pdf
                else None,
                "wf_exclusions_applied_report_json": str(wf_exclusions_applied_report_json),
            },
        )

        ckpt = save_checkpoint(
            checkpoint_file=ckpt_file,
            state=ckpt,
            stage=ProcessingStage.ANALYZER_COMPLETE,
            failed_stage=None,
            error=None,
            extra_fields={
                "footprinting_out_dir": str(footprinting_out_dir),
                "concat_footprints_grid_pdf": str(concat_grid_pdf) if inputs.plot_concat_footprints_grid_pdf else None,
                "concat_footprints_grid_curated_pdf": str(concat_grid_curated_pdf)
                if inputs.plot_concat_footprints_grid_pdf
                else None,
                "multi_source_footprints_dir": str(multi_source_dir) if inputs.plot_multi_source_footprints_pdf else None,
                "multi_source_footprints_summary_json": str(multi_source_summary_json)
                if inputs.plot_multi_source_footprints_pdf
                else None,
                "merged_union_by_unit_dir": str(merged_union_dir) if inputs.plot_multi_source_footprints_pdf else None,
                "merged_union_summary_json": str(merged_union_summary_json)
                if inputs.plot_multi_source_footprints_pdf
                else None,
                "summary_json": str(summary_json),
                "wf_exclusions_applied_report_json": str(wf_exclusions_applied_report_json),
            },
        )

        return FootprintingOutputs(
            well_out_dir=well_out_dir,
            templates_out_dir=templates_out_dir,
            footprinting_out_dir=footprinting_out_dir,
            concat_footprints_grid_pdf=(concat_grid_pdf if inputs.plot_concat_footprints_grid_pdf else None),
            concat_footprints_grid_curated_pdf=(
                concat_grid_curated_pdf if inputs.plot_concat_footprints_grid_pdf else None
            ),
            multi_source_footprints_dir=(multi_source_dir if inputs.plot_multi_source_footprints_pdf else None),
            multi_source_footprints_summary_json=(multi_source_summary_json if inputs.plot_multi_source_footprints_pdf else None),
            footprinting_summary_json=summary_json,
        )

    except Exception as e:
        save_checkpoint(
            checkpoint_file=ckpt_file,
            state=ckpt,
            stage=ProcessingStage.ANALYZER,
            failed_stage=ProcessingStage.ANALYZER.name,
            error=exception_to_error_dict(e),
            extra_fields={"footprinting_out_dir": str(footprinting_out_dir)},
        )
        raise
