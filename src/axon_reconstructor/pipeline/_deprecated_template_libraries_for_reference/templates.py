from __future__ import annotations

import json
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


def _check_unit_merging_available(*, logger) -> tuple[bool, str | None]:
    """Return whether SpikeInterface auto-merge can run in this environment.

    SpikeInterface's auto-merge presets rely on quality metrics that call
    `compute_refrac_period_violations`, which requires `numba`. When numba is
    not installed, upstream currently warns and returns None, which can crash
    auto-merge.
    """

    try:
        import spikeinterface.curation.auto_merge as _am  # type: ignore[import-not-found]

        have_numba = getattr(_am, "HAVE_NUMBA", None)
        if have_numba is False:
            return False, "numba is not installed (required by SpikeInterface auto-merge presets)"
    except Exception as e:
        logger.debug("Unable to import spikeinterface.curation.auto_merge: %s", e)
        # If we cannot import the module, unit merging cannot run.
        return False, "spikeinterface.curation.auto_merge is unavailable"

    # Belt-and-suspenders check.
    try:
        import numba  # type: ignore[import-not-found]  # noqa: F401
    except Exception:
        return False, "numba is not installed (required by SpikeInterface auto-merge presets)"

    return True, None


def _ensure_analyzer_extensions(*, analyzer, extension_names: list[str], logger, n_jobs: int) -> None:
    """Ensure a `SortingAnalyzer` has the requested extensions.

    SpikeInterface's `SortingAnalyzer` API has evolved; in 0.103.x the supported
    way to check is `has_extension()` (not `get_extension_names()`).
    """

    missing = [name for name in extension_names if not analyzer.has_extension(name)]
    if not missing:
        return

    logger.info("Computing extensions: %s", ", ".join(missing))
    analyzer.compute(missing, verbose=False, n_jobs=max(1, int(n_jobs)))


def _compute_templates_checkpoint_file(*, well_out_dir: Path, h5_path: Path, stream_id: str) -> Path:
    """Use a dedicated checkpoint file for template extraction/merging."""

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

    # Which waveforms/analyzers to use
    include_concat: bool = True
    include_segments: bool = True

    # Extraction controls
    unit_ids: Optional[list[Any]] = None
    unit_limit: Optional[int] = None

    # Unit merging (optional)
    run_unit_merging: bool = True
    merge_presets: Optional[list[str]] = None
    merge_recursive: bool = False
    n_jobs: int = 8

    # Resume/overwrite
    force_restart: bool = False


@dataclass(frozen=True)
class TemplateExtractOutputs:
    well_out_dir: Path
    templates_out_dir: Path
    merged_templates_dir: Path
    summary_json: Path

    unit_merging_dir: Optional[Path] = None
    merged_analyzer_dir: Optional[Path] = None
    unit_merging_summary_json: Optional[Path] = None


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _loc_key(loc: Any) -> tuple[int, int]:
    """Stable key for matching channel locations across segments.

    We scale/round to avoid float equality surprises.
    """

    x = float(loc[0])
    y = float(loc[1])
    return (int(round(x * 1000.0)), int(round(y * 1000.0)))


def _merge_templates_by_location(
    *,
    templates: list["Any"],
    channel_locations: list["Any"],
    logger,
) -> tuple["Any", "Any", dict[str, Any]]:
    """Merge multiple (samples x channels) templates into a single template.

    Matching is done by identical channel location (x, y). Overlapping channels are
    averaged. New channels are appended.
    """

    import numpy as np  # type: ignore[import-not-found]

    assert len(templates) == len(channel_locations)
    if len(templates) == 0:
        raise ValueError("No templates to merge")

    # Initialize with first template.
    merged = np.asarray(templates[0])
    merged_locs = np.asarray(channel_locations[0])
    if merged.ndim != 2:
        raise ValueError(f"Template must be 2D (samples x channels); got shape {merged.shape}")
    if merged_locs.ndim != 2 or merged_locs.shape[1] < 2:
        raise ValueError(f"Channel locations must be (n,2); got shape {merged_locs.shape}")
    if merged.shape[1] != merged_locs.shape[0]:
        raise ValueError("Template channels and channel_locations length mismatch")

    # Per-sample counts, to support NaN-masked merges.
    counts = np.ones_like(merged, dtype=np.uint16)
    if np.isnan(merged).any():
        counts = (~np.isnan(merged)).astype(np.uint16)
        merged = np.nan_to_num(merged, nan=0.0)

    loc_to_index: dict[tuple[int, int], int] = {_loc_key(loc): int(i) for i, loc in enumerate(merged_locs)}

    stats: dict[str, Any] = {
        "num_inputs": int(len(templates)),
        "inputs": [],
        "final_channels": None,
    }

    for idx in range(1, len(templates)):
        t_in = np.asarray(templates[idx])
        locs_in = np.asarray(channel_locations[idx])
        if t_in.ndim != 2:
            raise ValueError(f"Incoming template must be 2D; got shape {t_in.shape}")
        if t_in.shape[1] != locs_in.shape[0]:
            raise ValueError("Incoming template channels and channel_locations mismatch")
        if t_in.shape[0] != merged.shape[0]:
            raise ValueError(
                f"All templates must share the same #samples; got {t_in.shape[0]} vs {merged.shape[0]}"
            )

        # Build overlap/new index lists.
        in_keys = [_loc_key(loc) for loc in locs_in]
        overlap_in: list[int] = []
        overlap_merged: list[int] = []
        new_in: list[int] = []
        new_locs: list[Any] = []
        for j, key in enumerate(in_keys):
            existing = loc_to_index.get(key)
            if existing is None:
                new_in.append(j)
                new_locs.append(locs_in[j])
            else:
                overlap_in.append(j)
                overlap_merged.append(existing)

        stats["inputs"].append(
            {
                "index": int(idx),
                "incoming_channels": int(t_in.shape[1]),
                "overlap_channels": int(len(overlap_in)),
                "new_channels": int(len(new_in)),
            }
        )

        # Merge overlap channels by (masked) average.
        if overlap_in:
            merged_idx = np.asarray(overlap_merged, dtype=int)
            in_idx = np.asarray(overlap_in, dtype=int)
            incoming = t_in[:, in_idx]
            incoming_counts = (~np.isnan(incoming)).astype(np.uint16)
            incoming = np.nan_to_num(incoming, nan=0.0)

            merged_part = merged[:, merged_idx]
            counts_part = counts[:, merged_idx]

            total = counts_part + incoming_counts
            # Avoid division by zero when both are zero (should be rare).
            out = np.where(total > 0, (merged_part * counts_part + incoming) / total, 0.0)

            merged[:, merged_idx] = out.astype(merged.dtype, copy=False)
            counts[:, merged_idx] = total

        # Append new channels.
        if new_in:
            in_idx = np.asarray(new_in, dtype=int)
            incoming_new = t_in[:, in_idx]
            incoming_counts_new = (~np.isnan(incoming_new)).astype(np.uint16)
            incoming_new = np.nan_to_num(incoming_new, nan=0.0)

            merged = np.concatenate([merged, incoming_new], axis=1)
            counts = np.concatenate([counts, incoming_counts_new], axis=1)

            # Update locations and dict.
            start = int(merged_locs.shape[0])
            merged_locs = np.concatenate([merged_locs, np.asarray(new_locs)], axis=0)
            for offset, loc in enumerate(new_locs):
                loc_to_index[_loc_key(loc)] = start + int(offset)

    stats["final_channels"] = int(merged.shape[1])
    logger.info("Merged template channels: %d", int(merged.shape[1]))
    return merged, merged_locs, stats


def extract_and_merge_templates(
    *,
    inputs: TemplateExtractInputs,
    logger_name_prefix: str = "axon_reconstructor",
) -> TemplateExtractOutputs:
    """Extract per-unit templates and merge across segments (by channel location).

    Inputs are the waveforms analyzers produced by the waveforms stage:
      <well>/waveforms_outputs/concat_waveforms/
      <well>/waveforms_outputs/segment_waveforms/segXX_*/
    Outputs go to:
      <well>/templates_outputs/merged_templates/
    """

    import spikeinterface.full as si  # type: ignore[import-not-found]

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
        logger_name=f"{logger_name_prefix}.{inputs.stream_id}.templates",
        verbose=True,
    )

    templates_out_dir = well_out_dir / TEMPLATES_OUTPUTS_DIRNAME
    merged_templates_dir = templates_out_dir / "merged_templates"
    summary_json = templates_out_dir / "template_merge_summary.json"

    unit_merging_dir = templates_out_dir / "unit_merging"
    merged_analyzer_dir = unit_merging_dir / "merged_analyzer"
    unit_merging_summary_json = unit_merging_dir / "unit_merging_summary.json"

    unit_merging_available, unit_merging_unavailable_reason = _check_unit_merging_available(logger=logger)
    will_run_unit_merging = bool(inputs.run_unit_merging) and bool(unit_merging_available)

    ckpt_file = _compute_templates_checkpoint_file(
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

    # Resume shortcut.
    # If unit merging *will run* in this environment, require its artifacts too.
    resume_ok = merged_templates_dir.exists() and summary_json.exists()
    if will_run_unit_merging:
        resume_ok = resume_ok and merged_analyzer_dir.exists() and unit_merging_summary_json.exists()

    if not inputs.force_restart and resume_ok:
        logger.info("Resuming templates: existing outputs found at %s", merged_templates_dir)

        # If unit merging was requested but cannot run, ensure we leave a breadcrumb.
        if inputs.run_unit_merging and not will_run_unit_merging and not unit_merging_summary_json.exists():
            unit_merging_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                unit_merging_summary_json,
                {
                    "status": "skipped",
                    "reason": unit_merging_unavailable_reason,
                    "h5_path": str(inputs.h5_path),
                    "stream_id": inputs.stream_id,
                    "requested_presets": list(inputs.merge_presets) if inputs.merge_presets is not None else None,
                    "requested_recursive": bool(inputs.merge_recursive),
                },
            )

        return TemplateExtractOutputs(
            well_out_dir=well_out_dir,
            templates_out_dir=templates_out_dir,
            merged_templates_dir=merged_templates_dir,
            summary_json=summary_json,
            unit_merging_dir=(unit_merging_dir if inputs.run_unit_merging else None),
            merged_analyzer_dir=(merged_analyzer_dir if will_run_unit_merging else None),
            unit_merging_summary_json=(unit_merging_summary_json if inputs.run_unit_merging else None),
        )

    ckpt = save_checkpoint(
        checkpoint_file=ckpt_file,
        state=ckpt,
        stage=ProcessingStage.ANALYZER,
        failed_stage=None,
        error=None,
        extra_fields={
            "templates_out_dir": str(templates_out_dir),
        },
    )

    try:
        waveforms_out_dir = well_out_dir / "waveforms_outputs"
        concat_waveforms_dir = waveforms_out_dir / "concat_waveforms"
        segment_waveforms_dir = waveforms_out_dir / "segment_waveforms"

        if inputs.include_concat and not concat_waveforms_dir.exists():
            raise FileNotFoundError(f"Missing concat waveforms analyzer at {concat_waveforms_dir}")

        analyzers: list[tuple[str, Any]] = []
        if inputs.include_concat:
            logger.info("Loading concat analyzer: %s", concat_waveforms_dir)
            analyzers.append(("concat", si.load_sorting_analyzer(concat_waveforms_dir)))

        if inputs.include_segments and segment_waveforms_dir.exists():
            seg_dirs = sorted([p for p in segment_waveforms_dir.iterdir() if p.is_dir()])
            logger.info("Found %d segment analyzers", len(seg_dirs))
            for p in seg_dirs:
                try:
                    analyzers.append((p.name, si.load_sorting_analyzer(p)))
                except Exception:
                    logger.warning("Skipping unreadable segment analyzer: %s", p)

        if not analyzers:
            raise RuntimeError("No analyzers available for template extraction")

        # Ensure templates extension exists everywhere.
        for name, an in analyzers:
            try:
                _ensure_analyzer_extensions(
                    analyzer=an,
                    extension_names=["templates"],
                    logger=logger,
                    n_jobs=int(inputs.n_jobs),
                )
            except Exception as e:
                raise RuntimeError(f"Failed to compute templates for analyzer {name}") from e

        # Unit list from concat analyzer by default.
        if inputs.unit_ids is not None:
            unit_ids = list(inputs.unit_ids)
        else:
            unit_ids = list(analyzers[0][1].sorting.unit_ids)

        merged_templates_dir.mkdir(parents=True, exist_ok=True)

        summary: dict[str, Any] = {
            "h5_path": str(inputs.h5_path),
            "stream_id": inputs.stream_id,
            "well_out_dir": str(well_out_dir),
            "include_concat": bool(inputs.include_concat),
            "include_segments": bool(inputs.include_segments),
            "sources": [name for name, _ in analyzers],
            "units": [],
        }

        unit_count = 0
        for unit_id in unit_ids:
            template_list = []
            loc_list = []
            used_sources = []

            for name, an in analyzers:
                # Skip segments where the unit has no waveforms.
                try:
                    wf_ext = an.get_extension("waveforms")
                    wf = wf_ext.get_waveforms_one_unit(unit_id=unit_id)
                    if wf is None or getattr(wf, "shape", (0,))[0] == 0:
                        continue
                except Exception:
                    # If waveforms extension is missing/unexpected, fall back to templates anyway.
                    pass

                try:
                    t_ext = an.get_extension("templates")
                    if hasattr(t_ext, "get_unit_template"):
                        tmpl = t_ext.get_unit_template(unit_id=unit_id)
                    else:
                        tmpl = None
                        if hasattr(t_ext, "get_templates"):
                            all_templates = t_ext.get_templates()
                            try:
                                unit_index = list(an.sorting.unit_ids).index(unit_id)
                                tmpl = all_templates[unit_index]
                            except Exception:
                                tmpl = None
                except Exception:
                    continue

                if tmpl is None:
                    continue
                try:
                    import numpy as np  # type: ignore[import-not-found]

                    arr = np.asarray(tmpl)
                    if arr.size == 0:
                        continue
                    if np.isnan(arr).all():
                        continue
                except Exception:
                    pass

                try:
                    locs = an.recording.get_channel_locations()
                except Exception:
                    continue

                template_list.append(tmpl)
                loc_list.append(locs)
                used_sources.append(name)

            if not template_list:
                continue

            merged_template, merged_locs, merge_stats = _merge_templates_by_location(
                templates=template_list,
                channel_locations=loc_list,
                logger=logger,
            )

            unit_template_file = merged_templates_dir / f"{unit_id}.npy"
            unit_locs_file = merged_templates_dir / f"{unit_id}_channels.npy"

            import numpy as np  # type: ignore[import-not-found]

            np.save(unit_template_file, merged_template)
            np.save(unit_locs_file, merged_locs)

            summary["units"].append(
                {
                    "unit_id": int(unit_id) if str(unit_id).isdigit() else str(unit_id),
                    "sources_used": used_sources,
                    "merged_template_path": str(unit_template_file),
                    "merged_channel_locations_path": str(unit_locs_file),
                    "merged_channels": int(getattr(merged_template, "shape", (0, 0))[1]),
                    "merge_stats": merge_stats,
                }
            )

            unit_count += 1
            if inputs.unit_limit is not None and unit_count >= int(inputs.unit_limit):
                break

        _write_json(summary_json, summary)

        # Optional: automatic unit merging on the concat analyzer (SpikeInterface).
        unit_merging_summary: Optional[dict[str, Any]] = None
        if inputs.run_unit_merging and not will_run_unit_merging:
            logger.warning(
                "Skipping unit merging because it is unavailable in this environment: %s",
                unit_merging_unavailable_reason,
            )
            unit_merging_dir.mkdir(parents=True, exist_ok=True)
            unit_merging_summary = {
                "status": "skipped",
                "reason": unit_merging_unavailable_reason,
                "h5_path": str(inputs.h5_path),
                "stream_id": inputs.stream_id,
                "requested_presets": list(inputs.merge_presets) if inputs.merge_presets is not None else None,
                "requested_recursive": bool(inputs.merge_recursive),
                "n_jobs": int(inputs.n_jobs),
            }
            _write_json(unit_merging_summary_json, unit_merging_summary)

        # Optional: automatic unit merging on the concat analyzer (SpikeInterface).
        if will_run_unit_merging:
            try:
                import spikeinterface.curation as sc  # type: ignore[import-not-found]
            except Exception as e:
                raise RuntimeError("Unit merging requested but spikeinterface.curation is unavailable") from e

            if not inputs.include_concat:
                raise ValueError("run_unit_merging=True requires include_concat=True")

            concat_analyzer = analyzers[0][1]
            unit_merging_dir.mkdir(parents=True, exist_ok=True)

            # Ensure required extensions exist for the default 'similarity_correlograms' preset.
            required_exts = ["templates", "correlograms", "template_similarity"]
            try:
                _ensure_analyzer_extensions(
                    analyzer=concat_analyzer,
                    extension_names=list(required_exts),
                    logger=logger,
                    n_jobs=int(inputs.n_jobs),
                )
            except Exception as e:
                raise RuntimeError("Failed to compute required extensions for unit merging") from e

            presets = inputs.merge_presets if inputs.merge_presets is not None else ["similarity_correlograms"]

            logger.info(
                "Running unit auto-merge (presets=%s, recursive=%s)",
                presets,
                bool(inputs.merge_recursive),
            )

            merged_result = sc.auto_merge_units(
                concat_analyzer,
                presets=list(presets),
                recursive=bool(inputs.merge_recursive),
                extra_outputs=True,
                force_copy=True,
                n_jobs=max(1, int(inputs.n_jobs)),
            )

            merged_analyzer, resolved_merges, merge_unit_groups, _outs = merged_result

            # Persist merged analyzer folder for downstream debugging.
            if merged_analyzer_dir.exists() and inputs.force_restart:
                import shutil

                shutil.rmtree(merged_analyzer_dir)
            merged_analyzer.save_as(format="binary_folder", folder=merged_analyzer_dir)

            # JSON-friendly summary (avoid dumping large arrays from outs).
            def _jsonable(obj: Any) -> Any:
                if obj is None:
                    return None
                if isinstance(obj, (str, int, float, bool)):
                    return obj
                if isinstance(obj, (list, tuple)):
                    return [_jsonable(x) for x in obj]
                if isinstance(obj, dict):
                    return {str(k): _jsonable(v) for k, v in obj.items()}
                try:
                    return str(obj)
                except Exception:
                    return None

            unit_merging_summary = {
                "status": "ok",
                "h5_path": str(inputs.h5_path),
                "stream_id": inputs.stream_id,
                "presets": list(presets),
                "recursive": bool(inputs.merge_recursive),
                "n_jobs": int(inputs.n_jobs),
                "num_units_before": int(len(concat_analyzer.sorting.unit_ids)),
                "num_units_after": int(len(merged_analyzer.sorting.unit_ids)),
                "merged_analyzer_dir": str(merged_analyzer_dir),
                "merge_unit_groups": _jsonable(merge_unit_groups),
                "resolved_merges": _jsonable(resolved_merges),
            }

            _write_json(unit_merging_summary_json, unit_merging_summary)
            _write_json(unit_merging_summary_json, unit_merging_summary)

        save_checkpoint(
            checkpoint_file=ckpt_file,
            state=ckpt,
            stage=ProcessingStage.ANALYZER_COMPLETE,
            failed_stage=None,
            error=None,
            extra_fields={
                "templates_out_dir": str(templates_out_dir),
                "merged_templates_dir": str(merged_templates_dir),
                "template_merge_summary_json": str(summary_json),
                "unit_merging_dir": str(unit_merging_dir) if inputs.run_unit_merging else None,
                "merged_analyzer_dir": str(merged_analyzer_dir) if inputs.run_unit_merging else None,
                "unit_merging_summary_json": str(unit_merging_summary_json) if inputs.run_unit_merging else None,
            },
        )

        logger.info("Template extraction/merge complete")

        return TemplateExtractOutputs(
            well_out_dir=well_out_dir,
            templates_out_dir=templates_out_dir,
            merged_templates_dir=merged_templates_dir,
            summary_json=summary_json,
            unit_merging_dir=(unit_merging_dir if inputs.run_unit_merging else None),
            merged_analyzer_dir=(merged_analyzer_dir if inputs.run_unit_merging else None),
            unit_merging_summary_json=(unit_merging_summary_json if inputs.run_unit_merging else None),
        )

    except Exception as e:
        save_checkpoint(
            checkpoint_file=ckpt_file,
            state=ckpt,
            stage=ProcessingStage.ANALYZER,
            failed_stage="TEMPLATES",
            error=exception_to_error_dict(e),
        )
        logger.exception("Template extraction/merge FAILED")
        raise
