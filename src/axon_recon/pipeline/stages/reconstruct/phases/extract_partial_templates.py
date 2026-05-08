"""Extract partial templates phase.

This phase is the first half of the templates artifact pipeline. It opens each
cached segment analyzer once, extracts a per-(unit, source) payload — the
"partial template" for that unit on that segment/concat source — and writes it
under ``cache/source_payloads/<source>/<unit>/`` plus a ``partial_summary.json``
listing the units that were materialized.

The downstream ``build_templates`` phase reads these partial payloads off disk
and merges them per-unit. ``build_templates`` MUST NOT reopen segment analyzers.

Natural fanout: one worker per segment (``nested_shape: segment_workers``);
each worker holds one segment analyzer and iterates over the unit list.
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

from .build_templates import (
    BuildTemplatesContext,
    _apply_build_templates_unit_label_filter,
    _build_templates_applied_debug_limits,
    _clear_payload_root_for_force_restart,
    _collect_unit_ids_from_source_manifests,
    _discover_cached_analyzer_sources,
    _load_or_create_source_unit_manifests,
    _materialize_cached_analyzers_by_source,
    _materialize_cached_analyzers_by_unit,
    _payload_root_status,
    _resolve_build_templates_context,
)
from ..templates.io import write_json
from ..templates.models.inputs import TemplatesInputs

LOGGER = logging.getLogger("axon_recon.templates.extract_partial_templates")

PARTIAL_SUMMARY_RELPATH = "context/extract_partial_templates_summary.json"


def _log_extract_partial_templates_start(
    *, inputs: TemplatesInputs, context: BuildTemplatesContext
) -> None:
    LOGGER.info(
        "templates.extract_partial_templates start: well_out_dir=%s templates_out_dir=%s payload_root=%s force_restart=%s",
        str(context.well_out_dir),
        str(context.templates_out_dir),
        str(context.payload_root),
        bool(inputs.force_restart),
    )
    LOGGER.info(
        "templates.extract_partial_templates settings: lazy_load_analyzers=%s emit_unit_source_materialization_log=%s",
        bool(getattr(inputs.phases.build_templates, "lazy_load_analyzers", False)),
        bool(
            getattr(
                inputs.phases.build_templates, "emit_unit_source_materialization_log", False
            )
        ),
    )


def _resolve_unit_ids_for_extract(
    *,
    inputs: TemplatesInputs,
    context: BuildTemplatesContext,
    analyzer_well_out_dir: Path,
    analyzer_cache_dir: Path,
    source_names: list[str],
) -> tuple[list[Any] | None, dict[str, list[Any] | None]]:
    unit_ids: list[Any] | None = None if inputs.unit_ids is None else list(inputs.unit_ids)
    if unit_ids is not None and inputs.unit_limit is not None:
        unit_ids = unit_ids[: int(inputs.unit_limit)]
    if unit_ids is not None:
        unit_ids = _apply_build_templates_unit_label_filter(
            inputs=inputs,
            unit_ids=unit_ids,
            well_out_dir=context.well_out_dir,
        )
    source_unit_ids_by_source = _load_or_create_source_unit_manifests(
        inputs=inputs,
        context=context,
        analyzer_well_out_dir=analyzer_well_out_dir,
        analyzer_cache_dir=analyzer_cache_dir,
        source_names=source_names,
    )
    if unit_ids is None:
        manifest_unit_ids = _collect_unit_ids_from_source_manifests(
            source_unit_ids_by_source,
            source_names,
        )
        if manifest_unit_ids:
            if inputs.unit_limit is not None:
                manifest_unit_ids = manifest_unit_ids[: int(inputs.unit_limit)]
            unit_ids = _apply_build_templates_unit_label_filter(
                inputs=inputs,
                unit_ids=manifest_unit_ids,
                well_out_dir=context.well_out_dir,
            )
    return unit_ids, source_unit_ids_by_source


def _materialize_partials(
    *,
    inputs: TemplatesInputs,
    context: BuildTemplatesContext,
    analyzer_well_out_dir: Path,
    analyzer_cache_dir: Path,
    source_names: list[str],
    unit_ids: list[Any] | None,
    source_unit_ids_by_source: dict[str, list[Any] | None],
    lazy_load_analyzers: bool,
) -> tuple[list[Any], dict[str, Any]]:
    if lazy_load_analyzers:
        return _materialize_cached_analyzers_by_unit(
            inputs=inputs,
            context=context,
            analyzer_well_out_dir=analyzer_well_out_dir,
            analyzer_cache_dir=analyzer_cache_dir,
            source_names=source_names,
            unit_ids=unit_ids,
            source_unit_ids_by_source=source_unit_ids_by_source,
        )
    return _materialize_cached_analyzers_by_source(
        inputs=inputs,
        context=context,
        analyzer_well_out_dir=analyzer_well_out_dir,
        analyzer_cache_dir=analyzer_cache_dir,
        source_names=source_names,
        unit_ids=unit_ids,
        source_unit_ids_by_source=source_unit_ids_by_source,
    )


def _write_partial_summary(
    *,
    inputs: TemplatesInputs,
    context: BuildTemplatesContext,
    summary: dict[str, Any],
    phase_started: float,
) -> dict[str, Any]:
    summary["timing"] = {"duration_seconds": float(perf_counter() - phase_started)}
    summary["applied_debug_limits"] = _build_templates_applied_debug_limits(inputs)
    summary_path = context.templates_out_dir / PARTIAL_SUMMARY_RELPATH
    LOGGER.info(
        "templates.extract_partial_templates generating outputs: summary_json=%s",
        str(summary_path),
    )
    write_json(summary_path, summary)
    summary["summary_json"] = str(summary_path)
    LOGGER.info(
        "templates.extract_partial_templates wrote summary output: %s",
        str(summary_path),
    )
    LOGGER.info(
        "templates.extract_partial_templates run stats: duration_seconds=%.3f unit_count=%d source_count=%d",
        float(summary["timing"]["duration_seconds"]),
        int(len(summary.get("requested_units", []))),
        int(summary.get("source_count", 0)),
    )
    return summary


def run_reconstruct_templates_extract_partial_templates_phase(
    inputs: TemplatesInputs,
) -> dict[str, Any]:
    phase_started = perf_counter()

    # 1. Resolve build/extract paths and source payload cache location.
    context = _resolve_build_templates_context(inputs)
    _log_extract_partial_templates_start(inputs=inputs, context=context)

    # 2. Apply force-restart cleanup for source payloads (this is the artifact
    #    set this phase owns).
    _clear_payload_root_for_force_restart(inputs=inputs, context=context)
    payload_status = _payload_root_status(context.payload_root)

    # 3. Discover cached analyzer sources. If there are none, the phase cannot
    #    materialize partials; surface a clear error.
    analyzer_well_out_dir, analyzer_cache_dir, source_names = _discover_cached_analyzer_sources(
        inputs=inputs,
        context=context,
    )
    if analyzer_cache_dir is None or not source_names:
        raise FileNotFoundError(
            "No cached templates analyzers found for extract_partial_templates. "
            f"checked analyzer_cache_dir={analyzer_cache_dir}; run templates.analyzers "
            "before templates.extract_partial_templates."
        )

    lazy_load_analyzers = bool(
        getattr(inputs.phases.build_templates, "lazy_load_analyzers", False)
    )
    LOGGER.info(
        "templates.extract_partial_templates loading cached analyzers: analyzer_well_out_dir=%s analyzer_cache_dir=%s source_count=%d lazy_load_analyzers=%s payload_status=%s",
        str(analyzer_well_out_dir),
        str(analyzer_cache_dir),
        int(len(source_names)),
        bool(lazy_load_analyzers),
        str(payload_status),
    )

    source_names = [str(source_name) for source_name in source_names]
    unit_ids, source_unit_ids_by_source = _resolve_unit_ids_for_extract(
        inputs=inputs,
        context=context,
        analyzer_well_out_dir=analyzer_well_out_dir,
        analyzer_cache_dir=analyzer_cache_dir,
        source_names=source_names,
    )

    materialized_unit_ids, streamed_sources_summary = _materialize_partials(
        inputs=inputs,
        context=context,
        analyzer_well_out_dir=analyzer_well_out_dir,
        analyzer_cache_dir=analyzer_cache_dir,
        source_names=source_names,
        unit_ids=unit_ids,
        source_unit_ids_by_source=source_unit_ids_by_source,
        lazy_load_analyzers=lazy_load_analyzers,
    )

    summary: dict[str, Any] = {
        "phase": "extract_partial_templates",
        "stream_id": str(inputs.stream_id),
        "well_out_dir": str(context.well_out_dir),
        "templates_out_dir": str(context.templates_out_dir),
        "payload_root": str(context.payload_root),
        "payload_output_rel_root": str(context.payload_output_rel_root),
        "source_payload_well_out_dir": str(analyzer_well_out_dir),
        "analyzer_cache_dir": str(analyzer_cache_dir),
        "lazy_load_analyzers": bool(lazy_load_analyzers),
        "source_names": [str(name) for name in source_names],
        "source_count": int(len(source_names)),
        "requested_units": list(materialized_unit_ids),
        "source_payload_sources": streamed_sources_summary,
    }

    return _write_partial_summary(
        inputs=inputs,
        context=context,
        summary=summary,
        phase_started=phase_started,
    )


__all__ = [
    "PARTIAL_SUMMARY_RELPATH",
    "run_reconstruct_templates_extract_partial_templates_phase",
]
