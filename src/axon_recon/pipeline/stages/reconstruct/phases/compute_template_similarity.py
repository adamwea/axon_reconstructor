from __future__ import annotations

import concurrent.futures
import shutil
from time import perf_counter
from typing import Any

from axon_recon.pipeline.stages.reconstruct.templates import runner as templates_runner
from axon_recon.pipeline.stages.reconstruct.templates.models.inputs import TemplatesInputs


def run_reconstruct_templates_compute_template_similarity_phase(
    inputs: TemplatesInputs,
) -> dict[str, Any]:
    phase_started = perf_counter()
    well_out_dir, _, templates_out_dir, _ = templates_runner._resolve_templates_phase_environment(
        inputs
    )
    try:
        merged_units_dir, _ = templates_runner._resolve_templates_dirs(
            well_out_dir=well_out_dir,
            templates_out_dir=templates_out_dir,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Missing built template artifacts for compute_template_similarity; run templates.build_templates first"
        ) from exc
    unit_ids = templates_runner._build_unit_ids(inputs, merged_units_dir)
    unit_ids = templates_runner._apply_unit_label_filter(
        inputs, unit_ids, well_out_dir, context="compute_template_similarity"
    )
    if not unit_ids:
        raise FileNotFoundError(
            f"No built template artifacts found under {merged_units_dir}; run templates.build_templates first"
        )
    phase_cfg = inputs.phases.compute_template_similarity
    output_paths = templates_runner.resolve_similarity_output_paths(
        templates_out_dir=templates_out_dir,
        similarity=phase_cfg,
    )
    worker_count = int(max(1, min(len(unit_ids), int(max(1, int(inputs.n_jobs))))))
    pair_plot_dir = output_paths["template_similarity_candidate_pair_plots_dir"]
    if pair_plot_dir.exists():
        shutil.rmtree(pair_plot_dir)
    for output_key in (
        "template_similarity_matrix_png",
        "template_similarity_matrix_svg",
        "template_similarity_scores_json",
        "template_similarity_candidate_pairs_json",
    ):
        artifact_path = output_paths[output_key]
        if artifact_path.exists():
            artifact_path.unlink()
    summary_path = templates_out_dir / str(phase_cfg.summary_json_relpath)
    if summary_path.exists():
        summary_path.unlink()
    unit_payloads_by_key: dict[str, templates_runner.TemplateSimilarityUnitInput] = {}
    missing_units: list[dict[str, Any]] = []
    progress_interval = max(1, len(unit_ids) // 10)

    def _load_unit_payload(
        unit_id: Any,
    ) -> tuple[Any, templates_runner.TemplateSimilarityUnitInput | None, dict[str, Any] | None]:
        merged_dir = merged_units_dir / f"unit_{unit_id}"
        try:
            merged_template, merged_locs = templates_runner._load_merged_unit(merged_dir)
        except Exception as exc:
            return (
                unit_id,
                None,
                {
                    "unit_id": unit_id,
                    "reason": "failed_to_load_merged_template",
                    "error": str(exc),
                    "path": str(merged_dir),
                },
            )
        return (
            unit_id,
            templates_runner.TemplateSimilarityUnitInput(
                unit_id=unit_id,
                template_c_by_t=merged_template,
                locations_xy=merged_locs,
            ),
            None,
        )

    templates_runner.LOGGER.info(
        "templates.compute_template_similarity load start: units=%d worker_count=%d",
        int(len(unit_ids)),
        int(worker_count),
    )
    if worker_count <= 1 or len(unit_ids) <= 1:
        for idx, unit_id in enumerate(unit_ids, start=1):
            loaded_unit_id, payload, missing = _load_unit_payload(unit_id)
            if payload is not None:
                unit_payloads_by_key[str(loaded_unit_id)] = payload
            if missing is not None:
                missing_units.append(missing)
            if (idx % progress_interval == 0) or (idx == len(unit_ids)):
                templates_runner.LOGGER.info(
                    "templates.compute_template_similarity load progress: %d/%d units scanned",
                    int(idx),
                    int(len(unit_ids)),
                )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {pool.submit(_load_unit_payload, unit_id): unit_id for unit_id in unit_ids}
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                loaded_unit_id, payload, missing = future.result()
                if payload is not None:
                    unit_payloads_by_key[str(loaded_unit_id)] = payload
                if missing is not None:
                    missing_units.append(missing)
                completed += 1
                if (completed % progress_interval == 0) or (completed == len(unit_ids)):
                    templates_runner.LOGGER.info(
                        "templates.compute_template_similarity load progress: %d/%d units scanned",
                        int(completed),
                        int(len(unit_ids)),
                    )
    unit_payloads = [
        unit_payloads_by_key[str(unit_id)]
        for unit_id in unit_ids
        if str(unit_id) in unit_payloads_by_key
    ]
    if not unit_payloads:
        raise FileNotFoundError(
            "Missing built template artifacts for compute_template_similarity; run templates.build_templates first"
        )
    templates_runner.LOGGER.info(
        "templates.compute_template_similarity start: templates_out_dir=%s loaded_units=%d missing_units=%d method=%s worker_count=%d",
        str(templates_out_dir),
        len(unit_payloads),
        len(missing_units),
        str(phase_cfg.method),
        int(worker_count),
    )
    summary = templates_runner.build_template_similarity_phase_summary(
        unit_payloads=unit_payloads,
        templates_out_dir=templates_out_dir,
        config=phase_cfg,
        output_paths=output_paths,
        per_unit_outputs=inputs.per_unit_outputs,
        probe_geometry=inputs.probe_geometry,
        missing_units=missing_units,
    )
    summary["stream_id"] = str(inputs.stream_id)
    summary["well_out_dir"] = str(well_out_dir)
    summary["applied_debug_limits"] = templates_runner._templates_applied_debug_limits(inputs)
    summary["duration_seconds"] = float(perf_counter() - phase_started)
    templates_runner.write_json(summary_path, summary)
    summary["summary_json"] = str(summary_path)
    templates_runner.LOGGER.info(
        "templates.compute_template_similarity wrote summary output: %s", str(summary_path)
    )
    templates_runner.LOGGER.info(
        "templates.compute_template_similarity run stats: duration_seconds=%.3f unit_count=%d pair_count=%d candidate_pairs=%d missing_units=%d",
        float(summary["duration_seconds"]),
        int(summary.get("unit_count", 0)),
        int(summary.get("pair_count", 0)),
        int(summary.get("candidate_pair_count", 0)),
        int(len(summary.get("missing_units", []))),
    )
    return summary
