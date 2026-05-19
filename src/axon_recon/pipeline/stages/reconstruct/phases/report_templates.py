from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from axon_recon.pipeline.checkpoint import with_checkpoint_marker
from axon_recon.pipeline.stages.reconstruct.templates import runner as templates_runner
from axon_recon.pipeline.stages.reconstruct.templates.models.inputs import TemplatesInputs


def _resolve_report_templates_source(inputs: TemplatesInputs) -> tuple[str, str]:
    consume = str(inputs.phases.report_templates.consume).strip().lower()
    if consume != "plot_templates_v2":
        raise ValueError(
            f"report_templates.consume must be 'plot_templates_v2' (got {consume!r}); "
            "legacy plot_templates v1 has been removed."
        )
    return "template_circles_v2_png", "templates.plot_templates_v2"


def run_reconstruct_templates_report_templates_phase(inputs: TemplatesInputs) -> dict[str, Any]:
    from axon_recon.pipeline.config import get_no_plot_override

    phase_started = perf_counter()
    well_out_dir, _, templates_out_dir, _ = templates_runner._resolve_templates_phase_environment(
        inputs
    )
    summary_json_path = templates_out_dir / str(inputs.phases.report_templates.summary_json_relpath)
    with with_checkpoint_marker(
        summary_json_path,
        phase_name="report_templates",
        stage_name="reconstruct",
    ):
        if get_no_plot_override() is True:
            summary_path = summary_json_path
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            payload: dict[str, Any] = {
                "phase": "report_templates",
                "status": "skipped",
                "reason": "plots_disabled",
                "templates_out_dir": str(templates_out_dir),
                "well_out_dir": str(well_out_dir),
            }
            templates_runner.write_json(summary_path, payload)
            payload["summary_json"] = str(summary_path)
            templates_runner.LOGGER.info(
                "templates.report_templates: skipped (reason=plots_disabled, --no-plot override active)"
            )
            return payload
        return _run_reconstruct_templates_report_templates_phase_body(
            inputs=inputs,
            phase_started=phase_started,
            well_out_dir=well_out_dir,
            templates_out_dir=templates_out_dir,
            summary_json_path=summary_json_path,
        )


def _run_reconstruct_templates_report_templates_phase_body(
    *,
    inputs: TemplatesInputs,
    phase_started: float,
    well_out_dir: Path,
    templates_out_dir: Path,
    summary_json_path: Path,
) -> dict[str, Any]:
    source_output_key, source_phase_name = _resolve_report_templates_source(inputs)
    unit_ids = (
        list(inputs.unit_ids)
        if inputs.unit_ids is not None
        else templates_runner._discover_unit_ids_from_unit_summaries(templates_out_dir)
    )
    unit_ids = templates_runner._apply_unit_label_filter(
        inputs, unit_ids, well_out_dir, context="report_templates"
    )
    if inputs.unit_limit is not None:
        unit_ids = unit_ids[: int(inputs.unit_limit)]
    if not unit_ids:
        raise FileNotFoundError(
            f"No unit summaries found under {templates_out_dir}; run {source_phase_name} first"
        )

    render_units: list[dict[str, Any]] = []
    missing_units: list[dict[str, Any]] = []
    for unit_id in unit_ids:
        paths = templates_runner.resolve_unit_output_paths(
            templates_out_dir=templates_out_dir,
            unit_id=unit_id,
            per_unit_outputs=inputs.per_unit_outputs,
        )
        unit_result = templates_runner._load_unit_result_from_summary(
            unit_id=unit_id, unit_summary_json=paths["unit_summary_json"]
        )
        if unit_result is None:
            missing_units.append({"unit_id": unit_id, "reason": "missing_unit_summary"})
            continue
        circle_png = unit_result.outputs.get(source_output_key)
        if circle_png is None:
            missing_units.append({"unit_id": unit_id, "reason": f"missing_{source_output_key}"})
            continue
        circle_png_path = Path(str(circle_png))
        if not circle_png_path.exists():
            missing_units.append(
                {
                    "unit_id": unit_id,
                    "reason": f"missing_{source_output_key}_file",
                    "path": str(circle_png_path),
                }
            )
            continue
        render_units.append({"unit_id": unit_id, "image_path": str(circle_png_path)})

    if not render_units:
        raise FileNotFoundError(
            f"Missing circle plot assets required for report_templates; run {source_phase_name} first"
        )

    report_outputs: dict[str, str] = {}
    if templates_runner.report_templates_pdf_requested(inputs):
        report_path = templates_out_dir / str(inputs.phases.report_templates.relpath)
        templates_runner.LOGGER.info(
            "templates.report_templates start: templates_out_dir=%s units=%d",
            str(templates_out_dir),
            len(render_units),
        )
        report_outputs.update(
            templates_runner.render_template_report_pdf(units=render_units, pdf_path=report_path)
        )

    summary = templates_runner.build_report_templates_phase_summary(
        inputs=inputs,
        templates_out_dir=templates_out_dir,
        rendered_units=[unit["unit_id"] for unit in render_units],
        missing_units=missing_units,
        report_outputs=report_outputs,
        source_output_key=source_output_key,
        duration_seconds=float(perf_counter() - phase_started),
    )
    summary["applied_debug_limits"] = templates_runner._templates_applied_debug_limits(inputs)
    summary_path = templates_out_dir / str(inputs.phases.report_templates.summary_json_relpath)
    templates_runner.write_json(summary_path, summary)
    summary["summary_json"] = str(summary_path)
    templates_runner.LOGGER.info(
        "templates.report_templates wrote summary output: %s", str(summary_path)
    )
    templates_runner.LOGGER.info(
        "templates.report_templates run stats: duration_seconds=%.3f unit_count=%d rendered_units=%d missing_units=%d",
        float(summary["duration_seconds"]),
        int(summary.get("unit_count", 0)),
        int(len(summary.get("rendered_units", []))),
        int(len(summary.get("missing_units", []))),
    )
    return summary
