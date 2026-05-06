from __future__ import annotations

from dataclasses import replace
from typing import Any

from axon_recon.pipeline.stages.reconstruct.templates import runner as templates_runner
from axon_recon.pipeline.stages.reconstruct.templates.models.inputs import TemplatesInputs


def run_reconstruct_templates_reports_phase(
    inputs: TemplatesInputs, *, report_scope: str | None = None
) -> dict[str, Any]:
    _, _, templates_out_dir, _ = templates_runner._resolve_templates_phase_environment(inputs)
    unit_ids = (
        list(inputs.unit_ids)
        if inputs.unit_ids is not None
        else templates_runner._discover_unit_ids_from_unit_summaries(templates_out_dir)
    )
    if not unit_ids:
        raise FileNotFoundError(
            f"No unit summaries found under {templates_out_dir}; run templates.per_unit_processing first"
        )
    missing_unit_summaries: list[Any] = []
    for unit_id in unit_ids:
        paths = templates_runner.resolve_unit_output_paths(
            templates_out_dir=templates_out_dir,
            unit_id=unit_id,
            per_unit_outputs=inputs.per_unit_outputs,
        )
        if not paths["unit_summary_json"].exists():
            missing_unit_summaries.append(unit_id)
    if missing_unit_summaries:
        raise FileNotFoundError(
            "Missing unit summaries required for reports phase; rerun per_unit_processing first for unit_ids="
            + str(missing_unit_summaries)
        )
    report_inputs = replace(
        inputs,
        force_restart=False,
        force_replot=False,
        force_replot_per_unit=False,
        force_rereport=True,
        reports=templates_runner._report_scope_config(inputs.reports, report_scope),
    )
    templates_runner._run_reconstruct_templates_pipeline_monolithic(report_inputs)
    summary_json = templates_out_dir / "templates_summary.json"
    if summary_json.exists():
        summary = templates_runner.read_json(summary_json)
        if isinstance(summary, dict):
            summary["phase"] = "reports" if report_scope is None else f"reports.{report_scope}"
            return summary
    return {
        "phase": ("reports" if report_scope is None else f"reports.{report_scope}"),
        "templates_out_dir": str(templates_out_dir),
    }
