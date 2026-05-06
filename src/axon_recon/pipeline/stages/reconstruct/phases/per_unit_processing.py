from __future__ import annotations

from dataclasses import replace
from typing import Any

from axon_recon.pipeline.output_paths import compute_mea_analysis_output_dir
from axon_recon.pipeline.stages.reconstruct.phases.build_templates import (
    run_reconstruct_templates_build_templates_phase,
)
from axon_recon.pipeline.stages.reconstruct.templates import runner as templates_runner
from axon_recon.pipeline.stages.reconstruct.templates.models.inputs import TemplatesInputs


def run_reconstruct_templates_per_unit_processing_phase(inputs: TemplatesInputs) -> dict[str, Any]:
    _, _, templates_out_dir, _ = templates_runner._resolve_templates_phase_environment(inputs)
    run_reconstruct_templates_build_templates_phase(inputs)
    try:
        templates_runner._resolve_templates_dirs(
            well_out_dir=compute_mea_analysis_output_dir(
                output_root=inputs.mea_output_root,
                data_file=inputs.h5_path,
                well=inputs.stream_id,
            ),
            templates_out_dir=templates_out_dir,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError("Missing built template artifacts for per-unit processing") from exc
    unit_inputs = replace(
        inputs,
        force_restart=False,
        force_replot=True,
        force_replot_per_unit=False,
        force_rereport=False,
        reports=templates_runner._disable_reports_config(inputs.reports),
    )
    templates_runner._run_reconstruct_templates_pipeline_monolithic(unit_inputs)
    summary_json = templates_out_dir / "templates_summary.json"
    if summary_json.exists():
        summary = templates_runner.read_json(summary_json)
        if isinstance(summary, dict):
            summary["phase"] = "per_unit_processing"
            return summary
    return {"phase": "per_unit_processing", "templates_out_dir": str(templates_out_dir)}
