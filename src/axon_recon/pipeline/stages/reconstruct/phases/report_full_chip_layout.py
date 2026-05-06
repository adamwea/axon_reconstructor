from __future__ import annotations

from pathlib import Path
from typing import Any

from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner
from axon_recon.pipeline.stages.reconstruct.models.inputs import ReconstructionInputs
from axon_recon.pipeline.stages.reconstruct.models.results import UnitReconstructionResult


def run_reconstruct_report_full_chip_layout_phase(inputs: ReconstructionInputs) -> dict[str, Any]:
    env = reconstruct_runner._prepare_reconstruct_phase_environment(
        inputs=inputs, clear_output_root=False
    )
    unit_results = reconstruct_runner._load_full_chip_layout_unit_results(
        reconstruction_out_dir=env.reconstruction_out_dir,
        inputs=inputs,
        merged_units_dir=env.merged_units_dir,
        unit_ids=env.unit_ids,
    )
    stage_outputs = _run_reconstruct_report_full_chip_layout_phase_impl(
        inputs=inputs, env=env, unit_results=unit_results
    )
    summary_json = (
        env.reconstruction_out_dir
        / Path(str(inputs.phases.report_full_chip_layout.summary_json_relpath)).expanduser()
    )
    return reconstruct_runner._write_reconstruct_phase_summary(
        phase_name="report_full_chip_layout",
        summary_json=summary_json,
        inputs=inputs,
        well_out_dir=env.well_out_dir,
        reconstruction_out_dir=env.reconstruction_out_dir,
        unit_results=unit_results,
        stage_outputs=stage_outputs,
        failed_units_summary_json=reconstruct_runner._current_failed_units_summary_json(
            inputs=inputs,
            reconstruction_out_dir=env.reconstruction_out_dir,
        ),
        preserve_stage_reports=False,
    )


def _run_reconstruct_report_full_chip_layout_phase_impl(
    *,
    inputs: ReconstructionInputs,
    env: Any,
    unit_results: list[UnitReconstructionResult],
) -> dict[str, str]:
    reconstruct_runner.LOGGER.info(
        "reconstruct.report_full_chip_layout overwrite policy: action=rewrite preserve_stage_reports_requested=%s force_restart=%s force_replot=%s selected_units=%d discovered_units=%d existing_full_chip_outputs=%s",
        bool(env.preserve_stage_reports),
        bool(inputs.force_restart),
        bool(inputs.force_replot),
        len(env.unit_ids),
        len(unit_results),
        sorted(
            key for key in env.existing_stage_outputs if str(key).startswith("full_chip_layout_")
        ),
    )
    return reconstruct_runner.run_report_full_chip_layout_core_phase(
        inputs=inputs,
        reconstruction_out_dir=env.reconstruction_out_dir,
        merged_units_dir=env.merged_units_dir,
        full_channels_templates_dir=env.full_channels_templates_dir,
        unit_results=unit_results,
        preserve_stage_reports=False,
        existing_stage_outputs=env.existing_stage_outputs,
        load_templates_for_unit_fn=reconstruct_runner.load_templates_for_unit,
        write_full_chip_layout_plot_fn=reconstruct_runner.write_full_chip_layout_plot,
        write_json_fn=reconstruct_runner.write_json,
        resolve_unit_output_paths_fn=reconstruct_runner.resolve_unit_output_paths,
        resolve_full_chip_layout_output_paths_fn=reconstruct_runner.resolve_full_chip_layout_output_paths,
        logger=reconstruct_runner.LOGGER,
    )
