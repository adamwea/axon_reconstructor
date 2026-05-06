from __future__ import annotations

from pathlib import Path
from typing import Any

from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner
from axon_recon.pipeline.stages.reconstruct.models.inputs import ReconstructionInputs
from axon_recon.pipeline.stages.reconstruct.models.results import UnitReconstructionResult


def run_reconstruct_plot_branch_velocities_phase(inputs: ReconstructionInputs) -> dict[str, Any]:
    env = reconstruct_runner._prepare_reconstruct_phase_environment(
        inputs=inputs, clear_output_root=False
    )
    unit_results, failed_units_summary_json = _run_reconstruct_plot_branch_velocities_phase_impl(
        inputs=inputs, env=env
    )
    summary_json = (
        env.reconstruction_out_dir
        / Path(str(inputs.phases.plot_branch_velocities.summary_json_relpath)).expanduser()
    )
    return reconstruct_runner._write_reconstruct_phase_summary(
        phase_name="plot_branch_velocities",
        summary_json=summary_json,
        inputs=inputs,
        well_out_dir=env.well_out_dir,
        reconstruction_out_dir=env.reconstruction_out_dir,
        unit_results=unit_results,
        failed_units_summary_json=failed_units_summary_json,
        preserve_stage_reports=env.preserve_stage_reports,
    )


def _run_reconstruct_plot_branch_velocities_phase_impl(
    *,
    inputs: ReconstructionInputs,
    env: Any,
) -> tuple[list[UnitReconstructionResult], Path | None]:
    unit_results = reconstruct_runner.run_plot_branch_velocities_core_phase(
        inputs=inputs,
        reconstruction_out_dir=env.reconstruction_out_dir,
        merged_units_dir=env.merged_units_dir,
        full_channels_templates_dir=env.full_channels_templates_dir,
        unit_ids=env.unit_ids,
        load_templates_for_unit_fn=reconstruct_runner.load_templates_for_unit,
        write_unit_branch_velocity_plot_fn=reconstruct_runner.write_unit_branch_velocity_plot,
        read_json_fn=reconstruct_runner.read_json,
        write_json_fn=reconstruct_runner.write_json,
        resolve_unit_output_paths_fn=reconstruct_runner.resolve_unit_output_paths,
        resolve_branch_phase_output_paths_fn=reconstruct_runner.resolve_branch_phase_output_paths,
        resolve_branch_phase_branch_output_paths_fn=reconstruct_runner.resolve_branch_phase_branch_output_paths,
        logger=reconstruct_runner.LOGGER,
    )
    return reconstruct_runner._cleanup_failed_reconstruct_unit_outputs(
        reconstruction_out_dir=env.reconstruction_out_dir,
        inputs=inputs,
        unit_results=unit_results,
    )
