from __future__ import annotations

from .models.inputs import ReconstructionInputs
from .models.results import ReconstructionResult
from .runner import (
	run_reconstruct_generate_gtrs_phase,
	run_reconstruct_plot_branch_propagations_phase,
	run_reconstruct_plot_branch_velocities_phase,
	run_reconstruct_plot_unit_summary_phase,
	run_reconstruct_plot_recons_phase,
	run_reconstruct_report_full_chip_layout_phase,
	run_reconstruct_report_recons_phase,
	run_reconstruct_report_summaries_phase,
	run_reconstruct_stage,
)


def run_reconstruct(inputs: ReconstructionInputs) -> ReconstructionResult:
	return run_reconstruct_stage(inputs)


def run_reconstruct_generate_gtrs(inputs: ReconstructionInputs) -> dict[str, object]:
	return run_reconstruct_generate_gtrs_phase(inputs)


def run_reconstruct_plot_recons(inputs: ReconstructionInputs) -> dict[str, object]:
	return run_reconstruct_plot_recons_phase(inputs)


def run_reconstruct_plot_branch_propagations(inputs: ReconstructionInputs) -> dict[str, object]:
	return run_reconstruct_plot_branch_propagations_phase(inputs)


def run_reconstruct_plot_branch_velocities(inputs: ReconstructionInputs) -> dict[str, object]:
	return run_reconstruct_plot_branch_velocities_phase(inputs)


def run_reconstruct_plot_unit_summary(inputs: ReconstructionInputs) -> dict[str, object]:
	return run_reconstruct_plot_unit_summary_phase(inputs)


def run_reconstruct_report_recons(inputs: ReconstructionInputs) -> dict[str, object]:
	return run_reconstruct_report_recons_phase(inputs)


def run_reconstruct_report_full_chip_layout(inputs: ReconstructionInputs) -> dict[str, object]:
	return run_reconstruct_report_full_chip_layout_phase(inputs)


def run_reconstruct_report_summaries(inputs: ReconstructionInputs) -> dict[str, object]:
	return run_reconstruct_report_summaries_phase(inputs)
