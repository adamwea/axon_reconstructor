from __future__ import annotations

from typing import Any, Callable

from .models.inputs import ReconstructionInputs
from .models.results import ReconstructionResult
from .runner import (
	_quiet_unexpected_plot_logs,
	run_reconstruct_clear_templates_cache_phase,
	run_reconstruct_generate_gtrs_phase,
	run_reconstruct_plot_branch_propagations_phase,
	run_reconstruct_plot_branch_velocities_phase,
	run_reconstruct_plot_unit_summary_phase,
	run_reconstruct_plot_recons_phase,
	run_reconstruct_report_full_chip_layout_phase,
	run_reconstruct_report_recon_grid_phase,
	run_reconstruct_report_recons_phase,
	run_reconstruct_report_summaries_phase,
	run_reconstruct_stage,
	run_reconstruct_templates_analyzers_phase,
	run_reconstruct_templates_build_templates_phase,
	run_reconstruct_templates_compute_template_similarity_phase,
	run_reconstruct_templates_plot_templates_phase,
	run_reconstruct_templates_report_templates_phase,
	run_reconstruct_templates_reports_phase,
	run_reconstruct_templates_resolve_sources_phase,
)


def _run_with_quiet_unexpected_plot_logs(
	inputs: ReconstructionInputs,
	runner: Callable[[ReconstructionInputs], Any],
) -> Any:
	with _quiet_unexpected_plot_logs(inputs):
		return runner(inputs)


def run_reconstruct(inputs: ReconstructionInputs) -> ReconstructionResult:
	return _run_with_quiet_unexpected_plot_logs(inputs, run_reconstruct_stage)


def run_reconstruct_templates_resolve_sources(inputs: ReconstructionInputs) -> dict[str, object]:
	return _run_with_quiet_unexpected_plot_logs(inputs, run_reconstruct_templates_resolve_sources_phase)


def run_reconstruct_templates_analyzers(inputs: ReconstructionInputs) -> dict[str, object]:
	return _run_with_quiet_unexpected_plot_logs(inputs, run_reconstruct_templates_analyzers_phase)


def run_reconstruct_templates_build_templates(inputs: ReconstructionInputs) -> dict[str, object]:
	return _run_with_quiet_unexpected_plot_logs(inputs, run_reconstruct_templates_build_templates_phase)


def run_reconstruct_templates_compute_template_similarity(inputs: ReconstructionInputs) -> dict[str, object]:
	return _run_with_quiet_unexpected_plot_logs(inputs, run_reconstruct_templates_compute_template_similarity_phase)


def run_reconstruct_templates_plot_templates(inputs: ReconstructionInputs) -> dict[str, object]:
	return _run_with_quiet_unexpected_plot_logs(inputs, run_reconstruct_templates_plot_templates_phase)


def run_reconstruct_templates_report_templates(inputs: ReconstructionInputs) -> dict[str, object]:
	return _run_with_quiet_unexpected_plot_logs(inputs, run_reconstruct_templates_report_templates_phase)


def run_reconstruct_templates_reports(inputs: ReconstructionInputs) -> dict[str, object]:
	return _run_with_quiet_unexpected_plot_logs(inputs, run_reconstruct_templates_reports_phase)


def run_reconstruct_generate_gtrs(inputs: ReconstructionInputs) -> dict[str, object]:
	return _run_with_quiet_unexpected_plot_logs(inputs, run_reconstruct_generate_gtrs_phase)


def run_reconstruct_plot_recons(inputs: ReconstructionInputs) -> dict[str, object]:
	return _run_with_quiet_unexpected_plot_logs(inputs, run_reconstruct_plot_recons_phase)


def run_reconstruct_plot_branch_propagations(inputs: ReconstructionInputs) -> dict[str, object]:
	return _run_with_quiet_unexpected_plot_logs(inputs, run_reconstruct_plot_branch_propagations_phase)


def run_reconstruct_plot_branch_velocities(inputs: ReconstructionInputs) -> dict[str, object]:
	return _run_with_quiet_unexpected_plot_logs(inputs, run_reconstruct_plot_branch_velocities_phase)


def run_reconstruct_plot_unit_summary(inputs: ReconstructionInputs) -> dict[str, object]:
	return _run_with_quiet_unexpected_plot_logs(inputs, run_reconstruct_plot_unit_summary_phase)


def run_reconstruct_report_recons(inputs: ReconstructionInputs) -> dict[str, object]:
	return _run_with_quiet_unexpected_plot_logs(inputs, run_reconstruct_report_recons_phase)


def run_reconstruct_report_recon_grid(inputs: ReconstructionInputs) -> dict[str, object]:
	return _run_with_quiet_unexpected_plot_logs(inputs, run_reconstruct_report_recon_grid_phase)


def run_reconstruct_report_full_chip_layout(inputs: ReconstructionInputs) -> dict[str, object]:
	return _run_with_quiet_unexpected_plot_logs(inputs, run_reconstruct_report_full_chip_layout_phase)


def run_reconstruct_report_summaries(inputs: ReconstructionInputs) -> dict[str, object]:
	return _run_with_quiet_unexpected_plot_logs(inputs, run_reconstruct_report_summaries_phase)


def run_reconstruct_clear_templates_cache(inputs: ReconstructionInputs) -> dict[str, object]:
	return _run_with_quiet_unexpected_plot_logs(inputs, run_reconstruct_clear_templates_cache_phase)
