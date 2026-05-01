from __future__ import annotations

from .models.inputs import ReconstructionInputs
from .models.results import ReconstructionResult
from .runner import (
	run_reconstruct_clear_templates_cache_phase,
	run_reconstruct_generate_gtrs_phase,
	run_reconstruct_plot_branch_propagations_phase,
	run_reconstruct_plot_branch_velocities_phase,
	run_reconstruct_plot_unit_summary_phase,
	run_reconstruct_plot_recons_phase,
	run_reconstruct_report_full_chip_layout_phase,
	run_reconstruct_report_recons_phase,
	run_reconstruct_report_summaries_phase,
	run_reconstruct_stage,
	run_reconstruct_templates_analyzers_phase,
	run_reconstruct_templates_build_templates_phase,
	run_reconstruct_templates_compute_template_similarity_phase,
	run_reconstruct_templates_extract_template_segments_phase,
	run_reconstruct_templates_plot_templates_phase,
	run_reconstruct_templates_report_templates_phase,
	run_reconstruct_templates_reports_phase,
	run_reconstruct_templates_resolve_sources_phase,
)


def run_reconstruct(inputs: ReconstructionInputs) -> ReconstructionResult:
	return run_reconstruct_stage(inputs)


def run_reconstruct_templates_resolve_sources(inputs: ReconstructionInputs) -> dict[str, object]:
	return run_reconstruct_templates_resolve_sources_phase(inputs)


def run_reconstruct_templates_analyzers(inputs: ReconstructionInputs) -> dict[str, object]:
	return run_reconstruct_templates_analyzers_phase(inputs)


def run_reconstruct_templates_extract_template_segments(inputs: ReconstructionInputs) -> dict[str, object]:
	return run_reconstruct_templates_extract_template_segments_phase(inputs)


def run_reconstruct_templates_build_templates(inputs: ReconstructionInputs) -> dict[str, object]:
	return run_reconstruct_templates_build_templates_phase(inputs)


def run_reconstruct_templates_compute_template_similarity(inputs: ReconstructionInputs) -> dict[str, object]:
	return run_reconstruct_templates_compute_template_similarity_phase(inputs)


def run_reconstruct_templates_plot_templates(inputs: ReconstructionInputs) -> dict[str, object]:
	return run_reconstruct_templates_plot_templates_phase(inputs)


def run_reconstruct_templates_report_templates(inputs: ReconstructionInputs) -> dict[str, object]:
	return run_reconstruct_templates_report_templates_phase(inputs)


def run_reconstruct_templates_reports(inputs: ReconstructionInputs) -> dict[str, object]:
	return run_reconstruct_templates_reports_phase(inputs)


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


def run_reconstruct_clear_templates_cache(inputs: ReconstructionInputs) -> dict[str, object]:
	return run_reconstruct_clear_templates_cache_phase(inputs)
