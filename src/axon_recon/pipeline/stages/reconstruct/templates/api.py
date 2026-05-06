from __future__ import annotations

from ..phases.analyzers import run_reconstruct_templates_analyzers_phase
from ..phases.build_templates import run_reconstruct_templates_build_templates_phase
from ..phases.compute_template_similarity import (
    run_reconstruct_templates_compute_template_similarity_phase,
)
from ..phases.per_unit_processing import run_reconstruct_templates_per_unit_processing_phase
from ..phases.plot_templates import run_reconstruct_templates_plot_templates_phase
from ..phases.report_templates import run_reconstruct_templates_report_templates_phase
from ..phases.reports import run_reconstruct_templates_reports_phase
from ..phases.resolve_sources import run_reconstruct_templates_resolve_sources_phase
from .models.inputs import TemplatesInputs
from .models.results import TemplatesResult
from .runner import (
    collect_templates_result_from_outputs,
    run_reconstruct_templates_pipeline,
)


def run_reconstruct_templates(inputs: TemplatesInputs) -> TemplatesResult:
    return run_reconstruct_templates_pipeline(inputs)


def collect_templates_result(inputs: TemplatesInputs) -> TemplatesResult:
    return collect_templates_result_from_outputs(inputs)


def run_reconstruct_templates_resolve_sources(inputs: TemplatesInputs) -> dict[str, object]:
    return run_reconstruct_templates_resolve_sources_phase(inputs)


def run_reconstruct_templates_analyzers(
    inputs: TemplatesInputs, *, source_scope: str | None = None
) -> dict[str, object]:
    return run_reconstruct_templates_analyzers_phase(inputs, source_scope=source_scope)


def run_reconstruct_templates_build_templates(inputs: TemplatesInputs) -> dict[str, object]:
    return run_reconstruct_templates_build_templates_phase(inputs)


def run_reconstruct_templates_compute_template_similarity(
    inputs: TemplatesInputs,
) -> dict[str, object]:
    return run_reconstruct_templates_compute_template_similarity_phase(inputs)


def run_reconstruct_templates_plot_templates(inputs: TemplatesInputs) -> dict[str, object]:
    return run_reconstruct_templates_plot_templates_phase(inputs)


def run_reconstruct_templates_report_templates(inputs: TemplatesInputs) -> dict[str, object]:
    return run_reconstruct_templates_report_templates_phase(inputs)


def run_reconstruct_templates_per_unit_processing(inputs: TemplatesInputs) -> dict[str, object]:
    return run_reconstruct_templates_per_unit_processing_phase(inputs)


def run_reconstruct_templates_reports(
    inputs: TemplatesInputs, *, report_scope: str | None = None
) -> dict[str, object]:
    return run_reconstruct_templates_reports_phase(inputs, report_scope=report_scope)
