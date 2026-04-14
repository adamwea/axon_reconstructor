from __future__ import annotations

from .models.inputs import TemplatesInputs
from .models.results import TemplatesResult
from .runner import (
	run_templates_analyzers_phase,
	run_templates_build_templates_phase,
	run_templates_extract_template_segments_phase,
	run_templates_per_unit_processing_phase,
	run_templates_reports_phase,
	run_templates_resolve_sources_phase,
	run_templates_stage,
)


def run_templates(inputs: TemplatesInputs) -> TemplatesResult:
	return run_templates_stage(inputs)


def run_templates_resolve_sources(inputs: TemplatesInputs) -> dict[str, object]:
	return run_templates_resolve_sources_phase(inputs)


def run_templates_analyzers(inputs: TemplatesInputs, *, source_scope: str | None = None) -> dict[str, object]:
	return run_templates_analyzers_phase(inputs, source_scope=source_scope)


def run_templates_extract_template_segments(inputs: TemplatesInputs) -> dict[str, object]:
	return run_templates_extract_template_segments_phase(inputs)


def run_templates_build_templates(inputs: TemplatesInputs) -> dict[str, object]:
	return run_templates_build_templates_phase(inputs)


def run_templates_per_unit_processing(inputs: TemplatesInputs) -> dict[str, object]:
	return run_templates_per_unit_processing_phase(inputs)


def run_templates_reports(inputs: TemplatesInputs, *, report_scope: str | None = None) -> dict[str, object]:
	return run_templates_reports_phase(inputs, report_scope=report_scope)
