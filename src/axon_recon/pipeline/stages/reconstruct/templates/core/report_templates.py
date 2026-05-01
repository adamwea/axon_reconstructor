from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models.inputs import TemplatesInputs


def report_templates_pdf_requested(inputs: TemplatesInputs) -> bool:
	return bool(inputs.phases.report_templates.write_pdf)


def build_report_templates_phase_summary(
	*,
	inputs: TemplatesInputs,
	templates_out_dir: Path,
	rendered_units: list[Any],
	missing_units: list[dict[str, Any]],
	report_outputs: dict[str, str],
	duration_seconds: float,
) -> dict[str, Any]:
	return {
		"phase": "report_templates",
		"templates_out_dir": str(templates_out_dir),
		"duration_seconds": float(duration_seconds),
		"unit_count": int(len(rendered_units) + len(missing_units)),
		"rendered_units": list(rendered_units),
		"missing_units": list(missing_units),
		"write_pdf": bool(inputs.phases.report_templates.write_pdf),
		"report_relpath": str(inputs.phases.report_templates.relpath),
		"summary_json_relpath": str(inputs.phases.report_templates.summary_json_relpath),
		"source_output_key": "template_circles_png",
		"outputs": dict(report_outputs),
	}
