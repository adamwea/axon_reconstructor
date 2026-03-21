from __future__ import annotations

from .models.inputs import TemplatesInputs
from .models.results import TemplatesResult
from .runner import run_templates_stage


def run_templates(inputs: TemplatesInputs) -> TemplatesResult:
	return run_templates_stage(inputs)
