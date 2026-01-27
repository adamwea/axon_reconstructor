"""Templates step public API.

The heavy implementation lives in .runner to keep this module orchestration-only.
"""

from __future__ import annotations

from .runner import TEMPLATES_OUTPUTS_DIRNAME, TemplateExtractInputs, TemplateExtractOutputs, extract_and_merge_templates

__all__ = [
    "TEMPLATES_OUTPUTS_DIRNAME",
    "TemplateExtractInputs",
    "TemplateExtractOutputs",
    "extract_and_merge_templates",
]
