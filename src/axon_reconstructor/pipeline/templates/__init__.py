"""Templates step.

This package extracts per-source templates, builds merged_union templates, and
writes QC PDFs and JSON summaries.

Primary entry points are re-exported from .main for compatibility.
"""

from __future__ import annotations

from .main import TemplateExtractInputs, TemplateExtractOutputs, extract_and_merge_templates

__all__ = [
    "TemplateExtractInputs",
    "TemplateExtractOutputs",
    "extract_and_merge_templates",
]
