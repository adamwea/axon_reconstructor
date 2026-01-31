"""Templates step.

This package extracts per-source templates, builds merged_union templates, and
writes QC PDFs and JSON summaries.

Primary entry points are re-exported from .main for compatibility.
"""

from __future__ import annotations

from .main import TemplateExtractInputs, TemplateExtractOutputs, extract_and_merge_templates
from .replot import replot_templates_outputs_from_disk, replot_unit_from_disk

__all__ = [
    "TemplateExtractInputs",
    "TemplateExtractOutputs",
    "extract_and_merge_templates",
    "replot_templates_outputs_from_disk",
    "replot_unit_from_disk",
]
