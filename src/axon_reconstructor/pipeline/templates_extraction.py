"""Compatibility wrapper for the templates step.

Some debug harnesses (and older code) import:
  - `axon_reconstructor.pipeline.templates_extraction.TemplateExtractionInputs`
  - `axon_reconstructor.pipeline.templates_extraction.extract_templates`

The canonical implementation now lives in `axon_reconstructor.pipeline.templates`.
"""

from __future__ import annotations

from .templates import (  # noqa: F401
    TemplateExtractInputs as TemplateExtractionInputs,
    TemplateExtractOutputs as TemplateExtractionOutputs,
    extract_and_merge_templates as extract_templates,
)
