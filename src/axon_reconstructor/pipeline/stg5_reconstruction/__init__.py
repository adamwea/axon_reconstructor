"""Reconstruction step.

This package runs axon reconstruction (axon_velocity + additional logic) using
templates-stage artifacts.

Reconstruction is designed to consume dense full-channel templates under
`<well>/stg4_templates_outputs/templates/full/` for best compatibility with
axon_velocity. It reads merged-contributing metadata best-effort (e.g. sampling
frequency) from `<well>/stg4_templates_outputs/templates/merged/`.

Primary entry points are re-exported from .main for compatibility.
"""

from __future__ import annotations

from .main import ReconstructionInputs, ReconstructionOutputs, reconstruct_from_templates

__all__ = [
    "ReconstructionInputs",
    "ReconstructionOutputs",
    "reconstruct_from_templates",
]
