"""Pipeline-facing APIs.

This package is where the end-to-end reconstruction pipeline objects live.
"""

import sys

from .alias_modules import analysis, preprocessing, reconstruction, spikesorting, templates, waveforms
from .alias_modules.analysis import AnalysisInputs, AnalysisOutputs, analyze_units
from .alias_modules.preprocessing import run_preprocess_stage
from .alias_modules.reconstruction import ReconstructionInputs, ReconstructionOutputs, reconstruct_from_templates
from .alias_modules.templates import TemplateExtractInputs, TemplateExtractOutputs, extract_and_merge_templates
from .alias_modules.waveforms import WaveformExtractInputs, WaveformExtractOutputs, extract_waveforms

sys.modules[__name__ + ".preprocessing"] = preprocessing
sys.modules[__name__ + ".spikesorting"] = spikesorting
sys.modules[__name__ + ".waveforms"] = waveforms
sys.modules[__name__ + ".templates"] = templates
sys.modules[__name__ + ".reconstruction"] = reconstruction
sys.modules[__name__ + ".analysis"] = analysis

__all__ = [
    "AnalysisInputs",
    "AnalysisOutputs",
    "analyze_units",
    "run_preprocess_stage",
    "ReconstructionInputs",
    "ReconstructionOutputs",
    "reconstruct_from_templates",
    "TemplateExtractInputs",
    "TemplateExtractOutputs",
    "extract_and_merge_templates",
    "WaveformExtractInputs",
    "WaveformExtractOutputs",
    "extract_waveforms",
]
