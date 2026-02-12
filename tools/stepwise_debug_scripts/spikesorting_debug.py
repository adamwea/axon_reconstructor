#!/usr/bin/env python3
"""Compatibility shim.

Project scripts should import from `axon_reconstructor.devtools.spikesorting_debug`.
This module remains for older workflows that added `tools/stepwise_debug_scripts` to
`sys.path` and imported `spikesorting_debug` directly.
"""

from axon_reconstructor.devtools.spikesorting_debug import (  # noqa: F401
    PREPROCESS_OUTPUTS_DIRNAME,
    SPIKESORTING_OUTPUTS_DIRNAME,
    SpikeSortingInputs,
    SpikeSortingOutputs,
    run_spikesorting_only,
)

__all__ = [
    "PREPROCESS_OUTPUTS_DIRNAME",
    "SPIKESORTING_OUTPUTS_DIRNAME",
    "SpikeSortingInputs",
    "SpikeSortingOutputs",
    "run_spikesorting_only",
]
