"""Compatibility shim.

Project scripts should import from `axon_reconstructor.devtools.preprocessing_debug`.
This module remains for older workflows that added `tools/stepwise_debug_scripts` to
`sys.path` and imported `preprocessing_debug` directly.
"""

from axon_reconstructor.devtools.preprocessing_debug import (  # noqa: F401
    PreprocessInputs,
    PreprocessOutputs,
    run_preprocessing_with_validations,
    validate_preprocessed_recording,
)

__all__ = [
    "PreprocessInputs",
    "PreprocessOutputs",
    "run_preprocessing_with_validations",
    "validate_preprocessed_recording",
]
