"""Public pipeline API for lightweight signal processing helpers.

Historically, template processing helpers lived in various internal modules.
Tests and pipeline code import these via `axon_reconstructor.pipeline.signal_processing`.

This file is intentionally tiny: it re-exports the stable helper(s) from the
internal dependency bundle.
"""

from __future__ import annotations

from axon_reconstructor._dep.internal_dep.generate_templates.signal_processing import (  # noqa: F401
    get_time_derivative,
)
