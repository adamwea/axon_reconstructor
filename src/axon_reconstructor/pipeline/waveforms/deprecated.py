from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional


def _plot_and_curate_if_requested(
    *,
    inputs,
    waveforms_out_dir: Path,
    concat_waveforms_dir: Path,
    recording: Any,
    sorting: Any,
    window,
    logger: Any,
) -> tuple[Optional[Path], Optional[Path]]:
    """Deprecated legacy entrypoint.

    This function is intentionally not used anymore. It remains for backwards
    compatibility with older internal call sites.
    """

    warnings.warn(
        "_plot_and_curate_if_requested is deprecated; use waveforms.steps._curate_then_plot",
        DeprecationWarning,
        stacklevel=2,
    )
    raise RuntimeError("_plot_and_curate_if_requested is deprecated; use _curate_then_plot")


__all__ = [
    "_plot_and_curate_if_requested",
]
