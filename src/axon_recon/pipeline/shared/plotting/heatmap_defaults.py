from __future__ import annotations

from typing import Any

from axon_recon.runtime_config import RuntimeConfig


def _first_stage_block(runtime_config: RuntimeConfig, paths: tuple[str, ...]) -> dict[str, Any]:
    for path in paths:
        block = runtime_config.get(path, {})
        if isinstance(block, dict) and block:
            return dict(block)
    return {}


def build_stage_plot_block(
    *,
    runtime_config: RuntimeConfig,
    stage_paths: tuple[str, ...],
) -> dict[str, Any]:
    """Return the first non-empty stage-local plotting block from the ordered path list."""

    return _first_stage_block(runtime_config, stage_paths)
