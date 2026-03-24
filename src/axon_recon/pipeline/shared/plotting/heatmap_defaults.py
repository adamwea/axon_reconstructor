from __future__ import annotations

from typing import Any

from axon_reconstructor.runtime_config import RuntimeConfig


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(existing, value)
        else:
            merged[key] = value
    return merged


def _read_nested_block(root: dict[str, Any], path: str) -> dict[str, Any]:
    node: Any = root
    for part in path.split("."):
        if not isinstance(node, dict):
            return {}
        node = node.get(part)
    if isinstance(node, dict) and node:
        return dict(node)
    return {}


def _first_stage_block(runtime_config: RuntimeConfig, paths: tuple[str, ...]) -> dict[str, Any]:
    for path in paths:
        block = runtime_config.get(path, {})
        if isinstance(block, dict) and block:
            return dict(block)
    return {}


def _merge_global_defaults(
    *,
    runtime_config: RuntimeConfig,
    global_paths: tuple[str, ...],
) -> dict[str, Any]:
    root = runtime_config.get("global_heatmap_defaults", {})
    if not isinstance(root, dict) or not root:
        # Backward-compat alias during transition window.
        root = runtime_config.get("global_heatmap_plotting", {})
    if not isinstance(root, dict) or not root:
        return {}

    merged: dict[str, Any] = {}
    for path in global_paths:
        block = _read_nested_block(root, path)
        if block:
            merged = _deep_merge_dict(merged, block)
    return merged


def build_stage_plot_block(
    *,
    runtime_config: RuntimeConfig,
    stage_paths: tuple[str, ...],
    global_paths: tuple[str, ...],
) -> dict[str, Any]:
    """Resolve plotting config with precedence: stage-specific > global shared defaults.

    The caller passes ordered stage paths where the first non-empty dict wins.
    Global defaults are merged in order (earlier paths lower priority than later paths).
    """

    global_block = _merge_global_defaults(runtime_config=runtime_config, global_paths=global_paths)
    stage_block = _first_stage_block(runtime_config, stage_paths)
    if not global_block:
        return stage_block
    if not stage_block:
        return global_block
    return _deep_merge_dict(global_block, stage_block)
