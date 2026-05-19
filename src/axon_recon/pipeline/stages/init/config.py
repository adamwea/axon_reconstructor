from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from axon_recon.runtime_config import RuntimeConfig


LOGGER = logging.getLogger("axon_recon.init.config")

_DEFAULT_OUTPUT_REL_ROOT = "init_outputs"


def _as_bool(value: Any, default: bool) -> bool:
	if value is None:
		return bool(default)
	if isinstance(value, bool):
		return value
	token = str(value).strip().lower()
	if token in {"1", "true", "yes", "on"}:
		return True
	if token in {"0", "false", "no", "off"}:
		return False
	return bool(default)


def _as_list_of_strings(value: Any) -> list[str]:
	if value is None:
		return []
	if isinstance(value, str):
		items: list[Any] = [value]
	elif isinstance(value, (list, tuple, set)):
		items = list(value)
	else:
		items = [value]
	return [str(item).strip() for item in items if str(item).strip()]


def _normalize_output_rel_root(raw: Any) -> str:
	text = str(raw or _DEFAULT_OUTPUT_REL_ROOT).strip()
	if not text:
		return _DEFAULT_OUTPUT_REL_ROOT
	text = text.lstrip("/")
	return text or _DEFAULT_OUTPUT_REL_ROOT


@dataclass(frozen=True)
class InitStageConfig:
	"""Scaffolded init stage configuration.

	Slice 4 introduces the stage with NO phases yet. Slice 5 will move
	`copy_src_to_scratch` here. Until then this dataclass intentionally
	stays minimal: any field that would only be needed once a real phase
	lands belongs in that follow-up slice, not this one.
	"""

	output_rel_root: str = _DEFAULT_OUTPUT_REL_ROOT
	enabled: bool = False
	phase_sequence: tuple[str, ...] = ()
	force_restart: bool = False
	force_replot: bool = False


def parse_init_stage_config(
	*,
	runtime_config: RuntimeConfig,
	data_config: RuntimeConfig | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> InitStageConfig:
	"""Parse the `stages.init` block from the runtime config.

	Defaults match `InitStageConfig` (disabled, empty phase_sequence). When the
	YAML omits the block entirely the returned config is the all-defaults
	instance — the stage is a no-op in that case.
	"""

	del data_config  # currently unused; kept for signature parity with sibling stages

	stage_cfg = runtime_config.get("stages.init", {})
	stage_cfg = stage_cfg if isinstance(stage_cfg, dict) else {}

	output_rel_root = _normalize_output_rel_root(stage_cfg.get("output_rel_root", None))
	enabled = _as_bool(stage_cfg.get("enabled", False), False)
	phase_sequence = tuple(_as_list_of_strings(stage_cfg.get("phase_sequence", [])))

	return InitStageConfig(
		output_rel_root=output_rel_root,
		enabled=enabled,
		phase_sequence=phase_sequence,
		force_restart=bool(force_restart_override or False),
		force_replot=bool(force_replot_override or False),
	)
