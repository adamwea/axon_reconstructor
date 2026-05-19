from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

from axon_recon.runtime_config import RuntimeConfig

from .models.inputs import (
	DEFAULT_INIT_PHASE_SEQUENCE,
	InitCopySrcToScratchPhaseConfig,
	InitPhasesConfig,
)


LOGGER = logging.getLogger("axon_recon.init.config")

_DEFAULT_OUTPUT_REL_ROOT = "init_outputs"


_INIT_PHASE_ALIASES: dict[str, str] = {
	"copy_src_to_scratch": "copy_src_to_scratch",
	"copy_source_to_scratch": "copy_src_to_scratch",
	"copy_src": "copy_src_to_scratch",
}


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


def _canonical_init_phase_name(value: Any, *, context: str) -> str:
	text = str(value or "").strip()
	if not text:
		raise ValueError(f"Empty {context} for init stage")
	if "." in text and text.split(".", 1)[0].strip().lower() == "init":
		text = text.split(".", 1)[1]
	token = text.strip().replace("-", "_").replace(" ", "_").lower()
	resolved = _INIT_PHASE_ALIASES.get(token, token)
	if resolved not in set(DEFAULT_INIT_PHASE_SEQUENCE):
		supported = ", ".join(DEFAULT_INIT_PHASE_SEQUENCE)
		raise ValueError(
			f"Unknown init {context}: {value!r}. Supported phases: {supported}"
		)
	return resolved


def _normalize_output_rel_root(raw: Any) -> str:
	text = str(raw or _DEFAULT_OUTPUT_REL_ROOT).strip()
	if not text:
		return _DEFAULT_OUTPUT_REL_ROOT
	text = text.lstrip("/")
	return text or _DEFAULT_OUTPUT_REL_ROOT


def _normalize_init_phase_sequence(value: Any) -> tuple[str, ...]:
	if value is None:
		return ()
	items = _as_list_of_strings(value)
	if not items:
		return ()
	canonical: list[str] = []
	seen: set[str] = set()
	for item in items:
		token = _canonical_init_phase_name(item, context="phase_sequence entry")
		if token in seen:
			continue
		seen.add(token)
		canonical.append(token)
	return tuple(canonical)


def _phase_resource_class(raw_cfg: dict[str, Any], _phase_name: str) -> str | None:
	raw = raw_cfg.get("resource_class", None)
	if raw is None:
		return None
	text = str(raw).strip()
	return text or None


def _parse_copy_src_to_scratch_phase(raw_cfg: Any) -> InitCopySrcToScratchPhaseConfig:
	cfg = raw_cfg if isinstance(raw_cfg, dict) else {}
	return InitCopySrcToScratchPhaseConfig(
		enabled=_as_bool(cfg.get("enabled", cfg.get("enable", False)), False),
		requires_use_scratch_root=_as_bool(cfg.get("requires_use_scratch_root", False), False),
		summary_json_relpath=str(
			cfg.get("summary_json_relpath", "context/copy_src_to_scratch_summary.json")
			or "context/copy_src_to_scratch_summary.json"
		),
		resource_class=_phase_resource_class(cfg, "copy_src_to_scratch"),
	)


@dataclass(frozen=True)
class InitStageConfig:
	"""Init stage configuration.

	Slice 5 wires `copy_src_to_scratch` into the stage. The stage stays
	disabled-by-default in YAML (so existing runtime configs that never carry
	a `stages.init` block continue to no-op); flipping `enabled: true` plus
	listing `copy_src_to_scratch` in `phase_sequence` runs the phase.
	"""

	output_rel_root: str = _DEFAULT_OUTPUT_REL_ROOT
	enabled: bool = False
	phase_sequence: tuple[str, ...] = ()
	force_restart: bool = False
	force_replot: bool = False
	phases: InitPhasesConfig = field(default_factory=InitPhasesConfig)


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
	phase_sequence = _normalize_init_phase_sequence(stage_cfg.get("phase_sequence", []))

	phases_cfg = stage_cfg.get("phases", {})
	phases_cfg = phases_cfg if isinstance(phases_cfg, dict) else {}
	copy_phase = _parse_copy_src_to_scratch_phase(phases_cfg.get("copy_src_to_scratch", {}))

	return InitStageConfig(
		output_rel_root=output_rel_root,
		enabled=enabled,
		phase_sequence=phase_sequence,
		force_restart=bool(force_restart_override or False),
		force_replot=bool(force_replot_override or False),
		phases=InitPhasesConfig(copy_src_to_scratch=copy_phase),
	)
