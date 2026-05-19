from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

from axon_recon.runtime_config import RuntimeConfig

from .models.inputs import (
	DEFAULT_CLEANUP_PHASE_SEQUENCE,
	CleanupPhasesConfig,
	CleanupWipeSrcScratchPhaseConfig,
)


LOGGER = logging.getLogger("axon_recon.cleanup.config")

_DEFAULT_OUTPUT_REL_ROOT = "cleanup_outputs"


_CLEANUP_PHASE_ALIASES: dict[str, str] = {
	"wipe_src_scratch": "wipe_src_scratch",
	"wipe_source_scratch": "wipe_src_scratch",
	"cleanup_scratch_copy": "wipe_src_scratch",
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


def _canonical_cleanup_phase_name(value: Any, *, context: str) -> str:
	text = str(value or "").strip()
	if not text:
		raise ValueError(f"Empty {context} for cleanup stage")
	if "." in text and text.split(".", 1)[0].strip().lower() == "cleanup":
		text = text.split(".", 1)[1]
	token = text.strip().replace("-", "_").replace(" ", "_").lower()
	resolved = _CLEANUP_PHASE_ALIASES.get(token, token)
	if resolved not in set(DEFAULT_CLEANUP_PHASE_SEQUENCE):
		supported = ", ".join(DEFAULT_CLEANUP_PHASE_SEQUENCE)
		raise ValueError(
			f"Unknown cleanup {context}: {value!r}. Supported phases: {supported}"
		)
	return resolved


def _normalize_output_rel_root(raw: Any) -> str:
	text = str(raw or _DEFAULT_OUTPUT_REL_ROOT).strip()
	if not text:
		return _DEFAULT_OUTPUT_REL_ROOT
	text = text.lstrip("/")
	return text or _DEFAULT_OUTPUT_REL_ROOT


def _normalize_cleanup_phase_sequence(value: Any) -> tuple[str, ...]:
	if value is None:
		return ()
	items = _as_list_of_strings(value)
	if not items:
		return ()
	canonical: list[str] = []
	seen: set[str] = set()
	for item in items:
		token = _canonical_cleanup_phase_name(item, context="phase_sequence entry")
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


def _parse_wipe_src_scratch_phase(raw_cfg: Any) -> CleanupWipeSrcScratchPhaseConfig:
	cfg = raw_cfg if isinstance(raw_cfg, dict) else {}
	return CleanupWipeSrcScratchPhaseConfig(
		enabled=_as_bool(cfg.get("enabled", cfg.get("enable", False)), False),
		dry_run=_as_bool(cfg.get("dry_run", False), False),
		requires_use_scratch_root=_as_bool(cfg.get("requires_use_scratch_root", False), False),
		summary_json_relpath=str(
			cfg.get("summary_json_relpath", "context/wipe_src_scratch_summary.json")
			or "context/wipe_src_scratch_summary.json"
		),
		resource_class=_phase_resource_class(cfg, "wipe_src_scratch"),
	)


@dataclass(frozen=True)
class CleanupStageConfig:
	"""Cleanup stage configuration.

	Slice 6 wires `wipe_src_scratch` into the stage as its first (and currently
	only) phase. The stage is disabled-by-default in YAML so cleanup never runs
	implicitly — the user opts in by flipping `enabled: true`. The long-term
	roadmap (`tech_debt.md` §"Finalize the phase roster", and
	`phase_roster_cleanup_plan.md` slice 6 open question 2) is to consolidate
	the scattered per-stage cleanup phases (`spikesort.cleanup_concat_binary`,
	`spikesort.cleanup_analyzers`, `reconstruct.clear_templates_cache`) into
	this stage so end-of-run disk hygiene lives in one place.
	"""

	output_rel_root: str = _DEFAULT_OUTPUT_REL_ROOT
	enabled: bool = False
	phase_sequence: tuple[str, ...] = ()
	force_restart: bool = False
	replot: bool = False
	phases: CleanupPhasesConfig = field(default_factory=CleanupPhasesConfig)


def parse_cleanup_stage_config(
	*,
	runtime_config: RuntimeConfig,
	data_config: RuntimeConfig | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
) -> CleanupStageConfig:
	"""Parse the `stages.cleanup` block from the runtime config.

	Defaults match `CleanupStageConfig` (disabled, empty phase_sequence). When
	the YAML omits the block entirely the returned config is the all-defaults
	instance — the stage is a no-op in that case.
	"""

	del data_config  # currently unused; kept for signature parity with sibling stages

	stage_cfg = runtime_config.get("stages.cleanup", {})
	stage_cfg = stage_cfg if isinstance(stage_cfg, dict) else {}

	output_rel_root = _normalize_output_rel_root(stage_cfg.get("output_rel_root", None))
	enabled = _as_bool(stage_cfg.get("enabled", False), False)
	phase_sequence = _normalize_cleanup_phase_sequence(stage_cfg.get("phase_sequence", []))

	phases_cfg = stage_cfg.get("phases", {})
	phases_cfg = phases_cfg if isinstance(phases_cfg, dict) else {}
	wipe_phase = _parse_wipe_src_scratch_phase(phases_cfg.get("wipe_src_scratch", {}))

	return CleanupStageConfig(
		output_rel_root=output_rel_root,
		enabled=enabled,
		phase_sequence=phase_sequence,
		force_restart=bool(force_restart_override or False),
		replot=bool(replot_override or False),
		phases=CleanupPhasesConfig(wipe_src_scratch=wipe_phase),
	)
