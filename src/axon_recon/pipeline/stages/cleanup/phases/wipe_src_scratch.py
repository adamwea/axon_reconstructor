"""`cleanup.wipe_src_scratch` phase orchestrator.

Mirrors the init-stage `copy_src_to_scratch` orchestrator: validates the
per-target preconditions, invokes `run_wipe_src_scratch_core` to remove the
scratch-staged H5 + sidecar `.cfg` files, and writes the
`wipe_src_scratch_summary.json` marker under the well's cleanup-stage output
dir.

The marker path is `<well>/cleanup_outputs/context/wipe_src_scratch_summary.json`
by default. `status` in the marker is one of the standard checkpoint statuses
from `guardrails/stage_phase_architecture.md` (plus phase-specific values
`skipped`, `deferred`, `dry_run`, `already_missing` carried over from the
previous preprocess-side implementation).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import Any

from ....output_paths import compute_mea_analysis_output_dir
from ..core.wipe_src_scratch import run_wipe_src_scratch_core
from ..models.inputs import CleanupInputs


LOGGER = logging.getLogger("axon_recon.cleanup.wipe_src_scratch")


def _utc_now_iso() -> str:
	return dt.datetime.now(dt.timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, dict):
		return {str(k): _json_ready(v) for k, v in value.items()}
	if isinstance(value, (list, tuple, set)):
		return [_json_ready(v) for v in value]
	if isinstance(value, (str, int, float, bool)) or value is None:
		return value
	return str(value)


def _write_json(path: Path, payload: dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_cleanup_out_dir(inputs: CleanupInputs) -> Path:
	well_out_dir = compute_mea_analysis_output_dir(
		output_root=inputs.mea_output_root,
		data_file=inputs.h5_path,
		well=inputs.stream_id,
	)
	return well_out_dir / str(inputs.output_rel_root)


def _resolve_summary_json_path(inputs: CleanupInputs) -> Path:
	relpath = str(inputs.phases.wipe_src_scratch.summary_json_relpath).strip()
	if not relpath:
		relpath = "context/wipe_src_scratch_summary.json"
	cleanup_out_dir = _resolve_cleanup_out_dir(inputs)
	return Path(os.path.abspath(str(cleanup_out_dir / relpath)))


def _validate_wipe_phase_requirements(inputs: CleanupInputs) -> None:
	if bool(inputs.phases.wipe_src_scratch.requires_use_scratch_root) and not bool(inputs.copied_to_scratch):
		raise RuntimeError(
			"cleanup wipe_src_scratch phase requires scratch input materialization, but the selected target is using the source h5 path"
		)


def run_cleanup_wipe_src_scratch_phase(inputs: CleanupInputs) -> dict[str, Any]:
	"""Execute the cleanup-stage `wipe_src_scratch` phase for one target.

	Returns the marker payload written to `summary_json_relpath` under the
	target's cleanup-stage output dir. The cleanup stage owns its own
	output_rel_root (`cleanup_outputs/`) so its force-restart rmtree scope
	stays isolated from the preprocess / spikesort / reconstruct trees.

	Slice 6 ships this as a single-target orchestrator: each invocation
	receives a fully-resolved `CleanupInputs` and writes one summary marker.
	No cross-target coordination (the old preprocess `_acquire_scratch_input_usage`
	gating was tied to running wipe at the end of a single preprocess
	target's phase chain; since cleanup is a separate stage now, the user
	invokes it after all consumers of the scratch input have finished, and
	the gating shifts from per-process refcounting to per-invocation
	contract).
	"""

	_validate_wipe_phase_requirements(inputs)

	source_h5_path = Path(inputs.source_h5_path or inputs.h5_path)
	core_payload = run_wipe_src_scratch_core(
		h5_path=Path(inputs.h5_path),
		source_h5_path=source_h5_path,
		copied_to_scratch=bool(inputs.copied_to_scratch),
		dry_run=bool(inputs.phases.wipe_src_scratch.dry_run),
		requires_use_scratch_root=bool(inputs.phases.wipe_src_scratch.requires_use_scratch_root),
		# Single-target invocation: no shared-user refcounting on this side.
		active_shared_users_remaining=0,
	)

	summary_json_path = _resolve_summary_json_path(inputs)
	cleanup_out_dir = _resolve_cleanup_out_dir(inputs)
	payload: dict[str, Any] = {
		"phase": "wipe_src_scratch",
		"status": str(core_payload.get("status", "ok")),
		"well_out_dir": str(cleanup_out_dir.parent),
		"cleanup_out_dir": str(cleanup_out_dir),
		"completed_at": _utc_now_iso(),
		"inputs": {
			"h5_path": str(inputs.h5_path),
			"source_h5_path": str(source_h5_path),
			"stream_id": str(inputs.stream_id),
			"copied_to_scratch": bool(inputs.copied_to_scratch),
			"dry_run": bool(inputs.phases.wipe_src_scratch.dry_run),
			"requires_use_scratch_root": bool(inputs.phases.wipe_src_scratch.requires_use_scratch_root),
		},
		"outputs": {
			"source_h5_path": str(source_h5_path),
			"resolved_h5_path": str(inputs.h5_path),
		},
	}
	# Surface the core payload's removed/would_remove/missing lists at the top
	# level so consumers don't have to dig into a nested dict (matches the old
	# preprocess-side marker shape).
	for key in (
		"source_h5_path",
		"resolved_h5_path",
		"copied_to_scratch",
		"dry_run",
		"requires_use_scratch_root",
		"active_shared_users_remaining",
		"removed_paths",
		"would_remove_paths",
		"missing_paths",
		"reason",
	):
		if key in core_payload:
			payload[key] = core_payload[key]
	_write_json(summary_json_path, _json_ready(payload))
	payload["summary_json"] = str(summary_json_path)
	LOGGER.info(
		"cleanup wipe_src_scratch phase complete stream_id=%s copied_to_scratch=%s dry_run=%s status=%s summary_json=%s",
		str(inputs.stream_id),
		bool(inputs.copied_to_scratch),
		bool(inputs.phases.wipe_src_scratch.dry_run),
		str(payload["status"]),
		str(summary_json_path),
	)
	return payload
