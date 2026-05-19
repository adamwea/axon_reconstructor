"""`init.copy_src_to_scratch` phase orchestrator.

Mirrors the preprocess-side orchestrator (`stages/preprocess/orchestrators/
copy_src_to_scratch.py` before slice 5) but is wired to the init stage's
inputs / output_rel_root. The actual file copy occurs at execution-target
selection (`pipeline/config.py:select_execution_targets` materializes the
H5 + sidecar `.cfg` via `resolve_copy_src_to_scratch_input_path` when
`materialize_scratch_inputs=True`). This orchestrator's job is therefore to:

  - Validate the per-target preconditions (e.g. requires_use_scratch_root).
  - Invoke `run_copy_src_to_scratch_core` to assemble the phase payload.
  - Write the summary_json marker under the well's init-stage output dir.

The marker path is `<well>/init_outputs/context/copy_src_to_scratch_summary.json`
by default. `status` in the marker is one of the standard checkpoint statuses
from `guardrails/stage_phase_architecture.md`.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import Any

from ....output_paths import compute_mea_analysis_output_dir
from ..core.copy_src_to_scratch import run_copy_src_to_scratch_core
from ..models.inputs import InitInputs


LOGGER = logging.getLogger("axon_recon.init.copy_src_to_scratch")


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


def _resolve_init_out_dir(inputs: InitInputs) -> Path:
	well_out_dir = compute_mea_analysis_output_dir(
		output_root=inputs.mea_output_root,
		data_file=inputs.h5_path,
		well=inputs.stream_id,
	)
	return well_out_dir / str(inputs.output_rel_root)


def _resolve_summary_json_path(inputs: InitInputs) -> Path:
	relpath = str(inputs.phases.copy_src_to_scratch.summary_json_relpath).strip()
	if not relpath:
		relpath = "context/copy_src_to_scratch_summary.json"
	init_out_dir = _resolve_init_out_dir(inputs)
	return Path(os.path.abspath(str(init_out_dir / relpath)))


def _validate_copy_phase_requirements(inputs: InitInputs) -> None:
	if bool(inputs.phases.copy_src_to_scratch.requires_use_scratch_root) and not bool(inputs.copied_to_scratch):
		raise RuntimeError(
			"init copy_src_to_scratch phase requires scratch input materialization, but the selected target is using the source h5 path"
		)


def run_init_copy_src_to_scratch_phase(inputs: InitInputs) -> dict[str, Any]:
	"""Execute the init-stage `copy_src_to_scratch` phase for one target.

	Returns the marker payload written to `summary_json_relpath` under the
	target's init-stage output dir. Idempotent: re-running on an already-copied
	target re-confirms the source/resolved paths and re-writes a fresh `status: ok`
	summary. The heavy lifting (actual H5 + sidecar copy) is upstream in
	`select_execution_targets`'s `materialize_scratch_inputs=True` path.
	"""

	_validate_copy_phase_requirements(inputs)

	source_h5_path = Path(inputs.source_h5_path or inputs.h5_path)
	core_payload = run_copy_src_to_scratch_core(
		h5_path=inputs.h5_path,
		source_h5_path=source_h5_path,
		stream_id=str(inputs.stream_id),
		copied_to_scratch=bool(inputs.copied_to_scratch),
		requires_use_scratch_root=bool(inputs.phases.copy_src_to_scratch.requires_use_scratch_root),
	)

	summary_json_path = _resolve_summary_json_path(inputs)
	init_out_dir = _resolve_init_out_dir(inputs)
	payload: dict[str, Any] = {
		"phase": "copy_src_to_scratch",
		"status": "ok",
		"well_out_dir": str(init_out_dir.parent),
		"init_out_dir": str(init_out_dir),
		"completed_at": _utc_now_iso(),
		"inputs": {
			"h5_path": str(inputs.h5_path),
			"source_h5_path": str(source_h5_path),
			"stream_id": str(inputs.stream_id),
			"copied_to_scratch": bool(inputs.copied_to_scratch),
			"requires_use_scratch_root": bool(inputs.phases.copy_src_to_scratch.requires_use_scratch_root),
		},
		"outputs": {
			"source_h5_path": str(source_h5_path),
			"resolved_h5_path": str(inputs.h5_path),
		},
	}
	payload.update(
		{
			"source_h5_path": core_payload["source_h5_path"],
			"resolved_h5_path": core_payload["resolved_h5_path"],
			"copied_to_scratch": core_payload["copied_to_scratch"],
			"requires_use_scratch_root": core_payload["requires_use_scratch_root"],
		}
	)
	_write_json(summary_json_path, _json_ready(payload))
	payload["summary_json"] = str(summary_json_path)
	LOGGER.info(
		"init copy_src_to_scratch phase complete stream_id=%s copied_to_scratch=%s summary_json=%s",
		str(inputs.stream_id),
		bool(inputs.copied_to_scratch),
		str(summary_json_path),
	)
	return payload
