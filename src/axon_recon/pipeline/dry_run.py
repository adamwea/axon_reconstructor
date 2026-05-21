"""Shared `--dry-run` summary writer.

Per `guardrails/dry_run.md` + `plans/active/dry_run_rollout_plan.md`:
every phase that implements the dry-run short-circuit writes a
`<phase>_summary.json` with `status: dry_run_ok` instead of doing the
heavy work, so the caller can verify the phase's wiring + input
resolution in seconds. This module centralizes the schema enforcement
so individual phases don't drift.

Schema (per the guardrail §3):

```json
{
  "status": "dry_run_ok",
  "well_out_dir": "/path/to/well",
  "stage_output_root_dir": "/path/to/stage_output",
  "phase": "<phase_name>",
  "inputs_resolved": [
    {"name": "concat_recording", "path": "...", "exists": true},
    ...
  ],
  "outputs_would_produce": [
    {"name": "merged_template", "path": "..."},
    ...
  ],
  "validation": {
    "missing_prerequisites": [],
    "warnings": []
  }
}
```

Phase-specific extensions are allowed (e.g. spike count estimates,
segment counts) as long as the base fields stay present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DRY_RUN_STATUS = "dry_run_ok"


def write_dry_run_summary(
	*,
	phase_name: str,
	well_out_dir: Path,
	stage_output_root_dir: Path,
	summary_json_path: Path,
	inputs_resolved: list[dict[str, Any]],
	outputs_would_produce: list[dict[str, Any]],
	validation: dict[str, Any] | None = None,
	extra_fields: dict[str, Any] | None = None,
) -> Path:
	"""Write the standard dry-run summary JSON.

	Used by every phase's dry-run short-circuit (see
	`guardrails/dry_run.md` §3 for the contract). Returns the
	`summary_json_path` so callers can use it in their own return value.

	Args:
	  phase_name: Phase identifier (e.g. ``"reconstruct.kssynth"``).
	  well_out_dir: Per-target well output directory the phase WOULD
	    write under in a real run.
	  stage_output_root_dir: The stage's output root (e.g.
	    ``recon_outputs/`` for the recon stage).
	  summary_json_path: Where the summary lands. Same path as the real
	    summary_json would, so `axon-recon status` reads it cleanly.
	  inputs_resolved: List of `{name, path, exists}` dicts — one per
	    input the phase would consume. ``exists`` lets the caller know
	    which prerequisites were present.
	  outputs_would_produce: List of `{name, path}` dicts — one per
	    output the phase would write in a real run.
	  validation: Optional `{missing_prerequisites: [...], warnings: [...]}`.
	    Defaults to empty lists. If `missing_prerequisites` is non-empty,
	    the caller should usually raise / return a non-OK status; this
	    helper does not enforce that (see sub-rule 4 in the guardrail).
	  extra_fields: Optional phase-specific extension fields. Merged
	    into the payload AFTER the base fields so the base shape always
	    wins on conflict.

	Returns:
	  The `summary_json_path` (same as the input arg).
	"""

	if validation is None:
		validation = {"missing_prerequisites": [], "warnings": []}
	else:
		# Defensive: ensure both keys exist with list values.
		validation = {
			"missing_prerequisites": list(validation.get("missing_prerequisites", [])),
			"warnings": list(validation.get("warnings", [])),
		}

	payload: dict[str, Any] = {
		"status": DRY_RUN_STATUS,
		"well_out_dir": str(well_out_dir),
		"stage_output_root_dir": str(stage_output_root_dir),
		"phase": str(phase_name),
		"inputs_resolved": [
			{
				"name": str(item.get("name", "")),
				"path": str(item.get("path", "")),
				"exists": bool(item.get("exists", False)),
			}
			for item in (inputs_resolved or [])
		],
		"outputs_would_produce": [
			{
				"name": str(item.get("name", "")),
				"path": str(item.get("path", "")),
			}
			for item in (outputs_would_produce or [])
		],
		"validation": validation,
	}
	if extra_fields:
		for key, value in extra_fields.items():
			if key in payload:
				# Don't let extras shadow the base schema fields.
				continue
			payload[key] = value

	summary_json_path = Path(summary_json_path)
	summary_json_path.parent.mkdir(parents=True, exist_ok=True)
	with open(summary_json_path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)
	return summary_json_path
