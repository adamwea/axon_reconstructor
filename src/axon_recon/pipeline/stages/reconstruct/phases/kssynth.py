"""Reconstruct-stage `kssynth` phase — scaffold (Era 3, slice 1a).

This phase replaces the recon stage's `extract_partial_templates` +
`build_templates` pair with a single call to `kssynth.synthesize(...)`
(see `~/dev/pkgs/kssynth`). Output lands in
`<well>/recon_outputs/synth_sorter_output/` as a KS-shaped folder that
downstream recon phases (`plot_templates_v2`, `report_templates`,
`axon_velocity_gtrs`) consume after the slice-4 repointing.

**Status (2026-05-20)**: SCAFFOLD ONLY. The entry point
`run_reconstruct_kssynth_phase` validates the kssynth import and writes
a stub summary JSON, but does NOT yet load segment analyzers nor call
`kssynth.synthesize`. That wiring lands in slice 1b once the
analyzer-loading contract is settled (see
`plans/active/kssynth_recon_integration_plan.md` §"Open questions").

The phase is NOT yet wired into `stages.reconstruct.phase_sequence` in
either debug YAML. It is unreachable from a production CLI invocation;
only the unit tests below exercise it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..templates.io import write_json
from ..templates.models.inputs import TemplatesInputs

LOGGER = logging.getLogger("axon_recon.reconstruct.kssynth")

KSSYNTH_SUMMARY_RELPATH = "synth_sorter_output/kssynth_summary.json"
KSSYNTH_OUTPUT_RELDIR = "synth_sorter_output"


def _kssynth_summary_payload(
	*,
	status: str,
	well_out_dir: Path,
	templates_out_dir: Path,
	error: str | None = None,
	n_units: int | None = None,
	n_analyzers: int | None = None,
) -> dict[str, Any]:
	payload: dict[str, Any] = {
		"phase": "reconstruct.kssynth",
		"status": status,
		"well_out_dir": str(well_out_dir),
		"templates_out_dir": str(templates_out_dir),
		"synth_sorter_output_relpath": KSSYNTH_OUTPUT_RELDIR,
	}
	if error is not None:
		payload["error"] = error
	if n_units is not None:
		payload["n_units"] = int(n_units)
	if n_analyzers is not None:
		payload["n_analyzers"] = int(n_analyzers)
	return payload


def _resolve_kssynth_output_dirs(inputs: TemplatesInputs) -> tuple[Path, Path]:
	from .build_templates import _resolve_build_templates_context

	context = _resolve_build_templates_context(inputs)
	synth_out_dir = context.templates_out_dir / KSSYNTH_OUTPUT_RELDIR
	synth_out_dir.mkdir(parents=True, exist_ok=True)
	return context.well_out_dir, context.templates_out_dir


def run_reconstruct_kssynth_phase(inputs: TemplatesInputs) -> dict[str, Any]:
	"""Entry point for the recon-stage `kssynth` phase.

	v1a (this commit): validates the `kssynth` import path is live in
	the current process and writes a placeholder summary JSON marking
	the phase as scaffold-only. Returns the summary dict for the
	stage runner to inspect.

	v1b (next slice, per `kssynth_recon_integration_plan.md`): replaces
	this body with a real analyzer-loading + `kssynth.synthesize(...)`
	call. The function signature and return-dict shape are stable from
	v1a forward; v1b only fills in the analyzer-resolve / synthesize /
	WriterResult-to-summary translation.
	"""

	well_out_dir, templates_out_dir = _resolve_kssynth_output_dirs(inputs)

	try:
		import kssynth  # noqa: F401
		from kssynth.api import synthesize  # noqa: F401
	except ImportError as exc:
		LOGGER.error("kssynth import failed: %s", exc)
		summary = _kssynth_summary_payload(
			status="error",
			well_out_dir=well_out_dir,
			templates_out_dir=templates_out_dir,
			error=f"kssynth import failed: {exc}",
		)
		summary_path = templates_out_dir / KSSYNTH_SUMMARY_RELPATH
		write_json(summary_path, summary)
		return summary

	LOGGER.info(
		"reconstruct.kssynth scaffold (slice 1a): kssynth import ok; "
		"synthesize() call deferred to slice 1b. well_out_dir=%s",
		str(well_out_dir),
	)
	summary = _kssynth_summary_payload(
		status="scaffold_only",
		well_out_dir=well_out_dir,
		templates_out_dir=templates_out_dir,
	)
	summary_path = templates_out_dir / KSSYNTH_SUMMARY_RELPATH
	write_json(summary_path, summary)
	return summary
