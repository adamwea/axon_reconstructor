"""Reconstruct-stage `kssynth` phase (Era 3, slices 1a + 1b).

This phase replaces the recon stage's `extract_partial_templates` +
`build_templates` pair with a single call to `kssynth.synthesize(...)`
(see `~/dev/pkgs/kssynth`). Output lands in
`<well>/recon_outputs/synth_sorter_output/` as a KS-shaped folder that
downstream recon phases (`plot_templates_v2`, `report_templates`,
`axon_velocity_gtrs`) consume after the slice-4 repointing.

**Status (2026-05-20)**: slice 1b — real `kssynth.synthesize(...)` call.
Resolves segment analyzers via
`templates.runner._load_templates_phase_analyzers`, extracts the
analyzer instances from the `(source_name, analyzer)` tuples it
returns, and forwards them to `kssynth.synthesize(...)`. Output dir +
summary JSON contract carried forward from slice 1a.

The phase is NOT yet wired into `stages.reconstruct.phase_sequence` in
either debug YAML. Unit tests exercise the wiring with monkey-patched
analyzer loader + synthesize; real-data smoke is slice 3.
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
KSSYNTH_PER_UNIT_RELDIR = "per_unit"  # under synth_sorter_output/


def _kssynth_summary_payload(
	*,
	status: str,
	well_out_dir: Path,
	templates_out_dir: Path,
	error: str | None = None,
	n_units: int | None = None,
	n_analyzers: int | None = None,
	n_channels: int | None = None,
	channel_grid_mode: str | None = None,
	policy: str | None = None,
	files_written: list[str] | None = None,
	per_unit_dir: Path | None = None,
	per_unit_n_units_written: int | None = None,
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
	if n_channels is not None:
		payload["n_channels"] = int(n_channels)
	if channel_grid_mode is not None:
		payload["channel_grid_mode"] = str(channel_grid_mode)
	if policy is not None:
		payload["policy"] = str(policy)
	if files_written is not None:
		payload["files_written"] = list(files_written)
	if per_unit_dir is not None:
		payload["per_unit_dir"] = str(per_unit_dir)
	if per_unit_n_units_written is not None:
		payload["per_unit_n_units_written"] = int(per_unit_n_units_written)
	return payload


def _write_per_unit_templates_from_synth_output(
	*,
	synth_out_dir: Path,
	unit_ids: tuple[int, ...] | list[int],
) -> tuple[Path, int]:
	"""Postprocess kssynth output into per-unit merged_template files.

	Era 3 slice 4 (S4-B): write per-unit `merged_template.npy` +
	`merged_channel_locations.npy` files matching the `build_templates`
	layout so downstream phases (`plot_templates_v2`, `report_templates`,
	`axon_velocity_gtrs`) can consume kssynth's output without bespoke
	logic.

	Layout written:
	  ``synth_sorter_output/per_unit/unit_<id>/merged_template.npy`` —
	  shape ``(n_active_channels, n_samples)`` where n_active_channels is
	  the count of channels with any non-zero sample for this unit
	  (sparsification matches build_templates' "only non-zero contribu-
	  ting channels" semantic).
	  ``synth_sorter_output/per_unit/unit_<id>/merged_channel_locations.npy``
	  — shape ``(n_active_channels, 2)`` xy coords of those channels.

	Inputs read from `synth_out_dir`:
	  ``templates.npy`` — shape ``(n_units, n_samples, n_channels)``,
	  produced by `kssynth.synthesize`.
	  ``channel_positions.npy`` — shape ``(n_channels, 2)``.

	Returns ``(per_unit_dir, n_units_written)``.
	"""

	import numpy as np

	templates_npy = synth_out_dir / "templates.npy"
	positions_npy = synth_out_dir / "channel_positions.npy"
	if not templates_npy.exists() or not positions_npy.exists():
		raise FileNotFoundError(
			f"kssynth per-unit postprocess: missing {templates_npy} or {positions_npy}"
		)

	templates = np.load(templates_npy)  # (n_units, n_samples, n_channels)
	positions = np.load(positions_npy)  # (n_channels, 2)
	if templates.ndim != 3:
		raise ValueError(
			f"kssynth per-unit postprocess: expected templates.npy to be 3-D "
			f"(n_units, n_samples, n_channels), got shape {templates.shape}"
		)
	if positions.ndim != 2 or positions.shape[1] != 2:
		raise ValueError(
			f"kssynth per-unit postprocess: expected channel_positions.npy to be "
			f"(n_channels, 2), got shape {positions.shape}"
		)

	per_unit_dir = synth_out_dir / KSSYNTH_PER_UNIT_RELDIR
	per_unit_dir.mkdir(parents=True, exist_ok=True)

	unit_ids_list = list(unit_ids or [])
	if len(unit_ids_list) != templates.shape[0]:
		raise ValueError(
			f"kssynth per-unit postprocess: unit_ids length ({len(unit_ids_list)}) "
			f"does not match templates.shape[0] ({templates.shape[0]})"
		)

	n_written = 0
	for i, unit_id in enumerate(unit_ids_list):
		# templates[i] is (n_samples, n_channels); transpose to (n_channels, n_samples).
		unit_template_full = templates[i].T
		# Sparsify: keep only channels with any non-zero sample (mirrors
		# build_templates' sparse layout).
		nonzero_mask = np.any(unit_template_full != 0, axis=1)
		if not bool(nonzero_mask.any()):
			# No active channels for this unit — write empty arrays.
			sparse_template = unit_template_full[:0]
			sparse_positions = positions[:0]
		else:
			sparse_template = unit_template_full[nonzero_mask]
			sparse_positions = positions[nonzero_mask]
		unit_dir = per_unit_dir / f"unit_{int(unit_id)}"
		unit_dir.mkdir(parents=True, exist_ok=True)
		np.save(unit_dir / "merged_template.npy", sparse_template)
		np.save(unit_dir / "merged_channel_locations.npy", sparse_positions)
		n_written += 1

	return per_unit_dir, n_written


def _resolve_kssynth_output_dirs(inputs: TemplatesInputs) -> tuple[Path, Path, Path]:
	"""Returns (well_out_dir, templates_out_dir, synth_out_dir).

	Wraps `_resolve_build_templates_context` so the kssynth phase lands its
	output beside the same `recon_outputs/` tree the legacy phases write to.
	"""
	from .build_templates import _resolve_build_templates_context

	context = _resolve_build_templates_context(inputs)
	synth_out_dir = context.templates_out_dir / KSSYNTH_OUTPUT_RELDIR
	synth_out_dir.mkdir(parents=True, exist_ok=True)
	return context.well_out_dir, context.templates_out_dir, synth_out_dir


def _load_segment_analyzers(inputs: TemplatesInputs) -> list[Any]:
	"""Loads segment analyzers for the kssynth phase.

	Reuses `templates.runner._load_templates_phase_analyzers` which is the
	same loader `build_templates` uses; ensures kssynth sees the same
	analyzer set the retired phases consumed. Returns the raw analyzer
	instances (drops the `(source_name, analyzer)` tuples).

	`include_concat` is force-disabled inside `_load_templates_phase_analyzers`
	itself per the recon-stage retirement of the concat codepath
	(see commit history around `legacy_include_concat`).
	"""
	from .build_templates import _resolve_build_templates_context
	from ..templates.runner import _load_templates_phase_analyzers

	context = _resolve_build_templates_context(inputs)
	analyzer_pairs = _load_templates_phase_analyzers(
		inputs=inputs,
		well_out_dir=context.well_out_dir,
		alternate_well_out_dirs=list(context.alternate_well_out_dirs),
		analyzer_cache_dir=context.analyzer_cache_dir,
	)
	# _load_templates_phase_analyzers returns list[tuple[source_name, analyzer]].
	# kssynth.synthesize takes list[Any] — just the analyzer instances.
	return [analyzer for _, analyzer in analyzer_pairs]


def run_reconstruct_kssynth_phase(inputs: TemplatesInputs) -> dict[str, Any]:
	"""Entry point for the recon-stage `kssynth` phase.

	Slice 1b implementation:
	  1. Validate the `kssynth.synthesize` import path.
	  2. Resolve `(well_out_dir, templates_out_dir, synth_out_dir)` using
	     the same `_resolve_build_templates_context` the legacy phases used.
	  3. Load segment analyzers via `_load_templates_phase_analyzers`.
	  4. Call `kssynth.synthesize(analyzers=..., out_folder=synth_out_dir)`
	     with default options (channel_grid="union",
	     aggregation="spike_count_weighted_mean").
	  5. Translate the `WriterResult` to a `kssynth_summary.json` payload
	     and persist it.

	Returns the summary dict for the stage runner to inspect.
	"""

	well_out_dir, templates_out_dir, synth_out_dir = _resolve_kssynth_output_dirs(inputs)
	summary_path = templates_out_dir / KSSYNTH_SUMMARY_RELPATH

	try:
		from kssynth.api import synthesize
	except ImportError as exc:
		LOGGER.error("kssynth import failed: %s", exc)
		summary = _kssynth_summary_payload(
			status="error",
			well_out_dir=well_out_dir,
			templates_out_dir=templates_out_dir,
			error=f"kssynth import failed: {exc}",
		)
		write_json(summary_path, summary)
		return summary

	try:
		analyzers = _load_segment_analyzers(inputs)
	except Exception as exc:
		LOGGER.exception("reconstruct.kssynth: analyzer load failed")
		summary = _kssynth_summary_payload(
			status="error",
			well_out_dir=well_out_dir,
			templates_out_dir=templates_out_dir,
			error=f"analyzer load failed: {exc}",
		)
		write_json(summary_path, summary)
		return summary

	LOGGER.info(
		"reconstruct.kssynth: loaded %d segment analyzers, calling kssynth.synthesize",
		len(analyzers),
	)

	try:
		writer_result = synthesize(
			analyzers=analyzers,
			out_folder=synth_out_dir,
		)
	except Exception as exc:
		LOGGER.exception("reconstruct.kssynth: kssynth.synthesize failed")
		summary = _kssynth_summary_payload(
			status="error",
			well_out_dir=well_out_dir,
			templates_out_dir=templates_out_dir,
			n_analyzers=len(analyzers),
			error=f"kssynth.synthesize failed: {exc}",
		)
		write_json(summary_path, summary)
		return summary

	# Slice 4 S4-B: postprocess kssynth's KS-shaped output into per-unit
	# `merged_template.npy` + `merged_channel_locations.npy` files matching
	# the build_templates layout. Downstream phases (`plot_templates_v2`,
	# `report_templates`, `axon_velocity_gtrs`) can then consume kssynth's
	# output via the existing template loaders.
	unit_ids = tuple(getattr(writer_result, "unit_ids", ()) or ())
	per_unit_dir: Path | None = None
	per_unit_n_units_written: int | None = None
	try:
		per_unit_dir, per_unit_n_units_written = _write_per_unit_templates_from_synth_output(
			synth_out_dir=synth_out_dir,
			unit_ids=unit_ids,
		)
	except Exception as exc:
		LOGGER.exception("reconstruct.kssynth: per-unit postprocess failed")
		# Treat per-unit postprocess as non-fatal — the synthesize step
		# already succeeded, so the KS-shaped sorter_output is on disk.
		# Downstream consumers can fall back to reading the dense templates
		# directly if they handle that shape.
		summary = _kssynth_summary_payload(
			status="error",
			well_out_dir=well_out_dir,
			templates_out_dir=templates_out_dir,
			n_units=len(unit_ids),
			n_analyzers=len(analyzers),
			n_channels=int(getattr(writer_result, "n_channels", 0) or 0),
			channel_grid_mode=getattr(writer_result, "channel_grid_mode", None),
			policy=getattr(writer_result, "policy", None),
			files_written=list(getattr(writer_result, "files_written", ()) or ()),
			error=f"per-unit postprocess failed: {exc}",
		)
		write_json(summary_path, summary)
		return summary

	summary = _kssynth_summary_payload(
		status="ok",
		well_out_dir=well_out_dir,
		templates_out_dir=templates_out_dir,
		n_units=len(unit_ids),
		n_analyzers=len(analyzers),
		n_channels=int(getattr(writer_result, "n_channels", 0) or 0),
		channel_grid_mode=getattr(writer_result, "channel_grid_mode", None),
		policy=getattr(writer_result, "policy", None),
		files_written=list(getattr(writer_result, "files_written", ()) or ()),
		per_unit_dir=per_unit_dir,
		per_unit_n_units_written=per_unit_n_units_written,
	)
	write_json(summary_path, summary)
	return summary
