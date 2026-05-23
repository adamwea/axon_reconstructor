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


def _resolve_kssynth_output_dirs(
	inputs: TemplatesInputs, *, create_dirs: bool = True
) -> tuple[Path, Path, Path]:
	"""Returns (well_out_dir, templates_out_dir, synth_out_dir).

	Wraps `_resolve_build_templates_context` so the kssynth phase lands its
	output beside the same `recon_outputs/` tree the legacy phases write to.

	Set `create_dirs=False` to skip the `synth_out_dir.mkdir()` side-effect
	— the dry-run path uses this so a `--dry-run` smoke doesn't create
	empty directories under the resolved output_root (which, if pointed
	at the reference data tree, would otherwise pollute it).
	"""
	from .build_templates import _resolve_build_templates_context

	context = _resolve_build_templates_context(inputs)
	synth_out_dir = context.templates_out_dir / KSSYNTH_OUTPUT_RELDIR
	if create_dirs:
		synth_out_dir.mkdir(parents=True, exist_ok=True)
	return context.well_out_dir, context.templates_out_dir, synth_out_dir


def _make_segment_analyzer_iter_factory(inputs: TemplatesInputs):
	"""Build a callable that yields segment analyzers ONE AT A TIME.

	Returns a zero-arg callable; each invocation produces a fresh iterator
	over (source_name, analyzer) tuples from
	`_iter_templates_phase_analyzers`. The kssynth sibling's streaming
	`synthesize(analyzer_iter_factory=...)` path calls this twice (pass 1
	= metadata, pass 2 = template extraction) and drops each analyzer
	after use so peak memory stays bounded at ~1 analyzer's footprint.

	Mirrors the legacy `build_templates` / `extract_partial_templates`
	memory pattern (`del analyzer; gc.collect()`) — the kssynth sibling
	does the drop internally.

	**NEVER bootstraps analyzers.** The kssynth phase is strictly a
	CONSUMER of the analyzers produced by `reconstruct.analyzers`. The
	inputs are wrapped via `_inputs_with_analyzer_build_if_missing(
	concat_build_if_missing=False, segments_build_if_missing=False)`
	before the iterator is created. `iter_spikeinterface_analyzers`
	loads cached analyzers only; if none are found the iterator yields
	nothing and `synthesize()` raises a clear error.
	"""
	from .analyzers import _inputs_with_analyzer_build_if_missing
	from .build_templates import _resolve_build_templates_context
	from ..templates.runner import _iter_templates_phase_analyzers

	context = _resolve_build_templates_context(inputs)
	consumer_only_inputs = _inputs_with_analyzer_build_if_missing(
		inputs,
		concat_build_if_missing=False,
		segments_build_if_missing=False,
	)

	def _factory():
		# Each call returns a fresh generator. The kssynth sibling iterates
		# it twice (pass 1 = positions/metadata, pass 2 = template extraction).
		for _src_name, analyzer in _iter_templates_phase_analyzers(
			inputs=consumer_only_inputs,
			well_out_dir=context.well_out_dir,
			alternate_well_out_dirs=list(context.alternate_well_out_dirs),
			analyzer_cache_dir=context.analyzer_cache_dir,
		):
			yield analyzer

	return _factory


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

	Slice 4e — dry-run short-circuit:
	  When the process-wide `--dry-run` override is set, the phase
	  resolves output dirs + checks for analyzer cache existence + writes
	  a `dry_run_ok` summary, then returns without invoking
	  `kssynth.synthesize` or loading analyzers. Lets the caller verify
	  wiring + input resolution in seconds before kicking off the heavy
	  compute (per `guardrails/dry_run.md`).

	Returns the summary dict for the stage runner to inspect.
	"""

	# Resolve output dirs WITHOUT side-effect-creating them yet — the
	# dry-run path doesn't need them created on disk. The non-dry-run
	# path re-resolves below with `create_dirs=True` so `synthesize()`
	# can write into them.
	from axon_recon.pipeline.config import get_dry_run_override

	_is_dry_run = bool(get_dry_run_override())
	well_out_dir, templates_out_dir, synth_out_dir = _resolve_kssynth_output_dirs(
		inputs, create_dirs=not _is_dry_run
	)
	summary_path = templates_out_dir / KSSYNTH_SUMMARY_RELPATH

	# Dry-run short-circuit (slice 4e).
	if _is_dry_run:
		from axon_recon.pipeline.dry_run import write_dry_run_summary

		analyzer_cache_dir: Path | None = None
		try:
			from .build_templates import _resolve_build_templates_context

			context = _resolve_build_templates_context(inputs)
			analyzer_cache_dir = context.analyzer_cache_dir
		except Exception as exc:
			LOGGER.warning("kssynth dry-run: analyzer cache resolution failed: %s", exc)

		inputs_resolved: list[dict[str, Any]] = []
		validation_warnings: list[str] = []
		if analyzer_cache_dir is not None:
			cache_exists = analyzer_cache_dir.exists()
			inputs_resolved.append(
				{
					"name": "analyzer_cache_dir",
					"path": str(analyzer_cache_dir),
					"exists": bool(cache_exists),
				}
			)
			if not cache_exists:
				validation_warnings.append(
					f"analyzer_cache_dir not found at {analyzer_cache_dir}; run "
					"`reconstruct.analyzers` first to populate it."
				)

		outputs_would_produce: list[dict[str, Any]] = [
			{"name": "synth_sorter_output", "path": str(synth_out_dir)},
			{"name": "per_unit_dir", "path": str(synth_out_dir / KSSYNTH_PER_UNIT_RELDIR)},
			{"name": "summary_json", "path": str(summary_path)},
		]
		write_dry_run_summary(
			phase_name="reconstruct.kssynth",
			well_out_dir=well_out_dir,
			stage_output_root_dir=templates_out_dir,
			summary_json_path=summary_path,
			inputs_resolved=inputs_resolved,
			outputs_would_produce=outputs_would_produce,
			validation={
				"missing_prerequisites": [],
				"warnings": validation_warnings,
			},
		)
		LOGGER.info(
			"reconstruct.kssynth: dry-run complete; summary at %s", summary_path
		)
		return {
			"phase": "reconstruct.kssynth",
			"status": "dry_run_ok",
			"well_out_dir": str(well_out_dir),
			"templates_out_dir": str(templates_out_dir),
			"synth_sorter_output_relpath": KSSYNTH_OUTPUT_RELDIR,
		}

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

	# Build a STREAMING analyzer iterator factory — keeps at most one
	# analyzer in memory inside kssynth.synthesize() (matches the legacy
	# build_templates per-source iterate+drop+gc pattern). User feedback:
	# materializing all 19 segment analyzers at once OOMs on production
	# multi-segment runs.
	try:
		analyzer_iter_factory = _make_segment_analyzer_iter_factory(inputs)
	except Exception as exc:
		LOGGER.exception("reconstruct.kssynth: analyzer factory setup failed")
		summary = _kssynth_summary_payload(
			status="error",
			well_out_dir=well_out_dir,
			templates_out_dir=templates_out_dir,
			error=f"analyzer factory setup failed: {exc}",
		)
		write_json(summary_path, summary)
		return summary

	# Wrap the factory in a counter so we can report n_analyzers in
	# error / success summaries without materializing a list.
	# `n_per_pass` resets on each factory call (each pass starts fresh)
	# so the final value is the count from the LAST pass — whether that
	# was pass 1 (failed before pass 2 started) or pass 2 (full success).
	# Either way it equals the unique-analyzer count from one full walk.
	class _AnalyzerCounter:
		n_per_pass: int = 0

	counter = _AnalyzerCounter()

	def _counting_factory():
		counter.n_per_pass = 0
		for analyzer in analyzer_iter_factory():
			counter.n_per_pass += 1
			yield analyzer

	LOGGER.info(
		"reconstruct.kssynth: invoking kssynth.synthesize in STREAMING mode "
		"(one analyzer in memory at a time)"
	)

	try:
		writer_result = synthesize(
			analyzer_iter_factory=_counting_factory,
			out_folder=synth_out_dir,
		)
	except Exception as exc:
		LOGGER.exception("reconstruct.kssynth: kssynth.synthesize failed")
		summary = _kssynth_summary_payload(
			status="error",
			well_out_dir=well_out_dir,
			templates_out_dir=templates_out_dir,
			n_analyzers=int(counter.n_per_pass),
			error=f"kssynth.synthesize failed: {exc}",
		)
		write_json(summary_path, summary)
		return summary
	n_analyzers = int(counter.n_per_pass)

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
			n_analyzers=n_analyzers,
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
		n_analyzers=n_analyzers,
		n_channels=int(getattr(writer_result, "n_channels", 0) or 0),
		channel_grid_mode=getattr(writer_result, "channel_grid_mode", None),
		policy=getattr(writer_result, "policy", None),
		files_written=list(getattr(writer_result, "files_written", ()) or ()),
		per_unit_dir=per_unit_dir,
		per_unit_n_units_written=per_unit_n_units_written,
	)
	write_json(summary_path, summary)
	return summary
