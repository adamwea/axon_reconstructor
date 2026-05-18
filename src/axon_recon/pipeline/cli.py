from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Callable

# Per-rank CUDA partitioning MUST run before any module imports torch/cupy/kilosort
# so the visibility change is observed at CUDA init. Keep this import + call at
# the very top of axon_recon-side imports.
from .mpi_adapter import apply_per_rank_cuda_visible_devices, current_mpi_context

apply_per_rank_cuda_visible_devices()

from .cpu_allocation import detect_cpu_topology, format_cpu_topology
from .execution import install_process_lifecycle
from .execution.logging_context import (
	ensure_pipeline_target_in_format,
	install_pipeline_log_record_factory,
)
from .logging import (
	configure_pipeline_logging,
	finalize_pipeline_logging,
	install_noisy_external_log_filters,
	log_context,
)
from .runner import build_stage_allocation_previews, format_stage_allocation_previews, print_stage_allocation_preview
from .shared.maxwell_plugin import install_maxwell_hdf5_plugin_message_filter
from .stages.preprocess.cli import (
	_run_concat_segments_from_args as _run_preprocess_concat_segments_from_args,
)
from .stages.preprocess.cli import (
	_run_copy_src_to_scratch_from_args as _run_preprocess_copy_src_to_scratch_from_args,
)
from .stages.preprocess.cli import _run_from_args as _run_preprocess_from_args
from .stages.preprocess.cli import (
	_run_plot_concat_channel_layout_from_args as _run_preprocess_plot_concat_channel_layout_from_args,
)
from .stages.preprocess.cli import (
	_run_plot_concat_traces_from_args as _run_preprocess_plot_concat_traces_from_args,
)
from .stages.preprocess.cli import (
	_run_plot_raster_threshold_from_args as _run_preprocess_plot_raster_threshold_from_args,
)
from .stages.preprocess.cli import (
	_run_plot_segment_channel_layouts_from_args as _run_preprocess_plot_segment_channel_layouts_from_args,
)
from .stages.preprocess.cli import (
	_run_plot_segment_traces_from_args as _run_preprocess_plot_segment_traces_from_args,
)
from .stages.preprocess.cli import (
	_run_prepare_raw_binaries_from_args as _run_preprocess_prepare_raw_binaries_from_args,
)
from .stages.preprocess.cli import (
	_run_preprocess_segments_from_args as _run_preprocess_preprocess_segments_from_args,
)
from .stages.preprocess.cli import (
	_run_save_rec_metadata_from_args as _run_preprocess_save_rec_metadata_from_args,
)
from .stages.preprocess.cli import (
	_run_wipe_src_scratch_from_args as _run_preprocess_wipe_src_scratch_from_args,
)
from .stages.reconstruct.cli import (
	_run_clear_templates_cache_from_args as _run_reconstruct_clear_templates_cache_from_args,
)
from .stages.reconstruct.cli import _run_from_args as _run_reconstruct_from_args
from .stages.reconstruct.cli import (
	_run_generate_gtrs_from_args as _run_reconstruct_generate_gtrs_from_args,
)
from .stages.reconstruct.cli import (
	_run_plot_branch_propagations_from_args as _run_reconstruct_plot_branch_propagations_from_args,
)
from .stages.reconstruct.cli import (
	_run_plot_branch_velocities_from_args as _run_reconstruct_plot_branch_velocities_from_args,
)
from .stages.reconstruct.cli import (
	_run_plot_recons_from_args as _run_reconstruct_plot_recons_from_args,
)
from .stages.reconstruct.cli import (
	_run_plot_unit_summary_from_args as _run_reconstruct_plot_unit_summary_from_args,
)
from .stages.reconstruct.cli import (
	_run_reconstruct_analyzers_from_args,
	_run_reconstruct_build_templates_from_args,
	_run_reconstruct_compute_template_similarity_from_args,
	_run_reconstruct_extract_partial_templates_from_args,
	_run_reconstruct_plot_templates_from_args,
	_run_reconstruct_plot_templates_v2_from_args,
	_run_reconstruct_report_templates_from_args,
	_run_reconstruct_reports_from_args,
	_run_reconstruct_resolve_sources_from_args,
)
from .stages.reconstruct.cli import (
	_run_report_full_chip_layout_from_args as _run_reconstruct_report_full_chip_layout_from_args,
)
from .stages.reconstruct.cli import (
	_run_report_recon_grid_from_args as _run_reconstruct_report_recon_grid_from_args,
)
from .stages.reconstruct.cli import (
	_run_report_recons_from_args as _run_reconstruct_report_recons_from_args,
)
from .stages.reconstruct.cli import (
	_run_report_summaries_from_args as _run_reconstruct_report_summaries_from_args,
)
from .stages.spikesort.cli import _run_bombcell_from_args as _run_spikesort_bombcell_from_args
from .stages.spikesort.cli import (
	_run_bombcell_pass2_from_args as _run_spikesort_bombcell_pass2_from_args,
)
from .stages.spikesort.cli import (
	_run_bootstrap_concat_binary_from_args as _run_spikesort_bootstrap_concat_binary_from_args,
)
from .stages.spikesort.cli import (
	_run_cleanup_analyzers_from_args as _run_spikesort_cleanup_analyzers_from_args,
)
from .stages.spikesort.cli import (
	_run_cleanup_concat_binary_from_args as _run_spikesort_cleanup_concat_binary_from_args,
)
from .stages.spikesort.cli import _run_from_args as _run_spikesort_from_args
from .stages.spikesort.cli import _run_merge_from_args as _run_spikesort_merge_from_args
from .stages.spikesort.cli import _run_merge_slay_from_args as _run_spikesort_merge_slay_from_args
from .stages.spikesort.cli import (
	_run_concat_analyzer_from_args as _run_spikesort_concat_analyzer_from_args,
)
from .stages.spikesort.cli import (
	_run_restore_sorter_output_from_args as _run_spikesort_restore_sorter_output_from_args,
)
from .stages.spikesort.cli import (
	_run_snapshot_sorter_output_from_args as _run_spikesort_snapshot_sorter_output_from_args,
)
from .stages.spikesort.cli import (
	_run_summarize_sort_from_args as _run_spikesort_summarize_sort_from_args,
)
from .stages.spikesort.orchestrators import _run_sort_from_args as _run_spikesort_sort_from_args
from .stages.analysis.cli import (
	_run_compute_metrics_from_args as _run_analysis_compute_metrics_from_args,
)
from .stages.analysis.cli import _run_from_args as _run_analysis_from_args

StageHandler = Callable[[argparse.Namespace], int]

_CANONICAL_STAGE_ORDER: list[str] = [
	"preprocess",
	"spikesort",
	"reconstruct",
	"analysis",
]

_STAGE_ALIASES: dict[str, str] = {
	"pre": "preprocess",
	"prep": "preprocess",
	"preproc": "preprocess",
	"pre.copy_src_to_scratch": "preprocess.copy_src_to_scratch",
	"pre.save_rec_metadata": "preprocess.save_rec_metadata",
	"pre.prepare_raw_binaries": "preprocess.prepare_raw_binaries",
	"pre.wipe_src_scratch": "preprocess.wipe_src_scratch",
	"pre.preprocess_segments": "preprocess.preprocess_segments",
	"pre.plot_segment_traces": "preprocess.plot_segment_traces",
	"pre.plot_segment_channel_layouts": "preprocess.plot_segment_channel_layouts",
	"pre.concat_segments": "preprocess.concat_segments",
	"pre.plot_concat_traces": "preprocess.plot_concat_traces",
	"pre.plot_concat_channel_layout": "preprocess.plot_concat_channel_layout",
	"pre.plot_raster_threshold": "preprocess.plot_raster_threshold",
	"preproc.copy_src_to_scratch": "preprocess.copy_src_to_scratch",
	"preproc.save_rec_metadata": "preprocess.save_rec_metadata",
	"preproc.prepare_raw_binaries": "preprocess.prepare_raw_binaries",
	"preproc.wipe_src_scratch": "preprocess.wipe_src_scratch",
	"preproc.preprocess_segments": "preprocess.preprocess_segments",
	"preproc.plot_segment_traces": "preprocess.plot_segment_traces",
	"preproc.plot_segment_channel_layouts": "preprocess.plot_segment_channel_layouts",
	"preproc.concat_segments": "preprocess.concat_segments",
	"preproc.plot_concat_traces": "preprocess.plot_concat_traces",
	"preproc.plot_concat_channel_layout": "preprocess.plot_concat_channel_layout",
	"preproc.plot_raster_threshold": "preprocess.plot_raster_threshold",
	"preprocess.plot_segment_channel_layouts": "preprocess.plot_segment_channel_layouts",
	"preprocess.prepare_raw_binaries": "preprocess.prepare_raw_binaries",
	"preprocess.plot_segment_traces": "preprocess.plot_segment_traces",
	"preprocess.concat_segments": "preprocess.concat_segments",
	"preprocess.plot_concat_traces": "preprocess.plot_concat_traces",
	"preprocess.plot_concat_channel_layout": "preprocess.plot_concat_channel_layout",
	"preprocess.plot_raster_threshold": "preprocess.plot_raster_threshold",
	"sort": "spikesort",
	"bootstrap_concat_binary": "spikesort.bootstrap_concat_binary",
	"cleanup_concat_binary": "spikesort.cleanup_concat_binary",
	"clear_concat_binary": "spikesort.cleanup_concat_binary",
	"spikesort.clear_concat_binary": "spikesort.cleanup_concat_binary",
	"cleanup_analyzers": "spikesort.cleanup_analyzers",
	"cleanup_analyzer": "spikesort.cleanup_analyzers",
	"clear_analyzers": "spikesort.cleanup_analyzers",
	"clear_analyzer": "spikesort.cleanup_analyzers",
	"spikesort.cleanup_analyzer": "spikesort.cleanup_analyzers",
	"spikesort.clear_analyzers": "spikesort.cleanup_analyzers",
	"spikesort.clear_analyzer": "spikesort.cleanup_analyzers",
	"spikesort.cleanup_concat_analyzer": "spikesort.cleanup_analyzers",
	"bombcell": "spikesort.bombcell_label",
	"spikesort.bombcell": "spikesort.bombcell_label",
	"bombcell_pass2": "spikesort.bombcell_label_pass2",
	"bombcell_label_pass2": "spikesort.bombcell_label_pass2",
	"bombcell_label_post_merge": "spikesort.bombcell_label_pass2",
	"bombcell_post_merge": "spikesort.bombcell_label_pass2",
	"spikesort.bombcell_pass2": "spikesort.bombcell_label_pass2",
	"spikesort.bombcell_label_post_merge": "spikesort.bombcell_label_pass2",
	"spikesort.bombcell_post_merge": "spikesort.bombcell_label_pass2",
	"merge": "spikesort.merge",
	"merge_SLAy": "spikesort.merge_SLAy",
	"spikesort.summary": "spikesort.summarize_sort",
	"spikesort.snapshot": "spikesort.snapshot_sorter_output",
	"spikesort.restore": "spikesort.restore_sorter_output",
	"spikesort.analyzer": "spikesort.concat_analyzer",
	"spikesort.build_concat_analyzer": "spikesort.concat_analyzer",
	"spikesort.merge_units": "spikesort.merge",
	"spikesort.merge.slay": "spikesort.merge_SLAy",
	"spikesort.merge_SLAy": "spikesort.merge_SLAy",
	"spikesort.merge_slay": "spikesort.merge_SLAy",
	"spikesort.merge_units.slay": "spikesort.merge_SLAy",
	"spike": "spikesort",
	"spikesorting": "spikesort",
	"recon": "reconstruct",
	"reconstruction": "reconstruct",
	"recon.resolve_sources": "reconstruct.resolve_sources",
	"recon.analyzers": "reconstruct.analyzers",
	"recon.extract_partial_templates": "reconstruct.extract_partial_templates",
	"recon.build_templates": "reconstruct.build_templates",
	"recon.compute_template_similarity": "reconstruct.compute_template_similarity",
	"recon.plot_templates": "reconstruct.plot_templates",
	"recon.plot_templates_v2": "reconstruct.plot_templates_v2",
	"recon.report_templates": "reconstruct.report_templates",
	"recon.reports": "reconstruct.reports",
	"recon.templates_resolve_sources": "reconstruct.resolve_sources",
	"recon.templates_analyzers": "reconstruct.analyzers",
	"recon.templates_extract_partial_templates": "reconstruct.extract_partial_templates",
	"recon.templates_build_templates": "reconstruct.build_templates",
	"recon.templates_compute_template_similarity": "reconstruct.compute_template_similarity",
	"recon.templates_plot_templates": "reconstruct.plot_templates",
	"recon.templates_plot_templates_v2": "reconstruct.plot_templates_v2",
	"recon.templates_report_templates": "reconstruct.report_templates",
	"recon.templates_reports": "reconstruct.reports",
	"recon.generate_gtrs": "reconstruct.generate_gtrs",
	"recon.plot_recons": "reconstruct.plot_recons",
	"recon.plot_branch_propagations": "reconstruct.plot_branch_propagations",
	"recon.plot_branch_velocities": "reconstruct.plot_branch_velocities",
	"recon.plot_unit_summary": "reconstruct.plot_unit_summary",
	"recon.report_recons": "reconstruct.report_recons",
	"recon.report_recon_grid": "reconstruct.report_recon_grid",
	"recon.report_full_chip_layout": "reconstruct.report_full_chip_layout",
	"recon.report_summaries": "reconstruct.report_summaries",
	"recon.clear_templates_cache": "reconstruct.clear_templates_cache",
	"reconstruction.resolve_sources": "reconstruct.resolve_sources",
	"reconstruction.analyzers": "reconstruct.analyzers",
	"reconstruction.extract_partial_templates": "reconstruct.extract_partial_templates",
	"reconstruction.build_templates": "reconstruct.build_templates",
	"reconstruction.compute_template_similarity": "reconstruct.compute_template_similarity",
	"reconstruction.plot_templates": "reconstruct.plot_templates",
	"reconstruction.plot_templates_v2": "reconstruct.plot_templates_v2",
	"reconstruction.report_templates": "reconstruct.report_templates",
	"reconstruction.reports": "reconstruct.reports",
	"reconstruction.templates_resolve_sources": "reconstruct.resolve_sources",
	"reconstruction.templates_analyzers": "reconstruct.analyzers",
	"reconstruction.templates_extract_partial_templates": "reconstruct.extract_partial_templates",
	"reconstruction.templates_build_templates": "reconstruct.build_templates",
	"reconstruction.templates_compute_template_similarity": "reconstruct.compute_template_similarity",
	"reconstruction.templates_plot_templates": "reconstruct.plot_templates",
	"reconstruction.templates_plot_templates_v2": "reconstruct.plot_templates_v2",
	"reconstruction.templates_report_templates": "reconstruct.report_templates",
	"reconstruction.templates_reports": "reconstruct.reports",
	"reconstruct.templates_resolve_sources": "reconstruct.resolve_sources",
	"reconstruct.templates_analyzers": "reconstruct.analyzers",
	"reconstruct.templates_extract_partial_templates": "reconstruct.extract_partial_templates",
	"reconstruct.templates_build_templates": "reconstruct.build_templates",
	"reconstruct.templates_compute_template_similarity": "reconstruct.compute_template_similarity",
	"reconstruct.templates_plot_templates": "reconstruct.plot_templates",
	"reconstruct.templates_plot_templates_v2": "reconstruct.plot_templates_v2",
	"reconstruct.templates_report_templates": "reconstruct.report_templates",
	"reconstruct.templates_reports": "reconstruct.reports",
	"reconstruction.generate_gtrs": "reconstruct.generate_gtrs",
	"reconstruction.plot_recons": "reconstruct.plot_recons",
	"reconstruction.plot_branch_propagations": "reconstruct.plot_branch_propagations",
	"reconstruction.plot_branch_velocities": "reconstruct.plot_branch_velocities",
	"reconstruction.plot_unit_summary": "reconstruct.plot_unit_summary",
	"reconstruction.report_recons": "reconstruct.report_recons",
	"reconstruction.report_recon_grid": "reconstruct.report_recon_grid",
	"reconstruction.report_full_chip_layout": "reconstruct.report_full_chip_layout",
	"reconstruction.report_summaries": "reconstruct.report_summaries",
	"reconstruction.clear_templates_cache": "reconstruct.clear_templates_cache",
	"metrics": "analysis.compute_metrics",
	"compute_metrics": "analysis.compute_metrics",
	"analysis.metrics": "analysis.compute_metrics",
	"analysis.compute": "analysis.compute_metrics",
}

_STAGE_HANDLERS: dict[str, StageHandler] = {
	"preprocess": _run_preprocess_from_args,
	"preprocess.copy_src_to_scratch": _run_preprocess_copy_src_to_scratch_from_args,
	"preprocess.save_rec_metadata": _run_preprocess_save_rec_metadata_from_args,
	"preprocess.prepare_raw_binaries": _run_preprocess_prepare_raw_binaries_from_args,
	"preprocess.wipe_src_scratch": _run_preprocess_wipe_src_scratch_from_args,
	"preprocess.preprocess_segments": _run_preprocess_preprocess_segments_from_args,
	"preprocess.plot_segment_traces": _run_preprocess_plot_segment_traces_from_args,
	"preprocess.plot_segment_channel_layouts": _run_preprocess_plot_segment_channel_layouts_from_args,
	"preprocess.concat_segments": _run_preprocess_concat_segments_from_args,
	"preprocess.plot_concat_traces": _run_preprocess_plot_concat_traces_from_args,
	"preprocess.plot_concat_channel_layout": _run_preprocess_plot_concat_channel_layout_from_args,
	"preprocess.plot_raster_threshold": _run_preprocess_plot_raster_threshold_from_args,
	"spikesort": _run_spikesort_from_args,
	"spikesort.bootstrap_concat_binary": _run_spikesort_bootstrap_concat_binary_from_args,
	"spikesort.cleanup_concat_binary": _run_spikesort_cleanup_concat_binary_from_args,
	"spikesort.cleanup_analyzers": _run_spikesort_cleanup_analyzers_from_args,
	"spikesort.sort": _run_spikesort_sort_from_args,
	"spikesort.bombcell_label": _run_spikesort_bombcell_from_args,
	"spikesort.bombcell_label_pass2": _run_spikesort_bombcell_pass2_from_args,
	"spikesort.summarize_sort": _run_spikesort_summarize_sort_from_args,
	"spikesort.snapshot_sorter_output": _run_spikesort_snapshot_sorter_output_from_args,
	"spikesort.restore_sorter_output": _run_spikesort_restore_sorter_output_from_args,
	"spikesort.concat_analyzer": _run_spikesort_concat_analyzer_from_args,
	"spikesort.merge": _run_spikesort_merge_from_args,
	"spikesort.merge_SLAy": _run_spikesort_merge_slay_from_args,
	"reconstruct": _run_reconstruct_from_args,
	"reconstruct.resolve_sources": _run_reconstruct_resolve_sources_from_args,
	"reconstruct.analyzers": _run_reconstruct_analyzers_from_args,
	"reconstruct.extract_partial_templates": _run_reconstruct_extract_partial_templates_from_args,
	"reconstruct.build_templates": _run_reconstruct_build_templates_from_args,
	"reconstruct.compute_template_similarity": _run_reconstruct_compute_template_similarity_from_args,
	"reconstruct.plot_templates": _run_reconstruct_plot_templates_from_args,
	"reconstruct.plot_templates_v2": _run_reconstruct_plot_templates_v2_from_args,
	"reconstruct.report_templates": _run_reconstruct_report_templates_from_args,
	"reconstruct.reports": _run_reconstruct_reports_from_args,
	"reconstruct.generate_gtrs": _run_reconstruct_generate_gtrs_from_args,
	"reconstruct.plot_recons": _run_reconstruct_plot_recons_from_args,
	"reconstruct.plot_branch_propagations": _run_reconstruct_plot_branch_propagations_from_args,
	"reconstruct.plot_branch_velocities": _run_reconstruct_plot_branch_velocities_from_args,
	"reconstruct.plot_unit_summary": _run_reconstruct_plot_unit_summary_from_args,
	"reconstruct.report_recons": _run_reconstruct_report_recons_from_args,
	"reconstruct.report_recon_grid": _run_reconstruct_report_recon_grid_from_args,
	"reconstruct.report_full_chip_layout": _run_reconstruct_report_full_chip_layout_from_args,
	"reconstruct.report_summaries": _run_reconstruct_report_summaries_from_args,
	"reconstruct.clear_templates_cache": _run_reconstruct_clear_templates_cache_from_args,
	"analysis": _run_analysis_from_args,
	"analysis.compute_metrics": _run_analysis_compute_metrics_from_args,
}


def _parse_unit_ids_csv(raw: str) -> list[int]:
	tokens = [token.strip() for token in str(raw).split(",")]
	parsed: list[int] = []
	seen: set[int] = set()
	for token in tokens:
		if not token:
			continue
		try:
			value = int(token)
		except Exception as exc:
			raise argparse.ArgumentTypeError(f"Invalid unit id '{token}'") from exc
		if value < 0:
			raise argparse.ArgumentTypeError(f"Unit id must be >= 0, got {value}")
		if value in seen:
			continue
		seen.add(value)
		parsed.append(value)
	if not parsed:
		raise argparse.ArgumentTypeError("Expected at least one unit id")
	return parsed


def _parse_int_or_auto(raw: str) -> int | str:
	stripped = str(raw).strip().lower()
	if stripped == "auto":
		return "auto"
	try:
		value = int(stripped)
	except Exception as exc:
		raise argparse.ArgumentTypeError(f"Expected a positive integer or 'auto', got {raw!r}") from exc
	if value <= 0:
		raise argparse.ArgumentTypeError(f"Expected a positive integer or 'auto', got {value}")
	return value


def _register_task_allocation_override_arguments(parser: argparse.ArgumentParser) -> None:
	"""Add optional task-allocation override flags that map onto TaskAllocationConfig fields."""
	parser.add_argument(
		"--profile",
		"--task-profile",
		default=None,
		dest="active_profile_override",
		help=(
			"Override resources.active_profile from the YAML (must match a name "
			"under resources.profiles). --task-profile is kept as a legacy alias."
		),
	)
	parser.add_argument(
		"--task-backend",
		default=None,
		dest="task_allocation_backend",
		help="Override task allocation backend (e.g. local_affinity). Implies enabled=True.",
	)
	parser.add_argument(
		"--tasks-per-node",
		type=_parse_int_or_auto,
		default=None,
		dest="task_allocation_tasks_per_node",
		help="Override tasks per node (positive integer or 'auto')",
	)
	parser.add_argument(
		"--cpus-per-task",
		type=_parse_int_or_auto,
		default=None,
		dest="task_allocation_cpus_per_task",
		help="Override CPUs per task (positive integer or 'auto')",
	)
	parser.add_argument(
		"--bind",
		default=None,
		dest="task_allocation_bind",
		help="Override CPU bind policy (e.g. physical_cores, logical_cores, none)",
	)
	parser.add_argument(
		"--use-hyperthreads",
		action="store_const",
		const=True,
		default=None,
		dest="task_allocation_use_hyperthreads",
		help="Override: include sibling logical CPUs (hyperthreads) in each task slot",
	)
	parser.add_argument(
		"--reserve-cpus",
		type=_parse_positive_int,
		default=None,
		dest="task_allocation_reserve_cpus",
		help="Override: number of CPU cores to reserve from the allocatable pool",
	)


def _build_task_allocation_override_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
	override: dict[str, Any] = {}
	backend = getattr(args, "task_allocation_backend", None)
	if backend is not None:
		override["backend"] = str(backend)
		override["enabled"] = True
	tasks_per_node = getattr(args, "task_allocation_tasks_per_node", None)
	if tasks_per_node is not None:
		override["tasks_per_node"] = tasks_per_node
	cpus_per_task = getattr(args, "task_allocation_cpus_per_task", None)
	if cpus_per_task is not None:
		override["cpus_per_task"] = cpus_per_task
	bind = getattr(args, "task_allocation_bind", None)
	if bind is not None:
		override["bind"] = str(bind)
	use_hyperthreads = getattr(args, "task_allocation_use_hyperthreads", None)
	if use_hyperthreads is not None:
		override["use_hyperthreads"] = bool(use_hyperthreads)
	reserve_cpus = getattr(args, "task_allocation_reserve_cpus", None)
	if reserve_cpus is not None:
		override["reserve_cpus"] = int(reserve_cpus)
	return override if override else None


def _parse_target_dataset_indices_from_args(args: argparse.Namespace) -> list[int] | None:
	raw = getattr(args, "target_datasets", None)
	if raw is None:
		return None
	items = list(raw) if isinstance(raw, (list, tuple, set)) else [raw]
	parsed: list[int] = []
	seen: set[int] = set()
	for item in items:
		for token in str(item).split(","):
			text = str(token).strip()
			if not text:
				continue
			try:
				value = int(text)
			except Exception as exc:
				raise SystemExit(f"Invalid dataset index for --target-datasets: {text!r}") from exc
			if value < 0:
				raise SystemExit(f"Dataset indices for --target-datasets must be >= 0, got {value}")
			if value in seen:
				continue
			seen.add(value)
			parsed.append(value)
	if not parsed:
		raise SystemExit("--target-datasets requires at least one dataset index")
	return parsed


def _normalize_well_token(token: str) -> str:
	"""Accept either a wellNNN string or a plain integer index and return wellNNN.

	"3", "03", "003" → "well003"; "well003" → "well003" (verbatim). Anything else
	is returned verbatim so non-conforming well IDs still match data.yml.
	"""
	text = str(token).strip()
	if not text:
		return text
	try:
		idx = int(text)
	except ValueError:
		return text
	if idx < 0:
		raise SystemExit(f"Well index must be >= 0, got {idx}")
	return f"well{idx:03d}"


def _parse_target_well_ids_from_args(args: argparse.Namespace) -> list[str] | None:
	"""Parse `--target-wells well003 well005` (or comma-separated, or plain ints) into a list.

	Returns `None` if the flag was not provided. Plain ints are zero-padded to
	wellNNN; wellNNN strings are kept verbatim (no case folding) so they match
	`wells[*].well_id` in data.yml.
	"""
	raw = getattr(args, "target_wells", None)
	if raw is None:
		return None
	items = list(raw) if isinstance(raw, (list, tuple, set)) else [raw]
	parsed: list[str] = []
	seen: set[str] = set()
	for item in items:
		for token in str(item).split(","):
			normalized = _normalize_well_token(token)
			if not normalized or normalized in seen:
				continue
			seen.add(normalized)
			parsed.append(normalized)
	if not parsed:
		raise SystemExit("--target-wells requires at least one well id (e.g. well003 or 3)")
	return parsed


def _parse_targets_pairs_from_args(args: argparse.Namespace) -> dict[int, list[str]] | None:
	"""Parse `--targets 6:1,6:4,12:4` into {6: ["well001", "well004"], 12: ["well004"]}.

	Each token is `<dataset_idx>:<well>` where well is an integer index (zero-padded
	to wellNNN) or an explicit wellNNN string. Returns None if the flag was not
	provided. Datasets that appear multiple times have their wells merged.
	"""
	raw = getattr(args, "targets", None)
	if raw is None:
		return None
	items = list(raw) if isinstance(raw, (list, tuple, set)) else [raw]
	pairs: dict[int, list[str]] = {}
	seen_pairs: set[tuple[int, str]] = set()
	for item in items:
		for token in str(item).split(","):
			text = str(token).strip()
			if not text:
				continue
			if ":" not in text:
				raise SystemExit(
					f"--targets entries must be <dataset>:<well>, got {text!r}"
				)
			dataset_part, well_part = text.split(":", 1)
			dataset_part = dataset_part.strip()
			well_part = well_part.strip()
			try:
				dataset_idx = int(dataset_part)
			except ValueError as exc:
				raise SystemExit(
					f"--targets dataset index must be an integer, got {dataset_part!r}"
				) from exc
			if dataset_idx < 0:
				raise SystemExit(f"--targets dataset index must be >= 0, got {dataset_idx}")
			well_id = _normalize_well_token(well_part)
			if not well_id:
				raise SystemExit(f"--targets well must be non-empty, got {text!r}")
			key = (dataset_idx, well_id)
			if key in seen_pairs:
				continue
			seen_pairs.add(key)
			pairs.setdefault(dataset_idx, []).append(well_id)
	if not pairs:
		raise SystemExit("--targets requires at least one <dataset>:<well> pair")
	return pairs


def _parse_positive_int(raw: str) -> int:
	try:
		value = int(str(raw).strip())
	except Exception as exc:
		raise argparse.ArgumentTypeError(f"Expected a positive integer, got {raw!r}") from exc
	if value <= 0:
		raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value}")
	return value


def _register_debug_limit_arguments(parser: argparse.ArgumentParser) -> None:
	parser.add_argument(
		"--limit-segments",
		type=_parse_positive_int,
		default=None,
		help="Limit stage segment work for debug smoke runs where supported",
	)
	parser.add_argument(
		"--limit-datasets",
		type=_parse_positive_int,
		default=None,
		help="Limit datasets for debug smoke runs",
	)
	parser.add_argument(
		"--target-dataset",
		"--target-datasets",
		nargs="+",
		default=None,
		dest="target_datasets",
		help=(
			"Target specific 0-based dataset indices, for example --target-dataset 0 or "
			"--target-datasets 0,2,8"
		),
	)
	parser.add_argument(
		"--target-well",
		"--target-wells",
		nargs="+",
		default=None,
		dest="target_wells",
		help=(
			"Target specific wells (applied across all selected datasets). Accepts wellNNN ids "
			"or plain integer indices, e.g. --target-wells well003,well005 or --target-wells 3,5. "
			"Integers are zero-padded to wellNNN; strings are case-sensitive and must match "
			"wells[*].well_id in the data config."
		),
	)
	parser.add_argument(
		"--targets",
		nargs="+",
		default=None,
		dest="targets",
		help=(
			"Per-pair dataset:well filter, e.g. --targets 6:1,6:4,12:4. Each entry is "
			"<dataset_index>:<well> where well is an integer index or wellNNN. Takes "
			"precedence over --target-datasets / --target-wells when set."
		),
	)
	parser.add_argument(
		"--limit-wells",
		type=_parse_positive_int,
		default=None,
		dest="limit_wells_per_dataset",
		help="Limit wells selected per dataset for debug smoke runs",
	)
	parser.add_argument(
		"--limit-wells-per-dataset",
		type=_parse_positive_int,
		default=None,
		dest="limit_wells_per_dataset",
		help="Limit wells selected per dataset for debug smoke runs",
	)
	parser.add_argument(
		"--limit-units",
		type=_parse_positive_int,
		default=None,
		help="Limit units for debug runs",
	)


def _register_stage_sequence_parser(
	*,
	subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
	name: str,
	help_text: str,
) -> None:
	parser = subparsers.add_parser(name, help=help_text)
	parser.add_argument(
		"stages",
		nargs="+",
		help=(
			"Stage tokens. Accepts forms like: preprocess spikesort | preproc,sort | "
			"[preprocess, sort] | all"
		),
	)
	parser.add_argument("--config", type=str, required=True, help="Path to runtime YAML/JSON config")
	parser.add_argument("--force-restart", action="store_true", help="Force stage restart for selected stages")
	parser.add_argument(
		"--force-replot",
		action="store_true",
		help=(
			"Replot existing computed outputs without recomputing. For plot-heavy phases "
			"(merge_SLAy, plot_*, report_*) this reuses on-disk compute and regenerates "
			"only the figures/reports."
		),
	)
	parser.add_argument(
		"--no-plot",
		action="store_true",
		help=(
			"Skip plot/report generation for plot-heavy phases (merge_SLAy plots, "
			"plot_*, report_*). Compute work still runs. Useful for big re-runs where "
			"figures aren't needed."
		),
	)
	unit_group = parser.add_mutually_exclusive_group()
	unit_group.add_argument("--unit-id", type=int, default=None, help="Optional single unit override")
	unit_group.add_argument(
		"--unit-ids",
		type=_parse_unit_ids_csv,
		default=None,
		help="Optional comma-separated list of unit ids",
	)
	_register_debug_limit_arguments(parser)
	_register_task_allocation_override_arguments(parser)
	parser.add_argument(
		"--phase-tune",
		action="store_true",
		help="Run selected stages/phases normally, then write advisory resource-class tuning recommendations from phase resource logs",
	)
	parser.add_argument(
		"--confirm-full-scope",
		action="store_true",
		help="Allow --phase-tune without debug limit flags",
	)
	parser.add_argument(
		"--alloc",
		action="store_true",
		help="Print allocation details that would be used by the selected stage(s), without running stage work",
	)
	parser.add_argument(
		"--confirm",
		action="store_true",
		help=(
			"Confirm a destructive stage action. Required by stages that overwrite canonical state, "
			"such as spikesort.restore_sorter_output."
		),
	)
	parser.set_defaults(handler=_run_stage_sequence_from_args)


def _run_system_topology_from_args(_args: argparse.Namespace) -> int:
	print(format_cpu_topology(detect_cpu_topology()))
	return 0


def _register_system_topology_parser(
	*,
	subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
	parser = subparsers.add_parser(
		"systopo",
		help="Print visible CPU topology derived from current affinity and sysfs",
	)
	parser.set_defaults(handler=_run_system_topology_from_args)


def _run_status_from_args(args: argparse.Namespace) -> int:
	from . import status as status_module

	target_datasets = _parse_target_dataset_indices_from_args(args)
	target_wells = _parse_target_well_ids_from_args(args)
	stages = getattr(args, "status_stages", None)
	verbose = bool(getattr(args, "verbose", False))
	sort_by = str(getattr(args, "status_sort_by", "dataset"))
	report = status_module.scan_status(
		Path(str(args.config)).expanduser().resolve(),
		target_datasets=target_datasets,
		target_wells=target_wells,
		stages=stages,
		collect_phases=verbose,
	)
	if verbose:
		print(status_module.format_verbose_tables(report, sort_by=sort_by))
	else:
		print(status_module.format_default_tables(report, sort_by=sort_by))
	return 0


def _register_status_parser(
	*,
	subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
	parser = subparsers.add_parser(
		"status",
		help=(
			"Summarize per-stage processing completeness for every (dataset, well) "
			"in the active data config. Use -v for per-phase detail."
		),
	)
	parser.add_argument("--config", type=str, required=True, help="Path to runtime YAML/JSON config")
	parser.add_argument(
		"--target-dataset",
		"--target-datasets",
		nargs="+",
		default=None,
		dest="target_datasets",
		help="0-based dataset indices to scan (default: all included datasets)",
	)
	parser.add_argument(
		"--target-well",
		"--target-wells",
		nargs="+",
		default=None,
		dest="target_wells",
		help="Specific well IDs to scan (applied across all selected datasets).",
	)
	parser.add_argument(
		"--stage",
		"--stages",
		nargs="+",
		default=None,
		dest="status_stages",
		choices=list(_CANONICAL_STAGE_ORDER),
		help="Restrict scan to specific stages (default: preprocess spikesort reconstruct analysis)",
	)
	parser.add_argument(
		"-v",
		"--verbose",
		action="store_true",
		help="Show per-phase markers per well in addition to the per-dataset rollup",
	)
	parser.add_argument(
		"--sort-by",
		choices=("dataset", "chip-well"),
		default="dataset",
		dest="status_sort_by",
		help=(
			"Row ordering. 'dataset' (default) preserves data-config order. "
			"'chip-well' groups rows by (chip_id, well_id) and orders each "
			"group by DIV/dataset index so each well's evolution over time "
			"reads down consecutive rows. Most useful with -v."
		),
	)
	parser.set_defaults(handler=_run_status_from_args)


def _parse_stage_list_tokens(raw_tokens: list[str]) -> list[str]:
	text = " ".join(str(token) for token in list(raw_tokens or [])).strip()
	if not text:
		raise SystemExit("No stages provided. Example: stages preprocess spikesort")

	text = text.strip().strip("[]")
	if not text:
		raise SystemExit("No stages provided. Example: stages [preprocess, spikesort]")

	prelim: list[str] = []
	for chunk in text.split(","):
		for token in chunk.strip().split():
			if token:
				prelim.append(token)

	valid = set(_STAGE_HANDLERS.keys())
	out: list[str] = []
	for raw in prelim:
		token = str(raw).strip().lower()
		token = _STAGE_ALIASES.get(token, token)
		if token == "all":
			out.extend(_CANONICAL_STAGE_ORDER)
			continue
		if token not in valid:
			extra_tokens = [name for name in sorted(valid) if name not in _CANONICAL_STAGE_ORDER]
			supported_tokens = list(_CANONICAL_STAGE_ORDER)
			if extra_tokens:
				supported_tokens.extend(extra_tokens)
			raise SystemExit(
				f"Unsupported stage token: {raw}. Supported: {', '.join(supported_tokens)} (plus aliases preproc, sort, recon)."
			)
		out.append(token)

	dedup: list[str] = []
	seen: set[str] = set()
	for stage_name in out:
		if stage_name in seen:
			continue
		dedup.append(stage_name)
		seen.add(stage_name)
	return dedup


def _phase_tune_has_scope_limits(args: argparse.Namespace) -> bool:
	if getattr(args, "limit_segments", None) is not None:
		return True
	if getattr(args, "limit_datasets", None) is not None:
		return True
	if getattr(args, "target_datasets", None) is not None:
		return True
	if getattr(args, "limit_wells_per_dataset", None) is not None:
		return True
	if getattr(args, "limit_units", None) is not None:
		return True
	if getattr(args, "unit_id", None) is not None:
		return True
	if getattr(args, "unit_ids", None) is not None:
		return True
	return False


def _render_mpi_sample_worker_test(*, rank: int, size: int, config_path: str | None = None) -> str:
	"""Render a sample worker preview for this MPI rank as a text block."""
	from types import SimpleNamespace

	from .cpu_allocation import _THREAD_ENV_VARS, TaskSlot, capture_sample_worker_environment

	if size <= 1:
		return ""

	lines = ["", f"mpi_sample_worker_test: rank {rank}"]
	preview_logger = logging.Logger(f"axon_recon.pipeline.alloc_preview.rank_{rank}")
	preview_logger.addHandler(logging.NullHandler())
	preview_logger.propagate = False

	try:
		# Detect full topology for this rank (reflects mpirun --bind-to affinity)
		topology = detect_cpu_topology()
		visible_cpus = tuple(topology.visible_cpus)

		# Read task_allocation config to know set_thread_env / nested_thread_policy / use_hyperthreads
		synthetic_plan: object | None = None
		_use_ht: bool = True
		if config_path is not None:
			try:
				from pathlib import Path
				from axon_recon.runtime_config import RuntimeConfig
				from .resources import parse_resources_config
				_runtime_cfg = RuntimeConfig.load(Path(config_path).expanduser().resolve())
				_res_cfg = parse_resources_config(runtime_config=_runtime_cfg, logger=None)
				_task_cfg = _res_cfg.task_allocation
				_use_ht = bool(_task_cfg.use_hyperthreads)
				synthetic_plan = SimpleNamespace(
					set_thread_env=bool(_task_cfg.set_thread_env),
					nested_thread_policy=str(_task_cfg.nested_thread_policy or "preserve_existing"),
					use_hyperthreads=_use_ht,
					backend="mpi",
					bind="none",
				)
			except Exception:
				pass

		if _use_ht:
			test_cpus = visible_cpus if visible_cpus else (0,)
			test_cores = tuple(core.core_id for core in topology.cores) if topology.cores else tuple(test_cpus)
			test_packages = tuple(core.package_id for core in topology.cores) if topology.cores else tuple([0] * len(test_cpus))
		else:
			# Match local_affinity behavior when hyperthreads are disabled: one logical CPU per physical core.
			test_cpus = tuple(int(core.logical_cpus[0]) for core in topology.cores if core.logical_cpus)
			if not test_cpus:
				test_cpus = visible_cpus if visible_cpus else (0,)
			test_cores = tuple(core.core_id for core in topology.cores if core.logical_cpus)
			test_packages = tuple(core.package_id for core in topology.cores if core.logical_cpus)

		slot = TaskSlot(
			slot_id=0,
			logical_cpus=test_cpus,
			core_ids=test_cores if test_cores else tuple(test_cpus),
			package_ids=test_packages if test_packages else tuple([0] * len(test_cpus)),
		)

		# Capture what environment would be set for this rank's worker
		env_state = capture_sample_worker_environment(slot, plan=synthetic_plan, logger=preview_logger)

		# Show CPU topology for this rank; use physical cores when hyperthreads disabled
		effective_cpu_count = topology.logical_cpu_count if _use_ht else topology.physical_core_count
		lines.append(f"  visible_cpus: {list(test_cpus)} ({topology.physical_core_count} physical cores, {topology.logical_cpu_count} logical)")
		lines.append(f"  effective_thread_count (use_hyperthreads={_use_ht}): {effective_cpu_count}")
		if "error" not in env_state:
			thread_env = env_state.get("thread_env", {})
			any_set = any(v is not None for v in thread_env.values())
			if any_set:
				lines.append("  thread_env (would set):")
				for var, val in thread_env.items():
					lines.append(f"    {var}={val if val is not None else 'unset'}")
			else:
				lines.append("  thread_env: (not configured — set resources.task_allocation.set_thread_env: true)")
		else:
			lines.append(f"  error: {env_state['error']}")

	except Exception as exc:
		lines.append(f"  error_spawning_test_worker: {exc}")

	return "\n".join(lines)


def _run_stage_sequence_from_args(args: argparse.Namespace) -> int:
	stage_list = _parse_stage_list_tokens(list(getattr(args, "stages", []) or []))
	logger = logging.getLogger("axon_recon.pipeline.stages")
	if bool(getattr(args, "alloc", False)):
		task_allocation_override = _build_task_allocation_override_from_args(args)
		if task_allocation_override and str(task_allocation_override.get("backend", "")).strip().lower() == "mpi":
			import hashlib as _hashlib
			import os as _os
			import tempfile as _tempfile
			import time as _time

			mpi_ctx = current_mpi_context()
			_rank = int(mpi_ctx.rank) if mpi_ctx is not None else 0
			_size = int(mpi_ctx.size) if mpi_ctx is not None else 1

			allocation_preview_text: str | None = None
			if _rank == 0:
				previews = build_stage_allocation_previews(
					config_path=str(getattr(args, "config")),
					stages=stage_list,
					target_datasets_override=_parse_target_dataset_indices_from_args(args),
					unit_id_override=getattr(args, "unit_id", None),
					unit_ids_override=getattr(args, "unit_ids", None),
					unit_limit_override=getattr(args, "limit_units", None),
					limit_segments_override=getattr(args, "limit_segments", None),
					limit_datasets_override=getattr(args, "limit_datasets", None),
					limit_wells_per_dataset_override=getattr(args, "limit_wells_per_dataset", None),
					force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
					force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
					task_allocation_override=task_allocation_override,
				)
				allocation_preview_text = format_stage_allocation_previews(previews)

			local_preview_block = _render_mpi_sample_worker_test(
				rank=_rank,
				size=_size,
				config_path=str(getattr(args, "config", None) or "") or None,
			)

			preview_token = _hashlib.sha1(
				"|".join(
					[
						str(_os.getppid()),
						str(_size),
						str(getattr(args, "config", "")),
						",".join(stage_list),
					]
				).encode("utf-8")
			).hexdigest()[:16]
			preview_dir = Path(_tempfile.gettempdir()) / f"axon_recon_alloc_preview_{preview_token}"
			preview_dir.mkdir(parents=True, exist_ok=True)
			local_preview_path = preview_dir / f"rank_{_rank}.txt"
			done_path = preview_dir / "preview.done"

			local_preview_path.write_text(local_preview_block, encoding="utf-8")

			if _rank == 0:
				deadline = _time.time() + 15.0
				while _time.time() < deadline:
					if all((preview_dir / f"rank_{idx}.txt").exists() for idx in range(_size)):
						break
					_time.sleep(0.05)

				gathered_preview_blocks: list[str] = []
				for idx in range(_size):
					block_path = preview_dir / f"rank_{idx}.txt"
					if block_path.exists():
						gathered_preview_blocks.append(block_path.read_text(encoding="utf-8"))
					else:
						gathered_preview_blocks.append(f"\nmpi_sample_worker_test: rank {idx}\n  error: preview block missing")

				if allocation_preview_text:
					print(allocation_preview_text)
				if gathered_preview_blocks:
					print("\n".join(block for block in gathered_preview_blocks if block))
				done_path.write_text("done\n", encoding="utf-8")
			else:
				deadline = _time.time() + 20.0
				while _time.time() < deadline:
					if done_path.exists():
						break
					_time.sleep(0.05)
			return 0
		else:
			# Non-MPI backend
			print_stage_allocation_preview(
				config_path=str(getattr(args, "config")),
				stages=stage_list,
				target_datasets_override=_parse_target_dataset_indices_from_args(args),
				unit_id_override=getattr(args, "unit_id", None),
				unit_ids_override=getattr(args, "unit_ids", None),
				unit_limit_override=getattr(args, "limit_units", None),
				limit_segments_override=getattr(args, "limit_segments", None),
				limit_datasets_override=getattr(args, "limit_datasets", None),
				limit_wells_per_dataset_override=getattr(args, "limit_wells_per_dataset", None),
				force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
				force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
				task_allocation_override=task_allocation_override,
			)
			return 0
	if bool(getattr(args, "phase_tune", False)):
		if not _phase_tune_has_scope_limits(args) and not bool(getattr(args, "confirm_full_scope", False)):
			logger.error(
				"--phase-tune requires at least one limit flag unless --confirm-full-scope is provided",
				extra={"event": "phase_tuning_scope_rejected"},
			)
			return 2
		logger.info(
			"Starting resource tuning run stages=%s limits={datasets:%s,target_datasets:%s,wells_per_dataset:%s,segments:%s,units:%s}",
			stage_list,
			getattr(args, "limit_datasets", None),
			getattr(args, "target_datasets", None),
			getattr(args, "limit_wells_per_dataset", None),
			getattr(args, "limit_segments", None),
			getattr(args, "limit_units", None),
			extra={"event": "phase_tuning_started"},
		)

	for stage_name in stage_list:
		handler = _STAGE_HANDLERS.get(stage_name)
		if handler is None:
			raise SystemExit(f"No handler registered for stage '{stage_name}'")

		with log_context(stage=stage_name):
			logger.info("stages: starting %s", stage_name, extra={"event": "stage_started"})
			nested_args = argparse.Namespace(**vars(args))
			nested_args.stage = stage_name
			nested_args.task_allocation_override = _build_task_allocation_override_from_args(args)
			nested_args.active_profile_override = getattr(args, "active_profile_override", None)
			# Stash the active-profile override at process-wide scope so that any
			# load_pipeline_runtime_bundle call inside this handler picks it up. This
			# keeps every stage's resource-budget manager, phase-budget context, and
			# allocation plan consistent with the CLI-requested profile, without
			# threading active_profile_override through every run_*_from_runtime entry.
			from .config import set_active_profile_override
			set_active_profile_override(nested_args.active_profile_override)
			try:
				rc = int(handler(nested_args))
			finally:
				set_active_profile_override(None)
			if rc != 0:
				logger.error("stages: stage %s failed with code %d", stage_name, rc, extra={"event": "stage_failed"})
				return rc
			logger.info("stages: completed %s", stage_name, extra={"event": "stage_completed"})

	if bool(getattr(args, "phase_tune", False)):
		from .phase_tuning import emit_phase_tuning_recommendations

		emit_phase_tuning_recommendations(
			config_path=str(getattr(args, "config")),
			selected_stages=stage_list,
		)

	return 0


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(prog="axon_recon")
	subparsers = parser.add_subparsers(dest="command", required=True)

	_register_stage_sequence_parser(
		subparsers=subparsers,
		name="stages",
		help_text="Run one or more pipeline stages in sequence",
	)
	_register_stage_sequence_parser(
		subparsers=subparsers,
		name="stage",
		help_text="Alias for stages",
	)
	_register_system_topology_parser(subparsers=subparsers)
	_register_status_parser(subparsers=subparsers)
	_register_dashboard_parser(subparsers=subparsers)

	return parser


def _register_dashboard_parser(
	*,
	subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
	"""Register `axon-recon dashboard` (and the `dash` alias) as sibling
	subcommands of `stages`.

	Delegates argument parsing + entrypoint to `axon_recon.dashboard.cli` so
	the dashboard's CLI surface stays owned by the dashboard package.
	"""
	from ..dashboard import cli as dashboard_cli

	def _add_common_args(parser: argparse.ArgumentParser) -> None:
		parser.add_argument("--config", required=True, help="Path to runtime YAML/JSON config")
		parser.add_argument(
			"--target-dataset",
			"--target-datasets",
			nargs="+",
			default=None,
			dest="target_datasets",
			help="0-based dataset indices to load",
		)
		parser.add_argument("--limit-wells", type=_parse_positive_int, default=None, dest="limit_wells")
		parser.add_argument("--limit-datasets", type=_parse_positive_int, default=None)
		parser.add_argument("--limit-wells-per-dataset", type=_parse_positive_int, default=None)
		parser.add_argument("--port", type=_parse_positive_int, default=8050)
		parser.add_argument("--host", default="127.0.0.1")
		parser.add_argument(
			"--lan",
			action="store_true",
			help="Bind to 0.0.0.0 and print every LAN URL the server is reachable at",
		)
		parser.add_argument("--no-browser", action="store_true")
		parser.add_argument("--debug", action="store_true")

	def _handler(args: argparse.Namespace) -> int:
		argv: list[str] = ["--config", str(args.config)]
		if args.target_datasets is not None:
			argv.append("--target-dataset")
			argv.extend(str(item) for item in args.target_datasets)
		for flag_name, attr_name in (
			("--limit-wells", "limit_wells"),
			("--limit-datasets", "limit_datasets"),
			("--limit-wells-per-dataset", "limit_wells_per_dataset"),
			("--port", "port"),
		):
			value = getattr(args, attr_name, None)
			if value is not None:
				argv.extend([flag_name, str(value)])
		if getattr(args, "host", None) is not None:
			argv.extend(["--host", str(args.host)])
		if bool(getattr(args, "lan", False)):
			argv.append("--lan")
		if bool(getattr(args, "no_browser", False)):
			argv.append("--no-browser")
		if bool(getattr(args, "debug", False)):
			argv.append("--debug")
		return int(dashboard_cli.main(argv))

	parser = subparsers.add_parser(
		"dashboard",
		help="Serve a Plotly Dash dashboard over per-well analysis_outputs/ artifacts",
	)
	_add_common_args(parser)
	parser.set_defaults(handler=_handler)

	alias_parser = subparsers.add_parser(
		"dash",
		help="Alias for `dashboard` — same args, shorter to type.",
	)
	_add_common_args(alias_parser)
	alias_parser.set_defaults(handler=_handler)


def _configure_runtime_logging_from_args(args: argparse.Namespace) -> None:
	config_path = getattr(args, "config", None)
	if config_path is None:
		install_pipeline_log_record_factory()
		fmt = ensure_pipeline_target_in_format("[%(levelname)s] %(message)s")
		if not logging.getLogger().handlers:
			logging.basicConfig(level=logging.INFO, format=fmt)
		return
	configure_pipeline_logging(config_path=Path(str(config_path)).expanduser().resolve() if config_path is not None else None)


def _configure_phase_tuning_monitoring_from_args(args: argparse.Namespace) -> None:
	from axon_recon.runtime_config import RuntimeConfig

	from .phase_tuning import parse_phase_tuning_config
	from .resource_usage import configure_phase_tuning_monitoring

	if bool(getattr(args, "alloc", False)):
		configure_phase_tuning_monitoring(enabled=False)
		return
	if not bool(getattr(args, "phase_tune", False)):
		configure_phase_tuning_monitoring(enabled=False)
		return
	config_path = getattr(args, "config", None)
	if config_path is None:
		configure_phase_tuning_monitoring(enabled=True)
		return
	runtime_config = RuntimeConfig.load(Path(str(config_path)).expanduser().resolve())
	tuning_config = parse_phase_tuning_config(runtime_config)
	configure_phase_tuning_monitoring(
		enabled=True,
		system_tools_enabled=bool(tuning_config.system_tools_enabled),
		system_tool_interval_s=float(tuning_config.system_tool_interval_s),
		output_relpath=str(tuning_config.output_relpath),
		write_tool_logs=bool(tuning_config.write_tool_logs),
		tuning_config=tuning_config,
	)


def main(argv: list[str] | None = None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)
	install_process_lifecycle()
	install_noisy_external_log_filters()
	install_maxwell_hdf5_plugin_message_filter()
	_configure_runtime_logging_from_args(args)
	_configure_phase_tuning_monitoring_from_args(args)

	# Activate the process-wide --target-wells / --targets overrides before any
	# stage handler fires. select_execution_targets honors them at the leaf, so we
	# don't need to plumb override parameters through every runner helper.
	from .config import set_no_plot_override, set_target_pairs_override, set_target_wells_override

	target_wells_filter = _parse_target_well_ids_from_args(args)
	if target_wells_filter is not None:
		set_target_wells_override(target_wells_filter)

	target_pairs_filter = _parse_targets_pairs_from_args(args)
	if target_pairs_filter is not None:
		set_target_pairs_override(target_pairs_filter)

	# Same pattern for --no-plot: a process-wide toggle that each plot-heavy
	# phase consults via resolve_plots_enabled(). Reset to None in the finally
	# block so subsequent in-process invocations aren't poisoned.
	if bool(getattr(args, "no_plot", False)):
		set_no_plot_override(True)

	handler = getattr(args, "handler", None)
	if handler is None:
		parser.print_help()
		return 2
	logger = logging.getLogger("axon_recon.pipeline")
	logger.info("pipeline run started", extra={"event": "run_started"})
	status = "error"
	try:
		rc = int(handler(args))
		status = "ok" if rc == 0 else "error"
		if rc == 0:
			logger.info("pipeline run completed", extra={"event": "run_completed"})
		else:
			logger.error("pipeline run failed with code %d", rc, extra={"event": "run_failed"})
		return rc
	except Exception:
		logger.exception("pipeline run failed", extra={"event": "run_failed"})
		raise
	finally:
		finalize_pipeline_logging(status=status)
		set_target_wells_override(None)
		set_target_pairs_override(None)
		set_no_plot_override(None)
		try:
			from .resource_usage import configure_phase_tuning_monitoring

			configure_phase_tuning_monitoring(enabled=False)
		except Exception:
			pass


if __name__ == "__main__":
	raise SystemExit(main())
