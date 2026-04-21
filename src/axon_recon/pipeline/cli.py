from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Callable

from axon_reconstructor.runtime_config import RuntimeConfig

from .stages.analysis.cli import _run_from_args as _run_analysis_from_args
from .stages.preprocess.cli import _run_concat_segments_from_args as _run_preprocess_concat_segments_from_args
from .stages.preprocess.cli import _run_copy_src_to_scratch_from_args as _run_preprocess_copy_src_to_scratch_from_args
from .stages.preprocess.cli import _run_from_args as _run_preprocess_from_args
from .stages.preprocess.cli import _run_plot_concat_traces_from_args as _run_preprocess_plot_concat_traces_from_args
from .stages.preprocess.cli import _run_plot_segment_traces_from_args as _run_preprocess_plot_segment_traces_from_args
from .stages.preprocess.cli import _run_preprocess_segments_from_args as _run_preprocess_preprocess_segments_from_args
from .stages.preprocess.cli import _run_save_rec_metadata_from_args as _run_preprocess_save_rec_metadata_from_args
from .stages.preprocess.cli import _run_wipe_src_scratch_from_args as _run_preprocess_wipe_src_scratch_from_args
from .stages.reconstruct.cli import _run_from_args as _run_reconstruct_from_args
from .stages.reconstruct.cli import _run_generate_gtrs_from_args as _run_reconstruct_generate_gtrs_from_args
from .stages.reconstruct.cli import _run_plot_branch_propagations_from_args as _run_reconstruct_plot_branch_propagations_from_args
from .stages.reconstruct.cli import _run_plot_branch_velocities_from_args as _run_reconstruct_plot_branch_velocities_from_args
from .stages.reconstruct.cli import _run_plot_unit_summary_from_args as _run_reconstruct_plot_unit_summary_from_args
from .stages.reconstruct.cli import _run_plot_recons_from_args as _run_reconstruct_plot_recons_from_args
from .stages.reconstruct.cli import _run_report_full_chip_layout_from_args as _run_reconstruct_report_full_chip_layout_from_args
from .stages.reconstruct.cli import _run_report_recons_from_args as _run_reconstruct_report_recons_from_args
from .stages.reconstruct.cli import _run_report_summaries_from_args as _run_reconstruct_report_summaries_from_args
from .stages.spikesort.cli import _run_from_args as _run_spikesort_from_args
from .stages.spikesort.cli import _run_merge_auto_merge_from_args as _run_spikesort_merge_auto_merge_from_args
from .stages.spikesort.cli import _run_merge_from_args as _run_spikesort_merge_from_args
from .stages.spikesort.cli import _run_merge_slay_from_args as _run_spikesort_merge_slay_from_args
from .stages.templates.cli import _run_from_args as _run_templates_from_args
from .stages.templates.cli import _run_analyzers_concat_from_args as _run_templates_analyzers_concat_from_args
from .stages.templates.cli import _run_analyzers_from_args as _run_templates_analyzers_from_args
from .stages.templates.cli import _run_analyzers_segments_from_args as _run_templates_analyzers_segments_from_args
from .stages.templates.cli import _run_build_templates_from_args as _run_templates_build_templates_from_args
from .stages.templates.cli import _run_compute_template_similarity_from_args as _run_templates_compute_template_similarity_from_args
from .stages.templates.cli import _run_extract_template_segments_from_args as _run_templates_extract_template_segments_from_args
from .stages.templates.cli import _run_per_unit_processing_from_args as _run_templates_per_unit_processing_from_args
from .stages.templates.cli import _run_plot_templates_from_args as _run_templates_plot_templates_from_args
from .stages.templates.cli import _run_report_templates_from_args as _run_templates_report_templates_from_args
from .stages.templates.cli import _run_reports_footprints_from_args as _run_templates_reports_footprints_from_args
from .stages.templates.cli import _run_reports_from_args as _run_templates_reports_from_args
from .stages.templates.cli import _run_reports_locations_from_args as _run_templates_reports_locations_from_args
from .stages.templates.cli import _run_reports_overlays_from_args as _run_templates_reports_overlays_from_args
from .stages.templates.cli import _run_resolve_sources_from_args as _run_templates_resolve_sources_from_args


StageHandler = Callable[[argparse.Namespace], int]

_CANONICAL_STAGE_ORDER: list[str] = [
	"preprocess",
	"spikesort",
	"templates",
	"reconstruct",
	"analysis",
]

_STAGE_ALIASES: dict[str, str] = {
	"pre": "preprocess",
	"prep": "preprocess",
	"preproc": "preprocess",
	"pre.copy_src_to_scratch": "preprocess.copy_src_to_scratch",
	"pre.save_rec_metadata": "preprocess.save_rec_metadata",
	"pre.wipe_src_scratch": "preprocess.wipe_src_scratch",
	"pre.preprocess_segments": "preprocess.preprocess_segments",
	"pre.plot_segment_traces": "preprocess.plot_segment_traces",
	"pre.concat_segments": "preprocess.concat_segments",
	"pre.plot_concat_traces": "preprocess.plot_concat_traces",
	"preproc.copy_src_to_scratch": "preprocess.copy_src_to_scratch",
	"preproc.save_rec_metadata": "preprocess.save_rec_metadata",
	"preproc.wipe_src_scratch": "preprocess.wipe_src_scratch",
	"preproc.preprocess_segments": "preprocess.preprocess_segments",
	"preproc.plot_segment_traces": "preprocess.plot_segment_traces",
	"preproc.concat_segments": "preprocess.concat_segments",
	"preproc.plot_concat_traces": "preprocess.plot_concat_traces",
	"preprocess.plot_segment_traces": "preprocess.plot_segment_traces",
	"preprocess.concat_segments": "preprocess.concat_segments",
	"preprocess.plot_concat_traces": "preprocess.plot_concat_traces",
	"sort": "spikesort",
	"spikesort.sort": "spikesort",
	"merge": "spikesort.merge",
	"spikesort.merge_units": "spikesort.merge",
	"spikesort.merge.automerge": "spikesort.merge.auto_merge",
	"spikesort.merge_units.slay": "spikesort.merge.slay",
	"spikesort.merge_units.auto_merge": "spikesort.merge.auto_merge",
	"spike": "spikesort",
	"spikesorting": "spikesort",
	"template": "templates",
	"templates.resolve": "templates.resolve_sources",
	"template.resolve": "templates.resolve_sources",
	"templates.analyzer": "templates.analyzers",
	"template.analyzers": "templates.analyzers",
	"template.analyzer": "templates.analyzers",
	"templates.extract_template_segments": "templates.per_unit_processing.extract_template_segments",
	"templates.per_unit_processing.build_templates": "templates.build_templates",
	"template.compute_template_similarity": "templates.compute_template_similarity",
	"templates.per_unit_processing.plot_templates": "templates.plot_templates",
	"template.report_templates": "templates.report_templates",
	"template.per_unit_processing": "templates.per_unit_processing",
	"template.reports": "templates.reports",
	"recon": "reconstruct",
	"reconstruction": "reconstruct",
	"recon.generate_gtrs": "reconstruct.generate_gtrs",
	"recon.plot_recons": "reconstruct.plot_recons",
	"recon.plot_branch_propagations": "reconstruct.plot_branch_propagations",
	"recon.plot_branch_velocities": "reconstruct.plot_branch_velocities",
	"recon.plot_unit_summary": "reconstruct.plot_unit_summary",
	"recon.report_recons": "reconstruct.report_recons",
	"recon.report_full_chip_layout": "reconstruct.report_full_chip_layout",
	"recon.report_full_chip_recon": "reconstruct.report_full_chip_layout",
	"recon.report_summaries": "reconstruct.report_summaries",
	"reconstruction.generate_gtrs": "reconstruct.generate_gtrs",
	"reconstruction.plot_recons": "reconstruct.plot_recons",
	"reconstruction.plot_branch_propagations": "reconstruct.plot_branch_propagations",
	"reconstruction.plot_branch_velocities": "reconstruct.plot_branch_velocities",
	"reconstruction.plot_unit_summary": "reconstruct.plot_unit_summary",
	"reconstruction.report_recons": "reconstruct.report_recons",
	"reconstruction.report_full_chip_layout": "reconstruct.report_full_chip_layout",
	"reconstruction.report_full_chip_recon": "reconstruct.report_full_chip_layout",
	"reconstruction.report_summaries": "reconstruct.report_summaries",
	"analyse": "analysis",
	"analyze": "analysis",
}

_STAGE_HANDLERS: dict[str, StageHandler] = {
	"preprocess": _run_preprocess_from_args,
	"preprocess.copy_src_to_scratch": _run_preprocess_copy_src_to_scratch_from_args,
	"preprocess.save_rec_metadata": _run_preprocess_save_rec_metadata_from_args,
	"preprocess.wipe_src_scratch": _run_preprocess_wipe_src_scratch_from_args,
	"preprocess.preprocess_segments": _run_preprocess_preprocess_segments_from_args,
	"preprocess.plot_segment_traces": _run_preprocess_plot_segment_traces_from_args,
	"preprocess.concat_segments": _run_preprocess_concat_segments_from_args,
	"preprocess.plot_concat_traces": _run_preprocess_plot_concat_traces_from_args,
	"spikesort": _run_spikesort_from_args,
	"spikesort.merge": _run_spikesort_merge_from_args,
	"spikesort.merge.slay": _run_spikesort_merge_slay_from_args,
	"spikesort.merge.auto_merge": _run_spikesort_merge_auto_merge_from_args,
	"templates": _run_templates_from_args,
	"templates.resolve_sources": _run_templates_resolve_sources_from_args,
	"templates.analyzers": _run_templates_analyzers_from_args,
	"templates.analyzers.concat": _run_templates_analyzers_concat_from_args,
	"templates.analyzers.segments": _run_templates_analyzers_segments_from_args,
	"templates.per_unit_processing.extract_template_segments": _run_templates_extract_template_segments_from_args,
	"templates.build_templates": _run_templates_build_templates_from_args,
	"templates.compute_template_similarity": _run_templates_compute_template_similarity_from_args,
	"templates.plot_templates": _run_templates_plot_templates_from_args,
	"templates.report_templates": _run_templates_report_templates_from_args,
	"templates.per_unit_processing": _run_templates_per_unit_processing_from_args,
	"templates.reports": _run_templates_reports_from_args,
	"templates.reports.locations": _run_templates_reports_locations_from_args,
	"templates.reports.footprints": _run_templates_reports_footprints_from_args,
	"templates.reports.overlays": _run_templates_reports_overlays_from_args,
	"reconstruct": _run_reconstruct_from_args,
	"reconstruct.generate_gtrs": _run_reconstruct_generate_gtrs_from_args,
	"reconstruct.plot_recons": _run_reconstruct_plot_recons_from_args,
	"reconstruct.plot_branch_propagations": _run_reconstruct_plot_branch_propagations_from_args,
	"reconstruct.plot_branch_velocities": _run_reconstruct_plot_branch_velocities_from_args,
	"reconstruct.plot_unit_summary": _run_reconstruct_plot_unit_summary_from_args,
	"reconstruct.report_recons": _run_reconstruct_report_recons_from_args,
	"reconstruct.report_full_chip_layout": _run_reconstruct_report_full_chip_layout_from_args,
	"reconstruct.report_summaries": _run_reconstruct_report_summaries_from_args,
	"analysis": _run_analysis_from_args,
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
	parser.add_argument("--force-replot", action="store_true", help="Alias for force-restart compatibility")
	unit_group = parser.add_mutually_exclusive_group()
	unit_group.add_argument("--unit-id", type=int, default=None, help="Optional single unit override")
	unit_group.add_argument(
		"--unit-ids",
		type=_parse_unit_ids_csv,
		default=None,
		help="Optional comma-separated list of unit ids",
	)
	parser.set_defaults(handler=_run_stage_sequence_from_args)


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


def _run_stage_sequence_from_args(args: argparse.Namespace) -> int:
	stage_list = _parse_stage_list_tokens(list(getattr(args, "stages", []) or []))
	logger = logging.getLogger("axon_recon.pipeline.stages")

	for stage_name in stage_list:
		handler = _STAGE_HANDLERS.get(stage_name)
		if handler is None:
			raise SystemExit(f"No handler registered for stage '{stage_name}'")

		logger.info("stages: starting %s", stage_name)
		nested_args = argparse.Namespace(**vars(args))
		nested_args.stage = stage_name
		rc = int(handler(nested_args))
		if rc != 0:
			logger.error("stages: stage %s failed with code %d", stage_name, rc)
			return rc
		logger.info("stages: completed %s", stage_name)

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

	return parser


def _configure_runtime_logging_from_args(args: argparse.Namespace) -> None:
	default_level = logging.INFO
	default_format = "[%(levelname)s] %(message)s"
	level = default_level
	fmt = default_format

	config_path = getattr(args, "config", None)
	if config_path is not None:
		try:
			runtime_cfg = RuntimeConfig.load(Path(str(config_path)).expanduser().resolve())
			logger_block = runtime_cfg.get("global_logger", {})
			logger_block = logger_block if isinstance(logger_block, dict) else {}

			debug_block = logger_block.get("debug_mode", {})
			debug_block = debug_block if isinstance(debug_block, dict) else {}
			debug_enabled = bool(debug_block.get("enable", False))

			level_name: Any
			format_value: Any
			if debug_enabled:
				level_name = debug_block.get("level", logger_block.get("level", "DEBUG"))
				format_value = debug_block.get("format", logger_block.get("format", default_format))
			else:
				level_name = logger_block.get("level", "INFO")
				format_value = logger_block.get("format", default_format)

			level = getattr(logging, str(level_name).upper(), default_level)
			fmt = str(format_value or default_format)
		except Exception:
			level = default_level
			fmt = default_format

	root = logging.getLogger()
	if not root.handlers:
		logging.basicConfig(level=level, format=fmt)
		return

	root.setLevel(level)
	for handler in root.handlers:
		try:
			handler.setLevel(level)
			handler.setFormatter(logging.Formatter(fmt))
		except Exception:
			continue


def main(argv: list[str] | None = None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)
	_configure_runtime_logging_from_args(args)
	handler = getattr(args, "handler", None)
	if handler is None:
		parser.print_help()
		return 2
	return int(handler(args))


if __name__ == "__main__":
	raise SystemExit(main())
