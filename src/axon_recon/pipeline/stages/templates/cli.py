from __future__ import annotations

import argparse

from ...runner import (
	run_templates_analyzers_concat_from_runtime,
	run_templates_analyzers_from_runtime,
	run_templates_analyzers_segments_from_runtime,
	run_templates_build_templates_from_runtime,
	run_templates_extract_template_segments_from_runtime,
	run_templates_from_runtime,
	run_templates_per_unit_processing_from_runtime,
	run_templates_plot_templates_from_runtime,
	run_templates_report_templates_from_runtime,
	run_templates_reports_footprints_from_runtime,
	run_templates_reports_from_runtime,
	run_templates_reports_locations_from_runtime,
	run_templates_reports_overlays_from_runtime,
	run_templates_resolve_sources_from_runtime,
)


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


def register_templates_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
	parser = subparsers.add_parser("templates", help="Run templates stage")
	parser.add_argument("--config", type=str, required=True, help="Path to runtime YAML/JSON config")
	unit_group = parser.add_mutually_exclusive_group()
	unit_group.add_argument("--unit-id", type=int, default=None, help="Run templates for a single unit id")
	unit_group.add_argument(
		"--unit-ids",
		type=_parse_unit_ids_csv,
		default=None,
		help="Run templates for a comma-separated list of unit ids",
	)
	parser.add_argument("--force-restart", action="store_true", help="Recompute even if per-unit outputs exist")
	parser.add_argument("--force-replot", action="store_true", help="Alias for force-restart compatibility")
	parser.set_defaults(handler=_run_from_args)


def _run_from_args(args: argparse.Namespace) -> int:
	return _print_templates_aggregate(
		run_templates_from_runtime(
			config_path=str(args.config),
			unit_id_override=getattr(args, "unit_id", None),
			unit_ids_override=getattr(args, "unit_ids", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)


def _run_resolve_sources_from_args(args: argparse.Namespace) -> int:
	agg = run_templates_resolve_sources_from_runtime(
		config_path=str(args.config),
		unit_id_override=getattr(args, "unit_id", None),
		unit_ids_override=getattr(args, "unit_ids", None),
		force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
		force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
	)
	print(f"stage: {agg.stage}")
	print(f"targets_total: {agg.total_targets}")
	print(f"targets_succeeded: {agg.succeeded_targets}")
	print(f"targets_failed: {agg.failed_targets}")
	for item in agg.target_results:
		t = item.target
		if item.status == "ok" and isinstance(item.result, dict):
			sources = item.result.get("sources", {})
			concat = sources.get("concat_analyzer", {}) if isinstance(sources, dict) else {}
			segments = sources.get("preprocessed_segments", {}) if isinstance(sources, dict) else {}
			print(
				f"target[{t.dataset_index}:{t.stream_id}] status=ok "
				f"well_out_dir={item.result.get('well_out_dir', 'unknown')} "
				f"concat_analyzer={concat.get('first_existing', None)} "
				f"preprocessed_segments={segments.get('first_existing', None)}"
			)
		else:
			print(
				f"target[{t.dataset_index}:{t.stream_id}] status=error "
				f"error={item.error or 'unknown'}"
			)
	return 0


def _print_templates_aggregate(agg: object) -> int:
	print(f"stage: {agg.stage}")
	print(f"targets_total: {agg.total_targets}")
	print(f"targets_succeeded: {agg.succeeded_targets}")
	print(f"targets_failed: {agg.failed_targets}")
	for item in agg.target_results:
		t = item.target
		if item.status != "ok" or item.result is None:
			print(
				f"target[{t.dataset_index}:{t.stream_id}] status=error "
				f"error={item.error or 'unknown'}"
			)
			continue
		result = item.result
		if isinstance(result, dict):
			summary_json = result.get("summary_json", None)
			templates_out_dir = result.get("templates_out_dir", None)
			phase = result.get("phase", agg.stage)
			print(
				f"target[{t.dataset_index}:{t.stream_id}] status=ok "
				f"phase={phase} templates_out_dir={templates_out_dir} summary={summary_json}"
			)
			continue
		print(
			f"target[{t.dataset_index}:{t.stream_id}] status=ok "
			f"templates_out_dir={result.templates_out_dir} units_processed={len(result.units)}"
		)
	return 0


def _run_analyzers_from_args(args: argparse.Namespace) -> int:
	return _print_templates_aggregate(
		run_templates_analyzers_from_runtime(
			config_path=str(args.config),
			unit_id_override=getattr(args, "unit_id", None),
			unit_ids_override=getattr(args, "unit_ids", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)


def _run_analyzers_concat_from_args(args: argparse.Namespace) -> int:
	return _print_templates_aggregate(
		run_templates_analyzers_concat_from_runtime(
			config_path=str(args.config),
			unit_id_override=getattr(args, "unit_id", None),
			unit_ids_override=getattr(args, "unit_ids", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)


def _run_analyzers_segments_from_args(args: argparse.Namespace) -> int:
	return _print_templates_aggregate(
		run_templates_analyzers_segments_from_runtime(
			config_path=str(args.config),
			unit_id_override=getattr(args, "unit_id", None),
			unit_ids_override=getattr(args, "unit_ids", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)


def _run_extract_template_segments_from_args(args: argparse.Namespace) -> int:
	return _print_templates_aggregate(
		run_templates_extract_template_segments_from_runtime(
			config_path=str(args.config),
			unit_id_override=getattr(args, "unit_id", None),
			unit_ids_override=getattr(args, "unit_ids", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)


def _run_build_templates_from_args(args: argparse.Namespace) -> int:
	return _print_templates_aggregate(
		run_templates_build_templates_from_runtime(
			config_path=str(args.config),
			unit_id_override=getattr(args, "unit_id", None),
			unit_ids_override=getattr(args, "unit_ids", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)


def _run_plot_templates_from_args(args: argparse.Namespace) -> int:
	return _print_templates_aggregate(
		run_templates_plot_templates_from_runtime(
			config_path=str(args.config),
			unit_id_override=getattr(args, "unit_id", None),
			unit_ids_override=getattr(args, "unit_ids", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)


def _run_report_templates_from_args(args: argparse.Namespace) -> int:
	return _print_templates_aggregate(
		run_templates_report_templates_from_runtime(
			config_path=str(args.config),
			unit_id_override=getattr(args, "unit_id", None),
			unit_ids_override=getattr(args, "unit_ids", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)


def _run_per_unit_processing_from_args(args: argparse.Namespace) -> int:
	return _print_templates_aggregate(
		run_templates_per_unit_processing_from_runtime(
			config_path=str(args.config),
			unit_id_override=getattr(args, "unit_id", None),
			unit_ids_override=getattr(args, "unit_ids", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)


def _run_reports_from_args(args: argparse.Namespace) -> int:
	return _print_templates_aggregate(
		run_templates_reports_from_runtime(
			config_path=str(args.config),
			unit_id_override=getattr(args, "unit_id", None),
			unit_ids_override=getattr(args, "unit_ids", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)


def _run_reports_locations_from_args(args: argparse.Namespace) -> int:
	return _print_templates_aggregate(
		run_templates_reports_locations_from_runtime(
			config_path=str(args.config),
			unit_id_override=getattr(args, "unit_id", None),
			unit_ids_override=getattr(args, "unit_ids", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)


def _run_reports_footprints_from_args(args: argparse.Namespace) -> int:
	return _print_templates_aggregate(
		run_templates_reports_footprints_from_runtime(
			config_path=str(args.config),
			unit_id_override=getattr(args, "unit_id", None),
			unit_ids_override=getattr(args, "unit_ids", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)


def _run_reports_overlays_from_args(args: argparse.Namespace) -> int:
	return _print_templates_aggregate(
		run_templates_reports_overlays_from_runtime(
			config_path=str(args.config),
			unit_id_override=getattr(args, "unit_id", None),
			unit_ids_override=getattr(args, "unit_ids", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)
