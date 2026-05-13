from __future__ import annotations

import argparse
import logging

from ...runner import (
	run_reconstruct_clear_templates_cache_from_runtime,
	run_reconstruct_from_runtime,
	run_reconstruct_generate_gtrs_from_runtime,
	run_reconstruct_plot_branch_propagations_from_runtime,
	run_reconstruct_plot_branch_velocities_from_runtime,
	run_reconstruct_plot_unit_summary_from_runtime,
	run_reconstruct_plot_recons_from_runtime,
	run_reconstruct_report_full_chip_layout_from_runtime,
	run_reconstruct_report_recon_grid_from_runtime,
	run_reconstruct_report_recons_from_runtime,
	run_reconstruct_report_summaries_from_runtime,
	run_reconstruct_templates_analyzers_from_runtime,
	run_reconstruct_templates_build_templates_from_runtime,
	run_reconstruct_templates_compute_template_similarity_from_runtime,
	run_reconstruct_templates_extract_partial_templates_from_runtime,
	run_reconstruct_templates_plot_templates_from_runtime,
	run_reconstruct_templates_plot_templates_v2_from_runtime,
	run_reconstruct_templates_report_templates_from_runtime,
	run_reconstruct_templates_reports_from_runtime,
	run_reconstruct_templates_resolve_sources_from_runtime,
)


LOGGER = logging.getLogger("axon_recon.reconstruct.cli")


def _emit_reconstruct_aggregate(agg: object) -> int:
	from ...execution.results import stage_aggregate_summary_lines

	for line in stage_aggregate_summary_lines(agg):
		LOGGER.info(line)
	for item in agg.target_results:
		target = item.target
		if item.status != "ok" or item.result is None:
			LOGGER.info(
				"target[%s:%s] status=error error=%s",
				target.dataset_index,
				target.stream_id,
				item.error or "unknown",
			)
			continue
		result = item.result
		if isinstance(result, dict):
			phase = result.get("phase", agg.stage)
			LOGGER.info(
				"target[%s:%s] status=ok phase=%s reconstruction_out_dir=%s summary=%s units_ok=%s units_error=%s",
				target.dataset_index,
				target.stream_id,
				phase,
				result.get("reconstruction_out_dir", None),
				result.get("summary_json", None),
				result.get("units_ok", 0),
				result.get("units_error", 0),
			)
			continue
		units_ok = sum(1 for unit in result.units if str(getattr(unit, "status", "ok")).strip().lower() == "ok")
		units_error = sum(1 for unit in result.units if str(getattr(unit, "status", "ok")).strip().lower() != "ok")
		LOGGER.info(
			"target[%s:%s] status=ok reconstruct_out_dir=%s units_processed=%s units_ok=%s units_error=%s",
			target.dataset_index,
			target.stream_id,
			result.reconstruction_out_dir,
			len(result.units),
			units_ok,
			units_error,
		)
	return 0


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


def _parse_positive_int(raw: str) -> int:
	try:
		value = int(str(raw).strip())
	except Exception as exc:
		raise argparse.ArgumentTypeError(f"Expected a positive integer, got {raw!r}") from exc
	if value <= 0:
		raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value}")
	return value


def _parse_target_dataset_indices(raw: object) -> list[int] | None:
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
				raise ValueError(f"Invalid dataset index for --target-datasets: {text!r}") from exc
			if value < 0:
				raise ValueError(f"Dataset indices for --target-datasets must be >= 0, got {value}")
			if value in seen:
				continue
			seen.add(value)
			parsed.append(value)
	if not parsed:
		raise ValueError("--target-datasets requires at least one dataset index")
	return parsed


def register_reconstruct_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
	parser = subparsers.add_parser("recon", aliases=["reconstruct"], help="Run reconstruct stage")
	parser.add_argument("--config", type=str, required=True, help="Path to runtime YAML/JSON config")
	unit_group = parser.add_mutually_exclusive_group()
	unit_group.add_argument("--unit-id", type=int, default=None, help="Run reconstruct for a single unit id")
	unit_group.add_argument(
		"--unit-ids",
		type=_parse_unit_ids_csv,
		default=None,
		help="Run reconstruct for a comma-separated list of unit ids",
	)
	parser.add_argument("--force-restart", action="store_true", help="Recompute even if per-unit outputs exist")
	parser.add_argument("--force-replot", action="store_true", help="Reserved for parity with legacy CLI")
	parser.add_argument(
		"--limit-segments",
		type=_parse_positive_int,
		default=None,
		help="Limit segment analyzer sources for debug runs",
	)
	parser.add_argument(
		"--limit-units",
		type=_parse_positive_int,
		default=None,
		help="Limit units for debug runs",
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
		"--limit-wells-per-dataset",
		type=_parse_positive_int,
		default=None,
		help="Limit wells selected per dataset for debug smoke runs",
	)
	parser.set_defaults(handler=_run_from_args)


def _reconstruct_runtime_kwargs(args: argparse.Namespace) -> dict[str, object]:
	try:
		target_datasets_override = _parse_target_dataset_indices(getattr(args, "target_datasets", None))
	except ValueError as exc:
		raise SystemExit(str(exc)) from exc
	return {
		"config_path": str(args.config),
		"unit_id_override": getattr(args, "unit_id", None),
		"unit_ids_override": getattr(args, "unit_ids", None),
		"unit_limit_override": getattr(args, "limit_units", None),
		"limit_segments_override": getattr(args, "limit_segments", None),
		"limit_datasets_override": getattr(args, "limit_datasets", None),
		"target_datasets_override": target_datasets_override,
		"limit_wells_per_dataset_override": getattr(args, "limit_wells_per_dataset", None),
		"force_restart_override": (True if bool(getattr(args, "force_restart", False)) else None),
		"force_replot_override": (True if bool(getattr(args, "force_replot", False)) else None),
		"task_allocation_override": getattr(args, "task_allocation_override", None),
	}


def _run_from_args(args: argparse.Namespace) -> int:
	return _emit_reconstruct_aggregate(run_reconstruct_from_runtime(**_reconstruct_runtime_kwargs(args)))


def _run_reconstruct_resolve_sources_from_args(args: argparse.Namespace) -> int:
	return _print_reconstruct_aggregate(run_reconstruct_templates_resolve_sources_from_runtime(**_reconstruct_runtime_kwargs(args)))


def _run_reconstruct_analyzers_from_args(args: argparse.Namespace) -> int:
	return _print_reconstruct_aggregate(run_reconstruct_templates_analyzers_from_runtime(**_reconstruct_runtime_kwargs(args)))


def _run_reconstruct_extract_partial_templates_from_args(args: argparse.Namespace) -> int:
	return _print_reconstruct_aggregate(
		run_reconstruct_templates_extract_partial_templates_from_runtime(**_reconstruct_runtime_kwargs(args))
	)


def _run_reconstruct_build_templates_from_args(args: argparse.Namespace) -> int:
	return _print_reconstruct_aggregate(run_reconstruct_templates_build_templates_from_runtime(**_reconstruct_runtime_kwargs(args)))


def _run_reconstruct_compute_template_similarity_from_args(args: argparse.Namespace) -> int:
	return _print_reconstruct_aggregate(run_reconstruct_templates_compute_template_similarity_from_runtime(**_reconstruct_runtime_kwargs(args)))


def _run_reconstruct_plot_templates_from_args(args: argparse.Namespace) -> int:
	return _print_reconstruct_aggregate(run_reconstruct_templates_plot_templates_from_runtime(**_reconstruct_runtime_kwargs(args)))


def _run_reconstruct_plot_templates_v2_from_args(args: argparse.Namespace) -> int:
	return _print_reconstruct_aggregate(run_reconstruct_templates_plot_templates_v2_from_runtime(**_reconstruct_runtime_kwargs(args)))


def _run_reconstruct_report_templates_from_args(args: argparse.Namespace) -> int:
	return _print_reconstruct_aggregate(run_reconstruct_templates_report_templates_from_runtime(**_reconstruct_runtime_kwargs(args)))


def _run_reconstruct_reports_from_args(args: argparse.Namespace) -> int:
	return _print_reconstruct_aggregate(run_reconstruct_templates_reports_from_runtime(**_reconstruct_runtime_kwargs(args)))


def _run_generate_gtrs_from_args(args: argparse.Namespace) -> int:
	return _print_reconstruct_aggregate(run_reconstruct_generate_gtrs_from_runtime(**_reconstruct_runtime_kwargs(args)))


def _run_plot_recons_from_args(args: argparse.Namespace) -> int:
	return _print_reconstruct_aggregate(run_reconstruct_plot_recons_from_runtime(**_reconstruct_runtime_kwargs(args)))


def _run_plot_branch_propagations_from_args(args: argparse.Namespace) -> int:
	return _print_reconstruct_aggregate(run_reconstruct_plot_branch_propagations_from_runtime(**_reconstruct_runtime_kwargs(args)))


def _run_plot_branch_velocities_from_args(args: argparse.Namespace) -> int:
	return _print_reconstruct_aggregate(run_reconstruct_plot_branch_velocities_from_runtime(**_reconstruct_runtime_kwargs(args)))


def _run_plot_unit_summary_from_args(args: argparse.Namespace) -> int:
	return _print_reconstruct_aggregate(run_reconstruct_plot_unit_summary_from_runtime(**_reconstruct_runtime_kwargs(args)))


def _run_report_recons_from_args(args: argparse.Namespace) -> int:
	return _print_reconstruct_aggregate(run_reconstruct_report_recons_from_runtime(**_reconstruct_runtime_kwargs(args)))


def _run_report_recon_grid_from_args(args: argparse.Namespace) -> int:
	return _print_reconstruct_aggregate(run_reconstruct_report_recon_grid_from_runtime(**_reconstruct_runtime_kwargs(args)))


def _run_report_full_chip_layout_from_args(args: argparse.Namespace) -> int:
	return _print_reconstruct_aggregate(run_reconstruct_report_full_chip_layout_from_runtime(**_reconstruct_runtime_kwargs(args)))


def _run_report_summaries_from_args(args: argparse.Namespace) -> int:
	return _print_reconstruct_aggregate(run_reconstruct_report_summaries_from_runtime(**_reconstruct_runtime_kwargs(args)))


def _run_clear_templates_cache_from_args(args: argparse.Namespace) -> int:
	return _print_reconstruct_aggregate(run_reconstruct_clear_templates_cache_from_runtime(**_reconstruct_runtime_kwargs(args)))


def _print_reconstruct_aggregate(agg: object) -> int:
	from ...execution.results import stage_aggregate_summary_lines

	for line in stage_aggregate_summary_lines(agg):
		print(line)
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
			phase = result.get("phase", agg.stage)
			print(
				f"target[{t.dataset_index}:{t.stream_id}] status=ok "
				f"phase={phase} reconstruction_out_dir={result.get('reconstruction_out_dir', None)} "
				f"summary={result.get('summary_json', None)} units_ok={result.get('units_ok', 0)} units_error={result.get('units_error', 0)}"
			)
			continue
		units_ok = sum(1 for unit in result.units if str(getattr(unit, "status", "ok")).strip().lower() == "ok")
		units_error = sum(1 for unit in result.units if str(getattr(unit, "status", "ok")).strip().lower() != "ok")
		print(
			f"target[{t.dataset_index}:{t.stream_id}] status=ok "
			f"reconstruct_out_dir={result.reconstruction_out_dir} "
			f"units_processed={len(result.units)} units_ok={units_ok} units_error={units_error}"
		)
	return 0
