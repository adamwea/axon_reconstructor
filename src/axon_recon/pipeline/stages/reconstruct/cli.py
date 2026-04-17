from __future__ import annotations

import argparse

from ...runner import (
	run_reconstruct_from_runtime,
	run_reconstruct_generate_gtrs_from_runtime,
	run_reconstruct_plot_branch_propagations_from_runtime,
	run_reconstruct_plot_branch_velocities_from_runtime,
	run_reconstruct_plot_unit_summary_from_runtime,
	run_reconstruct_plot_recons_from_runtime,
	run_reconstruct_report_full_chip_layout_from_runtime,
	run_reconstruct_report_recons_from_runtime,
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
	parser.set_defaults(handler=_run_from_args)


def _run_from_args(args: argparse.Namespace) -> int:
	return _print_reconstruct_aggregate(
		run_reconstruct_from_runtime(
			config_path=str(args.config),
			unit_id_override=getattr(args, "unit_id", None),
			unit_ids_override=getattr(args, "unit_ids", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)


def _run_generate_gtrs_from_args(args: argparse.Namespace) -> int:
	return _print_reconstruct_aggregate(
		run_reconstruct_generate_gtrs_from_runtime(
			config_path=str(args.config),
			unit_id_override=getattr(args, "unit_id", None),
			unit_ids_override=getattr(args, "unit_ids", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)


def _run_plot_recons_from_args(args: argparse.Namespace) -> int:
	return _print_reconstruct_aggregate(
		run_reconstruct_plot_recons_from_runtime(
			config_path=str(args.config),
			unit_id_override=getattr(args, "unit_id", None),
			unit_ids_override=getattr(args, "unit_ids", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)


def _run_plot_branch_propagations_from_args(args: argparse.Namespace) -> int:
	return _print_reconstruct_aggregate(
		run_reconstruct_plot_branch_propagations_from_runtime(
			config_path=str(args.config),
			unit_id_override=getattr(args, "unit_id", None),
			unit_ids_override=getattr(args, "unit_ids", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)


def _run_plot_branch_velocities_from_args(args: argparse.Namespace) -> int:
	return _print_reconstruct_aggregate(
		run_reconstruct_plot_branch_velocities_from_runtime(
			config_path=str(args.config),
			unit_id_override=getattr(args, "unit_id", None),
			unit_ids_override=getattr(args, "unit_ids", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)


def _run_plot_unit_summary_from_args(args: argparse.Namespace) -> int:
	return _print_reconstruct_aggregate(
		run_reconstruct_plot_unit_summary_from_runtime(
			config_path=str(args.config),
			unit_id_override=getattr(args, "unit_id", None),
			unit_ids_override=getattr(args, "unit_ids", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)


def _run_report_recons_from_args(args: argparse.Namespace) -> int:
	return _print_reconstruct_aggregate(
		run_reconstruct_report_recons_from_runtime(
			config_path=str(args.config),
			unit_id_override=getattr(args, "unit_id", None),
			unit_ids_override=getattr(args, "unit_ids", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)


def _run_report_full_chip_layout_from_args(args: argparse.Namespace) -> int:
	return _print_reconstruct_aggregate(
		run_reconstruct_report_full_chip_layout_from_runtime(
			config_path=str(args.config),
			unit_id_override=getattr(args, "unit_id", None),
			unit_ids_override=getattr(args, "unit_ids", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)


def _print_reconstruct_aggregate(agg: object) -> int:
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
