from __future__ import annotations

import argparse

from ...runner import run_analysis_from_runtime


def register_analysis_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
	parser = subparsers.add_parser("analysis", help="Run analysis stage")
	parser.add_argument("--config", type=str, required=True, help="Path to runtime YAML/JSON config")
	parser.add_argument("--unit-id", type=int, default=None, help="Run analysis for a single unit id")
	parser.add_argument("--force-restart", action="store_true", help="Recompute analysis outputs for each target")
	parser.add_argument("--force-replot", action="store_true", help="Reserved for parity with legacy CLI")
	parser.set_defaults(handler=_run_from_args)


def _run_from_args(args: argparse.Namespace) -> int:
	agg = run_analysis_from_runtime(
		config_path=str(args.config),
		unit_id_override=getattr(args, "unit_id", None),
		force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
		force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
	)
	print(f"stage: {agg.stage}")
	print(f"targets_total: {agg.total_targets}")
	print(f"targets_succeeded: {agg.succeeded_targets}")
	print(f"targets_failed: {agg.failed_targets}")
	for item in agg.target_results:
		t = item.target
		if item.status == "ok" and item.result is not None:
			warnings = len(getattr(item.result, "deferred_warnings", []) or [])
			print(
				f"target[{t.dataset_index}:{t.stream_id}] status=ok "
				f"analysis_out_dir={item.result.analysis_out_dir} "
				f"outputs={len(item.result.outputs)} deferred_warnings={warnings}"
			)
		else:
			print(
				f"target[{t.dataset_index}:{t.stream_id}] status=error "
				f"error={item.error or 'unknown'}"
			)
	return 0
