from __future__ import annotations

import argparse

from ...runner import run_preprocess_from_runtime


def register_preprocess_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
	parser = subparsers.add_parser("preprocess", help="Run preprocess stage")
	parser.add_argument("--config", type=str, required=True, help="Path to runtime YAML/JSON config")
	parser.add_argument("--force-restart", action="store_true", help="Recompute preprocess outputs for each target")
	parser.add_argument("--force-replot", action="store_true", help="Alias for force-restart compatibility")
	parser.set_defaults(handler=_run_from_args)


def _run_from_args(args: argparse.Namespace) -> int:
	agg = run_preprocess_from_runtime(
		config_path=str(args.config),
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
			print(
				f"target[{t.dataset_index}:{t.stream_id}] status=ok "
				f"preprocess_out_dir={item.result.preprocess_out_dir} "
				f"outputs={len(item.result.outputs)}"
			)
		else:
			print(
				f"target[{t.dataset_index}:{t.stream_id}] status=error "
				f"error={item.error or 'unknown'}"
			)
	return 0
