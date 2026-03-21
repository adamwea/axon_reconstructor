from __future__ import annotations

import argparse

from ...runner import run_reconstruct_from_runtime


def register_reconstruct_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
	parser = subparsers.add_parser("recon", aliases=["reconstruct"], help="Run reconstruct stage")
	parser.add_argument("--config", type=str, required=True, help="Path to runtime YAML/JSON config")
	parser.add_argument("--unit-id", type=int, default=None, help="Run reconstruct for a single unit id")
	parser.add_argument("--force-restart", action="store_true", help="Recompute even if per-unit outputs exist")
	parser.add_argument("--force-replot", action="store_true", help="Reserved for parity with legacy CLI")
	parser.set_defaults(handler=_run_from_args)


def _run_from_args(args: argparse.Namespace) -> int:
	agg = run_reconstruct_from_runtime(
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
			print(
				f"target[{t.dataset_index}:{t.stream_id}] status=ok "
				f"reconstruct_out_dir={item.result.reconstruction_out_dir} units_processed={len(item.result.units)}"
			)
		else:
			print(
				f"target[{t.dataset_index}:{t.stream_id}] status=error "
				f"error={item.error or 'unknown'}"
			)
	return 0

