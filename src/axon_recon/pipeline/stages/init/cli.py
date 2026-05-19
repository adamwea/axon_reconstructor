from __future__ import annotations

import argparse


def _run_from_args(args: argparse.Namespace) -> int:
	"""CLI entry point for the init stage.

	Slice 4 scaffolds the stage but registers no phases; this handler exists
	so the `_STAGE_HANDLERS` dict in `pipeline/cli.py` can route `axon-recon
	stages init` to a no-op runner. Slice 5 will swap in real phase wiring
	once `copy_src_to_scratch` moves in.
	"""

	from ...runner import run_init_from_runtime

	config_path = str(getattr(args, "config", None) or "")
	if not config_path:
		raise SystemExit("--config is required for the init stage")
	result = run_init_from_runtime(
		config_path=config_path,
		force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
		force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
	)
	# No-op stages still report zero failures.
	return 0 if result.failed_targets == 0 else 1
