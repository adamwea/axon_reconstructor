from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from axon_recon.pipeline.config import load_pipeline_runtime_bundle
from axon_recon.pipeline.config import select_execution_targets
from axon_recon.pipeline.stages.reconstruct.api import run_reconstruct
from axon_recon.pipeline.stages.reconstruct.config import build_reconstruction_inputs_for_target
from axon_recon.pipeline.stages.reconstruct.config import parse_reconstruction_stage_config


CONFIG_PATH = REPO_ROOT / "tools" / "debug" / "debug.runtime.yml"
UNIT_ID = 94
FORCE_RESTART = True
FORCE_REPLOT = False


def main() -> int:
	# Put a breakpoint here when launching from VS Code.
	bundle = load_pipeline_runtime_bundle(config_path=str(CONFIG_PATH))
	targets = select_execution_targets(bundle=bundle)
	if not targets:
		raise RuntimeError("No execution targets found in runtime/data config")

	stage_cfg = parse_reconstruction_stage_config(
		runtime_config=bundle.runtime_config,
		unit_id_override=int(UNIT_ID),
		force_restart_override=bool(FORCE_RESTART),
		force_replot_override=bool(FORCE_REPLOT),
	)

	# Intentionally run a single target directly to bypass distribute_targets/thread fanout.
	target = targets[0]
	inputs = build_reconstruction_inputs_for_target(
		target=target,
		stage_config=stage_cfg,
		unit_workers=1,
	)
	result = run_reconstruct(inputs)

	failed_units = [u for u in result.units if str(getattr(u, "status", "ok")).strip().lower() != "ok"]

	print("stage: reconstruct")
	print("targets_total: 1")
	print(f"targets_succeeded: {0 if failed_units else 1}")
	print(f"targets_failed: {1 if failed_units else 0}")
	if failed_units:
		first = failed_units[0]
		print(
			f"target[1:{target.stream_id}] status=error "
			f"error=reconstruct unit failures: failed={len(failed_units)}/{len(result.units)} "
			f"first_unit={getattr(first, 'unit_id', 'unknown')} "
			f"first_error={getattr(first, 'error', None) or getattr(first, 'status', 'error')}"
		)
		return 1

	recon_out = getattr(result, "reconstruction_out_dir", None)
	units_processed = len(getattr(result, "units", []) or [])
	print(
		f"target[1:{target.stream_id}] status=ok "
		f"reconstruct_out_dir={recon_out} units_processed={units_processed}"
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
