#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUNTIME_CONFIG="${1:-$REPO_ROOT/tools/debug/debug.runtime.yml}"

if [[ ! -f "$RUNTIME_CONFIG" ]]; then
  echo "runtime config not found: $RUNTIME_CONFIG" >&2
  exit 2
fi

PYTHON_BIN="${AXON_RECON_PYTHON:-/home/adamm/miniconda3/envs/axon_recon/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

cd "$REPO_ROOT"

"$PYTHON_BIN" - <<'PY' "$RUNTIME_CONFIG"
from __future__ import annotations

import json
import sys
from pathlib import Path

from axon_recon.pipeline.config import (
    load_pipeline_runtime_bundle,
    resolve_stage_parallelism,
    select_execution_targets,
)
from axon_recon.pipeline.stages.preprocess.config import (
    build_preprocess_inputs_for_target,
    parse_preprocess_stage_config,
)
from axon_recon.pipeline.stages.spikesort.config import (
    build_spikesort_inputs_for_target,
    parse_spikesort_stage_config,
)
from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir
from axon_reconstructor.pipeline.stg1_preprocessing.constants import PREPROCESS_OUTPUTS_DIRNAME
from axon_reconstructor.pipeline.stg2_spikesorting.runner import SPIKESORTING_OUTPUTS_DIRNAME

runtime_config_path = Path(sys.argv[1]).expanduser().resolve()
bundle = load_pipeline_runtime_bundle(config_path=str(runtime_config_path))
targets = select_execution_targets(bundle=bundle)
if not targets:
    raise RuntimeError("No execution targets found from runtime/data config")

target = targets[0]
pre_parallel = resolve_stage_parallelism(bundle=bundle, stage_name="preprocess")
spk_parallel = resolve_stage_parallelism(bundle=bundle, stage_name="spikesort")

pre_stage_cfg = parse_preprocess_stage_config(runtime_config=bundle.runtime_config)
spk_stage_cfg = parse_spikesort_stage_config(runtime_config=bundle.runtime_config)

pre_inputs = build_preprocess_inputs_for_target(
    target=target,
    stage_config=pre_stage_cfg,
    unit_workers=int(pre_parallel.unit_workers),
)
spk_inputs = build_spikesort_inputs_for_target(
    target=target,
    stage_config=spk_stage_cfg,
    unit_workers=int(spk_parallel.unit_workers),
)

well_out_dir = compute_mea_analysis_output_dir(
    output_root=target.mea_output_root,
    data_file=target.h5_path,
    well=target.stream_id,
)

pre_v2_dir = well_out_dir / str(pre_inputs.output_rel_root)
pre_legacy_dir = well_out_dir / PREPROCESS_OUTPUTS_DIRNAME
pre_recording_dir = pre_legacy_dir / "preprocessed_recording"
spk_v2_dir = well_out_dir / str(spk_inputs.output_rel_root)
spk_legacy_dir = well_out_dir / SPIKESORTING_OUTPUTS_DIRNAME

checks = {
    "same_well_out_dir": bool(pre_v2_dir.parent == well_out_dir == spk_v2_dir.parent),
    "spikesort_consumes_preprocess_recording_contract": bool(pre_recording_dir.parent == pre_legacy_dir),
    "legacy_stage_dirs_are_distinct": bool(pre_legacy_dir != spk_legacy_dir),
}

payload = {
    "runtime_config": str(runtime_config_path),
    "target": {
        "dataset_id": target.dataset_id,
        "h5_path": str(target.h5_path),
        "stream_id": target.stream_id,
        "mea_output_root": str(target.mea_output_root),
    },
    "resolved_paths": {
        "well_out_dir": str(well_out_dir),
        "preprocess": {
            "output_rel_root": str(pre_inputs.output_rel_root),
            "v2_out_dir": str(pre_v2_dir),
            "legacy_out_dir": str(pre_legacy_dir),
            "recording_dir_for_spikesort": str(pre_recording_dir),
        },
        "spikesort": {
            "output_rel_root": str(spk_inputs.output_rel_root),
            "v2_out_dir": str(spk_v2_dir),
            "legacy_out_dir": str(spk_legacy_dir),
        },
    },
    "checks": checks,
}

print(json.dumps(payload, indent=2))

if not all(checks.values()):
    raise SystemExit(1)

print("dry-run chain check passed")
PY

printf '\nDry-run chain command completed successfully.\n'
