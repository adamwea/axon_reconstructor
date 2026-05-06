from __future__ import annotations

from pathlib import Path
from typing import Any

from axon_recon.pipeline.output_paths import compute_mea_analysis_output_dir
from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner
from axon_recon.pipeline.stages.reconstruct.core.clear_templates_cache import (
    run_clear_templates_cache_phase,
)
from axon_recon.pipeline.stages.reconstruct.models.inputs import ReconstructionInputs


def run_reconstruct_clear_templates_cache_phase(inputs: ReconstructionInputs) -> dict[str, Any]:
    well_out_dir = compute_mea_analysis_output_dir(
        output_root=inputs.mea_output_root,
        data_file=inputs.h5_path,
        well=inputs.stream_id,
    )
    cfg = inputs.phases.clear_templates_cache
    summary = run_clear_templates_cache_phase(
        well_out_dir=well_out_dir,
        enabled=bool(cfg.enabled),
        keep_merged_per_unit_outputs=bool(cfg.keep_merged_per_unit_outputs),
        keep_full_channels_templates=bool(cfg.keep_full_channels_templates),
        templates_output_rel_root=(
            str(inputs.templates_inputs.output_rel_root)
            if inputs.templates_inputs is not None
            else str(inputs.output_rel_root)
        ),
        logger=reconstruct_runner.LOGGER,
    )
    summary["applied_debug_limits"] = reconstruct_runner._reconstruct_applied_debug_limits(inputs)
    summary_json = (
        well_out_dir
        / str(inputs.output_rel_root)
        / Path(str(cfg.summary_json_relpath)).expanduser()
    )
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    reconstruct_runner.write_json(summary_json, summary)
    summary["summary_json"] = str(summary_json)
    return summary
