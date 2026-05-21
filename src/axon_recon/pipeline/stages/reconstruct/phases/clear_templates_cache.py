from __future__ import annotations

from pathlib import Path
from typing import Any

from axon_recon.pipeline.output_paths import compute_mea_analysis_output_dir
from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner
# Import the SUBMODULE rather than the function directly so monkeypatch.setattr
# on `core.clear_templates_cache.run_clear_templates_cache_phase` is observed
# by this phase module — the call-time attribute lookup picks up the patched
# version (see test_runner.py
# `test_reconstruct_clear_templates_cache_phase_uses_templates_output_root`
# which patches the core module and relies on the lookup hitting the patch).
from axon_recon.pipeline.stages.reconstruct.core import clear_templates_cache as _core_clear_cache
from axon_recon.pipeline.stages.reconstruct.models.inputs import ReconstructionInputs


def run_reconstruct_clear_templates_cache_phase(inputs: ReconstructionInputs) -> dict[str, Any]:
    from axon_recon.pipeline.config import get_dry_run_override

    well_out_dir = compute_mea_analysis_output_dir(
        output_root=inputs.mea_output_root,
        data_file=inputs.h5_path,
        well=inputs.stream_id,
    )
    cfg = inputs.phases.clear_templates_cache
    summary_json = (
        well_out_dir
        / str(inputs.output_rel_root)
        / Path(str(cfg.summary_json_relpath)).expanduser()
    )
    templates_output_rel_root = (
        str(inputs.templates_inputs.output_rel_root)
        if inputs.templates_inputs is not None
        else str(inputs.output_rel_root)
    )
    cache_dir = well_out_dir / templates_output_rel_root / "cache" / "templates"

    # Dry-run short-circuit (dry_run_rollout slice 5 final sub-slice).
    # Skip the rmtree(cache_dir) entirely; report it as a would-be-output
    # so caller knows what would be deleted.
    if get_dry_run_override():
        from axon_recon.pipeline.dry_run import write_dry_run_summary

        write_dry_run_summary(
            phase_name="reconstruct.clear_templates_cache",
            well_out_dir=well_out_dir,
            stage_output_root_dir=well_out_dir / str(inputs.output_rel_root),
            summary_json_path=summary_json,
            inputs_resolved=[
                {
                    "name": "cache_dir",
                    "path": str(cache_dir),
                    "exists": bool(cache_dir.exists()),
                },
            ],
            outputs_would_produce=[
                {"name": "summary_json", "path": str(summary_json)},
                {"name": "cache_dir_would_be_removed", "path": str(cache_dir)},
            ],
            extra_fields={
                "keep_merged_per_unit_outputs": bool(cfg.keep_merged_per_unit_outputs),
                "keep_full_channels_templates": bool(cfg.keep_full_channels_templates),
                "phase_enabled": bool(cfg.enabled),
            },
        )
        reconstruct_runner.LOGGER.info(
            "reconstruct.clear_templates_cache: dry-run complete; summary at %s",
            str(summary_json),
        )
        return {
            "phase": "reconstruct.clear_templates_cache",
            "status": "dry_run_ok",
            "well_out_dir": str(well_out_dir),
            "cache_dir": str(cache_dir),
        }

    summary = _core_clear_cache.run_clear_templates_cache_phase(
        well_out_dir=well_out_dir,
        enabled=bool(cfg.enabled),
        keep_merged_per_unit_outputs=bool(cfg.keep_merged_per_unit_outputs),
        keep_full_channels_templates=bool(cfg.keep_full_channels_templates),
        templates_output_rel_root=templates_output_rel_root,
        logger=reconstruct_runner.LOGGER,
    )
    summary["applied_debug_limits"] = reconstruct_runner._reconstruct_applied_debug_limits(inputs)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    reconstruct_runner.write_json(summary_json, summary)
    summary["summary_json"] = str(summary_json)
    return summary
