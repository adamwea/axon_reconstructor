from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from axon_recon.pipeline.checkpoint import with_checkpoint_marker
from axon_recon.pipeline.execution import install_linux_parent_death_signal
from axon_recon.pipeline.execution.progress import (
    add_current_progress_total,
    advance_current_progress,
)
from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner
from axon_recon.pipeline.stages.reconstruct.models.inputs import ReconstructionInputs
from axon_recon.pipeline.stages.reconstruct.models.results import UnitReconstructionResult


def run_reconstruct_axon_velocity_gtrs_phase(inputs: ReconstructionInputs) -> dict[str, Any]:
    env = reconstruct_runner._prepare_reconstruct_phase_environment(
        inputs=inputs, clear_output_root=True
    )
    summary_json = (
        env.reconstruction_out_dir
        / Path(str(inputs.phases.axon_velocity_gtrs.summary_json_relpath)).expanduser()
    )

    # Dry-run short-circuit (dry_run_rollout slice 5 sub-slice). When set,
    # resolve outputs + report unit count without invoking the GTRS
    # per-unit fanout (which is the heaviest compute in the recon stage
    # after kssynth).
    from axon_recon.pipeline.config import get_dry_run_override

    if get_dry_run_override():
        from axon_recon.pipeline.dry_run import write_dry_run_summary

        # Resolve templates dir for inputs reporting — same fallback the
        # real path uses; non-fatal here.
        templates_dir_reported: Path | None = None
        try:
            from axon_recon.pipeline.stages.reconstruct.templates.runner import (
                _resolve_templates_dirs,
            )

            templates_dir_reported, _ = _resolve_templates_dirs(
                well_out_dir=env.well_out_dir,
                templates_out_dir=env.well_out_dir / "recon_outputs",
            )
        except Exception as exc:
            reconstruct_runner.LOGGER.warning(
                "axon_velocity_gtrs dry-run: templates dir resolution failed: %s", exc
            )

        inputs_resolved: list[dict[str, Any]] = []
        if templates_dir_reported is not None:
            inputs_resolved.append(
                {
                    "name": "merged_templates_dir",
                    "path": str(templates_dir_reported),
                    "exists": bool(templates_dir_reported.exists()),
                }
            )

        outputs_would_produce: list[dict[str, Any]] = [
            {"name": "summary_json", "path": str(summary_json)},
            {"name": "reconstruction_out_dir", "path": str(env.reconstruction_out_dir)},
        ]
        write_dry_run_summary(
            phase_name="reconstruct.axon_velocity_gtrs",
            well_out_dir=env.well_out_dir,
            stage_output_root_dir=env.reconstruction_out_dir,
            summary_json_path=summary_json,
            inputs_resolved=inputs_resolved,
            outputs_would_produce=outputs_would_produce,
            extra_fields={"n_units_resolved": int(len(env.unit_ids))},
        )
        reconstruct_runner.LOGGER.info(
            "reconstruct.axon_velocity_gtrs: dry-run complete; summary at %s",
            str(summary_json),
        )
        return {
            "phase": "reconstruct.axon_velocity_gtrs",
            "status": "dry_run_ok",
            "well_out_dir": str(env.well_out_dir),
            "reconstruction_out_dir": str(env.reconstruction_out_dir),
            "n_units": int(len(env.unit_ids)),
        }

    with with_checkpoint_marker(
        summary_json,
        phase_name="axon_velocity_gtrs",
        stage_name="reconstruct",
    ):
        reconstruct_runner.LOGGER.info(
            "reconstruct.axon_velocity_gtrs phase start: well_out_dir=%s reconstruction_out_dir=%s units=%d applied_debug_limits=%s",
            str(env.well_out_dir),
            str(env.reconstruction_out_dir),
            len(env.unit_ids),
            reconstruct_runner._reconstruct_applied_debug_limits(inputs),
        )
        unit_results, failed_units_summary_json = _run_reconstruct_axon_velocity_gtrs_phase_impl(
            inputs=inputs, env=env
        )
        summary = reconstruct_runner._write_reconstruct_phase_summary(
            phase_name="axon_velocity_gtrs",
            summary_json=summary_json,
            inputs=inputs,
            well_out_dir=env.well_out_dir,
            reconstruction_out_dir=env.reconstruction_out_dir,
            unit_results=unit_results,
            failed_units_summary_json=failed_units_summary_json,
            preserve_stage_reports=env.preserve_stage_reports,
        )
        reconstruct_runner.LOGGER.info(
            "reconstruct.axon_velocity_gtrs wrote summary output: %s",
            str(summary_json),
        )
        reconstruct_runner.LOGGER.info(
            "reconstruct.axon_velocity_gtrs run stats: units_total=%d units_ok=%d units_error=%d",
            int(summary.get("unit_count", 0)),
            int(summary.get("units_ok", 0)),
            int(summary.get("units_error", 0)),
        )
        return summary


def _as_positive_int_or_none(value: Any) -> int | None:
    try:
        parsed = int(value)
    except Exception:
        return None
    if parsed <= 0:
        return None
    return int(parsed)


def _chunk_unit_ids(unit_ids: list[Any], *, batch_size: int) -> list[list[Any]]:
    resolved_batch_size = max(1, int(batch_size))
    return [
        list(unit_ids[idx : idx + resolved_batch_size])
        for idx in range(0, len(unit_ids), resolved_batch_size)
    ]


def _resolve_axon_velocity_gtrs_execution_plan(
    *,
    inputs: ReconstructionInputs,
    unit_ids: list[Any],
) -> tuple[int, int, int, list[list[Any]]]:
    unit_count = len(unit_ids)
    if unit_count <= 0:
        return 1, 1, 1, []
    derived_unit_workers = max(1, int(inputs.n_jobs))
    phase_cfg = inputs.phases.axon_velocity_gtrs
    unit_procs = _as_positive_int_or_none(getattr(phase_cfg, "unit_procs", None))
    if unit_procs is None:
        unit_procs = min(derived_unit_workers, 6)
    unit_procs = max(1, min(int(unit_procs), derived_unit_workers, unit_count))
    unit_batch_size = _as_positive_int_or_none(getattr(phase_cfg, "unit_batch_size", None))
    if unit_batch_size is None:
        unit_batch_size = max(1, (unit_count + unit_procs - 1) // unit_procs)
    batches = _chunk_unit_ids(unit_ids, batch_size=unit_batch_size)
    process_workers = max(1, min(unit_procs, len(batches)))
    return derived_unit_workers, process_workers, int(unit_batch_size), batches


@dataclass(frozen=True)
class _AxonVelocityGtrsBatchInputs:
    inputs: ReconstructionInputs
    reconstruction_out_dir: Path
    merged_units_dir: Path
    full_channels_templates_dir: Path


def _run_axon_velocity_gtrs_batch(
    batch_inputs: _AxonVelocityGtrsBatchInputs,
) -> list[UnitReconstructionResult]:
    return reconstruct_runner.run_axon_velocity_gtrs_core_phase(
        inputs=batch_inputs.inputs,
        reconstruction_out_dir=batch_inputs.reconstruction_out_dir,
        merged_units_dir=batch_inputs.merged_units_dir,
        full_channels_templates_dir=batch_inputs.full_channels_templates_dir,
        unit_ids=list(batch_inputs.inputs.unit_ids or []),
        import_axon_velocity_fn=reconstruct_runner.import_axon_velocity,
        load_templates_for_unit_fn=reconstruct_runner.load_templates_for_unit,
        compute_graph_tracking_fn=reconstruct_runner.compute_graph_tracking,
        compute_raw_branches_payload_fn=reconstruct_runner.compute_raw_branches_payload,
        compute_branches_with_polyline_fn=reconstruct_runner.compute_branches_with_polyline,
        compute_detection_filter_payload_fn=reconstruct_runner.compute_detection_filter_payload,
        compute_kurtosis_filter_payload_fn=reconstruct_runner.compute_kurtosis_filter_payload,
        compute_peak_std_filter_payload_fn=reconstruct_runner.compute_peak_std_filter_payload,
        compute_delay_filter_payload_fn=reconstruct_runner.compute_delay_filter_payload,
        compute_all_filters_payload_fn=reconstruct_runner.compute_all_filters_payload,
        compute_heuristics_payload_fn=reconstruct_runner.compute_heuristics_payload,
        compute_gtr_json_payload_fn=reconstruct_runner.compute_gtr_json_payload,
        write_unit_channel_selection_diagnostic_figure_fn=reconstruct_runner.write_unit_channel_selection_diagnostic_figure,
        write_unit_axon_reconstruction_diagnostic_figure_fn=reconstruct_runner.write_unit_axon_reconstruction_diagnostic_figure,
        read_json_fn=reconstruct_runner.read_json,
        write_json_fn=reconstruct_runner.write_json,
        resolve_unit_output_paths_fn=reconstruct_runner.resolve_unit_output_paths,
        is_empty_signal_selection_error_fn=reconstruct_runner._is_empty_signal_selection_error,
        is_expected_reconstruct_unit_failure_fn=reconstruct_runner._is_expected_reconstruct_unit_failure,
        normalize_template_for_tracking_fn=reconstruct_runner._normalize_template_for_tracking,
        logger=reconstruct_runner.LOGGER,
    )


def _run_reconstruct_axon_velocity_gtrs_batches(
    *,
    inputs: ReconstructionInputs,
    env: Any,
) -> list[UnitReconstructionResult]:
    derived_unit_workers, unit_procs, unit_batch_size, batches = (
        _resolve_axon_velocity_gtrs_execution_plan(
            inputs=inputs,
            unit_ids=env.unit_ids,
        )
    )
    reconstruct_runner.LOGGER.info(
        "reconstruct.axon_velocity_gtrs execution plan: requested_units=%d derived_unit_workers=%d unit_procs=%d unit_batch_size=%d unit_batches=%d",
        len(env.unit_ids),
        int(derived_unit_workers),
        int(unit_procs),
        int(unit_batch_size),
        len(batches),
    )
    if len(batches) <= 1 or unit_procs <= 1:
        return reconstruct_runner.run_axon_velocity_gtrs_core_phase(
            inputs=inputs,
            reconstruction_out_dir=env.reconstruction_out_dir,
            merged_units_dir=env.merged_units_dir,
            full_channels_templates_dir=env.full_channels_templates_dir,
            unit_ids=env.unit_ids,
            import_axon_velocity_fn=reconstruct_runner.import_axon_velocity,
            load_templates_for_unit_fn=reconstruct_runner.load_templates_for_unit,
            compute_graph_tracking_fn=reconstruct_runner.compute_graph_tracking,
            compute_raw_branches_payload_fn=reconstruct_runner.compute_raw_branches_payload,
            compute_branches_with_polyline_fn=reconstruct_runner.compute_branches_with_polyline,
            compute_detection_filter_payload_fn=reconstruct_runner.compute_detection_filter_payload,
            compute_kurtosis_filter_payload_fn=reconstruct_runner.compute_kurtosis_filter_payload,
            compute_peak_std_filter_payload_fn=reconstruct_runner.compute_peak_std_filter_payload,
            compute_delay_filter_payload_fn=reconstruct_runner.compute_delay_filter_payload,
            compute_all_filters_payload_fn=reconstruct_runner.compute_all_filters_payload,
            compute_heuristics_payload_fn=reconstruct_runner.compute_heuristics_payload,
            compute_gtr_json_payload_fn=reconstruct_runner.compute_gtr_json_payload,
            write_unit_channel_selection_diagnostic_figure_fn=reconstruct_runner.write_unit_channel_selection_diagnostic_figure,
            write_unit_axon_reconstruction_diagnostic_figure_fn=reconstruct_runner.write_unit_axon_reconstruction_diagnostic_figure,
            read_json_fn=reconstruct_runner.read_json,
            write_json_fn=reconstruct_runner.write_json,
            resolve_unit_output_paths_fn=reconstruct_runner.resolve_unit_output_paths,
            is_empty_signal_selection_error_fn=reconstruct_runner._is_empty_signal_selection_error,
            is_expected_reconstruct_unit_failure_fn=reconstruct_runner._is_expected_reconstruct_unit_failure,
            normalize_template_for_tracking_fn=reconstruct_runner._normalize_template_for_tracking,
            logger=reconstruct_runner.LOGGER,
        )

    batch_inputs_list = [
        _AxonVelocityGtrsBatchInputs(
            inputs=replace(inputs, n_jobs=1, unit_ids=list(batch_unit_ids)),
            reconstruction_out_dir=env.reconstruction_out_dir,
            merged_units_dir=env.merged_units_dir,
            full_channels_templates_dir=env.full_channels_templates_dir,
        )
        for batch_unit_ids in batches
    ]
    batch_results: list[UnitReconstructionResult] = []
    try:
        add_current_progress_total(len(env.unit_ids))
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=unit_procs,
            initializer=install_linux_parent_death_signal,
        ) as pool:
            futures = {
                pool.submit(_run_axon_velocity_gtrs_batch, batch_inputs): list(
                    batch_inputs.inputs.unit_ids or []
                )
                for batch_inputs in batch_inputs_list
            }
            completed = 0
            completed_units = 0
            total_batches = len(futures)
            for future in concurrent.futures.as_completed(futures):
                batch_result = future.result()
                batch_results.extend(batch_result)
                completed += 1
                completed_batch_units = len(batch_result)
                completed_units += completed_batch_units
                advance_current_progress(completed_batch_units)
                reconstruct_runner.LOGGER.info(
                    "reconstruct.axon_velocity_gtrs unified progress: %d/%d units completed (%d/%d batches)",
                    completed_units,
                    len(env.unit_ids),
                    completed,
                    total_batches,
                )
    except Exception as exc:
        reconstruct_runner.LOGGER.warning(
            "reconstruct.axon_velocity_gtrs process pool execution failed, falling back to in-process execution: %s",
            exc,
        )
        return reconstruct_runner.run_axon_velocity_gtrs_core_phase(
            inputs=inputs,
            reconstruction_out_dir=env.reconstruction_out_dir,
            merged_units_dir=env.merged_units_dir,
            full_channels_templates_dir=env.full_channels_templates_dir,
            unit_ids=env.unit_ids,
            import_axon_velocity_fn=reconstruct_runner.import_axon_velocity,
            load_templates_for_unit_fn=reconstruct_runner.load_templates_for_unit,
            compute_graph_tracking_fn=reconstruct_runner.compute_graph_tracking,
            compute_raw_branches_payload_fn=reconstruct_runner.compute_raw_branches_payload,
            compute_branches_with_polyline_fn=reconstruct_runner.compute_branches_with_polyline,
            compute_detection_filter_payload_fn=reconstruct_runner.compute_detection_filter_payload,
            compute_kurtosis_filter_payload_fn=reconstruct_runner.compute_kurtosis_filter_payload,
            compute_peak_std_filter_payload_fn=reconstruct_runner.compute_peak_std_filter_payload,
            compute_delay_filter_payload_fn=reconstruct_runner.compute_delay_filter_payload,
            compute_all_filters_payload_fn=reconstruct_runner.compute_all_filters_payload,
            compute_heuristics_payload_fn=reconstruct_runner.compute_heuristics_payload,
            compute_gtr_json_payload_fn=reconstruct_runner.compute_gtr_json_payload,
            write_unit_channel_selection_diagnostic_figure_fn=reconstruct_runner.write_unit_channel_selection_diagnostic_figure,
            write_unit_axon_reconstruction_diagnostic_figure_fn=reconstruct_runner.write_unit_axon_reconstruction_diagnostic_figure,
            read_json_fn=reconstruct_runner.read_json,
            write_json_fn=reconstruct_runner.write_json,
            resolve_unit_output_paths_fn=reconstruct_runner.resolve_unit_output_paths,
            is_empty_signal_selection_error_fn=reconstruct_runner._is_empty_signal_selection_error,
            is_expected_reconstruct_unit_failure_fn=reconstruct_runner._is_expected_reconstruct_unit_failure,
            normalize_template_for_tracking_fn=reconstruct_runner._normalize_template_for_tracking,
            logger=reconstruct_runner.LOGGER,
            progress_total_already_added=True,
        )

    batch_results.sort(key=lambda item: str(item.unit_id))
    return batch_results


def _run_reconstruct_axon_velocity_gtrs_phase_impl(
    *,
    inputs: ReconstructionInputs,
    env: Any,
) -> tuple[list[UnitReconstructionResult], Path | None]:
    unit_results = _run_reconstruct_axon_velocity_gtrs_batches(inputs=inputs, env=env)
    return reconstruct_runner._cleanup_failed_reconstruct_unit_outputs(
        reconstruction_out_dir=env.reconstruction_out_dir,
        inputs=inputs,
        unit_results=unit_results,
    )
