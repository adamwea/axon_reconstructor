from __future__ import annotations

import json
import logging
from pathlib import Path

from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir
from axon_reconstructor.pipeline.stg2_spikesorting.runner import (
	SPIKESORTING_OUTPUTS_DIRNAME,
	SpikeSortingInputs as LegacySpikeSortingInputs,
	run_spikesorting_stage as run_legacy_spikesorting_stage,
)

from .models.inputs import SpikesortInputs
from .models.results import SpikesortResult


LOGGER = logging.getLogger("axon_recon.spikesort")


def _write_json(path: Path, payload: dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_spikesort_stage(inputs: SpikesortInputs) -> SpikesortResult:
	well_out_dir = compute_mea_analysis_output_dir(
		output_root=inputs.mea_output_root,
		data_file=inputs.h5_path,
		well=inputs.stream_id,
	)

	legacy_inputs = LegacySpikeSortingInputs(
		h5_path=inputs.h5_path,
		stream_id=inputs.stream_id,
		mea_output_root=inputs.mea_output_root,
		log_enabled=bool(inputs.logging_enabled),
		log_verbose=bool(inputs.logging_verbose),
		log_file_override=inputs.logging_file_relpath,
		limit_segments_per_well=(int(inputs.debug_limit_segments_per_well) if inputs.debug_limit_segments_per_well is not None else None),
		sorter=inputs.sorter,
		docker_image=inputs.docker_image,
		recording_num=inputs.recording_num,
		verbose=inputs.verbose,
		ks_batch_duration_s=inputs.ks_batch_duration_s,
		ks_batch_size=inputs.ks_batch_size,
		ks_th_universal=inputs.ks_th_universal,
		ks_th_learned=inputs.ks_th_learned,
		ks_th_single_ch=inputs.ks_th_single_ch,
		ks_cluster_downsampling=inputs.ks_cluster_downsampling,
		ks_nearest_chans=inputs.ks_nearest_chans,
		ks_max_channel_distance=inputs.ks_max_channel_distance,
		n_jobs=inputs.n_jobs,
		chunk_duration=inputs.chunk_duration,
		cuda_visible_devices=inputs.cuda_visible_devices,
		run_analyzer=inputs.run_analyzer,
		run_reports=inputs.run_reports,
		plot_mode=inputs.plot_mode,
		plot_debug=inputs.plot_debug,
		raster_sort=inputs.raster_sort,
		fixed_y=inputs.fixed_y,
		no_curation=inputs.no_curation,
		export_to_phy=inputs.export_to_phy,
		force_rerun_analyzer=inputs.force_rerun_analyzer,
		um_kwargs=(dict(inputs.um_kwargs) if isinstance(inputs.um_kwargs, dict) else None),
		am_kwargs=(dict(inputs.am_kwargs) if isinstance(inputs.am_kwargs, dict) else None),
		option_kwargs=(dict(inputs.option_kwargs) if isinstance(inputs.option_kwargs, dict) else None),
		force_restart=bool(inputs.force_restart or inputs.force_replot),
		resume_from=inputs.resume_from,
	)
	legacy_outputs = run_legacy_spikesorting_stage(inputs=legacy_inputs, logger=LOGGER)

	legacy_out_dir = Path(legacy_outputs.output_dir)
	if str(inputs.output_rel_root).strip() != SPIKESORTING_OUTPUTS_DIRNAME:
		LOGGER.info(
			"Ignoring spikesort output_rel_root=%s; using canonical directory=%s",
			inputs.output_rel_root,
			SPIKESORTING_OUTPUTS_DIRNAME,
		)
	spikesort_out_dir = legacy_out_dir
	summary_json = spikesort_out_dir / "spikesort_summary.json"

	outputs: dict[str, str] = {
		"legacy.spikesort_out_dir": str(legacy_out_dir),
		"recording_dir": str(legacy_outputs.recording_dir),
		"sorter_output_dir": str(legacy_outputs.sorter_output_dir),
		"analyzer_dir": str(legacy_outputs.analyzer_dir),
	}
	if legacy_outputs.merged_sorting_dir is not None:
		outputs["merged_sorting_dir"] = str(legacy_outputs.merged_sorting_dir)
	if legacy_outputs.merged_sorter_output_dir is not None:
		outputs["merged_sorter_output_dir"] = str(legacy_outputs.merged_sorter_output_dir)

	_write_json(
		summary_json,
		{
			"h5_path": str(inputs.h5_path),
			"stream_id": str(inputs.stream_id),
			"well_out_dir": str(well_out_dir),
			"spikesort_out_dir": str(spikesort_out_dir),
			"legacy_spikesort_out_dir": str(legacy_out_dir),
			"output_rel_root": str(inputs.output_rel_root),
			"inputs": {
				"logging_enabled": bool(inputs.logging_enabled),
				"logging_verbose": bool(inputs.logging_verbose),
				"logging_file_relpath": inputs.logging_file_relpath,
				"debug_limit_segments_per_well": (
					int(inputs.debug_limit_segments_per_well)
					if inputs.debug_limit_segments_per_well is not None
					else None
				),
				"sorter": str(inputs.sorter),
				"docker_image": inputs.docker_image,
				"recording_num": str(inputs.recording_num),
				"verbose": bool(inputs.verbose),
				"n_jobs": inputs.n_jobs,
				"chunk_duration": inputs.chunk_duration,
				"cuda_visible_devices": inputs.cuda_visible_devices,
				"run_analyzer": bool(inputs.run_analyzer),
				"run_reports": bool(inputs.run_reports),
				"plot_enabled": bool(inputs.plot_enabled),
				"plot_mode": str(inputs.plot_mode),
				"plot_debug": bool(inputs.plot_debug),
				"raster_sort": inputs.raster_sort,
				"fixed_y": bool(inputs.fixed_y),
				"no_curation": bool(inputs.no_curation),
				"export_to_phy": bool(inputs.export_to_phy),
				"force_restart": bool(inputs.force_restart),
				"force_replot": bool(inputs.force_replot),
				"resume_from": inputs.resume_from,
			},
			"outputs": outputs,
		},
	)

	return SpikesortResult(
		well_out_dir=well_out_dir,
		spikesort_out_dir=spikesort_out_dir,
		summary_json=summary_json,
		outputs=outputs,
	)
