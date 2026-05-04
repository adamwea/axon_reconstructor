from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SpikesortInputs:
	h5_path: Path
	stream_id: str
	mea_output_root: Path
	final_output_root: Path | None = None

	output_rel_root: str = "spikesort_outputs"
	preprocess_concat_recording_relpath: str | None = None
	sort_original_preprocess_concat_recording_relpath: str | None = None
	sort_bootstrapped_concat_recording_relpath: str | None = None
	logging_enabled: bool = True
	logging_verbose: bool = False
	logging_file_relpath: str | None = None
	debug_outputs: bool = False
	debug_mode_enabled: bool = False
	debug_limit_datasets: int | None = None
	debug_limit_wells: int | None = None
	debug_limit_wells_per_dataset: int | None = None
	debug_limit_segments_per_well: int | None = None
	sort_engine: str = "mea_analysis"
	sorter: str = "kilosort4"
	docker_image: str | None = None
	mea_analysis_enabled: bool = True
	mea_analysis_docker_image: str | None = None
	local_spikeinterface_enabled: bool = False
	local_spikeinterface_output_relpath: str = "sorter_output"
	local_spikeinterface_remove_existing_on_force_restart: bool = True
	local_spikeinterface_run_sorter_kwargs: dict[str, Any] | None = field(default=None)
	local_spikeinterface_analyzer_enabled: bool = True
	local_spikeinterface_analyzer_output_relpath: str = "analyzer_output"
	recording_num: str = "rec0000"
	verbose: bool = False
	progress_bar: bool = True

	ks_batch_duration_s: float | None = None
	ks_batch_size: int | None = None
	ks_th_universal: float | None = None
	ks_th_learned: float | None = None
	ks_th_single_ch: float | None = None
	ks_cluster_downsampling: int | None = None
	ks_nearest_chans: int | None = None
	ks_max_channel_distance: float | None = None

	n_jobs: int | None = None
	chunk_duration: str | None = None
	cuda_visible_devices: str | None = None

	run_analyzer: bool = True
	run_reports: bool = True
	plot_enabled: bool = True
	plot_mode: str = "separate"
	plot_debug: bool = False
	raster_sort: str | None = None
	fixed_y: bool = False
	no_curation: bool = False
	export_to_phy: bool = False
	force_rerun_analyzer: bool = False
	summarize_sort_enabled: bool = False
	summarize_sort_emit_logs: bool = True
	summarize_sort_generate_artifacts: bool = False
	um_kwargs: dict[str, Any] | None = field(default=None)
	am_kwargs: dict[str, Any] | None = field(default=None)
	option_kwargs: dict[str, Any] | None = field(default=None)
	sort_enabled: bool = True
	sort_delete_outputs_on_force_restart: bool = False
	sort_use_bootstrapped_concat_binary: bool = False
	sort_use_lazy_source: bool = True
	sort_assert_one_source: bool = False

	force_restart: bool = False
	force_replot: bool = False
	resume_from: str | None = None
	merge_analyzer_compute_sparsity: bool = True
	merge_analyzer_density_mode: str = "auto"
	merge_template_random_spikes_method: str = "default"
	merge_template_random_spikes_percentage: float | None = None
	merge_template_random_spikes_max_spikes_per_unit: int | None = 500
	merge_template_random_spikes_min_spikes_per_unit: int | None = None
	merge_template_random_spikes_log_before_after_spike_counts: bool = False
	merge_template_random_spikes_margin_size: int | None = None
	merge_template_random_spikes_seed: int | None = None
	merge_analyzer_n_jobs: int | None = None
	merge_analyzer_chunk_duration: str | None = None
	merge_analyzer_sparsity_method: str = "radius"
	merge_analyzer_sparsity_radius_um: float | None = 100.0
	merge_analyzer_sparsity_num_channels: int | None = 5
	merge_analyzer_sparsity_threshold: float | None = 5.0
	merge_analyzer_sparsity_peak_sign: str = "neg"
	merge_analyzer_sparsity_num_spikes_for_sparsity: int | None = 100
	merge_analyzer_sparsity_by_property: str | None = None
	merge_analyzer_waveforms_ms_before: float | None = 1.0
	merge_analyzer_waveforms_ms_after: float | None = 2.0
	merge_analyzer_waveforms_dtype: str | None = None
