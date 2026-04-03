from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from axon_reconstructor.pipeline.stg2_spikesorting.runner import SPIKESORTING_OUTPUTS_DIRNAME


@dataclass(frozen=True)
class SpikesortInputs:
	h5_path: Path
	stream_id: str
	mea_output_root: Path
	final_output_root: Path | None = None

	output_rel_root: str = SPIKESORTING_OUTPUTS_DIRNAME
	logging_enabled: bool = True
	logging_verbose: bool = False
	logging_file_relpath: str | None = None
	debug_limit_segments_per_well: int | None = None
	sorter: str = "kilosort4"
	docker_image: str | None = None
	recording_num: str = "rec0000"
	verbose: bool = False

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
	um_kwargs: dict[str, Any] | None = field(default=None)
	am_kwargs: dict[str, Any] | None = field(default=None)
	option_kwargs: dict[str, Any] | None = field(default=None)

	force_restart: bool = False
	force_replot: bool = False
	resume_from: str | None = None
