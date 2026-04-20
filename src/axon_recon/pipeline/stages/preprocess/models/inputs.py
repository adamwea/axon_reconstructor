from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from axon_reconstructor.pipeline.stg1_preprocessing.constants import PREPROCESS_OUTPUTS_DIRNAME


@dataclass(frozen=True)
class PreprocessPhaseConfig:
	enabled: bool = True
	summary_json_relpath: str = "context/preprocess_phase_summary.json"


@dataclass(frozen=True)
class PreprocessCopySrcToScratchPhaseConfig:
	enabled: bool = False
	requires_use_scratch_root: bool = False
	summary_json_relpath: str = "context/copy_src_to_scratch_summary.json"


@dataclass(frozen=True)
class PreprocessWipeSrcScratchPhaseConfig:
	enabled: bool = False
	dry_run: bool = False
	requires_use_scratch_root: bool = False
	summary_json_relpath: str = "context/wipe_src_scratch_summary.json"


@dataclass(frozen=True)
class PreprocessPlotConfig:
	disable_all_png_diagnostics: bool | None = None
	layouts: bool = True
	concat_trace: bool = True
	segment_traces: bool = True
	output_dir: str | None = None
	epoch_markers_output_dir: str | None = None
	assay_stats_relpath: str = "assay_stats_{stream_id}.txt"
	channel_layouts_subdir: str = "channel_layouts"
	segment_traces_subdir: str = "segment_traces"
	concat_trace_relpath: str = "concat_cluster_reps_{stream_id}.png"
	n_representative_channels: int = 4
	concat_trace_n_reps: int = 4
	segment_trace_n_reps: int = 4
	n_jobs: int | None = None
	trace_downsample_hz: float | None = None
	trace_max_points: int = 150000


@dataclass(frozen=True)
class PreprocessPhaseOutputsConfig:
	save_chunk_duration: str = "1s"
	save_progress_bar: bool = False
	concat_save_n_jobs: int | None = None
	segment_save_n_jobs: int | None = None
	print_n_jobs_used: bool = False


@dataclass(frozen=True)
class PreprocessSegmentsPhaseConfig:
	enabled: bool = True
	summary_json_relpath: str = "context/segment_recordings_summary.json"
	rel_output_root: str = "per_segment_preprocessed"
	plot: PreprocessPlotConfig = field(
		default_factory=lambda: PreprocessPlotConfig(
			concat_trace=False,
		)
	)
	outputs: PreprocessPhaseOutputsConfig = field(default_factory=PreprocessPhaseOutputsConfig)


@dataclass(frozen=True)
class PreprocessConcatenatePreprocessedRecordingsPhaseConfig:
	enabled: bool = True
	summary_json_relpath: str = "context/concatenated_recording_summary.json"
	rel_output_root: str = "preprocessed_recording"
	plot: PreprocessPlotConfig = field(
		default_factory=lambda: PreprocessPlotConfig(
			layouts=False,
			segment_traces=False,
		)
	)
	outputs: PreprocessPhaseOutputsConfig = field(default_factory=PreprocessPhaseOutputsConfig)
	save_common_electrodes: PreprocessPhaseConfig = field(
		default_factory=lambda: PreprocessPhaseConfig(
			enabled=True,
			summary_json_relpath="context/save_common_electrodes_summary.json",
		)
	)


@dataclass(frozen=True)
class PreprocessPhasesConfig:
	copy_src_to_scratch: PreprocessCopySrcToScratchPhaseConfig = field(
		default_factory=PreprocessCopySrcToScratchPhaseConfig
	)
	save_rec_metadata: PreprocessPhaseConfig = field(
		default_factory=lambda: PreprocessPhaseConfig(
			enabled=False,
			summary_json_relpath="context/recording_metadata_summary.json",
		)
	)
	wipe_src_scratch: PreprocessWipeSrcScratchPhaseConfig = field(
		default_factory=PreprocessWipeSrcScratchPhaseConfig
	)
	preprocess_segments: PreprocessSegmentsPhaseConfig = field(
		default_factory=PreprocessSegmentsPhaseConfig
	)
	concatenate_preprocessed_recordings: PreprocessConcatenatePreprocessedRecordingsPhaseConfig = field(
		default_factory=PreprocessConcatenatePreprocessedRecordingsPhaseConfig
	)


@dataclass(frozen=True)
class PreprocessInputs:
	h5_path: Path
	stream_id: str
	mea_output_root: Path
	final_output_root: Path | None = None
	source_h5_path: Path | None = None
	copied_to_scratch: bool = False

	output_rel_root: str = PREPROCESS_OUTPUTS_DIRNAME
	force_restart: bool = False
	force_replot: bool = False
	debug_limit_segments_per_well: int | None = None
	logging_enabled: bool = True
	logging_verbose: bool = True
	logging_file_relpath: str | None = None
	logging_suppress_h5_plugin_messages: bool = False
	logging_phase_dividers: bool = True
	enable_checkpointing: bool = True
	n_jobs: int = 1
	plot_layouts: bool = True
	plot_concat_trace: bool = True
	plot_segment_traces: bool = True
	plot_output_dir: str | None = None
	epoch_markers_output_dir: str | None = None
	assay_stats_relpath: str = "assay_stats_{stream_id}.txt"
	channel_layouts_subdir: str = "channel_layouts"
	segment_traces_subdir: str = "segment_traces"
	concat_trace_relpath: str = "concat_cluster_reps_{stream_id}.png"
	n_representative_channels: int = 4
	concat_trace_n_reps: int = 4
	segment_trace_n_reps: int = 4
	plot_n_jobs: int = 1
	trace_downsample_hz: float | None = None
	trace_max_points: int = 150000
	observability_mode: str = "off"
	observability_output_subdir: str = "run_metadata"
	observability_save_run_manifest: bool = False
	observability_save_event_timeline: bool = False
	observability_save_environment: bool = False
	observability_save_artifact_inventory: bool = False
	observability_save_stage_log: bool = False
	observability_stage_log_relpath: str = "logs/preprocess_pipeline.log"
	temporal_resample_factor: int | None = None
	temporal_resample_rate_hz: int | None = None
	temporal_resample_margin_ms: float = 100.0
	temporal_resample_dtype: str | None = None
	save_recording: bool = True
	overwrite_saved_recording: bool = True
	save_concat_recording: bool = True
	save_segment_recordings: bool = True
	save_chunk_duration: str = "1s"
	save_progress_bar: bool = False
	concat_save_n_jobs: int | None = None
	segment_save_n_jobs: int | None = None
	print_n_jobs_used: bool = False
	phases: PreprocessPhasesConfig = field(default_factory=PreprocessPhasesConfig)
