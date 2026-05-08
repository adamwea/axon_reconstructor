from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..constants import PREPROCESS_OUTPUTS_DIRNAME


DEFAULT_PREPROCESS_PHASE_SEQUENCE: tuple[str, ...] = (
	"copy_src_to_scratch",
	"save_rec_metadata",
	"prepare_raw_binaries",
	"preprocess_segments",
	"plot_segment_traces",
	"plot_segment_channel_layouts",
	"plot_raster_threshold",
	"concat_segments",
	"plot_concat_traces",
	"plot_concat_channel_layout",
	"wipe_src_scratch",
)


@dataclass(frozen=True)
class PreprocessPhaseConfig:
	enabled: bool = True
	summary_json_relpath: str = "context/preprocess_phase_summary.json"
	resource_class: str | None = None


@dataclass(frozen=True)
class PreprocessCopySrcToScratchPhaseConfig:
	enabled: bool = False
	requires_use_scratch_root: bool = False
	summary_json_relpath: str = "context/copy_src_to_scratch_summary.json"
	resource_class: str | None = None


@dataclass(frozen=True)
class PreprocessSaveRecMetadataPhaseConfig:
	enabled: bool = False
	verbose: bool = False
	metadata_source: str = "source_h5"
	debug_mode_enabled: bool = False
	debug_limit_datasets: int | None = None
	debug_limit_wells: int | None = None
	debug_limit_wells_per_dataset: int | None = None
	report_step_timers: bool = False
	summary_json_relpath: str = "context/recording_metadata_summary.json"
	segment_epochs_relpath: str = "segment_epochs.json"
	contiguous_epochs_relpath: str = "continuous_epochs.json"
	sampling_metadata_relpath: str = "sampling_rate_metadata.json"
	common_electrodes_relpath: str = "common_electrodes.npy"
	common_electrodes_summary_json_relpath: str = "context/save_common_electrodes_summary.json"
	resource_class: str | None = None


@dataclass(frozen=True)
class PreprocessPrepareRawBinariesPhaseConfig:
	enabled: bool = True
	summary_json_relpath: str = "context/prepare_raw_binaries_summary.json"
	resource_class: str | None = None
	rel_output_root: str = "raw_binary_recording"
	manifest_relpath: str = "context/raw_binary_manifest.json"
	outputs: PreprocessPhaseOutputsConfig = field(default_factory=lambda: PreprocessPhaseOutputsConfig())


@dataclass(frozen=True)
class PreprocessWipeSrcScratchPhaseConfig:
	enabled: bool = False
	dry_run: bool = False
	requires_use_scratch_root: bool = False
	summary_json_relpath: str = "context/wipe_src_scratch_summary.json"
	resource_class: str | None = None


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
	print_n_jobs_used: bool = False


@dataclass(frozen=True)
class PreprocessSegmentsPhaseConfig:
	enabled: bool = True
	output_mode: str = "lazy"
	lazy_source: str = "scratch"
	summary_json_relpath: str = "context/segment_recordings_summary.json"
	resource_class: str | None = None
	rel_output_root: str = "preprocessed_segments"
	outputs: PreprocessPhaseOutputsConfig = field(default_factory=PreprocessPhaseOutputsConfig)


@dataclass(frozen=True)
class PreprocessPlotSegmentTracesPhaseConfig:
	enabled: bool = True
	summary_json_relpath: str = "context/plot_segment_traces_summary.json"
	resource_class: str | None = None
	plot: PreprocessPlotConfig = field(
		default_factory=lambda: PreprocessPlotConfig(
			concat_trace=False,
		)
	)


@dataclass(frozen=True)
class PreprocessConcatSegmentsPhaseConfig:
	enabled: bool = True
	concatenate_preprocessed_recordings: bool = True
	debug_mode_enabled: bool = False
	debug_limit_datasets: int | None = None
	debug_limit_wells: int | None = None
	debug_limit_wells_per_dataset: int | None = None
	output_mode: str = "binary"
	summary_json_relpath: str = "context/concat_segments_summary.json"
	resource_class: str | None = None
	rel_output_root: str = "concatenated_recording"
	manifest_relpath: str = "context/concat_segments_manifest.json"
	outputs: PreprocessPhaseOutputsConfig = field(default_factory=PreprocessPhaseOutputsConfig)


@dataclass(frozen=True)
class PreprocessPlotConcatTracesPhaseConfig:
	enabled: bool = True
	summary_json_relpath: str = "context/plot_concat_traces_summary.json"
	resource_class: str | None = None
	plot: PreprocessPlotConfig = field(
		default_factory=lambda: PreprocessPlotConfig(
			layouts=False,
			segment_traces=False,
		)
	)


@dataclass(frozen=True)
class PreprocessPlotSegmentChannelLayoutsPhaseConfig:
	enabled: bool = True
	summary_json_relpath: str = "context/plot_segment_channel_layouts_summary.json"
	resource_class: str | None = None
	plot: PreprocessPlotConfig = field(
		default_factory=lambda: PreprocessPlotConfig(
			concat_trace=False,
			segment_traces=False,
		)
	)


@dataclass(frozen=True)
class PreprocessPlotConcatChannelLayoutPhaseConfig:
	enabled: bool = False
	summary_json_relpath: str = "context/plot_concat_channel_layout_summary.json"
	resource_class: str | None = None
	plot: PreprocessPlotConfig = field(
		default_factory=lambda: PreprocessPlotConfig(
			segment_traces=False,
			concat_trace=False,
		)
	)


@dataclass(frozen=True)
class PreprocessPlotRasterThresholdPhaseConfig:
	enabled: bool = False
	debug_mode_enabled: bool = False
	debug_limit_datasets: int | None = None
	debug_limit_wells: int | None = None
	debug_limit_wells_per_dataset: int | None = None
	report_step_timers: bool = False
	summary_json_relpath: str = "context/plot_raster_threshold_summary.json"
	resource_class: str | None = None
	rel_output_root: str = "raster_threshold"


@dataclass(frozen=True)
class PreprocessReportPreprocessingPhaseConfig:
	enabled: bool = False
	summary_json_relpath: str = "context/report_preprocessing_summary.json"
	resource_class: str | None = None
	report_relpath: str = "report/preprocessing_report.md"
	json_summary_relpath: str = "report/preprocessing_report.json"


@dataclass(frozen=True)
class PreprocessCleanupOutputsPhaseConfig:
	enabled: bool = False
	summary_json_relpath: str = "context/cleanup_preprocessing_outputs_summary.json"
	resource_class: str | None = None


PreprocessConcatenateRecordingsPhaseConfig = PreprocessConcatSegmentsPhaseConfig
PreprocessConcatenatePreprocessedRecordingsPhaseConfig = PreprocessConcatSegmentsPhaseConfig


@dataclass(frozen=True)
class PreprocessPhasesConfig:
	copy_src_to_scratch: PreprocessCopySrcToScratchPhaseConfig = field(
		default_factory=PreprocessCopySrcToScratchPhaseConfig
	)
	save_rec_metadata: PreprocessSaveRecMetadataPhaseConfig = field(
		default_factory=lambda: PreprocessSaveRecMetadataPhaseConfig(
			enabled=False,
			verbose=False,
			metadata_source="source_h5",
			debug_mode_enabled=False,
			debug_limit_datasets=None,
			debug_limit_wells=None,
			debug_limit_wells_per_dataset=None,
			report_step_timers=False,
			summary_json_relpath="context/recording_metadata_summary.json",
			segment_epochs_relpath="segment_epochs.json",
			contiguous_epochs_relpath="continuous_epochs.json",
			sampling_metadata_relpath="sampling_rate_metadata.json",
			common_electrodes_relpath="common_electrodes.npy",
			common_electrodes_summary_json_relpath="context/save_common_electrodes_summary.json",
		)
	)
	prepare_raw_binaries: PreprocessPrepareRawBinariesPhaseConfig = field(
		default_factory=PreprocessPrepareRawBinariesPhaseConfig
	)
	wipe_src_scratch: PreprocessWipeSrcScratchPhaseConfig = field(
		default_factory=PreprocessWipeSrcScratchPhaseConfig
	)
	preprocess_segments: PreprocessSegmentsPhaseConfig = field(
		default_factory=PreprocessSegmentsPhaseConfig
	)
	plot_segment_traces: PreprocessPlotSegmentTracesPhaseConfig = field(
		default_factory=PreprocessPlotSegmentTracesPhaseConfig
	)
	plot_segment_channel_layouts: PreprocessPlotSegmentChannelLayoutsPhaseConfig = field(
		default_factory=PreprocessPlotSegmentChannelLayoutsPhaseConfig
	)
	concat_segments: PreprocessConcatSegmentsPhaseConfig = field(
		default_factory=PreprocessConcatSegmentsPhaseConfig
	)
	plot_concat_traces: PreprocessPlotConcatTracesPhaseConfig = field(
		default_factory=PreprocessPlotConcatTracesPhaseConfig
	)
	plot_concat_channel_layout: PreprocessPlotConcatChannelLayoutPhaseConfig = field(
		default_factory=PreprocessPlotConcatChannelLayoutPhaseConfig
	)
	plot_raster_threshold: PreprocessPlotRasterThresholdPhaseConfig = field(
		default_factory=PreprocessPlotRasterThresholdPhaseConfig
	)
	report_preprocessing: PreprocessReportPreprocessingPhaseConfig = field(
		default_factory=PreprocessReportPreprocessingPhaseConfig
	)
	cleanup_preprocessing_outputs: PreprocessCleanupOutputsPhaseConfig = field(
		default_factory=PreprocessCleanupOutputsPhaseConfig
	)

	@property
	def concatenate_recordings(self) -> PreprocessConcatSegmentsPhaseConfig:
		return self.concat_segments

	@property
	def concatenate_preprocessed_recordings(self) -> PreprocessConcatSegmentsPhaseConfig:
		return self.concat_segments


@dataclass(frozen=True)
class PreprocessInputs:
	h5_path: Path
	stream_id: str
	mea_output_root: Path
	final_output_root: Path | None = None
	source_h5_path: Path | None = None
	copied_to_scratch: bool = False
	phase_sequence: tuple[str, ...] = DEFAULT_PREPROCESS_PHASE_SEQUENCE

	output_rel_root: str = PREPROCESS_OUTPUTS_DIRNAME
	force_restart: bool = False
	force_replot: bool = False
	debug_limit_datasets: int | None = None
	debug_limit_wells: int | None = None
	debug_limit_wells_per_dataset: int | None = None
	debug_limit_segments_per_well: int | None = None
	logging_enabled: bool = True
	logging_verbose: bool = True
	logging_file_relpath: str | None = None
	logging_suppress_h5_plugin_messages: bool = False
	logging_phase_dividers: bool = True
	logging_subphase_dividers_to_stdout: bool = True
	enable_checkpointing: bool = True
	n_jobs: int = 1
	runtime_stage_workers: int | None = None
	runtime_well_workers: int | None = None
	runtime_n_jobs_source: str | None = None
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
	print_n_jobs_used: bool = False
	phases: PreprocessPhasesConfig = field(default_factory=PreprocessPhasesConfig)
