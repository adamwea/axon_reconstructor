from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from axon_recon.runtime_config import RuntimeConfig
from axon_recon.pipeline.execution.context import ExecutionTarget
from axon_recon.pipeline.stages.preprocess.models.inputs import DEFAULT_PREPROCESS_PHASE_SEQUENCE
from axon_recon.pipeline.stages.preprocess.config import (
    build_preprocess_inputs_for_target,
    load_preprocess_inputs_from_runtime,
    parse_preprocess_stage_config,
)


def test_parse_preprocess_stage_config_defaults() -> None:
    cfg = RuntimeConfig({"stages": {"preprocess": {}}})

    parsed = parse_preprocess_stage_config(runtime_config=cfg)

    assert parsed.output_rel_root == "preprocess_outputs"
    assert parsed.phase_sequence == DEFAULT_PREPROCESS_PHASE_SEQUENCE
    assert parsed.force_restart is False
    assert parsed.force_replot is False
    assert parsed.debug_limit_datasets is None
    assert parsed.debug_limit_wells is None
    assert parsed.debug_limit_wells_per_dataset is None
    assert parsed.debug_limit_segments_per_well is None
    assert parsed.logging_enabled is True
    assert parsed.logging_verbose is True
    assert parsed.logging_file_relpath is None
    assert parsed.logging_suppress_h5_plugin_messages is False
    assert parsed.logging_phase_dividers is True
    assert parsed.enable_checkpointing is True
    assert parsed.n_jobs is None
    assert parsed.plot_layouts is True
    assert parsed.plot_concat_trace is True
    assert parsed.plot_segment_traces is True
    assert parsed.plot_output_dir is None
    assert parsed.epoch_markers_output_dir is None
    assert parsed.assay_stats_relpath == "assay_stats_{stream_id}.txt"
    assert parsed.channel_layouts_subdir == "channel_layouts"
    assert parsed.segment_traces_subdir == "segment_traces"
    assert parsed.concat_trace_relpath == "concat_cluster_reps_{stream_id}.png"
    assert parsed.n_representative_channels == 4
    assert parsed.concat_trace_n_reps == 4
    assert parsed.segment_trace_n_reps == 4
    assert parsed.plot_n_jobs is None
    assert parsed.trace_downsample_hz is None
    assert parsed.trace_max_points == 150000
    assert parsed.observability_mode == "off"
    assert parsed.observability_output_subdir == "run_metadata"
    assert parsed.observability_save_run_manifest is False
    assert parsed.observability_save_event_timeline is False
    assert parsed.observability_save_environment is False
    assert parsed.observability_save_artifact_inventory is False
    assert parsed.observability_save_stage_log is False
    assert parsed.observability_stage_log_relpath == "logs/preprocess_pipeline.log"
    assert parsed.temporal_resample_factor is None
    assert parsed.temporal_resample_rate_hz is None
    assert parsed.temporal_resample_margin_ms == 100.0
    assert parsed.temporal_resample_dtype is None
    assert parsed.save_recording is True
    assert parsed.overwrite_saved_recording is True
    assert parsed.save_concat_recording is True
    assert parsed.save_segment_recordings is True
    assert parsed.save_chunk_duration == "1s"
    assert parsed.save_progress_bar is False
    assert parsed.print_n_jobs_used is False
    assert parsed.phases.copy_src_to_scratch.enabled is False
    assert parsed.phases.copy_src_to_scratch.summary_json_relpath == "context/copy_src_to_scratch_summary.json"
    assert parsed.phases.save_rec_metadata.enabled is False
    assert parsed.phases.save_rec_metadata.verbose is False
    assert parsed.phases.save_rec_metadata.metadata_source == "source_h5"
    assert parsed.phases.save_rec_metadata.summary_json_relpath == "context/recording_metadata_summary.json"
    assert parsed.phases.save_rec_metadata.segment_epochs_relpath == "segment_epochs.json"
    assert parsed.phases.save_rec_metadata.contiguous_epochs_relpath == "continuous_epochs.json"
    assert parsed.phases.save_rec_metadata.sampling_metadata_relpath == "sampling_rate_metadata.json"
    assert parsed.phases.prepare_raw_binaries.enabled is True
    assert parsed.phases.prepare_raw_binaries.summary_json_relpath == "context/prepare_raw_binaries_summary.json"
    assert parsed.phases.prepare_raw_binaries.rel_output_root == "raw_binary_recording"
    assert parsed.phases.prepare_raw_binaries.manifest_relpath == "context/raw_binary_manifest.json"
    assert parsed.phases.wipe_src_scratch.enabled is False
    assert parsed.phases.wipe_src_scratch.dry_run is False
    assert parsed.phases.wipe_src_scratch.requires_use_scratch_root is False
    assert parsed.phases.wipe_src_scratch.summary_json_relpath == "context/wipe_src_scratch_summary.json"
    assert parsed.phases.preprocess_segments.enabled is True
    assert parsed.phases.preprocess_segments.output_mode == "lazy"
    assert parsed.phases.preprocess_segments.lazy_source == "scratch"
    assert parsed.phases.preprocess_segments.summary_json_relpath == "context/segment_recordings_summary.json"
    assert parsed.phases.preprocess_segments.rel_output_root == "preprocessed_segments"
    assert parsed.phases.plot_segment_traces.enabled is True
    assert parsed.phases.plot_segment_traces.summary_json_relpath == "context/plot_segment_traces_summary.json"
    assert parsed.phases.plot_segment_channel_layouts.enabled is True
    assert parsed.phases.plot_segment_channel_layouts.summary_json_relpath == "context/plot_segment_channel_layouts_summary.json"
    assert parsed.phases.concat_segments.enabled is True
    assert parsed.phases.concat_segments.concatenate_preprocessed_recordings is True
    assert parsed.phases.concat_segments.debug_mode_enabled is False
    assert parsed.phases.concat_segments.debug_limit_datasets is None
    assert parsed.phases.concat_segments.debug_limit_wells is None
    assert parsed.phases.concat_segments.debug_limit_wells_per_dataset is None
    assert parsed.phases.concat_segments.output_mode == "binary"
    assert parsed.phases.concat_segments.summary_json_relpath == "context/concat_segments_summary.json"
    assert parsed.phases.concat_segments.rel_output_root == "concatenated_recording"
    assert parsed.phases.plot_concat_traces.enabled is True
    assert parsed.phases.plot_concat_traces.summary_json_relpath == "context/plot_concat_traces_summary.json"
    assert parsed.phases.plot_concat_channel_layout.enabled is False
    assert parsed.phases.plot_concat_channel_layout.summary_json_relpath == "context/plot_concat_channel_layout_summary.json"
    assert parsed.phases.plot_raster_threshold.enabled is False
    assert parsed.phases.plot_raster_threshold.debug_mode_enabled is False
    assert parsed.phases.plot_raster_threshold.debug_limit_datasets is None
    assert parsed.phases.plot_raster_threshold.debug_limit_wells is None
    assert parsed.phases.plot_raster_threshold.debug_limit_wells_per_dataset is None
    assert parsed.phases.plot_raster_threshold.report_step_timers is False
    assert parsed.phases.plot_raster_threshold.summary_json_relpath == "context/plot_raster_threshold_summary.json"
    assert parsed.phases.report_preprocessing.enabled is False
    assert parsed.phases.report_preprocessing.summary_json_relpath == "context/report_preprocessing_summary.json"
    assert parsed.phases.cleanup_preprocessing_outputs.enabled is False
    assert parsed.phases.cleanup_preprocessing_outputs.summary_json_relpath == "context/cleanup_preprocessing_outputs_summary.json"
    assert parsed.phases.save_rec_metadata.common_electrodes_relpath == "common_electrodes.npy"
    assert (
        parsed.phases.save_rec_metadata.common_electrodes_summary_json_relpath
        == "context/save_common_electrodes_summary.json"
    )


def test_parse_preprocess_stage_config_force_overrides_take_precedence() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "preprocess": {
                    "execution": {
                        "force_restart": False,
                        "force_replot": True,
                    },
                    "outputs": {
                        "output_rel_root": "/preprocess_v2",
                    },
                }
            }
        }
    )

    parsed = parse_preprocess_stage_config(
        runtime_config=cfg,
        force_restart_override=True,
        force_replot_override=False,
    )

    assert parsed.force_restart is True
    assert parsed.force_replot is False
    assert parsed.output_rel_root == "preprocess_v2"


def test_parse_preprocess_stage_config_reads_debug_limits() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "preprocess": {
                    "debug": {
                        "limit_wells": 3,
                        "limit_segments_per_well": 2,
                    }
                }
            }
        }
    )

    parsed = parse_preprocess_stage_config(runtime_config=cfg)

    assert parsed.debug_limit_wells == 3
    assert parsed.debug_limit_segments_per_well == 2


def test_parse_preprocess_stage_config_reads_plot_block_knobs() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "preprocess": {
                    "plot": {
                        "concat_trace": False,
                        "segment_traces": False,
                        "output_dir": "plots/{stream_id}",
                        "epoch_markers_output_dir": "epochs/{stream_id}",
                        "assay_stats_relpath": "logs/assay_stats_{stream_id}.txt",
                        "channel_layouts_subdir": "layouts",
                        "segment_traces_subdir": "trace_segments",
                        "concat_trace_relpath": "trace_concat/concat_{stream_id}.png",
                        "n_representative_channels": 6,
                        "n_jobs": 3,
                        "trace_downsample_hz": 250.0,
                        "trace_max_points": 42000,
                    }
                }
            }
        }
    )

    parsed = parse_preprocess_stage_config(runtime_config=cfg)

    assert parsed.plot_concat_trace is False
    assert parsed.plot_segment_traces is False
    assert parsed.plot_output_dir == "plots/{stream_id}"
    assert parsed.epoch_markers_output_dir == "epochs/{stream_id}"
    assert parsed.assay_stats_relpath == "logs/assay_stats_{stream_id}.txt"
    assert parsed.channel_layouts_subdir == "layouts"
    assert parsed.segment_traces_subdir == "trace_segments"
    assert parsed.concat_trace_relpath == "trace_concat/concat_{stream_id}.png"
    assert parsed.n_representative_channels == 6
    assert parsed.concat_trace_n_reps == 6
    assert parsed.segment_trace_n_reps == 6
    assert parsed.plot_n_jobs == 3
    assert parsed.trace_downsample_hz == 250.0
    assert parsed.trace_max_points == 42000


def test_parse_preprocess_stage_config_reads_nested_trace_rep_controls() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "preprocess": {
                    "plot": {
                        "concat_trace": {
                            "enabled": True,
                            "n_reps": 2,
                        },
                        "segment_traces": True,
                        "per_segment_traces": {
                            "n_reps": 5,
                        },
                    }
                }
            }
        }
    )

    parsed = parse_preprocess_stage_config(runtime_config=cfg)

    assert parsed.plot_concat_trace is True
    assert parsed.plot_segment_traces is True
    assert parsed.concat_trace_n_reps == 2
    assert parsed.segment_trace_n_reps == 5


def test_parse_preprocess_stage_config_master_plot_override_disables_all_pngs() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "preprocess": {
                    "execution": {
                        "plot_layouts": True,
                        "plot_concat_trace": True,
                        "plot_segment_traces": True,
                    },
                    "plot": {
                        "disable_all_png_diagnostics": True,
                        "layouts": True,
                        "concat_trace": {
                            "enabled": True,
                            "n_reps": 3,
                        },
                        "per_segment_traces": {
                            "enabled": True,
                            "n_reps": 2,
                        },
                    },
                }
            }
        }
    )

    parsed = parse_preprocess_stage_config(runtime_config=cfg)

    assert parsed.plot_layouts is False
    assert parsed.plot_concat_trace is False
    assert parsed.plot_segment_traces is False


def test_parse_preprocess_stage_config_master_plot_override_enables_all_pngs() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "preprocess": {
                    "execution": {
                        "plot_layouts": False,
                        "plot_concat_trace": False,
                        "plot_segment_traces": False,
                    },
                    "plot": {
                        "disable_all_png_diagnostics": False,
                        "layouts": False,
                        "concat_trace": False,
                        "segment_traces": False,
                        "per_segment_traces": {
                            "enabled": False,
                            "n_reps": 1,
                        },
                    },
                }
            }
        }
    )

    parsed = parse_preprocess_stage_config(runtime_config=cfg)

    assert parsed.plot_layouts is True
    assert parsed.plot_concat_trace is True
    assert parsed.plot_segment_traces is True


def test_parse_preprocess_stage_config_supports_n_reps_per_segment_alias() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "preprocess": {
                    "plot": {
                        "n_reps_per_segment": 5,
                    }
                }
            }
        }
    )

    parsed = parse_preprocess_stage_config(runtime_config=cfg)

    assert parsed.n_representative_channels == 5
    assert parsed.concat_trace_n_reps == 5
    assert parsed.segment_trace_n_reps == 5


def test_parse_preprocess_stage_config_normalizes_legacy_plot_output_paths() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "preprocess": {
                    "plot": {
                        "output_dir": "stg1_preprocess_outputs",
                        "epoch_markers_output_dir": "stg1_preprocess_outputs/epochs",
                    }
                }
            }
        }
    )

    parsed = parse_preprocess_stage_config(runtime_config=cfg)

    assert parsed.plot_output_dir == "preprocess_outputs"
    assert parsed.epoch_markers_output_dir == "preprocess_outputs/epochs"


def test_parse_preprocess_stage_config_reads_logging_block_knobs() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "preprocess": {
                    "logging": {
                        "enabled": False,
                        "verbose": False,
                        "file_relpath": "logs/custom_preprocess.log",
                        "suppress_h5_plugin_messages": True,
                        "phase_dividers": False,
                    }
                }
            }
        }
    )

    parsed = parse_preprocess_stage_config(runtime_config=cfg)

    assert parsed.logging_enabled is False
    assert parsed.logging_verbose is False
    assert parsed.logging_file_relpath == "logs/custom_preprocess.log"
    assert parsed.logging_suppress_h5_plugin_messages is True
    assert parsed.logging_phase_dividers is False


def test_parse_preprocess_stage_config_reads_save_output_knobs() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "preprocess": {
                    "execution": {
                        "save_recording": True,
                    },
                    "outputs": {
                        "save_concat_recording": True,
                        "save_segment_recordings": False,
                        "save_chunk_duration": "2s",
                        "save_progress_bar": True,
                        "print_n_jobs_used": True,
                    },
                }
            }
        }
    )

    parsed = parse_preprocess_stage_config(runtime_config=cfg)

    assert parsed.save_recording is True
    assert parsed.save_concat_recording is True
    assert parsed.save_segment_recordings is False
    assert parsed.save_chunk_duration == "2s"
    assert parsed.save_progress_bar is True
    assert parsed.print_n_jobs_used is True


def test_parse_preprocess_stage_config_reads_phase_overrides() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "preprocess": {
                    "phases": {
                        "copy_src_to_scratch": {
                            "enabled": True,
                            "requires_use_scratch_root": True,
                            "summary_json_relpath": "context/custom_copy_summary.json",
                        },
                        "save_rec_metadata": {
                            "enabled": True,
                            "verbose": True,
                            "metadata_source": "scratch_copy",
                            "debug_mode": {
                                "enabled": True,
                                "limit_datasets": 1,
                                "limit_wells": 3,
                                "limit_wells_per_dataset": 2,
                                "report_step_timers": True,
                            },
                            "summary_json_relpath": "context/custom_recording_metadata_summary.json",
                            "segment_epochs_relpath": "meta/segments.json",
                            "contiguous_epochs_relpath": "meta/contiguous.json",
                            "sampling_metadata_relpath": "meta/sampling.json",
                            "common_electrodes_summary_json_relpath": "context/custom_common_summary.json",
                        },
                        "prepare_raw_binaries": {
                            "enabled": True,
                            "summary_json_relpath": "context/custom_prepare_raw_binaries_summary.json",
                            "rel_output_root": "raw_binary_recording",
                            "manifest_relpath": "context/custom_raw_binary_manifest.json",
                            "outputs": {
                                "save_chunk_duration": "3s",
                                "save_progress_bar": True,
                            },
                        },
                        "wipe_src_scratch": {
                            "enabled": True,
                            "dry_run": True,
                            "requires_use_scratch_root": True,
                            "summary_json_relpath": "context/custom_wipe_src_scratch_summary.json",
                        },
                        "preprocess_segments": {
                            "enabled": False,
                            "output_mode": "binary",
                            "lazy_source": "src",
                            "summary_json_relpath": "context/custom_segments_summary.json",
                            "rel_output_root": "preprocessed_segments",
                            "outputs": {
                                "save_chunk_duration": "2s",
                                "save_progress_bar": True,
                                "print_n_jobs_used": True,
                            },
                        },
                        "plot_segment_traces": {
                            "enabled": True,
                            "summary_json_relpath": "context/custom_plot_segment_traces_summary.json",
                            "plot": {
                                "layouts": False,
                                "segment_traces": False,
                                "output_dir": "plots/segments/{stream_id}",
                            },
                        },
                        "plot_segment_channel_layouts": {
                            "enabled": True,
                            "summary_json_relpath": "context/custom_plot_segment_channel_layouts_summary.json",
                            "plot": {
                                "layouts": True,
                                "channel_layouts_subdir": "custom_segment_layouts",
                            },
                        },
                        "concat_segments": {
                            "enabled": True,
                            "concatenate_preprocessed_recordings": False,
                            "debug_mode": {
                                "enabled": True,
                                "limit_datasets": 1,
                                "limit_wells": 2,
                                "limit_wells_per_dataset": 2,
                            },
                            "output_mode": "lazy",
                            "summary_json_relpath": "context/custom_concat_segments_summary.json",
                            "rel_output_root": "concatenated_recording",
                            "outputs": {},
                        },
                        "plot_concat_traces": {
                            "enabled": True,
                            "summary_json_relpath": "context/custom_plot_concat_traces_summary.json",
                            "plot": {
                                "concat_trace": {
                                    "enabled": False,
                                    "n_reps": 3,
                                }
                            },
                        },
                        "plot_concat_channel_layout": {
                            "enabled": True,
                            "summary_json_relpath": "context/custom_plot_concat_channel_layout_summary.json",
                            "plot": {
                                "layouts": True,
                                "channel_layouts_subdir": "custom_concat_layouts",
                            },
                        },
                        "plot_raster_threshold": {
                            "enabled": True,
                            "debug_mode": {
                                "enabled": True,
                                "limit_datasets": 1,
                                "limit_wells": 1,
                                "limit_wells_per_dataset": 1,
                                "report_step_timers": True,
                            },
                            "summary_json_relpath": "context/custom_plot_raster_threshold_summary.json",
                            "rel_output_root": "raster_threshold_outputs",
                        },
                        "report_preprocessing": {
                            "enabled": True,
                            "summary_json_relpath": "context/custom_report_preprocessing_summary.json",
                            "report_relpath": "report/custom_preprocessing_report.md",
                            "json_summary_relpath": "report/custom_preprocessing_report.json",
                        },
                        "cleanup_preprocessing_outputs": {
                            "enabled": True,
                            "summary_json_relpath": "context/custom_cleanup_preprocessing_outputs_summary.json",
                        },
                    },
                }
            }
        }
    )

    parsed = parse_preprocess_stage_config(runtime_config=cfg)

    assert parsed.phases.copy_src_to_scratch.enabled is True
    assert parsed.phases.copy_src_to_scratch.requires_use_scratch_root is True
    assert parsed.phases.copy_src_to_scratch.summary_json_relpath == "context/custom_copy_summary.json"
    assert parsed.phases.save_rec_metadata.enabled is True
    assert parsed.phases.save_rec_metadata.verbose is True
    assert parsed.phases.save_rec_metadata.metadata_source == "scratch_copy"
    assert parsed.phases.save_rec_metadata.debug_mode_enabled is True
    assert parsed.phases.save_rec_metadata.debug_limit_datasets == 1
    assert parsed.phases.save_rec_metadata.debug_limit_wells == 3
    assert parsed.phases.save_rec_metadata.debug_limit_wells_per_dataset == 2
    assert parsed.phases.save_rec_metadata.report_step_timers is True
    assert parsed.phases.save_rec_metadata.summary_json_relpath == "context/custom_recording_metadata_summary.json"
    assert parsed.phases.save_rec_metadata.segment_epochs_relpath == "meta/segments.json"
    assert parsed.phases.save_rec_metadata.contiguous_epochs_relpath == "meta/contiguous.json"
    assert parsed.phases.save_rec_metadata.sampling_metadata_relpath == "meta/sampling.json"
    assert parsed.phases.prepare_raw_binaries.enabled is True
    assert parsed.phases.prepare_raw_binaries.summary_json_relpath == "context/custom_prepare_raw_binaries_summary.json"
    assert parsed.phases.prepare_raw_binaries.rel_output_root == "raw_binary_recording"
    assert parsed.phases.prepare_raw_binaries.manifest_relpath == "context/custom_raw_binary_manifest.json"
    assert parsed.phases.prepare_raw_binaries.outputs.save_chunk_duration == "3s"
    assert parsed.phases.prepare_raw_binaries.outputs.save_progress_bar is True
    assert parsed.phases.wipe_src_scratch.enabled is True
    assert parsed.phases.wipe_src_scratch.dry_run is True
    assert parsed.phases.wipe_src_scratch.requires_use_scratch_root is True
    assert parsed.phases.wipe_src_scratch.summary_json_relpath == "context/custom_wipe_src_scratch_summary.json"
    assert parsed.phases.preprocess_segments.enabled is False
    assert parsed.phases.preprocess_segments.output_mode == "binary"
    assert parsed.phases.preprocess_segments.lazy_source == "src"
    assert parsed.phases.preprocess_segments.summary_json_relpath == "context/custom_segments_summary.json"
    assert parsed.phases.preprocess_segments.rel_output_root == "preprocessed_segments"
    assert parsed.phases.preprocess_segments.outputs.save_chunk_duration == "2s"
    assert parsed.phases.preprocess_segments.outputs.save_progress_bar is True
    assert parsed.phases.preprocess_segments.outputs.print_n_jobs_used is True
    assert parsed.phases.plot_segment_traces.enabled is True
    assert parsed.phases.plot_segment_traces.summary_json_relpath == "context/custom_plot_segment_traces_summary.json"
    assert parsed.phases.plot_segment_traces.plot.layouts is False
    assert parsed.phases.plot_segment_traces.plot.segment_traces is False
    assert parsed.phases.plot_segment_traces.plot.output_dir == "plots/segments/{stream_id}"
    assert parsed.phases.plot_segment_channel_layouts.enabled is True
    assert parsed.phases.plot_segment_channel_layouts.summary_json_relpath == "context/custom_plot_segment_channel_layouts_summary.json"
    assert parsed.phases.plot_segment_channel_layouts.plot.layouts is True
    assert parsed.phases.plot_segment_channel_layouts.plot.channel_layouts_subdir == "custom_segment_layouts"
    assert parsed.phases.concat_segments.enabled is True
    assert parsed.phases.concat_segments.concatenate_preprocessed_recordings is False
    assert parsed.phases.concat_segments.debug_mode_enabled is True
    assert parsed.phases.concat_segments.debug_limit_datasets == 1
    assert parsed.phases.concat_segments.debug_limit_wells == 2
    assert parsed.phases.concat_segments.debug_limit_wells_per_dataset == 2
    assert parsed.phases.concat_segments.output_mode == "lazy"
    assert parsed.phases.concat_segments.summary_json_relpath == "context/custom_concat_segments_summary.json"
    assert parsed.phases.concat_segments.rel_output_root == "concatenated_recording"
    assert parsed.phases.plot_concat_traces.enabled is True
    assert parsed.phases.plot_concat_traces.summary_json_relpath == "context/custom_plot_concat_traces_summary.json"
    assert parsed.phases.plot_concat_traces.plot.concat_trace is False
    assert parsed.phases.plot_concat_traces.plot.concat_trace_n_reps == 3
    assert parsed.phases.plot_concat_channel_layout.enabled is True
    assert parsed.phases.plot_concat_channel_layout.summary_json_relpath == "context/custom_plot_concat_channel_layout_summary.json"
    assert parsed.phases.plot_concat_channel_layout.plot.layouts is True
    assert parsed.phases.plot_concat_channel_layout.plot.channel_layouts_subdir == "custom_concat_layouts"
    assert parsed.phases.plot_raster_threshold.enabled is True
    assert parsed.phases.plot_raster_threshold.debug_mode_enabled is True
    assert parsed.phases.plot_raster_threshold.debug_limit_datasets == 1
    assert parsed.phases.plot_raster_threshold.debug_limit_wells == 1
    assert parsed.phases.plot_raster_threshold.debug_limit_wells_per_dataset == 1
    assert parsed.phases.plot_raster_threshold.report_step_timers is True
    assert parsed.phases.plot_raster_threshold.summary_json_relpath == "context/custom_plot_raster_threshold_summary.json"
    assert parsed.phases.plot_raster_threshold.rel_output_root == "raster_threshold_outputs"
    assert parsed.phases.report_preprocessing.enabled is True
    assert parsed.phases.report_preprocessing.summary_json_relpath == "context/custom_report_preprocessing_summary.json"
    assert parsed.phases.report_preprocessing.report_relpath == "report/custom_preprocessing_report.md"
    assert parsed.phases.report_preprocessing.json_summary_relpath == "report/custom_preprocessing_report.json"
    assert parsed.phases.cleanup_preprocessing_outputs.enabled is True
    assert parsed.phases.cleanup_preprocessing_outputs.summary_json_relpath == "context/custom_cleanup_preprocessing_outputs_summary.json"
    assert (
        parsed.phases.save_rec_metadata.common_electrodes_summary_json_relpath
        == "context/custom_common_summary.json"
    )


def test_parse_preprocess_stage_config_reads_phase_sequence() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "preprocess": {
                    "phase_sequence": [
                        "preprocess.copy_src_to_scratch",
                        "save_segment_recordings",
                        "concat_segments",
                    ]
                }
            }
        }
    )

    parsed = parse_preprocess_stage_config(runtime_config=cfg)

    assert parsed.phase_sequence == (
        "copy_src_to_scratch",
        "preprocess_segments",
        "concat_segments",
    )


def test_parse_preprocess_stage_config_reads_resources_n_jobs() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "preprocess": {
                    "execution": {"n_jobs": 2},
                    "resources": {"n_jobs": 5},
                }
            }
        }
    )

    parsed = parse_preprocess_stage_config(runtime_config=cfg)

    assert parsed.n_jobs == 5


def test_parse_preprocess_stage_config_reads_global_debug_mode_alias() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "preprocess": {
                    "debug_mode": {
                        "limit_datasets": 1,
                        "limit_wells": 2,
                        "limit_wells_per_dataset": 2,
                        "limit_segments_per_well": 3,
                    }
                }
            }
        }
    )

    parsed = parse_preprocess_stage_config(runtime_config=cfg)

    assert parsed.debug_limit_datasets == 1
    assert parsed.debug_limit_wells == 2
    assert parsed.debug_limit_wells_per_dataset == 2
    assert parsed.debug_limit_segments_per_well == 3


def test_parse_preprocess_stage_config_treats_non_positive_trace_max_points_as_uncapped() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "preprocess": {
                    "plot": {
                        "trace_max_points": -1,
                    }
                }
            }
        }
    )

    parsed = parse_preprocess_stage_config(runtime_config=cfg)

    assert parsed.trace_max_points == -1


def test_parse_preprocess_stage_config_resolves_detailed_observability_mode() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "preprocess": {
                    "observability": {
                        "mode": "detailed",
                        "output_subdir": "meta/run_artifacts",
                        "stage_log_relpath": "logs/custom_preprocess_pipeline.log",
                    }
                }
            }
        }
    )

    parsed = parse_preprocess_stage_config(runtime_config=cfg)

    assert parsed.observability_mode == "detailed"
    assert parsed.observability_output_subdir == "meta/run_artifacts"
    assert parsed.observability_save_run_manifest is True
    assert parsed.observability_save_event_timeline is True
    assert parsed.observability_save_environment is True
    assert parsed.observability_save_artifact_inventory is True
    assert parsed.observability_save_stage_log is True
    assert parsed.observability_stage_log_relpath == "logs/custom_preprocess_pipeline.log"


def test_load_preprocess_inputs_from_runtime_defaults_and_overrides(tmp_path: Path) -> None:
    data_path = tmp_path / "data.yml"
    data_path.write_text(
        dedent(
            """
            output_root: /tmp/out
            datasets:
              - raw_data_h5_path: /tmp/input.raw.h5
                include_in_runtime: true
                wells:
                  - well_id: well005
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    runtime_path = tmp_path / "runtime.yml"
    runtime_path.write_text(
                (
                        f"data: {data_path}\n"
                        "stages:\n"
                        "  preprocess:\n"
                        "    execution:\n"
                        "      force_restart: false\n"
                        "      n_jobs: 6\n"
                        "      temporal_resample_factor: 4\n"
                        "      plot_layouts: false\n"
                        "    debug:\n"
                        "      limit_segments_per_well: 2\n"
                        "    logging:\n"
                        "      enabled: true\n"
                        "      verbose: false\n"
                        "      file_relpath: logs/preprocess_pipeline.log\n"
                        "      suppress_h5_plugin_messages: true\n"
                        "      phase_dividers: false\n"
                        "    observability:\n"
                        "      mode: detailed\n"
                        "      output_subdir: run_metadata\n"
                        "      stage_log_relpath: logs/preprocess_pipeline.log\n"
                        "    outputs:\n"
                        "      output_rel_root: preprocess_stage_outputs\n"
                        "      save_concat_recording: true\n"
                        "      save_segment_recordings: false\n"
                        "    phases:\n"
                        "      copy_src_to_scratch:\n"
                        "        enabled: true\n"
                        "        requires_use_scratch_root: false\n"
                        "      save_rec_metadata:\n"
                        "        enabled: true\n"
                        "        verbose: true\n"
                        "        metadata_source: source_h5\n"
                        "        summary_json_relpath: context/recording_metadata_summary.json\n"
                        "        segment_epochs_relpath: segment_epochs.json\n"
                        "        contiguous_epochs_relpath: continuous_epochs.json\n"
                        "        sampling_metadata_relpath: sampling_rate_metadata.json\n"
                        "        common_electrodes_summary_json_relpath: context/save_common_electrodes_summary.json\n"
                        "      prepare_raw_binaries:\n"
                        "        enabled: true\n"
                        "        rel_output_root: raw_binary_recording\n"
                        "        manifest_relpath: context/raw_binary_manifest.json\n"
                        "      wipe_src_scratch:\n"
                        "        enabled: true\n"
                        "        dry_run: true\n"
                        "        requires_use_scratch_root: true\n"
                        "        summary_json_relpath: context/wipe_src_scratch_summary.json\n"
                        "      preprocess_segments:\n"
                        "        enabled: false\n"
                        "        output_mode: lazy\n"
                        "        lazy_source: src\n"
                        "        rel_output_root: preprocessed_segments\n"
                        "        plot:\n"
                        "          layouts: false\n"
                        "          segment_traces: false\n"
                        "          n_reps_per_segment: 7\n"
                        "          n_jobs: 4\n"
                        "          trace_downsample_hz: 200.0\n"
                        "          trace_max_points: 32000\n"
                        "          output_dir: preprocess_outputs/plots\n"
                        "        outputs:\n"
                        "          save_chunk_duration: 2s\n"
                        "          save_progress_bar: true\n"
                        "          print_n_jobs_used: true\n"
                        "      concat_segments:\n"
                        "        enabled: true\n"
                        "        concatenate_preprocessed_recordings: false\n"
                        "        output_mode: binary\n"
                        "        rel_output_root: concatenated_recording\n"
                        "        outputs: {}\n"
                        "      plot_concat_traces:\n"
                        "        enabled: true\n"
                        "        plot:\n"
                        "          concat_trace: false\n"
                        "      plot_segment_channel_layouts:\n"
                        "        enabled: true\n"
                        "      plot_concat_channel_layout:\n"
                        "        enabled: false\n"
                        "      plot_raster_threshold:\n"
                        "        enabled: false\n"
                        "      report_preprocessing:\n"
                        "        enabled: false\n"
                        "      cleanup_preprocessing_outputs:\n"
                        "        enabled: false\n"
                ),
        encoding="utf-8",
    )

    inputs = load_preprocess_inputs_from_runtime(
        config_path=str(runtime_path),
        force_restart_override=True,
    )

    assert inputs.stream_id == "well005"
    assert inputs.output_rel_root == "preprocess_stage_outputs"
    assert inputs.force_restart is True
    assert inputs.force_replot is False
    assert inputs.debug_limit_segments_per_well == 2
    assert inputs.logging_enabled is True
    assert inputs.logging_verbose is False
    assert inputs.logging_file_relpath == "logs/preprocess_pipeline.log"
    assert inputs.logging_suppress_h5_plugin_messages is True
    assert inputs.logging_phase_dividers is False
    assert inputs.n_jobs == 6
    assert inputs.temporal_resample_factor == 4
    assert inputs.observability_mode == "detailed"
    assert inputs.observability_output_subdir == "run_metadata"
    assert inputs.observability_save_run_manifest is True
    assert inputs.observability_save_event_timeline is True
    assert inputs.observability_save_environment is True
    assert inputs.observability_save_artifact_inventory is True
    assert inputs.observability_save_stage_log is True
    assert inputs.observability_stage_log_relpath == "logs/preprocess_pipeline.log"
    assert inputs.save_concat_recording is True
    assert inputs.save_segment_recordings is False
    assert inputs.phases.copy_src_to_scratch.enabled is True
    assert inputs.phases.save_rec_metadata.enabled is True
    assert inputs.phases.save_rec_metadata.verbose is True
    assert inputs.phases.save_rec_metadata.metadata_source == "source_h5"
    assert inputs.phases.save_rec_metadata.summary_json_relpath == "context/recording_metadata_summary.json"
    assert inputs.phases.save_rec_metadata.segment_epochs_relpath == "segment_epochs.json"
    assert inputs.phases.save_rec_metadata.contiguous_epochs_relpath == "continuous_epochs.json"
    assert inputs.phases.save_rec_metadata.sampling_metadata_relpath == "sampling_rate_metadata.json"
    assert inputs.phases.prepare_raw_binaries.enabled is True
    assert inputs.phases.prepare_raw_binaries.rel_output_root == "raw_binary_recording"
    assert inputs.phases.prepare_raw_binaries.manifest_relpath == "context/raw_binary_manifest.json"
    assert inputs.phases.wipe_src_scratch.enabled is True
    assert inputs.phases.wipe_src_scratch.dry_run is True
    assert inputs.phases.wipe_src_scratch.requires_use_scratch_root is True
    assert inputs.phases.wipe_src_scratch.summary_json_relpath == "context/wipe_src_scratch_summary.json"
    assert inputs.phases.preprocess_segments.enabled is False
    assert inputs.phases.preprocess_segments.output_mode == "lazy"
    assert inputs.phases.preprocess_segments.lazy_source == "src"
    assert inputs.phases.preprocess_segments.rel_output_root == "preprocessed_segments"
    assert inputs.phases.preprocess_segments.outputs.save_chunk_duration == "2s"
    assert inputs.phases.preprocess_segments.outputs.save_progress_bar is True
    assert inputs.phases.preprocess_segments.outputs.print_n_jobs_used is True
    assert inputs.phases.plot_segment_traces.enabled is True
    assert inputs.phases.plot_segment_traces.plot.layouts is False
    assert inputs.phases.plot_segment_traces.plot.segment_traces is False
    assert inputs.phases.plot_segment_traces.plot.n_representative_channels == 7
    assert inputs.phases.plot_segment_traces.plot.segment_trace_n_reps == 7
    assert inputs.phases.plot_segment_traces.plot.n_jobs == 4
    assert inputs.phases.plot_segment_traces.plot.trace_downsample_hz == 200.0
    assert inputs.phases.plot_segment_traces.plot.trace_max_points == 32000
    assert inputs.phases.plot_segment_traces.plot.output_dir == "preprocess_outputs/plots"
    assert inputs.phases.plot_segment_channel_layouts.enabled is True
    assert inputs.phases.concat_segments.enabled is True
    assert inputs.phases.concat_segments.concatenate_preprocessed_recordings is False
    assert inputs.phases.concat_segments.output_mode == "binary"
    assert inputs.phases.concat_segments.rel_output_root == "concatenated_recording"
    assert inputs.phases.plot_concat_traces.enabled is True
    assert inputs.phases.plot_concat_traces.plot.concat_trace is False
    assert inputs.phases.plot_concat_channel_layout.enabled is False
    assert inputs.phases.plot_raster_threshold.enabled is False
    assert inputs.phases.report_preprocessing.enabled is False
    assert inputs.phases.cleanup_preprocessing_outputs.enabled is False
    assert inputs.phases.save_rec_metadata.common_electrodes_summary_json_relpath == "context/save_common_electrodes_summary.json"


def test_load_preprocess_inputs_plot_n_jobs_null_inherits_preprocess_n_jobs(tmp_path: Path) -> None:
    data_path = tmp_path / "data.yml"
    data_path.write_text(
        dedent(
            """
            output_root: /tmp/out
            datasets:
              - raw_data_h5_path: /tmp/input.raw.h5
                include_in_runtime: true
                wells:
                  - well_id: well005
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    runtime_path = tmp_path / "runtime.yml"
    runtime_path.write_text(
        (
            f"data: {data_path}\n"
            "stages:\n"
            "  preprocess:\n"
            "    execution:\n"
            "      n_jobs: 6\n"
            "    plot:\n"
            "      n_jobs: null\n"
        ),
        encoding="utf-8",
    )

    inputs = load_preprocess_inputs_from_runtime(config_path=str(runtime_path))

    assert inputs.n_jobs == 6
    assert inputs.plot_n_jobs == 6
    assert inputs.debug_limit_segments_per_well is None


def test_build_preprocess_inputs_plot_n_jobs_inherits_unit_workers_when_unset() -> None:
    stage_cfg = parse_preprocess_stage_config(runtime_config=RuntimeConfig({"stages": {"preprocess": {}}}))
    target = ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test.h5",
        h5_path=Path("/tmp/test.h5"),
        stream_id="well001",
        mea_output_root=Path("/tmp/out"),
    )

    inputs = build_preprocess_inputs_for_target(
        target=target,
        stage_config=stage_cfg,
        unit_workers=7,
    )

    assert inputs.n_jobs == 7
    assert inputs.plot_n_jobs == 7
    assert inputs.source_h5_path == target.h5_path
    assert inputs.copied_to_scratch is False
    assert inputs.logging_subphase_dividers_to_stdout is True


def test_per_phase_yaml_n_jobs_no_longer_recognized() -> None:
    """Per-phase parallelism knobs are silently ignored after slice 7.
    The runtime n_jobs is determined by the phase budget / task slot, not the YAML knob.
    """
    cfg = RuntimeConfig(
        {
            "stages": {
                "preprocess": {
                    "phases": {
                        "preprocess_segments": {
                            "outputs": {
                                "segment_save_n_jobs": 8,
                            },
                        },
                        "concat_segments": {
                            "outputs": {
                                "concat_save_n_jobs": 8,
                            },
                        },
                    }
                }
            }
        }
    )
    parsed = parse_preprocess_stage_config(runtime_config=cfg)
    assert not hasattr(parsed.phases.preprocess_segments.outputs, "segment_save_n_jobs"), (
        "segment_save_n_jobs field must be removed from PreprocessPhaseOutputsConfig"
    )
    assert not hasattr(parsed.phases.concat_segments.outputs, "concat_save_n_jobs"), (
        "concat_save_n_jobs field must be removed from PreprocessPhaseOutputsConfig"
    )
