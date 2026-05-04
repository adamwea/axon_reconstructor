from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from axon_recon.runtime_config import RuntimeConfig
from axon_recon.pipeline.stages.spikesort.config import (
    DEFAULT_SPIKESORT_PHASE_SEQUENCE,
    load_spikesort_inputs_from_runtime,
    parse_spikesort_stage_config,
)


def test_parse_spikesort_stage_config_defaults() -> None:
    cfg = RuntimeConfig({"stages": {"spikesort": {}}})

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.output_rel_root == "spikesort_outputs"
    assert parsed.preprocess_concat_recording_relpath is None
    assert parsed.merge_sequence == ("SLAy", "auto_merge", "unitmatch")
    assert parsed.phase_sequence == DEFAULT_SPIKESORT_PHASE_SEQUENCE
    assert parsed.logging_enabled is True
    assert parsed.logging_verbose is False
    assert parsed.logging_file_relpath is None
    assert parsed.debug_mode_enabled is False
    assert parsed.debug_limit_datasets is None
    assert parsed.debug_limit_wells is None
    assert parsed.debug_limit_wells_per_dataset is None
    assert parsed.debug_limit_segments_per_well is None
    assert parsed.sort_debug_mode_enabled is False
    assert parsed.sort_debug_limit_datasets is None
    assert parsed.sort_debug_limit_wells is None
    assert parsed.sort_debug_limit_wells_per_dataset is None
    assert parsed.bootstrap_concat_binary_debug_mode_enabled is False
    assert parsed.bootstrap_concat_binary_debug_limit_datasets is None
    assert parsed.bootstrap_concat_binary_debug_limit_wells is None
    assert parsed.bootstrap_concat_binary_debug_limit_wells_per_dataset is None
    assert parsed.bootstrap_concat_binary_debug_limit_segments_per_well is None
    assert parsed.cleanup_concat_binary_debug_mode_enabled is False
    assert parsed.cleanup_concat_binary_debug_limit_datasets is None
    assert parsed.cleanup_concat_binary_debug_limit_wells is None
    assert parsed.cleanup_concat_binary_debug_limit_wells_per_dataset is None
    assert parsed.summarize_sort_debug_mode_enabled is False
    assert parsed.summarize_sort_debug_limit_datasets is None
    assert parsed.summarize_sort_debug_limit_wells is None
    assert parsed.summarize_sort_debug_limit_wells_per_dataset is None
    assert parsed.bombcell_label_debug_mode_enabled is False
    assert parsed.bombcell_label_debug_limit_datasets is None
    assert parsed.bombcell_label_debug_limit_wells is None
    assert parsed.bombcell_label_debug_limit_wells_per_dataset is None
    assert parsed.sort_engine == "mea_analysis"
    assert parsed.sorter == "kilosort4"
    assert parsed.docker_image is None
    assert parsed.mea_analysis_enabled is True
    assert parsed.mea_analysis_docker_image is None
    assert parsed.local_spikeinterface_enabled is False
    assert parsed.local_spikeinterface_output_relpath == "sorter_output"
    assert parsed.local_spikeinterface_remove_existing_on_force_restart is True
    assert parsed.local_spikeinterface_run_sorter_kwargs == {}
    assert parsed.local_spikeinterface_analyzer_enabled is True
    assert parsed.local_spikeinterface_analyzer_output_relpath == "analyzer_output"
    assert parsed.recording_num == "rec0000"
    assert parsed.verbose is False
    assert parsed.progress_bar is True
    assert parsed.n_jobs is None
    assert parsed.chunk_duration is None
    assert parsed.cuda_visible_devices is None
    assert parsed.force_single_well_sort is False
    assert parsed.run_analyzer is True
    assert parsed.run_reports is True
    assert parsed.sort_enabled is True
    assert parsed.sort_delete_outputs_on_force_restart is False
    assert parsed.sort_original_preprocess_concat_recording_relpath is None
    assert parsed.sort_bootstrapped_concat_recording_relpath == "spikesort_outputs/cache/bootstrap_concat_binary/recording"
    assert parsed.sort_use_bootstrapped_concat_binary is False
    assert parsed.sort_use_lazy_source is True
    assert parsed.sort_assert_one_source is False
    assert parsed.plot_enabled is True
    assert parsed.plot_mode == "separate"
    assert parsed.plot_debug is False
    assert parsed.raster_sort is None
    assert parsed.fixed_y is False
    assert parsed.no_curation is False
    assert parsed.export_to_phy is False
    assert parsed.force_rerun_analyzer is False
    assert parsed.bootstrap_concat_binary_enabled is False
    assert parsed.bootstrap_concat_binary_cache_relpath == "cache/bootstrap_concat_binary"
    assert parsed.bootstrap_concat_binary_recording_relpath == "cache/bootstrap_concat_binary/recording"
    assert parsed.bootstrap_concat_binary_manifest_relpath == "cache/bootstrap_concat_binary/concat_segments_manifest.json"
    assert parsed.bootstrap_concat_binary_summary_json_relpath == "cache/bootstrap_concat_binary/bootstrap_concat_binary_summary.json"
    assert parsed.bootstrap_concat_binary_source_segment_manifest_relpath == "preprocess_outputs/preprocessed_segments/manifest.json"
    assert parsed.bootstrap_concat_binary_use_as_preprocess_concat_recording is True
    assert parsed.bootstrap_concat_binary_overwrite_existing is False
    assert parsed.bootstrap_concat_binary_overwrite_on_force_restart is True
    assert parsed.bootstrap_concat_binary_n_jobs is None
    assert parsed.bootstrap_concat_binary_chunk_duration is None
    assert parsed.bootstrap_concat_binary_progress_bar is True
    assert parsed.cleanup_concat_binary_enabled is False
    assert parsed.cleanup_concat_binary_relpath == "cache/bootstrap_concat_binary"
    assert parsed.cleanup_concat_binary_summary_json_relpath == "cache/bootstrap_concat_binary_cleanup_summary.json"
    assert parsed.summarize_sort_enabled is False
    assert parsed.summarize_sort_emit_logs is True
    assert parsed.summarize_sort_generate_artifacts is False
    assert parsed.bombcell_label_enabled is False
    assert parsed.bombcell_label_relpath == "bombcell_label_outputs"
    assert parsed.bombcell_label_delete_outputs_on_force_restart is True
    assert parsed.bombcell_label_cache_sorter_output_before_analyzer_gen is False
    assert parsed.bombcell_label_publish_cached_sorter_output_on_success is False
    assert parsed.bombcell_label_publish_cached_analyzer_on_success is False
    assert parsed.bombcell_label_cleanup_analyzer_on_success is False
    assert parsed.bombcell_label_cleanup_cached_sorter_output_on_success is False
    assert parsed.bombcell_label_thresholds is None
    assert parsed.bombcell_label_thresholds_path is None
    assert parsed.bombcell_label_label_non_somatic is True
    assert parsed.bombcell_label_split_non_somatic_good_mua is True
    assert parsed.bombcell_label_apply_to_sorter_output is True
    assert parsed.bombcell_label_write_cluster_group is True
    assert parsed.bombcell_label_fail_on_error is False
    assert parsed.bombcell_label_reports_enabled is True
    assert parsed.bombcell_label_reports_summary_json_enabled is True
    assert parsed.bombcell_label_reports_summary_json_relpath == "bombcell_label_summary.json"
    assert parsed.slay_enabled is False
    assert parsed.slay_relpath == "SLAy_outputs"
    assert parsed.slay_sorter_output_relpath is None
    assert parsed.slay_output_json_relpath == "run-output.json"
    assert parsed.slay_candidate_pairs_relpath == "recommended_merge_candidates.tsv"
    assert parsed.slay_merge_groups_relpath == "recommended_merge_groups.json"
    assert parsed.slay_allow_numpy_fallback is True
    assert parsed.slay_plot_merges is False
    assert parsed.slay_auto_accept_merges is False
    assert parsed.slay_copy_automerge_artifacts is True
    assert parsed.slay_delete_outputs_on_force_restart is True
    assert parsed.slay_recompute_analyzer is False
    assert parsed.slay_model_cache_relpath == "cache/slay_model/ae.pt"
    assert parsed.slay_model_cache_use_cached_model is True
    assert parsed.slay_model_cache_write_model is True
    assert parsed.slay_force_restart_retrain_model is False
    assert parsed.slay_params is None
    assert parsed.auto_merge_enabled is False
    assert parsed.auto_merge_relpath == "automerge_outputs"
    assert parsed.auto_merge_delete_outputs_on_force_restart is True
    assert parsed.auto_merge_candidate_pairs_reldir == "recommended_merge_candidates"
    assert parsed.auto_merge_merged_units_reldir == "merged_units"
    assert parsed.auto_merge_auto_accept_merges is False
    assert parsed.auto_merge_template_diff_thresholds == (0.25,)
    assert parsed.merge_slay_enabled is False
    assert parsed.merge_slay_rel_output_root == "merge_SLAy"
    assert parsed.merge_slay_use_canonical_workspace is True
    assert parsed.merge_slay_canonical_workspace_relpath == "cache/merge_workspace"
    assert parsed.merge_slay_canonical_workspace_rebuild_analyzer is False
    assert parsed.merge_slay_publish_canonical_to_stage_outputs_on_success is True
    assert parsed.merge_slay_debug_mode_enabled is False
    assert parsed.merge_slay_debug_limit_datasets is None
    assert parsed.merge_slay_debug_limit_wells is None
    assert parsed.merge_slay_debug_limit_wells_per_dataset is None
    assert parsed.merge_si_auto_enabled is False
    assert parsed.merge_si_auto_rel_output_root == "merge_si_auto"
    assert parsed.merge_si_auto_use_canonical_workspace is True
    assert parsed.merge_si_auto_debug_mode_enabled is False
    assert parsed.merge_si_auto_debug_limit_datasets is None
    assert parsed.merge_si_auto_debug_limit_wells is None
    assert parsed.merge_si_auto_debug_limit_wells_per_dataset is None
    assert parsed.merge_unitmatch_enabled is False
    assert parsed.merge_unitmatch_rel_output_root == "merge_unitmatch"
    assert parsed.merge_unitmatch_use_canonical_workspace is True
    assert parsed.merge_unitmatch_debug_mode_enabled is False
    assert parsed.merge_unitmatch_debug_limit_datasets is None
    assert parsed.merge_unitmatch_debug_limit_wells is None
    assert parsed.merge_unitmatch_debug_limit_wells_per_dataset is None
    assert parsed.merge_units_enabled is True
    assert parsed.merge_rel_output_root is None
    assert parsed.merge_delete_outputs_on_force_restart is False
    assert parsed.merge_force_restart is False
    assert parsed.merge_force_replot is False
    assert parsed.merge_cleanup_generated_analyzers_on_success is True
    assert parsed.merge_analyzer_regenerate_on_replot is True
    assert parsed.merge_analyzer_check_if_regen_is_needed is True
    assert parsed.merge_analyzer_compute_sparsity is True
    assert parsed.merge_template_random_spikes_method == "default"
    assert parsed.merge_template_random_spikes_percentage is None
    assert parsed.merge_template_random_spikes_max_spikes_per_unit == 500
    assert parsed.merge_template_random_spikes_min_spikes_per_unit is None
    assert parsed.merge_template_random_spikes_log_before_after_spike_counts is False
    assert parsed.merge_template_random_spikes_margin_size is None
    assert parsed.merge_template_random_spikes_seed is None
    assert parsed.merge_analyzer_n_jobs is None
    assert parsed.merge_analyzer_chunk_duration is None
    assert parsed.merge_analyzer_sparsity_method == "radius"
    assert parsed.merge_analyzer_sparsity_radius_um == 100.0
    assert parsed.merge_analyzer_sparsity_num_channels == 5
    assert parsed.merge_analyzer_sparsity_threshold == 5.0
    assert parsed.merge_analyzer_sparsity_peak_sign == "neg"
    assert parsed.merge_analyzer_sparsity_num_spikes_for_sparsity == 100
    assert parsed.merge_analyzer_sparsity_by_property is None
    assert parsed.merge_analyzer_waveforms_ms_before == 1.0
    assert parsed.merge_analyzer_waveforms_ms_after == 2.0
    assert parsed.merge_analyzer_waveforms_dtype is None
    assert parsed.cache_sorting_outputs_before_merge is False
    assert parsed.cache_sorting_outputs_before_merge_relpath == "pre_merge_cache"
    assert parsed.cache_sorting_outputs_before_merge_cleanup_on_success is False
    assert parsed.cache_sorting_outputs_before_merge_use_cache_on_force_restart is False
    assert parsed.cache_sorting_outputs_before_merge_replace_sorting_with_cache_before_force_restart is False
    assert parsed.cache_sorting_outputs_before_merge_refresh_on_run is False
    assert parsed.cache_sorting_outputs_before_merge_strict_restore_on_force_restart is True
    assert parsed.cache_sorting_outputs_before_merge_use_canonical_workspace is False
    assert parsed.cache_sorting_outputs_before_merge_canonical_workspace_relpath == "cache/merge_workspace"
    assert parsed.cache_sorting_outputs_before_merge_canonical_workspace_refresh_on_run is True
    assert parsed.cache_sorting_outputs_before_merge_canonical_workspace_rebuild_analyzer is False
    assert parsed.cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_success is False
    assert parsed.cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_failure is False
    assert parsed.cache_sorting_outputs_before_merge_assert_slay_uses_canonical_workspace is True
    assert parsed.cache_sorting_outputs_before_merge_assert_auto_merge_uses_canonical_workspace is True
    assert parsed.merge_reports_enabled is False
    assert parsed.merge_reports_unit_diff_json_enabled is False
    assert parsed.merge_reports_unit_diff_json_relpath == "unit_diffs_after_merge.json"
    assert parsed.merge_reports_unit_diff_map_enabled is False
    assert parsed.merge_reports_unit_diff_map_relpath == "unit_diff_map.json"
    assert parsed.merge_reports_unit_diff_map_flat_enabled is False
    assert parsed.merge_reports_unit_diff_map_flat_relpath == "unit_diff_map_flat.json"
    assert parsed.merge_reports_post_merge_unit_locations_enabled is False
    assert parsed.merge_reports_post_merge_unit_locations_relpath == "post_merge_unit_locations.json"
    assert parsed.merge_reports_2panel_enabled is False
    assert parsed.merge_reports_2panel_point_size == 9.0
    assert parsed.merge_reports_2panel_relpath == "unit_locations_before_after_merge.png"
    assert parsed.merge_reports_2panel_label_pre_and_post_units is False


def test_parse_spikesort_stage_config_reads_phase_sequence() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phase_sequence": [
                        "bootstrap_concat_binary",
                        "spikesort.sort",
                        "bombcell",
                        "merge_slay",
                        "auto_merge",
                        "cleanup",
                    ]
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.phase_sequence == (
        "bootstrap_concat_binary",
        "sort",
        "bombcell_label",
        "merge_SLAy",
        "merge_si_auto",
        "cleanup_concat_binary",
    )


def test_parse_spikesort_stage_config_reads_global_debug_mode() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "debug_mode": {
                        "enabled": True,
                        "limit_datasets": 1,
                        "limit_wells": 2,
                        "limit_wells_per_dataset": 2,
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.debug_mode_enabled is True
    assert parsed.debug_limit_datasets == 1
    assert parsed.debug_limit_wells == 2
    assert parsed.debug_limit_wells_per_dataset == 2


def test_parse_spikesort_stage_config_summarize_sort_phase() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "summarize_sort": {
                            "enabled": True,
                            "emit_logs": False,
                            "generate_artifacts": True,
                        }
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.summarize_sort_enabled is True
    assert parsed.summarize_sort_emit_logs is False
    assert parsed.summarize_sort_generate_artifacts is True
    assert parsed.merge_reports_2panel_write_png is True
    assert parsed.merge_reports_2panel_write_svg is False
    assert parsed.merge_reports_2panel_before_relpath == "unit_locations_before_merge.png"
    assert parsed.merge_reports_2panel_before_write_png is True
    assert parsed.merge_reports_2panel_before_write_svg is False
    assert parsed.merge_reports_2panel_before_point_color == "#7a7a7a"
    assert parsed.merge_reports_2panel_after_relpath == "unit_locations_after_merge.png"
    assert parsed.merge_reports_2panel_after_write_png is True
    assert parsed.merge_reports_2panel_after_write_svg is False
    assert parsed.merge_reports_2panel_after_point_color == "#7a7a7a"
    assert parsed.merge_reports_2panel_highlight_merges_enabled is False
    assert parsed.merge_reports_2panel_highlight_merges_linked is True
    assert parsed.merge_reports_2panel_highlight_plot_after_other_units is False
    assert parsed.merge_reports_2panel_highlight_label_affected_units is False
    assert parsed.merge_reports_2panel_highlight_show_legend is False
    assert parsed.merge_reports_2panel_highlight_legend_position == "center left"
    assert parsed.merge_reports_2panel_highlight_legend_x == -0.2
    assert parsed.merge_reports_2panel_highlight_legend_y == 0.5
    assert parsed.merge_reports_2panel_highlight_sort_pre_legend_by_groups is False
    assert parsed.merge_reports_2panel_highlight_debug_json_enabled is True
    assert parsed.merge_reports_2panel_highlight_debug_json_relpath == "unit_locations_highlight_linkage.json"
    assert parsed.merge_reports_2panel_highlight_before_color == "#ff7f0e"
    assert parsed.merge_reports_2panel_highlight_after_color == "#2ca02c"
    assert parsed.merge_reports_2panel_highlight_palette == "tab20"
    assert parsed.merge_reports_2panel_inherit_probe_dimensions is False
    assert parsed.merge_reports_2panel_zoom_to_affected_units is False
    assert parsed.merge_reports_2panel_probe_dim_x_um is None
    assert parsed.merge_reports_2panel_probe_dim_y_um is None
    assert parsed.merge_reports_template_heatmaps_enabled is False
    assert parsed.merge_reports_template_heatmaps_relpath == "template_heatmaps_per_merge"
    assert parsed.merge_reports_template_heatmaps_assets_reldir == "assets"
    assert parsed.merge_reports_template_heatmaps_write_png is True
    assert parsed.merge_reports_template_heatmaps_write_svg is False
    assert parsed.merge_reports_template_heatmaps_write_assets_png is True
    assert parsed.merge_reports_template_heatmaps_write_assets_svg is False
    assert parsed.merge_reports_template_heatmaps_panel_width_in == 11.0
    assert parsed.merge_reports_template_heatmaps_panel_height_in == 6.0
    assert parsed.merge_reports_template_heatmaps_marker_size == 10.0
    assert parsed.merge_reports_template_heatmaps_cmap == "viridis"
    assert parsed.merge_reports_template_heatmaps_show_colorbar is True
    assert parsed.merge_reports_template_heatmaps_relative_color_bar_height == 1.0
    assert parsed.merge_reports_template_heatmaps_color_scale == "linear"
    assert parsed.merge_reports_template_heatmaps_log_epsilon == 1e-3
    assert parsed.merge_reports_template_heatmaps_magnitude_mode == "ptp"
    assert parsed.merge_reports_template_heatmaps_max_merges is None
    assert parsed.merge_reports_template_heatmaps_debug_json_relpath == "template_heatmaps_per_merge_report.json"
    assert parsed.merge_reports_template_heatmaps_inherit_probe_dimensions is False
    assert parsed.merge_reports_template_heatmaps_probe_dim_x_um is None
    assert parsed.merge_reports_template_heatmaps_probe_dim_y_um is None
    assert parsed.merge_reports_template_heatmaps_probe_pitch_um is None
    assert parsed.merge_reports_template_heatmaps_probe_electrode_size_um_x is None
    assert parsed.merge_reports_template_heatmaps_probe_electrode_size_um_y is None
    assert parsed.merge_metadata_enabled is False
    assert parsed.merge_metadata_write_json is True
    assert parsed.merge_metadata_json_relpath == "merge_metadata_summary.json"
    assert parsed.merge_metadata_include_unit_locations is True
    assert parsed.merge_metadata_log_summary_details is False
    assert parsed.pre_merge_metadata_enabled is False
    assert parsed.pre_merge_metadata_write_json is True
    assert parsed.pre_merge_metadata_json_relpath == "pre_merge_metadata_summary.json"
    assert parsed.pre_merge_metadata_include_unit_locations is True
    assert parsed.pre_merge_metadata_log_summary_details is False
    assert parsed.post_merge_metadata_enabled is False
    assert parsed.post_merge_metadata_write_json is True
    assert parsed.post_merge_metadata_json_relpath == "post_merge_metadata_summary.json"
    assert parsed.post_merge_metadata_include_unit_locations is True
    assert parsed.post_merge_metadata_log_summary_details is False
    assert parsed.force_restart is False
    assert parsed.force_replot is False


def test_parse_spikesort_stage_config_ignores_force_single_well_sort_resource() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "resources": {
                        "force_single_well_sort": True,
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.force_single_well_sort is False


def test_parse_spikesort_stage_config_reads_sort_debug_mode() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "sort": {
                            "debug_mode": {
                                "enabled": True,
                                "limit_datasets": 1,
                                "limit_wells": 1,
                                "limit_wells_per_dataset": 2,
                            }
                        }
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.sort_debug_mode_enabled is True
    assert parsed.sort_debug_limit_datasets == 1
    assert parsed.sort_debug_limit_wells == 1
    assert parsed.sort_debug_limit_wells_per_dataset == 2


def test_parse_spikesort_stage_config_reads_summarize_sort_debug_mode() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "summarize_sort": {
                            "debug_mode": {
                                "enabled": True,
                                "limit_datasets": 1,
                                "limit_wells": 1,
                            }
                        }
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.summarize_sort_debug_mode_enabled is True
    assert parsed.summarize_sort_debug_limit_datasets == 1
    assert parsed.summarize_sort_debug_limit_wells == 1


def test_parse_spikesort_stage_config_reads_bootstrap_concat_binary_phase() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "execution": {
                        "output_root": "spikesort_outputs_v2",
                    },
                    "inputs": {
                        "preprocess_concat_recording_relpath": "preprocess_outputs/concatenated_recording",
                    },
                    "phases": {
                        "bootstrap_concat_binary": {
                            "enabled": True,
                            "cache_relpath": "cache/local_concat",
                            "recording_relpath": "cache/local_concat/recording",
                            "manifest_relpath": "cache/local_concat/manifest.json",
                            "summary_json_relpath": "cache/local_concat/summary.json",
                            "source_segment_manifest_relpath": "preprocess_outputs/preprocessed_segments/manifest.json",
                            "overwrite_existing": True,
                            "overwrite_on_force_restart": False,
                            "n_jobs": 3,
                            "chunk_duration": "2s",
                            "progress_bar": False,
                            "debug_mode": {
                                "enabled": True,
                                "limit_datasets": 1,
                                "limit_wells": 2,
                                "limit_segments_per_well": 3,
                            },
                        }
                    },
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.bootstrap_concat_binary_enabled is True
    assert parsed.bootstrap_concat_binary_cache_relpath == "cache/local_concat"
    assert parsed.bootstrap_concat_binary_recording_relpath == "cache/local_concat/recording"
    assert parsed.bootstrap_concat_binary_manifest_relpath == "cache/local_concat/manifest.json"
    assert parsed.bootstrap_concat_binary_summary_json_relpath == "cache/local_concat/summary.json"
    assert parsed.bootstrap_concat_binary_source_segment_manifest_relpath == "preprocess_outputs/preprocessed_segments/manifest.json"
    assert parsed.bootstrap_concat_binary_overwrite_existing is True
    assert parsed.bootstrap_concat_binary_overwrite_on_force_restart is False
    assert parsed.bootstrap_concat_binary_n_jobs == 3
    assert parsed.bootstrap_concat_binary_chunk_duration == "2s"
    assert parsed.bootstrap_concat_binary_progress_bar is False
    assert parsed.bootstrap_concat_binary_debug_mode_enabled is True
    assert parsed.bootstrap_concat_binary_debug_limit_datasets == 1
    assert parsed.bootstrap_concat_binary_debug_limit_wells == 2
    assert parsed.bootstrap_concat_binary_debug_limit_segments_per_well == 3
    assert parsed.sort_use_bootstrapped_concat_binary is True
    assert parsed.sort_use_lazy_source is True
    assert parsed.sort_assert_one_source is False
    assert parsed.sort_original_preprocess_concat_recording_relpath == "preprocess_outputs/concatenated_recording"
    assert parsed.sort_bootstrapped_concat_recording_relpath == "spikesort_outputs_v2/cache/local_concat/recording"
    assert parsed.preprocess_concat_recording_relpath == "spikesort_outputs_v2/cache/local_concat/recording"


def test_parse_spikesort_stage_config_reads_sort_source_controls() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "output_root": "spikesort_outputs_v2",
                    "inputs": {
                        "preprocess_concat_recording_relpath": "preprocess_outputs/concatenated_recording",
                    },
                    "phases": {
                        "bootstrap_concat_binary": {
                            "recording_relpath": "cache/local_concat/recording",
                        },
                        "sort": {
                            "use_bootstrapped_concat_binary": True,
                            "use_lazy_source": False,
                            "assert_one_source": True,
                        },
                    },
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.sort_original_preprocess_concat_recording_relpath == "preprocess_outputs/concatenated_recording"
    assert parsed.sort_bootstrapped_concat_recording_relpath == "spikesort_outputs_v2/cache/local_concat/recording"
    assert parsed.preprocess_concat_recording_relpath == "spikesort_outputs_v2/cache/local_concat/recording"
    assert parsed.sort_use_bootstrapped_concat_binary is True
    assert parsed.sort_use_lazy_source is False
    assert parsed.sort_assert_one_source is True


def test_parse_spikesort_stage_config_sort_source_controls_can_keep_original_source() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "output_root": "spikesort_outputs_v2",
                    "inputs": {
                        "preprocess_concat_recording_relpath": "preprocess_outputs/concatenated_recording",
                    },
                    "phases": {
                        "bootstrap_concat_binary": {
                            "enabled": True,
                            "recording_relpath": "cache/local_concat/recording",
                        },
                        "sort": {
                            "use_bootstrapped_concat_binary": False,
                        },
                    },
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.sort_use_bootstrapped_concat_binary is False
    assert parsed.preprocess_concat_recording_relpath == "preprocess_outputs/concatenated_recording"


def test_parse_spikesort_stage_config_reads_sectioned_sort_engine_layout() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "output_root": "spikesort_outputs_v2",
                    "inputs": {
                        "preprocess_concat_recording_relpath": "preprocess_outputs/concatenated_recording",
                    },
                    "phases": {
                        "bootstrap_concat_binary": {
                            "recording_relpath": "cache/local_concat/recording",
                        },
                        "sort": {
                            "engine": "local_spikeinterface",
                            "source": {
                                "use_bootstrapped_concat_binary": True,
                                "use_lazy_source": False,
                                "assert_one_source": True,
                            },
                            "sorter": {
                                "name": "kilosort4",
                                "kilosort": {
                                    "batch_duration_s": 0.25,
                                    "thresholds": {
                                        "universal": 9,
                                        "learned": 8,
                                        "single_ch": 6,
                                    },
                                    "clustering": {
                                        "downsampling": 13,
                                    },
                                    "channels": {
                                        "nearest": 11,
                                        "max_distance": 37,
                                    },
                                },
                            },
                            "local_spikeinterface": {
                                "enabled": True,
                                "progress_bar": False,
                                "output_relpath": "local_sorter",
                                "remove_existing_on_force_restart": False,
                                "run_sorter_kwargs": {
                                    "delete_output_folder": True,
                                },
                                "analyzer": {
                                    "enabled": False,
                                    "output_relpath": "local_analyzer",
                                },
                            },
                            "mea_analysis": {
                                "enabled": True,
                                "docker_image": "legacy/image:kept",
                                "resume_from": "sort",
                                "plot": {
                                    "mode": "merged",
                                    "plot_debug": True,
                                },
                                "report": {
                                    "no_curation": True,
                                    "export_to_phy": True,
                                },
                            },
                        },
                    },
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.sort_engine == "local_spikeinterface"
    assert parsed.sorter == "kilosort4"
    assert parsed.sort_use_bootstrapped_concat_binary is True
    assert parsed.sort_use_lazy_source is False
    assert parsed.sort_assert_one_source is True
    assert parsed.preprocess_concat_recording_relpath == "spikesort_outputs_v2/cache/local_concat/recording"
    assert parsed.ks_batch_duration_s == 0.25
    assert parsed.ks_th_universal == 9.0
    assert parsed.ks_th_learned == 8.0
    assert parsed.ks_th_single_ch == 6.0
    assert parsed.ks_cluster_downsampling == 13
    assert parsed.ks_nearest_chans == 11
    assert parsed.ks_max_channel_distance == 37.0
    assert parsed.local_spikeinterface_enabled is True
    assert parsed.local_spikeinterface_output_relpath == "local_sorter"
    assert parsed.local_spikeinterface_remove_existing_on_force_restart is False
    assert parsed.local_spikeinterface_run_sorter_kwargs == {"delete_output_folder": True}
    assert parsed.local_spikeinterface_analyzer_enabled is False
    assert parsed.local_spikeinterface_analyzer_output_relpath == "local_analyzer"
    assert parsed.progress_bar is False
    assert parsed.mea_analysis_enabled is True
    assert parsed.mea_analysis_docker_image == "legacy/image:kept"
    assert parsed.docker_image == "legacy/image:kept"
    assert parsed.resume_from == "sort"
    assert parsed.plot_mode == "merged"
    assert parsed.plot_debug is True
    assert parsed.no_curation is True
    assert parsed.export_to_phy is True


def test_parse_spikesort_stage_config_rejects_unknown_sort_engine() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "sort": {
                            "engine": "external_container",
                        }
                    }
                }
            }
        }
    )

    with pytest.raises(ValueError, match="Unsupported spikesort sort engine"):
        parse_spikesort_stage_config(runtime_config=cfg)


def test_parse_spikesort_stage_config_reads_clear_concat_binary_alias() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "clear_concat_binary": {
                            "enabled": True,
                            "relpath": "cache/local_concat",
                            "summary_json_relpath": "cache/local_concat_cleanup.json",
                            "debug_mode": {
                                "enabled": True,
                                "limit_datasets": 1,
                                "limit_wells": 1,
                            },
                        }
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.cleanup_concat_binary_enabled is True
    assert parsed.cleanup_concat_binary_relpath == "cache/local_concat"
    assert parsed.cleanup_concat_binary_summary_json_relpath == "cache/local_concat_cleanup.json"
    assert parsed.cleanup_concat_binary_debug_mode_enabled is True
    assert parsed.cleanup_concat_binary_debug_limit_datasets == 1
    assert parsed.cleanup_concat_binary_debug_limit_wells == 1


def test_parse_spikesort_stage_config_parses_sectioned_stage_layout() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "resources": {
                        "chunk_duration": "1s",
                    },
                    "execution": {
                        "force_restart": True,
                        "limit_wells": 4,
                        "rerun_analyzer": True,
                        "auto_merge_units": True,
                        "auto_merge_template_diff_thresh": "0.05,0.15,0.25",
                    },
                    "inputs": {
                        "preprocess_concat_recording_relpath": "preprocess_outputs/preprocessed_recording",
                    },
                    "output_root": "spikesort_outputs_v2",
                    "outputs": {
                        "output_root": "legacy_outputs_should_not_win",
                    },
                    "sorter": "kilosort4",
                    "docker_image": "adammwea/benshalomlab_spikesorter_pythonpatch:v3",
                    "kilosort": {
                        "batch_duration_s": 0.75,
                        "thresholds": {
                            "universal": 8,
                            "learned": 7,
                            "single_ch": 5,
                        },
                        "clustering": {
                            "downsampling": 15,
                        },
                        "channels": {
                            "nearest": 12,
                            "max_distance": 40,
                        },
                    },
                    "unitmatch": {
                        "merge_units": False,
                        "dry_run": False,
                        "output_subdir_name": "unitmatch_outputs",
                        "throughput_subdir_name": "unitmatch_throughput",
                        "max_spikes_per_unit": -1,
                        "oversplit_min_probability": 0.95,
                        "limits": {
                            "max_candidate_pairs": -1,
                            "oversplit_max_suggestions": -1,
                        },
                        "apply_merges": False,
                        "recursive": False,
                        "iterations": {
                            "max": -1,
                            "keep_all": False,
                        },
                    },
                    "phases": {
                        "merge_units": {
                            "sequence": ["SLAy", "auto_merge", "unitmatch"],
                            "auto_merge": {
                                "enabled": True,
                                "template_diff_thresh": "0.05,0.15,0.25",
                                "relpath": "automerge_outputs",
                                "delete_outputs_on_force_restart": False,
                                "candidate_pairs_reldir": "recommended_merge_candidates",
                                "merged_units_reldir": "merged_units",
                                "auto_accept_merges": True,
                            },
                            "SLAy": {
                                "recompute_analyzer": True,
                            },
                        }
                    },
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.output_rel_root == "spikesort_outputs_v2"
    assert parsed.preprocess_concat_recording_relpath == "preprocess_outputs/preprocessed_recording"
    assert parsed.force_restart is True
    assert parsed.debug_limit_wells == 4
    assert parsed.chunk_duration == "1s"
    assert parsed.sorter == "kilosort4"
    assert parsed.docker_image == "adammwea/benshalomlab_spikesorter_pythonpatch:v3"
    assert parsed.ks_batch_duration_s == 0.75
    assert parsed.ks_th_universal == 8.0
    assert parsed.ks_th_learned == 7.0
    assert parsed.ks_th_single_ch == 5.0
    assert parsed.ks_cluster_downsampling == 15
    assert parsed.ks_nearest_chans == 12
    assert parsed.ks_max_channel_distance == 40.0

    assert isinstance(parsed.um_kwargs, dict)
    assert parsed.um_kwargs.get("merge_units") is False
    assert parsed.um_kwargs.get("dry_run") is False
    assert parsed.um_kwargs.get("output_subdir_name") == "unitmatch_outputs"
    assert parsed.um_kwargs.get("throughput_subdir_name") == "unitmatch_throughput"
    assert parsed.um_kwargs.get("max_spikes_per_unit") == -1
    assert parsed.um_kwargs.get("oversplit_min_probability") == 0.95
    assert parsed.um_kwargs.get("max_candidate_pairs") == -1
    assert parsed.um_kwargs.get("oversplit_max_suggestions") == -1
    assert parsed.um_kwargs.get("max_iterations") == -1
    assert parsed.um_kwargs.get("keep_all_iterations") is False

    assert isinstance(parsed.am_kwargs, dict)
    assert parsed.am_kwargs.get("enabled") is True
    assert parsed.am_kwargs.get("template_diff_thresh") == "0.05,0.15,0.25"
    assert parsed.force_rerun_analyzer is True
    assert isinstance(parsed.option_kwargs, dict)
    assert parsed.option_kwargs.get("force_rerun_analyzer") is True
    assert parsed.merge_sequence == ("SLAy", "auto_merge", "unitmatch")
    assert parsed.slay_recompute_analyzer is True
    assert parsed.auto_merge_enabled is True
    assert parsed.auto_merge_relpath == "automerge_outputs"
    assert parsed.auto_merge_delete_outputs_on_force_restart is False
    assert parsed.auto_merge_candidate_pairs_reldir == "recommended_merge_candidates"
    assert parsed.auto_merge_merged_units_reldir == "merged_units"
    assert parsed.auto_merge_auto_accept_merges is True
    assert parsed.auto_merge_template_diff_thresholds == (0.05, 0.15, 0.25)


def test_parse_spikesort_stage_config_promotes_percentage_sampling_to_effective_method(caplog) -> None:
    caplog.set_level("WARNING")
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "merge_units": {
                            "analyzer": {
                                "template_extraction": {
                                    "random_spikes_percentage": 75,
                                    "min_spikes_per_unit": 1000,
                                    "log_before_after_spike_counts": True,
                                    "max_spikes_per_unit": 5000,
                                    "seed": 7,
                                }
                            }
                        }
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.merge_template_random_spikes_method == "percentage"
    assert parsed.merge_template_random_spikes_percentage == pytest.approx(0.75)
    assert parsed.merge_template_random_spikes_min_spikes_per_unit == 1000
    assert parsed.merge_template_random_spikes_log_before_after_spike_counts is True
    assert parsed.merge_template_random_spikes_max_spikes_per_unit == 5000
    assert parsed.merge_template_random_spikes_seed == 7
    assert not caplog.records


def test_parse_spikesort_stage_config_accepts_legacy_percentage_alias_with_warning(caplog) -> None:
    caplog.set_level("WARNING")
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "merge_units": {
                            "analyzer": {
                                "template_extraction": {
                                    "min_perc_spikes_per_unit": 75,
                                }
                            }
                        }
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.merge_template_random_spikes_method == "percentage"
    assert parsed.merge_template_random_spikes_percentage == pytest.approx(0.75)
    assert any("min_perc_spikes_per_unit is deprecated" in record.getMessage() for record in caplog.records)


def test_parse_spikesort_stage_config_percentage_mode_does_not_impose_default_max_cap() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "merge_units": {
                            "analyzer": {
                                "template_extraction": {
                                    "random_spikes_percentage": 75,
                                }
                            }
                        }
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.merge_template_random_spikes_method == "percentage"
    assert parsed.merge_template_random_spikes_percentage == pytest.approx(0.75)
    assert parsed.merge_template_random_spikes_max_spikes_per_unit is None


def test_parse_spikesort_stage_config_rejects_invalid_percentage_sampling_values() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "merge_units": {
                            "analyzer": {
                                "template_extraction": {
                                    "random_spikes_percentage": 150,
                                }
                            }
                        }
                    }
                }
            }
        }
    )

    with pytest.raises(ValueError, match="random_spikes_percentage"):
        parse_spikesort_stage_config(runtime_config=cfg)


def test_parse_spikesort_stage_config_phase_blocks_take_precedence() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "resources": {
                        "n_jobs": 6,
                        "chunk_duration": "2s",
                    },
                    "execution": {
                        "output_root": "spikesort_outputs_exec",
                        "resume_from": "merge",
                        "sorter": "legacy_sorter_should_lose",
                        "docker_image": "legacy/image:old",
                        "run_reports": False,
                        "rerun_analyzer": False,
                    },
                    "phases": {
                        "sort": {
                            "sorter": "kilosort4",
                            "docker_image": "phase/image:new",
                            "run_reports": True,
                            "plot": {
                                "enabled": True,
                                "mode": "merged",
                                "plot_debug": True,
                                "raster_sort": "unit_id",
                                "fixed_y": True,
                            },
                            "report": {
                                "enabled": True,
                                "no_curation": True,
                                "export_to_phy": True,
                            },
                            "kilosort": {
                                "batch_duration_s": 0.5,
                                "thresholds": {
                                    "universal": 9,
                                    "learned": 8,
                                    "single_ch": 6,
                                },
                                "clustering": {
                                    "downsampling": 11,
                                },
                                "channels": {
                                    "nearest": 10,
                                    "max_distance": 38,
                                },
                            },
                        },
                        "merge_units": {
                            "rerun_analyzer": True,
                            "auto_merge": {
                                "enabled": True,
                                "template_diff_thresh": "0.11,0.22",
                            },
                            "unitmatch": {
                                "merge_units": True,
                                "dry_run": False,
                                "output_subdir_name": "um_outputs",
                                "throughput_subdir_name": "um_throughput",
                                "max_spikes_per_unit": 123,
                                "oversplit_min_probability": 0.97,
                                "limits": {
                                    "max_candidate_pairs": 456,
                                    "oversplit_max_suggestions": 789,
                                },
                                "apply_merges": True,
                                "recursive": True,
                                "iterations": {
                                    "max": 7,
                                    "keep_all": False,
                                },
                            },
                        },
                    },
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.output_rel_root == "spikesort_outputs_exec"
    assert parsed.resume_from == "merge"
    assert parsed.n_jobs == 6
    assert parsed.chunk_duration == "2s"

    assert parsed.sorter == "kilosort4"
    assert parsed.docker_image == "phase/image:new"
    assert parsed.run_reports is True
    assert parsed.plot_mode == "merged"
    assert parsed.plot_debug is True
    assert parsed.raster_sort == "unit_id"
    assert parsed.fixed_y is True
    assert parsed.no_curation is True
    assert parsed.export_to_phy is True

    assert parsed.ks_batch_duration_s == 0.5
    assert parsed.ks_th_universal == 9.0
    assert parsed.ks_th_learned == 8.0
    assert parsed.ks_th_single_ch == 6.0
    assert parsed.ks_cluster_downsampling == 11
    assert parsed.ks_nearest_chans == 10
    assert parsed.ks_max_channel_distance == 38.0

    assert parsed.force_rerun_analyzer is True
    assert isinstance(parsed.option_kwargs, dict)
    assert parsed.option_kwargs.get("force_rerun_analyzer") is True

    assert isinstance(parsed.am_kwargs, dict)
    assert parsed.am_kwargs.get("enabled") is True
    assert parsed.am_kwargs.get("template_diff_thresh") == "0.11,0.22"

    assert isinstance(parsed.um_kwargs, dict)
    assert parsed.um_kwargs.get("merge_units") is True
    assert parsed.um_kwargs.get("dry_run") is False
    assert parsed.um_kwargs.get("output_subdir_name") == "um_outputs"
    assert parsed.um_kwargs.get("throughput_subdir_name") == "um_throughput"
    assert parsed.um_kwargs.get("max_spikes_per_unit") == 123
    assert parsed.um_kwargs.get("oversplit_min_probability") == 0.97
    assert parsed.um_kwargs.get("max_candidate_pairs") == 456
    assert parsed.um_kwargs.get("oversplit_max_suggestions") == 789
    assert parsed.um_kwargs.get("apply_merges") is True
    assert parsed.um_kwargs.get("recursive") is True
    assert parsed.um_kwargs.get("max_iterations") == 7
    assert parsed.um_kwargs.get("keep_all_iterations") is False


def test_parse_spikesort_stage_config_reads_merge_analyzer_policy_knobs() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "merge_units": {
                            "analyzer": {
                                "regenerate_on_replot": False,
                                "check_if_regen_is_needed": False,
                                "compute_sparsity": False,
                                "template_random_spikes_method": "all",
                                "max_spikes_per_unit": 321,
                                "margin_size": 17,
                                "seed": 42,
                                "n_jobs": 3,
                                "chunk_duration": "0.25s",
                                "sparsity_method": "best_channels",
                                "num_channels": 9,
                                "peak_sign": "both",
                                "num_spikes_for_sparsity": 222,
                                "waveforms_ms_before": 0.75,
                                "waveforms_ms_after": 1.75,
                                "waveforms_dtype": "float32",
                            }
                        }
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.merge_analyzer_regenerate_on_replot is False
    assert parsed.merge_analyzer_check_if_regen_is_needed is False
    assert parsed.merge_analyzer_compute_sparsity is False
    assert parsed.merge_template_random_spikes_method == "all"
    assert parsed.merge_template_random_spikes_max_spikes_per_unit == 321
    assert parsed.merge_template_random_spikes_margin_size == 17
    assert parsed.merge_template_random_spikes_seed == 42
    assert parsed.merge_analyzer_n_jobs == 3
    assert parsed.merge_analyzer_chunk_duration == "0.25s"
    assert parsed.merge_analyzer_sparsity_method == "best_channels"
    assert parsed.merge_analyzer_sparsity_num_channels == 9
    assert parsed.merge_analyzer_sparsity_peak_sign == "both"
    assert parsed.merge_analyzer_sparsity_num_spikes_for_sparsity == 222
    assert parsed.merge_analyzer_waveforms_ms_before == 0.75
    assert parsed.merge_analyzer_waveforms_ms_after == 1.75
    assert parsed.merge_analyzer_waveforms_dtype == "float32"


def test_parse_spikesort_stage_config_reads_grouped_merge_analyzer_policy_knobs() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "merge_units": {
                            "analyzer": {
                                "regenerate_on_replot": False,
                                "check_if_regen_is_needed": False,
                                "n_jobs": 3,
                                "chunk_duration": "0.25s",
                                "template_extraction": {
                                    "random_spikes_method": "all",
                                    "max_spikes_per_unit": 321,
                                    "margin_size": 17,
                                    "seed": 42,
                                },
                                "sparsity": {
                                    "compute_sparsity": False,
                                    "method": "best_channels",
                                    "num_channels": 9,
                                    "peak_sign": "both",
                                    "num_spikes_for_sparsity": 222,
                                },
                                "waveforms": {
                                    "ms_before": 0.75,
                                    "ms_after": 1.75,
                                    "dtype": "float32",
                                },
                            }
                        }
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.merge_analyzer_regenerate_on_replot is False
    assert parsed.merge_analyzer_check_if_regen_is_needed is False
    assert parsed.merge_analyzer_compute_sparsity is False
    assert parsed.merge_template_random_spikes_method == "all"
    assert parsed.merge_template_random_spikes_max_spikes_per_unit == 321
    assert parsed.merge_template_random_spikes_margin_size == 17
    assert parsed.merge_template_random_spikes_seed == 42
    assert parsed.merge_analyzer_n_jobs == 3
    assert parsed.merge_analyzer_chunk_duration == "0.25s"
    assert parsed.merge_analyzer_sparsity_method == "best_channels"
    assert parsed.merge_analyzer_sparsity_num_channels == 9
    assert parsed.merge_analyzer_sparsity_peak_sign == "both"
    assert parsed.merge_analyzer_sparsity_num_spikes_for_sparsity == 222
    assert parsed.merge_analyzer_waveforms_ms_before == 0.75
    assert parsed.merge_analyzer_waveforms_ms_after == 1.75
    assert parsed.merge_analyzer_waveforms_dtype == "float32"


def test_parse_spikesort_stage_config_compute_sparsity_false_disables_sparsity_masking() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "merge_units": {
                            "analyzer": {
                                "template_extraction": {
                                },
                                "sparsity": {
                                    "compute_sparsity": False,
                                    "method": "best_channels",
                                    "num_channels": 11,
                                },
                            }
                        }
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.merge_analyzer_compute_sparsity is False
    assert parsed.merge_analyzer_sparsity_method == "best_channels"
    assert parsed.merge_analyzer_sparsity_num_channels == 11


def test_parse_spikesort_stage_config_reads_legacy_merge_analyzer_regenereate_on_replot_key() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "merge_units": {
                            "analyzer": {
                                "regenereate_on_replot": False,
                            }
                        }
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.merge_analyzer_regenerate_on_replot is False


def test_parse_spikesort_stage_config_unitmatch_enabled_false_disables_merge_units() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "execution": {
                        "um_kwargs": {
                            "merge_units": True,
                        }
                    },
                    "phases": {
                        "merge_unitmatch": {
                            "enabled": False,
                            "merge_units": True,
                        }
                    },
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert isinstance(parsed.um_kwargs, dict)
    assert parsed.um_kwargs.get("enabled") is False
    assert parsed.um_kwargs.get("merge_units") is False


def test_parse_spikesort_stage_config_reads_slay_merge_knobs() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "sort": {
                            "enabled": False,
                            "delete_outputs_on_force_restart": True,
                        },
                        "merge_SLAy": {
                            "enabled": True,
                            "relpath": "SLAy_outputs_custom",
                            "sorter_output_relpath": "spikesort_outputs/sorter_output/sorter_output",
                            "output_json_relpath": "reports/slay_run_output.json",
                            "candidate_pairs_relpath": "candidates/recommended.tsv",
                            "merge_groups_relpath": "candidates/groups.json",
                            "allow_numpy_fallback": False,
                            "plot_merges": True,
                            "auto_accept_merges": False,
                            "copy_automerge_artifacts": False,
                            "delete_outputs_on_force_restart": False,
                            "recompute_analyzer": True,
                            "model_cache": {
                                "enabled": True,
                                "relpath": "cache/slay_model/custom_ae.pt",
                                "use_cached_model": False,
                                "write_model_cache": True,
                            },
                            "force_restart_retrain_model": True,
                            "params": {
                                "max_spikes": 250,
                                "final_thresh": 0.6,
                            },
                        },
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.sort_enabled is False
    assert parsed.sort_delete_outputs_on_force_restart is True
    assert parsed.slay_enabled is True
    assert parsed.slay_relpath == "SLAy_outputs_custom"
    assert parsed.slay_sorter_output_relpath == "spikesort_outputs/sorter_output/sorter_output"
    assert parsed.slay_output_json_relpath == "reports/slay_run_output.json"
    assert parsed.slay_candidate_pairs_relpath == "candidates/recommended.tsv"
    assert parsed.slay_merge_groups_relpath == "candidates/groups.json"
    assert parsed.slay_allow_numpy_fallback is False
    assert parsed.slay_plot_merges is True
    assert parsed.slay_auto_accept_merges is False
    assert parsed.slay_copy_automerge_artifacts is False
    assert parsed.slay_delete_outputs_on_force_restart is False
    assert parsed.slay_recompute_analyzer is True
    assert parsed.slay_model_cache_relpath == "cache/slay_model/custom_ae.pt"
    assert parsed.slay_model_cache_use_cached_model is False
    assert parsed.slay_model_cache_write_model is True
    assert parsed.slay_force_restart_retrain_model is True
    assert isinstance(parsed.slay_params, dict)
    assert parsed.slay_params.get("max_spikes") == 250
    assert parsed.slay_params.get("final_thresh") == 0.6


def test_parse_spikesort_stage_config_reads_nested_merge_slay_block() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "merge_SLAy": {
                            "enabled": True,
                            "slay": {
                                "relpath": "SLAy_nested",
                                "output_json_relpath": "nested/run-output.json",
                                "recompute_analyzer": False,
                                "params": {"final_thresh": 0.42},
                            },
                        }
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.slay_enabled is True
    assert parsed.slay_relpath == "SLAy_nested"
    assert parsed.slay_output_json_relpath == "nested/run-output.json"
    assert parsed.slay_recompute_analyzer is False
    assert isinstance(parsed.slay_params, dict)
    assert parsed.slay_params.get("final_thresh") == 0.42


def test_parse_spikesort_stage_config_reads_bombcell_phase_knobs() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "bombcell_label": {
                            "enabled": True,
                            "relpath": "merge_output/bombcell_labels",
                            "delete_outputs_on_force_restart": False,
                            "cache_sorter_output_before_analyzer_gen": True,
                            "publish_cached_sorter_output_on_success": False,
                            "publish_cached_analyzer_on_success": True,
                            "cleanup_on_success": {
                                "analyzer": True,
                                "cached_sorter_output": True,
                            },
                            "params": {
                                "thresholds": {
                                    "noise": {
                                        "snr": {
                                            "greater": 4.0,
                                        }
                                    }
                                },
                                "thresholds_path": "/tmp/bombcell_thresholds.json",
                                "label_non_somatic": False,
                                "split_non_somatic_good_mua": True,
                            },
                            "apply_to_sorter_output": True,
                            "write_cluster_group": False,
                            "fail_on_error": True,
                            "reports": {
                                "enabled": True,
                                "summary_json": {
                                    "enabled": True,
                                    "relpath": "reports/bombcell_label_summary.json",
                                },
                            },
                        }
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.bombcell_label_enabled is True
    assert parsed.bombcell_label_relpath == "merge_output/bombcell_labels"
    assert parsed.bombcell_label_delete_outputs_on_force_restart is False
    assert parsed.bombcell_label_cache_sorter_output_before_analyzer_gen is True
    assert parsed.bombcell_label_publish_cached_sorter_output_on_success is False
    assert parsed.bombcell_label_publish_cached_analyzer_on_success is True
    assert parsed.bombcell_label_cleanup_analyzer_on_success is True
    assert parsed.bombcell_label_cleanup_cached_sorter_output_on_success is True
    assert isinstance(parsed.bombcell_label_thresholds, dict)
    assert parsed.bombcell_label_thresholds.get("noise", {}).get("snr", {}).get("greater") == 4.0
    assert parsed.bombcell_label_thresholds_path == "/tmp/bombcell_thresholds.json"
    assert parsed.bombcell_label_label_non_somatic is False
    assert parsed.bombcell_label_split_non_somatic_good_mua is True
    assert parsed.bombcell_label_apply_to_sorter_output is True
    assert parsed.bombcell_label_write_cluster_group is False
    assert parsed.bombcell_label_fail_on_error is True
    assert parsed.bombcell_label_reports_enabled is True
    assert parsed.bombcell_label_reports_summary_json_enabled is True
    assert parsed.bombcell_label_reports_summary_json_relpath == "reports/bombcell_label_summary.json"


def test_parse_spikesort_stage_config_bombcell_params_take_precedence_over_top_level() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "bombcell_label": {
                            "label_non_somatic": True,
                            "split_non_somatic_good_mua": False,
                            "params": {
                                "label_non_somatic": False,
                                "split_non_somatic_good_mua": True,
                            },
                        }
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.bombcell_label_label_non_somatic is False
    assert parsed.bombcell_label_split_non_somatic_good_mua is True


def test_parse_spikesort_stage_config_enables_bombcell_when_phase_block_present() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "bombcell_label": {},
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.bombcell_label_enabled is True


def test_parse_spikesort_stage_config_reads_auto_merge_knobs() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "merge_sequence": ["SLAy", "auto_merge"],
                        "merge_si_auto": {
                            "enabled": True,
                            "template_diff_thresh": "0.07,0.11",
                            "relpath": "merge_outputs/automerge",
                            "delete_outputs_on_force_restart": False,
                            "candidate_pairs_reldir": "pairs_by_iteration",
                            "merged_units_reldir": "merged_units_by_iteration",
                            "auto_accept_merges": True,
                        },
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.merge_sequence == ("SLAy", "auto_merge")
    assert parsed.auto_merge_enabled is True
    assert parsed.auto_merge_relpath == "merge_outputs/automerge"
    assert parsed.auto_merge_delete_outputs_on_force_restart is False
    assert parsed.auto_merge_candidate_pairs_reldir == "pairs_by_iteration"
    assert parsed.auto_merge_merged_units_reldir == "merged_units_by_iteration"
    assert parsed.auto_merge_auto_accept_merges is True
    assert parsed.auto_merge_template_diff_thresholds == (0.07, 0.11)


def test_parse_spikesort_stage_config_reads_phase_local_merge_common_overrides() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "merge_sequence": ["SLAy", "auto_merge"],
                        "merge_SLAy": {
                            "enabled": True,
                            "rel_output_root": "merge_output/custom_slay",
                            "force_restart": True,
                            "analyzer": {
                                "n_jobs": 7,
                            },
                            "pre_merge_metadata": {
                                "enabled": False,
                            },
                            "post_merge_metadata": {
                                "json_relpath": "reports/slay_post.json",
                            },
                            "reports": {
                                "enabled": False,
                                "unit_diff_json": {
                                    "enabled": False,
                                },
                            },
                            "working_cache": {
                                "enabled": True,
                                "relpath": "cache/slay_workspace",
                                "refresh_on_run": False,
                                "publish_to_canonical_on_success": True,
                                "publish_to_canonical_on_failure": False,
                            },
                        },
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.merge_sequence == ("SLAy", "auto_merge")
    assert isinstance(parsed.merge_phase_runtime_overrides, dict)
    slay_overrides = parsed.merge_phase_runtime_overrides.get("merge_slay")
    assert isinstance(slay_overrides, dict)
    assert slay_overrides.get("merge_rel_output_root") == "merge_output/custom_slay"
    assert slay_overrides.get("merge_force_restart") is True
    assert slay_overrides.get("merge_analyzer_n_jobs") == 7
    assert slay_overrides.get("pre_merge_metadata_enabled") is False
    assert slay_overrides.get("post_merge_metadata_json_relpath") == "reports/slay_post.json"
    assert slay_overrides.get("merge_reports_enabled") is False
    assert slay_overrides.get("merge_reports_unit_diff_json_enabled") is False
    assert slay_overrides.get("cache_sorting_outputs_before_merge_use_canonical_workspace") is True
    assert (
        slay_overrides.get("cache_sorting_outputs_before_merge_canonical_workspace_relpath")
        == "cache/slay_workspace"
    )


def test_parse_spikesort_stage_config_reads_merge_slay_phase_knobs() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "merge_SLAy": {
                            "enabled": True,
                            "rel_output_root": "merge_outputs/slay_only",
                            "delete_outputs_on_force_restart": True,
                            "force_restart": True,
                            "force_replot": False,
                            "debug_mode": {
                                "enabled": True,
                                "limit_datasets": 1,
                                "limit_wells": 1,
                            },
                            "working_cache": {
                                "enabled": True,
                                "relpath": "cache/slay_workspace",
                                "refresh_on_run": False,
                                "publish_to_canonical_on_success": True,
                                "publish_to_canonical_on_failure": False,
                                "assert_selected_sorter_output": False,
                            },
                        }
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.merge_slay_enabled is True
    assert parsed.merge_slay_rel_output_root == "merge_outputs/slay_only"
    assert parsed.merge_slay_delete_outputs_on_force_restart is True
    assert parsed.merge_slay_force_restart is True
    assert parsed.merge_slay_force_replot is False
    assert parsed.merge_slay_use_canonical_workspace is True
    assert parsed.merge_slay_canonical_workspace_relpath == "cache/slay_workspace"
    assert parsed.merge_slay_canonical_workspace_refresh_on_run is False
    assert parsed.merge_slay_canonical_workspace_rebuild_analyzer is False
    assert parsed.merge_slay_publish_canonical_to_stage_outputs_on_success is True
    assert parsed.merge_slay_publish_canonical_to_stage_outputs_on_failure is False
    assert parsed.merge_slay_assert_uses_canonical_workspace is False
    assert parsed.merge_slay_debug_mode_enabled is True
    assert parsed.merge_slay_debug_limit_datasets == 1
    assert parsed.merge_slay_debug_limit_wells == 1


def test_parse_spikesort_stage_config_reads_merge_metadata_knobs() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "merge_units": {
                            "merge_metadata": {
                                "enabled": True,
                                "write_json": True,
                                "json_relpath": "metadata/merge_metadata_summary.json",
                                "include_unit_locations": False,
                                "log_summary_details": True,
                            }
                        }
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.merge_metadata_enabled is True
    assert parsed.merge_metadata_write_json is True
    assert parsed.merge_metadata_json_relpath == "metadata/merge_metadata_summary.json"
    assert parsed.merge_metadata_include_unit_locations is False
    assert parsed.merge_metadata_log_summary_details is True


def test_parse_spikesort_stage_config_reads_merge_phase_master_enable_and_cache_knobs() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "merge_units": {
                            "enabled": False,
                            "rel_output_root": "merge_outputs",
                            "delete_outputs_on_force_restart": True,
                            "cache_sorting_outputs_before_merge": {
                                "enabled": True,
                                "relpath": "cache/pre_merge_cache",
                                "cleanup_on_success": True,
                                "use_cache_on_force_restart": True,
                            },
                        }
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.merge_units_enabled is False
    assert parsed.merge_rel_output_root == "merge_outputs"
    assert parsed.merge_delete_outputs_on_force_restart is True
    assert parsed.merge_force_restart is False
    assert parsed.merge_force_replot is False
    assert parsed.cache_sorting_outputs_before_merge is True
    assert parsed.cache_sorting_outputs_before_merge_relpath == "cache/pre_merge_cache"
    assert parsed.cache_sorting_outputs_before_merge_cleanup_on_success is True
    assert parsed.cache_sorting_outputs_before_merge_use_cache_on_force_restart is True
    assert parsed.cache_sorting_outputs_before_merge_replace_sorting_with_cache_before_force_restart is True
    assert parsed.cache_sorting_outputs_before_merge_use_canonical_workspace is False
    assert parsed.cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_success is False
    assert parsed.cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_failure is False


def test_parse_spikesort_stage_config_reads_working_cache_knobs() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "merge_units": {
                            "working_cache": {
                                "enabled": True,
                                "relpath": "cache/merge_workspace",
                                "refresh_on_run": False,
                                "publish_to_canonical_on_success": True,
                                "publish_to_canonical_on_failure": True,
                                "assert_selected_sorter_output": False,
                            },
                        }
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.cache_sorting_outputs_before_merge is False
    assert parsed.cache_sorting_outputs_before_merge_use_canonical_workspace is True
    assert parsed.cache_sorting_outputs_before_merge_canonical_workspace_relpath == "cache/merge_workspace"
    assert parsed.cache_sorting_outputs_before_merge_canonical_workspace_refresh_on_run is False
    assert parsed.cache_sorting_outputs_before_merge_canonical_workspace_rebuild_analyzer is False
    assert parsed.cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_success is True
    assert parsed.cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_failure is True
    assert parsed.cache_sorting_outputs_before_merge_assert_slay_uses_canonical_workspace is False
    assert parsed.cache_sorting_outputs_before_merge_assert_auto_merge_uses_canonical_workspace is False


def test_parse_spikesort_stage_config_merge_force_knobs_inherit_global_toggles() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "execution": {
                        "force_restart": True,
                        "force_replot": True,
                    },
                    "phases": {
                        "merge_units": {
                            "force_restart": False,
                            "force_replot": False,
                        }
                    },
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.force_restart is True
    assert parsed.force_replot is True
    assert parsed.merge_force_restart is True
    assert parsed.merge_force_replot is True


def test_parse_spikesort_stage_config_reads_merge_reports_2panel_knobs() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "merge_units": {
                            "reports": {
                                "enabled": True,
                                "unit_diff_json": {
                                    "enabled": True,
                                    "relpath": "reports/unit_diffs_after_merge.json",
                                },
                                "2panels_unit_locations_before_after_merge": {
                                    "enabled": True,
                                    "point_size": 7.5,
                                    "relpath": "reports/2panels_unit_locations_before_after_merge",
                                    "label_pre_and_post_units": True,
                                    "write_png": True,
                                    "write_svg": True,
                                    "inherit_probe_dimensions": True,
                                    "zoom_to_affected_units": True,
                                    "probe_dim_x_um": 3850,
                                    "probe_dim_y_um": 2100,
                                    "highlight_merges": {
                                        "enabled": True,
                                        "linked_highlight": True,
                                        "plot_after_other_units": True,
                                        "label_affected_units": True,
                                        "show_legend": True,
                                        "legend_position": "upper left",
                                        "legend_x": -0.35,
                                        "legend_y": 0.85,
                                        "sort_pre_legend_by_groups": True,
                                        "debug_json": False,
                                        "debug_json_relpath": "reports/highlight_linkage.json",
                                        "before_color": "#ff0000",
                                        "after_color": "#00ff00",
                                        "palette": "Set1",
                                    },
                                    "assets": {
                                        "before": {
                                            "relpath": "reports/unit_locations_before_merge",
                                            "write_png": False,
                                            "write_svg": True,
                                            "point_color": "#111111",
                                        },
                                        "after": {
                                            "relpath": "reports/unit_locations_after_merge",
                                            "write_png": True,
                                            "write_svg": False,
                                            "point_color": "#222222",
                                        },
                                    },
                                },
                            }
                        }
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.merge_reports_enabled is True
    assert parsed.merge_reports_unit_diff_json_enabled is True
    assert parsed.merge_reports_unit_diff_json_relpath == "reports/unit_diffs_after_merge.json"
    assert parsed.merge_reports_2panel_enabled is True
    assert parsed.merge_reports_2panel_point_size == 7.5
    assert parsed.merge_reports_2panel_relpath == "reports/2panels_unit_locations_before_after_merge"
    assert parsed.merge_reports_2panel_label_pre_and_post_units is True
    assert parsed.merge_reports_2panel_write_png is True
    assert parsed.merge_reports_2panel_write_svg is True
    assert parsed.merge_reports_2panel_before_relpath == "reports/unit_locations_before_merge"
    assert parsed.merge_reports_2panel_before_write_png is False
    assert parsed.merge_reports_2panel_before_write_svg is True
    assert parsed.merge_reports_2panel_before_point_color == "#111111"
    assert parsed.merge_reports_2panel_after_relpath == "reports/unit_locations_after_merge"
    assert parsed.merge_reports_2panel_after_write_png is True
    assert parsed.merge_reports_2panel_after_write_svg is False
    assert parsed.merge_reports_2panel_after_point_color == "#222222"
    assert parsed.merge_reports_2panel_highlight_merges_enabled is True
    assert parsed.merge_reports_2panel_highlight_merges_linked is True
    assert parsed.merge_reports_2panel_highlight_plot_after_other_units is True
    assert parsed.merge_reports_2panel_highlight_label_affected_units is True
    assert parsed.merge_reports_2panel_highlight_show_legend is True
    assert parsed.merge_reports_2panel_highlight_legend_position == "upper left"
    assert parsed.merge_reports_2panel_highlight_legend_x == -0.35
    assert parsed.merge_reports_2panel_highlight_legend_y == 0.85
    assert parsed.merge_reports_2panel_highlight_sort_pre_legend_by_groups is True
    assert parsed.merge_reports_2panel_highlight_debug_json_enabled is False
    assert parsed.merge_reports_2panel_highlight_debug_json_relpath == "reports/highlight_linkage.json"
    assert parsed.merge_reports_2panel_highlight_before_color == "#ff0000"
    assert parsed.merge_reports_2panel_highlight_after_color == "#00ff00"
    assert parsed.merge_reports_2panel_highlight_palette == "Set1"
    assert parsed.merge_reports_2panel_inherit_probe_dimensions is True
    assert parsed.merge_reports_2panel_zoom_to_affected_units is True
    assert parsed.merge_reports_2panel_probe_dim_x_um == 3850.0
    assert parsed.merge_reports_2panel_probe_dim_y_um == 2100.0


def test_parse_spikesort_stage_config_reads_merge_template_heatmap_report_knobs() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "merge_units": {
                            "reports": {
                                "enabled": True,
                                "template_heatmaps_per_merge": {
                                    "enabled": True,
                                    "relpath": "reports/template_heatmaps",
                                    "write_png": True,
                                    "write_svg": True,
                                    "inherit_probe_dimensions": True,
                                    "probe_dim_x_um": 3850,
                                    "probe_dim_y_um": 2100,
                                    "pitch_um": 17.5,
                                    "electrode_size_um": {
                                        "x": 12.0,
                                        "y": 8.8,
                                    },
                                    "panel_width_in": 12.5,
                                    "panel_height_in": 7.25,
                                    "marker_size": 22.0,
                                    "cmap": "magma",
                                    "show_colorbar": False,
                                    "relative_color_bar_height": 0.5,
                                    "color_scale": "log",
                                    "log_epsilon": 0.005,
                                    "max_merges": 14,
                                    "debug_json_relpath": "reports/template_heatmap_debug.json",
                                    "assets": {
                                        "relpath": "reports/template_heatmaps/assets",
                                        "write_png": True,
                                        "write_svg": True,
                                    },
                                },
                            }
                        }
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.merge_reports_template_heatmaps_enabled is True
    assert parsed.merge_reports_template_heatmaps_relpath == "reports/template_heatmaps"
    assert parsed.merge_reports_template_heatmaps_assets_reldir == "reports/template_heatmaps/assets"
    assert parsed.merge_reports_template_heatmaps_write_png is True
    assert parsed.merge_reports_template_heatmaps_write_svg is True
    assert parsed.merge_reports_template_heatmaps_write_assets_png is True
    assert parsed.merge_reports_template_heatmaps_write_assets_svg is True
    assert parsed.merge_reports_template_heatmaps_panel_width_in == 12.5
    assert parsed.merge_reports_template_heatmaps_panel_height_in == 7.25
    assert parsed.merge_reports_template_heatmaps_marker_size == 22.0
    assert parsed.merge_reports_template_heatmaps_cmap == "magma"
    assert parsed.merge_reports_template_heatmaps_show_colorbar is False
    assert parsed.merge_reports_template_heatmaps_relative_color_bar_height == 0.5
    assert parsed.merge_reports_template_heatmaps_color_scale == "log"
    assert parsed.merge_reports_template_heatmaps_log_epsilon == 0.005
    assert parsed.merge_reports_template_heatmaps_magnitude_mode == "ptp"
    assert parsed.merge_reports_template_heatmaps_max_merges == 14
    assert parsed.merge_reports_template_heatmaps_debug_json_relpath == "reports/template_heatmap_debug.json"
    assert parsed.merge_reports_template_heatmaps_inherit_probe_dimensions is True
    assert parsed.merge_reports_template_heatmaps_probe_dim_x_um == 3850.0
    assert parsed.merge_reports_template_heatmaps_probe_dim_y_um == 2100.0
    assert parsed.merge_reports_template_heatmaps_probe_pitch_um == 17.5
    assert parsed.merge_reports_template_heatmaps_probe_electrode_size_um_x == 12.0
    assert parsed.merge_reports_template_heatmaps_probe_electrode_size_um_y == 8.8


def test_parse_spikesort_stage_config_reads_merge_template_heatmap_magnitude_mode() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "merge_units": {
                            "reports": {
                                "template_heatmaps_per_merge": {
                                    "magnitude_mode": "abs_peak",
                                },
                            }
                        }
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.merge_reports_template_heatmaps_magnitude_mode == "abs_peak"


def test_parse_spikesort_stage_config_reads_merge_reports_unit_diff_json_knobs() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "merge_units": {
                            "reports": {
                                "enabled": True,
                                "unit_diff_json": {
                                    "enabled": True,
                                    "relpath": "merge_outputs/reports/unit_diffs_after_merge.json",
                                },
                            }
                        }
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.merge_reports_enabled is True
    assert parsed.merge_reports_unit_diff_json_enabled is True
    assert parsed.merge_reports_unit_diff_json_relpath == "merge_outputs/reports/unit_diffs_after_merge.json"


def test_parse_spikesort_stage_config_reads_merge_reports_unit_diff_map_knobs() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "merge_units": {
                            "reports": {
                                "enabled": True,
                                "unit_diff_map": {
                                    "enabled": True,
                                    "relpath": "merge_outputs/reports/unit_diff_map.json",
                                },
                                "unit_diff_map_flat": {
                                    "enabled": True,
                                    "relpath": "merge_outputs/reports/unit_diff_map_flat.json",
                                },
                                "post_merge_unit_locations": {
                                    "enabled": True,
                                    "relpath": "merge_outputs/reports/post_merge_unit_locations.json",
                                },
                            }
                        }
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.merge_reports_enabled is True
    assert parsed.merge_reports_unit_diff_map_enabled is True
    assert parsed.merge_reports_unit_diff_map_relpath == "merge_outputs/reports/unit_diff_map.json"
    assert parsed.merge_reports_unit_diff_map_flat_enabled is True
    assert parsed.merge_reports_unit_diff_map_flat_relpath == "merge_outputs/reports/unit_diff_map_flat.json"
    assert parsed.merge_reports_post_merge_unit_locations_enabled is True
    assert (
        parsed.merge_reports_post_merge_unit_locations_relpath
        == "merge_outputs/reports/post_merge_unit_locations.json"
    )


def test_parse_spikesort_stage_config_supports_legacy_boolean_cache_flag() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "merge_units": {
                            "cache_sorting_outputs_before_merge": True,
                        }
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.cache_sorting_outputs_before_merge is True
    assert parsed.cache_sorting_outputs_before_merge_relpath == "pre_merge_cache"
    assert parsed.cache_sorting_outputs_before_merge_cleanup_on_success is False
    assert parsed.cache_sorting_outputs_before_merge_use_cache_on_force_restart is False
    assert parsed.cache_sorting_outputs_before_merge_replace_sorting_with_cache_before_force_restart is False


def test_parse_spikesort_stage_config_reads_replace_sorting_cache_alias() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "merge_units": {
                            "cache_sorting_outputs_before_merge": {
                                "enabled": True,
                                "replace_sorting_with_cache_before_force_restart": True,
                            },
                        }
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.cache_sorting_outputs_before_merge is True
    assert parsed.cache_sorting_outputs_before_merge_replace_sorting_with_cache_before_force_restart is True
    assert parsed.cache_sorting_outputs_before_merge_use_cache_on_force_restart is True


def test_parse_spikesort_stage_config_reads_legacy_pre_post_blocks_independently() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "merge_units": {
                            "pre_merge_metadata": {
                                "enabled": True,
                                "write_json": True,
                                "json_relpath": "legacy/pre_merge_sorting_metadata.json",
                            },
                            "post_merge_metadata": {
                                "enabled": True,
                                "write_json": False,
                                "json_relpath": "legacy/post_merge_sorting_metadata.json",
                            },
                        }
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.merge_metadata_enabled is False
    assert parsed.merge_metadata_write_json is True
    assert parsed.merge_metadata_json_relpath == "merge_metadata_summary.json"
    assert parsed.pre_merge_metadata_enabled is True
    assert parsed.pre_merge_metadata_write_json is True
    assert parsed.pre_merge_metadata_json_relpath == "legacy/pre_merge_sorting_metadata.json"
    assert parsed.post_merge_metadata_enabled is True
    assert parsed.post_merge_metadata_write_json is False
    assert parsed.post_merge_metadata_json_relpath == "legacy/post_merge_sorting_metadata.json"


def test_parse_spikesort_stage_config_force_overrides_take_precedence() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "execution": {
                        "force_restart": False,
                        "force_replot": True,
                        "sorter": "kilosort2_5",
                        "n_jobs": 8,
                    },
                    "outputs": {
                        "output_root": "/spikesort_v2",
                    },
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(
        runtime_config=cfg,
        force_restart_override=True,
        force_replot_override=False,
    )

    assert parsed.force_restart is True
    assert parsed.force_replot is False
    assert parsed.sorter == "kilosort2_5"
    assert parsed.n_jobs == 8
    assert parsed.output_rel_root == "spikesort_v2"


def test_parse_spikesort_stage_config_reads_logging_debug_plot_report_blocks() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "execution": {
                        "run_reports": True,
                        "no_curation": False,
                        "export_to_phy": False,
                    },
                    "logging": {
                        "enabled": False,
                        "verbose": True,
                        "file_relpath": "logs/custom_spikesort.log",
                    },
                    "debug_outputs": True,
                    "debug": {
                        "limit_wells": 2,
                    },
                    "plot": {
                        "enabled": True,
                        "mode": "merged",
                        "plot_debug": True,
                        "raster_sort": "firing_rate",
                        "fixed_y": True,
                    },
                    "report": {
                        "enabled": True,
                        "no_curation": True,
                        "export_to_phy": True,
                    },
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.logging_enabled is False
    assert parsed.logging_verbose is True
    assert parsed.logging_file_relpath == "logs/custom_spikesort.log"
    assert parsed.debug_outputs is True
    assert parsed.debug_limit_wells == 2
    assert parsed.run_reports is True
    assert parsed.plot_enabled is True
    assert parsed.plot_mode == "merged"
    assert parsed.plot_debug is True
    assert parsed.raster_sort == "firing_rate"
    assert parsed.fixed_y is True
    assert parsed.no_curation is True
    assert parsed.export_to_phy is True


def test_parse_spikesort_stage_config_plot_disabled_forces_reports_off() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "execution": {
                        "run_reports": True,
                    },
                    "plot": {
                        "enabled": False,
                    },
                    "report": {
                        "enabled": True,
                    },
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.plot_enabled is False
    assert parsed.run_reports is False


def test_parse_spikesort_stage_config_execution_delete_outputs_on_force_restart_alias_enabled() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "execution": {
                        "delete_outputs_on_force_restart": True,
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.sort_delete_outputs_on_force_restart is True


def test_parse_spikesort_stage_config_sort_phase_delete_flag_overrides_execution_alias() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "execution": {
                        "delete_outputs_on_force_restart": True,
                    },
                    "phases": {
                        "sort": {
                            "delete_outputs_on_force_restart": False,
                        }
                    },
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.sort_delete_outputs_on_force_restart is False


def test_parse_spikesort_stage_config_supports_top_level_legacy_keys() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "sorter": "kilosort4",
                    "docker_image": "test/image:legacy",
                    "rerun_analyzer": True,
                    "resume_from": "merge",
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.sorter == "kilosort4"
    assert parsed.docker_image == "test/image:legacy"
    assert parsed.force_rerun_analyzer is True
    assert parsed.no_curation is False
    assert parsed.resume_from == "merge"


def test_parse_spikesort_stage_config_ignores_legacy_curation_flag() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "curation": False,
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.no_curation is False


def test_parse_spikesort_stage_config_normalizes_legacy_output_rel_root() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "outputs": {
                        "output_rel_root": "stg2_spikesorting_outputs",
                    }
                }
            }
        }
    )

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.output_rel_root == "spikesort_outputs"


def test_load_spikesort_inputs_from_runtime_defaults_and_overrides(tmp_path: Path) -> None:
    data_path = tmp_path / "data.yml"
    data_path.write_text(
        dedent(
            """
            output_root: /tmp/out
            datasets:
              - raw_data_h5_path: /tmp/input.raw.h5
                include_in_runtime: true
                wells:
                  - well_id: well006
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    runtime_path = tmp_path / "runtime.yml"
    runtime_path.write_text(
        "\n".join(
            [
                f"data: {data_path}",
                "stages:",
                "  spikesort:",
                "    execution:",
                "      sorter: kilosort4",
                "      docker_image: test/image:latest",
                "      n_jobs: 4",
                "      run_reports: false",
                "    logging: {enabled: true, verbose: true, file_relpath: logs/spikesort_pipeline.log}",
                "    plot: {mode: merged, plot_debug: true, raster_sort: unit_id, fixed_y: true}",
                "    outputs:",
                "      output_root: spikesort_stage_outputs",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inputs = load_spikesort_inputs_from_runtime(
        config_path=str(runtime_path),
        force_replot_override=True,
    )

    assert inputs.stream_id == "well006"
    assert inputs.output_rel_root == "spikesort_stage_outputs"
    assert inputs.sorter == "kilosort4"
    assert inputs.docker_image == "test/image:latest"
    assert inputs.logging_enabled is True
    assert inputs.logging_verbose is True
    assert inputs.logging_file_relpath == "logs/spikesort_pipeline.log"
    assert inputs.n_jobs == 4
    assert inputs.run_reports is False
    assert inputs.plot_mode == "merged"
    assert inputs.plot_debug is True
    assert inputs.raster_sort == "unit_id"
    assert inputs.fixed_y is True
    assert inputs.force_restart is False
    assert inputs.force_replot is True
    assert inputs.merge_analyzer_compute_sparsity is True
    assert inputs.merge_template_random_spikes_method == "default"
    assert inputs.merge_template_random_spikes_max_spikes_per_unit == 500
    assert inputs.merge_analyzer_sparsity_peak_sign == "neg"
    assert inputs.merge_analyzer_sparsity_num_spikes_for_sparsity == 100
    assert inputs.merge_analyzer_waveforms_ms_before == 1.0
    assert inputs.merge_analyzer_waveforms_ms_after == 2.0


def test_load_spikesort_inputs_from_runtime_reads_stage_level_input_and_output_root(tmp_path: Path) -> None:
    data_path = tmp_path / "data.yml"
    data_path.write_text(
        dedent(
            """
            output_root: /tmp/out
            datasets:
              - raw_data_h5_path: /tmp/input.raw.h5
                include_in_runtime: true
                wells:
                  - well_id: well006
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    runtime_path = tmp_path / "runtime.yml"
    runtime_path.write_text(
        "\n".join(
            [
                f"data: {data_path}",
                "stages:",
                "  spikesort:",
                "    resources:",
                "      chunk_duration: 1s",
                "    execution:",
                "      sorter: kilosort4",
                "    inputs:",
                "      preprocess_concat_recording_relpath: preprocess_outputs/preprocessed_recording",
                "    output_root: spikesort_stage_outputs",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inputs = load_spikesort_inputs_from_runtime(config_path=str(runtime_path))

    assert inputs.stream_id == "well006"
    assert inputs.output_rel_root == "spikesort_stage_outputs"
    assert inputs.preprocess_concat_recording_relpath == "preprocess_outputs/preprocessed_recording"
    assert inputs.sort_original_preprocess_concat_recording_relpath == "preprocess_outputs/preprocessed_recording"
    assert inputs.sort_bootstrapped_concat_recording_relpath == "spikesort_stage_outputs/cache/bootstrap_concat_binary/recording"
    assert inputs.sort_use_bootstrapped_concat_binary is False
    assert inputs.sort_use_lazy_source is True
    assert inputs.sort_assert_one_source is False
    assert inputs.chunk_duration == "1s"


def test_load_spikesort_inputs_from_runtime_reads_grouped_merge_analyzer_policy_knobs(tmp_path: Path) -> None:
    data_path = tmp_path / "data.yml"
    data_path.write_text(
        dedent(
            """
            output_root: /tmp/out
            datasets:
              - raw_data_h5_path: /tmp/input.raw.h5
                include_in_runtime: true
                wells:
                  - well_id: well006
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    runtime_path = tmp_path / "runtime.yml"
    runtime_path.write_text(
        "\n".join(
            [
                f"data: {data_path}",
                "stages:",
                "  spikesort:",
                "    phases:",
                "      merge_units:",
                "        analyzer:",
                "          n_jobs: 2",
                "          chunk_duration: 0.5s",
                "          template_extraction:",
                "            random_spikes_method: all",
                "            max_spikes_per_unit: 321",
                "            margin_size: 11",
                "            seed: 7",
                "          sparsity:",
                "            compute_sparsity: false",
                "            method: threshold",
                "            threshold: 4.5",
                "            peak_sign: both",
                "            num_spikes_for_sparsity: 222",
                "          waveforms:",
                "            ms_before: 0.8",
                "            ms_after: 1.6",
                "            dtype: float32",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inputs = load_spikesort_inputs_from_runtime(config_path=str(runtime_path))

    assert inputs.merge_analyzer_compute_sparsity is False
    assert inputs.merge_template_random_spikes_method == "all"
    assert inputs.merge_template_random_spikes_max_spikes_per_unit == 321
    assert inputs.merge_template_random_spikes_margin_size == 11
    assert inputs.merge_template_random_spikes_seed == 7
    assert inputs.merge_analyzer_n_jobs == 2
    assert inputs.merge_analyzer_chunk_duration == "0.5s"
    assert inputs.merge_analyzer_sparsity_method == "threshold"
    assert inputs.merge_analyzer_sparsity_threshold == 4.5
    assert inputs.merge_analyzer_sparsity_peak_sign == "both"
    assert inputs.merge_analyzer_sparsity_num_spikes_for_sparsity == 222
    assert inputs.merge_analyzer_waveforms_ms_before == 0.8
    assert inputs.merge_analyzer_waveforms_ms_after == 1.6
    assert inputs.merge_analyzer_waveforms_dtype == "float32"


def test_load_spikesort_inputs_from_runtime_reads_percentage_sampling_threshold_knob(tmp_path: Path) -> None:
    data_path = tmp_path / "data.yml"
    data_path.write_text(
        dedent(
            """
            output_root: /tmp/out
            datasets:
              - raw_data_h5_path: /tmp/input.raw.h5
                include_in_runtime: true
                wells:
                  - well_id: well006
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    runtime_path = tmp_path / "runtime.yml"
    runtime_path.write_text(
        "\n".join(
            [
                f"data: {data_path}",
                "stages:",
                "  spikesort:",
                "    phases:",
                "      merge_units:",
                "        analyzer:",
                "          template_extraction:",
                "            random_spikes_percentage: 75",
                "            min_spikes_per_unit: 1000",
                "            log_before_after_spike_counts: true",
                "            max_spikes_per_unit: 5000",
                "            seed: 7",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inputs = load_spikesort_inputs_from_runtime(config_path=str(runtime_path))

    assert inputs.merge_template_random_spikes_method == "percentage"
    assert inputs.merge_template_random_spikes_percentage == pytest.approx(0.75)
    assert inputs.merge_template_random_spikes_min_spikes_per_unit == 1000
    assert inputs.merge_template_random_spikes_log_before_after_spike_counts is True
    assert inputs.merge_template_random_spikes_max_spikes_per_unit == 5000
    assert inputs.merge_template_random_spikes_seed == 7


def test_load_spikesort_inputs_from_runtime_percentage_mode_keeps_max_cap_unset_when_omitted(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "data.yml"
    data_path.write_text(
        dedent(
            """
            output_root: /tmp/out
            datasets:
              - raw_data_h5_path: /tmp/input.raw.h5
                include_in_runtime: true
                wells:
                  - well_id: well006
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    runtime_path = tmp_path / "runtime.yml"
    runtime_path.write_text(
        "\n".join(
            [
                f"data: {data_path}",
                "stages:",
                "  spikesort:",
                "    phases:",
                "      merge_units:",
                "        analyzer:",
                "          template_extraction:",
                "            random_spikes_percentage: 75",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inputs = load_spikesort_inputs_from_runtime(config_path=str(runtime_path))

    assert inputs.merge_template_random_spikes_method == "percentage"
    assert inputs.merge_template_random_spikes_percentage == pytest.approx(0.75)
    assert inputs.merge_template_random_spikes_max_spikes_per_unit is None
