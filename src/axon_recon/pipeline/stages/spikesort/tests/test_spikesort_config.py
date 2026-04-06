from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from axon_reconstructor.runtime_config import RuntimeConfig
from axon_recon.pipeline.stages.spikesort.config import (
    load_spikesort_inputs_from_runtime,
    parse_spikesort_stage_config,
)


def test_parse_spikesort_stage_config_defaults() -> None:
    cfg = RuntimeConfig({"stages": {"spikesort": {}}})

    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.output_rel_root == "spikesort_outputs"
    assert parsed.preprocess_concat_recording_relpath is None
    assert parsed.logging_enabled is True
    assert parsed.logging_verbose is False
    assert parsed.logging_file_relpath is None
    assert parsed.debug_limit_wells is None
    assert parsed.sorter == "kilosort4"
    assert parsed.docker_image is None
    assert parsed.recording_num == "rec0000"
    assert parsed.verbose is False
    assert parsed.n_jobs is None
    assert parsed.chunk_duration is None
    assert parsed.cuda_visible_devices is None
    assert parsed.run_analyzer is True
    assert parsed.run_reports is True
    assert parsed.sort_enabled is True
    assert parsed.sort_delete_outputs_on_force_restart is False
    assert parsed.plot_enabled is True
    assert parsed.plot_mode == "separate"
    assert parsed.plot_debug is False
    assert parsed.raster_sort is None
    assert parsed.fixed_y is False
    assert parsed.no_curation is False
    assert parsed.export_to_phy is False
    assert parsed.force_rerun_analyzer is False
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
    assert parsed.slay_params is None
    assert parsed.force_restart is False
    assert parsed.force_replot is False


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
                        "merge_units": {
                            "unitmatch": {
                                "enabled": False,
                                "merge_units": True,
                            }
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
                        "merge_units": {
                            "SLAy": {
                                "enabled": True,
                                "relpath": "SLAy_outputs_custom",
                                "package_root": "/tmp/slay",
                                "sorter_output_relpath": "spikesort_outputs/sorter_output/sorter_output",
                                "output_json_relpath": "reports/slay_run_output.json",
                                "candidate_pairs_relpath": "candidates/recommended.tsv",
                                "merge_groups_relpath": "candidates/groups.json",
                                "allow_numpy_fallback": False,
                                "plot_merges": True,
                                "auto_accept_merges": False,
                                "copy_automerge_artifacts": False,
                                "delete_outputs_on_force_restart": False,
                                "params": {
                                    "max_spikes": 250,
                                    "final_thresh": 0.6,
                                },
                            }
                        }
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
    assert parsed.slay_package_root == "/tmp/slay"
    assert parsed.slay_sorter_output_relpath == "spikesort_outputs/sorter_output/sorter_output"
    assert parsed.slay_output_json_relpath == "reports/slay_run_output.json"
    assert parsed.slay_candidate_pairs_relpath == "candidates/recommended.tsv"
    assert parsed.slay_merge_groups_relpath == "candidates/groups.json"
    assert parsed.slay_allow_numpy_fallback is False
    assert parsed.slay_plot_merges is True
    assert parsed.slay_auto_accept_merges is False
    assert parsed.slay_copy_automerge_artifacts is False
    assert parsed.slay_delete_outputs_on_force_restart is False
    assert isinstance(parsed.slay_params, dict)
    assert parsed.slay_params.get("max_spikes") == 250
    assert parsed.slay_params.get("final_thresh") == 0.6


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
    assert inputs.chunk_duration == "1s"
