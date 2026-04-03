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
    assert parsed.logging_enabled is True
    assert parsed.logging_verbose is False
    assert parsed.logging_file_relpath is None
    assert parsed.debug_limit_wells is None
    assert parsed.debug_limit_segments_per_well is None
    assert parsed.sorter == "kilosort4"
    assert parsed.docker_image is None
    assert parsed.recording_num == "rec0000"
    assert parsed.verbose is False
    assert parsed.n_jobs is None
    assert parsed.chunk_duration is None
    assert parsed.cuda_visible_devices is None
    assert parsed.run_analyzer is True
    assert parsed.run_reports is True
    assert parsed.plot_enabled is True
    assert parsed.plot_mode == "separate"
    assert parsed.plot_debug is False
    assert parsed.raster_sort is None
    assert parsed.fixed_y is False
    assert parsed.no_curation is False
    assert parsed.export_to_phy is False
    assert parsed.force_rerun_analyzer is False
    assert parsed.force_restart is False
    assert parsed.force_replot is False


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
                        "limit_segments_per_well": 3,
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
    assert parsed.debug_limit_segments_per_well == 3
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
                "    debug: {limit_segments_per_well: 2}",
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
    assert inputs.debug_limit_segments_per_well == 2
    assert inputs.n_jobs == 4
    assert inputs.run_reports is False
    assert inputs.plot_mode == "merged"
    assert inputs.plot_debug is True
    assert inputs.raster_sort == "unit_id"
    assert inputs.fixed_y is True
    assert inputs.force_restart is False
    assert inputs.force_replot is True
