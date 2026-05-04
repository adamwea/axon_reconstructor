from __future__ import annotations

import json
from pathlib import Path

from axon_recon.pipeline.phase_tuning import (
    PhaseTuningConfig,
    build_phase_tuning_summary,
    collect_phase_resource_observations_from_jsonl,
    format_phase_tuning_report,
)
from axon_recon.pipeline.resources import parse_resources_config
from axon_recon.runtime_config import RuntimeConfig


def test_collect_phase_resource_observations_filters_run_and_stage(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "pipeline.jsonl"
    records = [
        {
            "event": "phase_resource_usage",
            "run_id": "run-a",
            "stage": "preprocess",
            "phase": "save_rec_metadata",
            "resource_class": "h5_metadata",
            "dataset_id": "dataset-a",
            "well_id": "well001",
            "source_h5_path": str(tmp_path / "source.h5"),
            "status": "success",
            "resource_usage": {
                "wall_time_s": 2.0,
                "total_peak_rss_gb": 1.2,
                "cpu_time_user_s": 1.0,
                "cpu_time_system_s": 0.5,
                "max_threads": 2,
                "observed_process_max_threads": 8,
                "disk_read_gb": 1.0,
                "disk_write_gb": 0.5,
            },
        },
        {
            "event": "phase_resource_usage",
            "run_id": "run-b",
            "stage": "preprocess",
            "phase": "save_rec_metadata",
            "resource_usage": {"wall_time_s": 1.0},
        },
    ]
    jsonl_path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")

    observations = collect_phase_resource_observations_from_jsonl(
        jsonl_path,
        run_id="run-a",
        selected_stages=["preprocess"],
    )

    assert len(observations) == 1
    assert observations[0]["stage"] == "preprocess"
    assert observations[0]["phase"] == "save_rec_metadata"
    assert observations[0]["disk_read_gb_per_s"] == 0.5
    assert observations[0]["disk_write_gb_per_s"] == 0.25


def test_build_phase_tuning_summary_recommends_resource_class_updates() -> None:
    runtime_config = RuntimeConfig(
        {
            "resources": {
                "active_profile": "test_profile",
                "profiles": {"test_profile": {"cpu_cores": 16, "ram_gb": 64, "h5_read_slots": 2}},
                "phase_resource_classes": {
                    "h5_metadata": {"cpu_cores": 1, "ram_gb": 2, "h5_read_slots": 0, "disk_heavy_slots": 0}
                },
            }
        }
    )
    resources = parse_resources_config(runtime_config=runtime_config)
    observations = [
        {
            "stage": "preprocess",
            "phase": "save_rec_metadata",
            "resource_class": "h5_metadata",
            "wall_time_s": 10.0,
            "total_peak_rss_gb": 3.1,
            "cpu_time_user_s": 20.0,
            "cpu_time_system_s": 1.0,
            "max_threads": 2,
            "observed_process_max_threads": 4,
            "disk_read_gb": 5.0,
            "disk_read_gb_per_s": 0.5,
            "disk_write_gb": 2.0,
            "disk_write_gb_per_s": 0.2,
        }
    ]

    summary = build_phase_tuning_summary(
        resources=resources,
        tuning_config=PhaseTuningConfig(),
        observations=observations,
        selected_stages=["preprocess.save_rec_metadata"],
        run_id="run-a",
    )
    recommendation = summary["recommendations"][0]

    assert summary["advisory_only"] is True
    assert recommendation["resource_class"] == "h5_metadata"
    assert recommendation["recommended_class_ram_gb"] > recommendation["current_class_ram_gb"]
    assert recommendation["recommended_class_cpu_cores"] > recommendation["current_class_cpu_cores"]
    assert recommendation["recommended_h5_read_slots"] == 1
    assert recommendation["recommended_disk_heavy_slots"] == 1
    assert "runtime YAML was not modified" in format_phase_tuning_report(summary)


def test_build_phase_tuning_summary_keeps_cpu_when_current_class_covers_observed_basis() -> None:
    runtime_config = RuntimeConfig(
        {
            "resources": {
                "active_profile": "test_profile",
                "profiles": {"test_profile": {"cpu_cores": 16, "ram_gb": 64}},
                "phase_resource_classes": {"h5_metadata": {"cpu_cores": 1, "ram_gb": 4}},
            }
        }
    )
    resources = parse_resources_config(runtime_config=runtime_config)
    summary = build_phase_tuning_summary(
        resources=resources,
        tuning_config=PhaseTuningConfig(),
        observations=[
            {
                "stage": "preprocess.save_rec_metadata",
                "phase": "save_rec_metadata",
                "resource_class": "h5_metadata",
                "wall_time_s": 1.0,
                "total_peak_rss_gb": 0.25,
                "cpu_time_user_s": 0.95,
                "cpu_time_system_s": 0.06,
                "max_threads": 1,
            }
        ],
        selected_stages=["preprocess.save_rec_metadata"],
        run_id="run-a",
    )
    recommendation = summary["recommendations"][0]

    assert recommendation["recommended_class_cpu_cores"] == 1
    assert "### preprocess.save_rec_metadata" in format_phase_tuning_report(summary)