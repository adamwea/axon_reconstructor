from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def test_build_phase_tuning_summary_recommends_active_profile_io_slot_increase_from_bandwidth_underuse() -> None:
    runtime_config = RuntimeConfig(
        {
            "resources": {
                "active_profile": "test_profile",
                "profiles": {"test_profile": {"cpu_cores": 16, "ram_gb": 64, "h5_read_slots": 1, "disk_heavy_slots": 1}},
                "phase_resource_classes": {
                    "h5_metadata": {"cpu_cores": 1, "ram_gb": 2, "h5_read_slots": 0, "disk_heavy_slots": 0}
                },
            }
        }
    )
    resources = parse_resources_config(runtime_config=runtime_config)
    observations = [
        {
            "timestamp": f"2026-05-04T00:00:1{index}+00:00",
            "stage": "preprocess",
            "phase": "save_rec_metadata",
            "resource_class": "h5_metadata",
            "source_h5_path": "/src/data.raw.h5",
            "wall_time_s": 10.0,
            "total_peak_rss_gb": 0.5,
            "cpu_time_user_s": 1.0,
            "cpu_time_system_s": 0.1,
            "max_threads": 1,
            "disk_read_gb": 5.0,
            "disk_read_gb_per_s": 0.5,
            "disk_write_gb": 2.0,
            "disk_write_gb_per_s": 0.2,
        }
        for index in range(3)
    ]

    summary = build_phase_tuning_summary(
        resources=resources,
        tuning_config=PhaseTuningConfig(),
        observations=observations,
        selected_stages=["preprocess.save_rec_metadata"],
        run_id="run-a",
        run_root="/out",
        disk_measurements=[
            {
                "path": "/src/data.raw.h5",
                "path_kind": "source_h5",
                "device_id": "src",
                "read_capacity_gb_per_s": 6.0,
                "write_capacity_gb_per_s": None,
            },
            {
                "path": "/out",
                "path_kind": "run_output",
                "device_id": "out",
                "read_capacity_gb_per_s": 6.0,
                "write_capacity_gb_per_s": 2.4,
            },
        ],
    )
    profile_recommendation = summary["active_profile_recommendation"]

    assert summary["recommendations"][0]["recommended_h5_read_slots"] == 1
    assert summary["recommendations"][0]["recommended_disk_heavy_slots"] == 1
    assert profile_recommendation["recommended_h5_read_slots"] == 3
    assert profile_recommendation["recommended_disk_heavy_slots"] == 3
    assert profile_recommendation["max_current_h5_read_slot_demand"] is None
    assert profile_recommendation["max_recommended_h5_read_slot_demand"] == 3
    assert profile_recommendation["max_h5_read_bandwidth_utilization"] == pytest.approx(0.25)
    assert profile_recommendation["max_disk_heavy_bandwidth_utilization"] == pytest.approx(0.25)
    report = format_phase_tuning_report(summary)
    assert "## Active Profile IO Slots" in report
    assert "## Disk Bandwidth Measurements" in report
    assert "## Disk Bandwidth Utilization" in report
    assert "- h5_read_slots: 1 -> 3" in report
    assert "- disk_heavy_slots: 1 -> 3" in report


def test_build_phase_tuning_summary_recommends_active_profile_io_slot_decrease_from_bandwidth_saturation() -> None:
    runtime_config = RuntimeConfig(
        {
            "resources": {
                "active_profile": "test_profile",
                "profiles": {"test_profile": {"cpu_cores": 16, "ram_gb": 64, "h5_read_slots": 4, "disk_heavy_slots": 4}},
                "phase_resource_classes": {
                    "h5_to_binary": {"cpu_cores": 1, "ram_gb": 2, "h5_read_slots": 1, "disk_heavy_slots": 1}
                },
            }
        }
    )
    resources = parse_resources_config(runtime_config=runtime_config)
    observations = [
        {
            "timestamp": f"2026-05-04T00:00:1{index}+00:00",
            "stage": "preprocess",
            "phase": "prepare_raw_binaries",
            "resource_class": "h5_to_binary",
            "source_h5_path": "/src/data.raw.h5",
            "wall_time_s": 10.0,
            "total_peak_rss_gb": 0.5,
            "cpu_time_user_s": 0.4,
            "cpu_time_system_s": 0.1,
            "max_threads": 1,
            "disk_read_gb": 3.0,
            "disk_read_gb_per_s": 0.3,
            "disk_write_gb": 3.0,
            "disk_write_gb_per_s": 0.3,
        }
        for index in range(4)
    ]

    summary = build_phase_tuning_summary(
        resources=resources,
        tuning_config=PhaseTuningConfig(min_observations_for_underuse=5),
        observations=observations,
        selected_stages=["preprocess.prepare_raw_binaries"],
        run_id="run-a",
        run_root="/out",
        disk_measurements=[
            {
                "path": "/src/data.raw.h5",
                "path_kind": "source_h5",
                "device_id": "src",
                "read_capacity_gb_per_s": 1.0,
                "write_capacity_gb_per_s": None,
            },
            {
                "path": "/out",
                "path_kind": "run_output",
                "device_id": "out",
                "read_capacity_gb_per_s": 1.0,
                "write_capacity_gb_per_s": 1.0,
            },
        ],
    )
    profile_recommendation = summary["active_profile_recommendation"]

    assert profile_recommendation["max_recommended_h5_read_slot_demand"] == 4
    assert profile_recommendation["max_recommended_disk_heavy_slot_demand"] == 4
    assert profile_recommendation["max_h5_read_bandwidth_utilization"] == pytest.approx(1.2)
    assert profile_recommendation["max_disk_heavy_bandwidth_utilization"] == pytest.approx(1.2)
    assert profile_recommendation["recommended_h5_read_slots"] == 2
    assert profile_recommendation["recommended_disk_heavy_slots"] == 2