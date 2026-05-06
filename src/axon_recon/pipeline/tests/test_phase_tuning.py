from __future__ import annotations

import json
from pathlib import Path

import pytest

from axon_recon.pipeline.phase_tuning import (
    PhaseTuningConfig,
    build_phase_tuning_summary,
    collect_disk_bandwidth_measurements,
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
                "phase_tune_tools": ["pidstat", "iostat"],
                "phase_tune_sample_count": 3,
                "phase_tune_avg_cpu_pct": 125.0,
                "phase_tune_peak_cpu_pct": 225.0,
                "phase_tune_peak_rss_gb": 2.5,
                "phase_tune_peak_read_gb_per_s": 0.75,
                "phase_tune_peak_write_gb_per_s": 0.25,
                "phase_tune_peak_device_util_pct": 88.0,
            },
            "resource_gate": {
                "wait_s": 1.25,
                "waited": True,
                "slot_demands": {"h5_read_slots": 1},
                "keyed_requests": {"source_h5_path": {"key": str(tmp_path / "scratch.h5"), "demand": 1}},
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
    assert observations[0]["source_h5_path"] == str(tmp_path / "source.h5")
    assert observations[0]["phase_read_h5_path"] == str(tmp_path / "scratch.h5")
    assert observations[0]["resource_gate_wait_s"] == 1.25
    assert observations[0]["resource_gate_slot_demands"] == {"h5_read_slots": 1}
    assert observations[0]["phase_tune_tools"] == ["pidstat", "iostat"]
    assert observations[0]["phase_tune_peak_cpu_pct"] == 225.0
    assert observations[0]["phase_tune_peak_device_util_pct"] == 88.0


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
            "phase_tune_tools": ["pidstat", "iostat"],
            "phase_tune_sample_count": 3,
            "phase_tune_avg_cpu_pct": 125.0,
            "phase_tune_peak_cpu_pct": 225.0,
            "phase_tune_peak_rss_gb": 3.2,
            "phase_tune_peak_read_gb_per_s": 0.75,
            "phase_tune_peak_write_gb_per_s": 0.25,
            "phase_tune_peak_device_util_pct": 88.0,
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
    report = format_phase_tuning_report(summary)
    assert "runtime YAML was not modified" in report
    assert "phase_tune_tools: iostat, pidstat" in report
    assert "max_phase_tune_peak_cpu_pct: 225.0" in report


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


def test_build_phase_tuning_summary_explains_flat_io_slots_when_bandwidth_underused_but_demand_covered() -> None:
    runtime_config = RuntimeConfig(
        {
            "resources": {
                "active_profile": "test_profile",
                "profiles": {"test_profile": {"cpu_cores": 16, "ram_gb": 64, "h5_read_slots": 3, "disk_heavy_slots": 3}},
                "phase_resource_classes": {
                    "preprocess_segments": {"cpu_cores": 1, "ram_gb": 2, "h5_read_slots": 1, "disk_heavy_slots": 1}
                },
            }
        }
    )
    resources = parse_resources_config(runtime_config=runtime_config)
    observations = [
        {
            "timestamp": f"2026-05-04T00:00:1{index}+00:00",
            "stage": "preprocess",
            "phase": "preprocess_segments",
            "resource_class": "preprocess_segments",
            "source_h5_path": "/src/data.raw.h5",
            "wall_time_s": 10.0,
            "total_peak_rss_gb": 0.5,
            "cpu_time_user_s": 1.0,
            "cpu_time_system_s": 0.1,
            "max_threads": 1,
            "disk_read_gb": 0.2,
            "disk_read_gb_per_s": 0.02,
            "disk_write_gb": 0.2,
            "disk_write_gb_per_s": 0.02,
        }
        for index in range(2)
    ]

    summary = build_phase_tuning_summary(
        resources=resources,
        tuning_config=PhaseTuningConfig(),
        observations=observations,
        selected_stages=["preprocess.preprocess_segments"],
        run_id="run-a",
        run_root="/out",
        disk_measurements=[
            {
                "path": "/src/data.raw.h5",
                "path_kind": "source_h5",
                "device_id": "src",
                "read_capacity_gb_per_s": 0.1,
                "write_capacity_gb_per_s": None,
            },
            {
                "path": "/out",
                "path_kind": "run_output",
                "device_id": "out",
                "read_capacity_gb_per_s": 1.0,
                "write_capacity_gb_per_s": 0.1,
            },
        ],
    )
    profile_recommendation = summary["active_profile_recommendation"]
    notes = "\n".join(profile_recommendation["notes"])

    assert profile_recommendation["max_recommended_h5_read_slot_demand"] == 2
    assert profile_recommendation["max_recommended_disk_heavy_slot_demand"] == 2
    assert profile_recommendation["recommended_h5_read_slots"] == 3
    assert profile_recommendation["recommended_disk_heavy_slots"] == 3
    assert "peak requested recommended slot demand including gate waits (2) did not exceed current profile slots (3)" in notes
    assert "increasing h5_read_slots would not change this run" in notes
    assert "increasing disk_heavy_slots would not change this run" in notes


def test_build_phase_tuning_summary_uses_phase_read_h5_path_for_bandwidth_pressure() -> None:
    runtime_config = RuntimeConfig(
        {
            "resources": {
                "active_profile": "test_profile",
                "profiles": {"test_profile": {"cpu_cores": 16, "ram_gb": 64, "h5_read_slots": 1, "disk_heavy_slots": 1}},
                "phase_resource_classes": {
                    "h5_metadata": {"cpu_cores": 1, "ram_gb": 2, "h5_read_slots": 1, "disk_heavy_slots": 0}
                },
            }
        }
    )
    resources = parse_resources_config(runtime_config=runtime_config)

    summary = build_phase_tuning_summary(
        resources=resources,
        tuning_config=PhaseTuningConfig(),
        observations=[
            {
                "timestamp": "2026-05-04T00:00:10+00:00",
                "stage": "preprocess",
                "phase": "save_rec_metadata",
                "resource_class": "h5_metadata",
                "source_h5_path": "/src/data.raw.h5",
                "phase_read_h5_path": "/scratch/data.raw.h5",
                "wall_time_s": 10.0,
                "total_peak_rss_gb": 0.5,
                "cpu_time_user_s": 1.0,
                "cpu_time_system_s": 0.1,
                "max_threads": 1,
                "disk_read_gb": 1.0,
                "disk_read_gb_per_s": 0.1,
                "disk_write_gb": 0.0,
                "disk_write_gb_per_s": 0.0,
            }
        ],
        selected_stages=["preprocess.save_rec_metadata"],
        run_id="run-a",
        run_root="/out",
        disk_measurements=[
            {
                "path": "/src/data.raw.h5",
                "path_kind": "source_h5",
                "device_id": "src",
                "read_capacity_gb_per_s": 0.1,
                "write_capacity_gb_per_s": None,
            },
            {
                "path": "/scratch/data.raw.h5",
                "path_kind": "phase_read_h5",
                "device_id": "scratch",
                "read_capacity_gb_per_s": 0.4,
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
    pressure = profile_recommendation["disk_bandwidth_pressure"]

    assert profile_recommendation["max_h5_read_bandwidth_utilization"] == pytest.approx(0.25)
    assert any(item["path"] == "/scratch/data.raw.h5" for item in pressure)
    assert not any(item["path"] == "/src/data.raw.h5" for item in pressure)


def test_collect_disk_bandwidth_measurements_includes_phase_read_h5_on_output_device(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs"
    run_root.mkdir()
    scratch_h5_path = tmp_path / "inputs" / "data.raw.h5"
    scratch_h5_path.parent.mkdir(parents=True)
    scratch_h5_path.write_bytes(b"x" * 1024 * 1024)
    second_scratch_h5_path = tmp_path / "inputs" / "second.raw.h5"
    second_scratch_h5_path.write_bytes(b"y" * 1024 * 1024)

    measurements = collect_disk_bandwidth_measurements(
        run_root=run_root,
        tuning_config=PhaseTuningConfig(disk_benchmark_size_mb=1, disk_benchmark_chunk_mb=1),
        observations=[
            {
                "source_h5_path": "/nas/data.raw.h5",
                "phase_read_h5_path": str(scratch_h5_path),
            }
        ],
    )

    phase_read_measurements = [item for item in measurements if item.get("path_kind") == "phase_read_h5"]
    assert len(phase_read_measurements) == 1
    assert phase_read_measurements[0]["path"] == str(scratch_h5_path)
    assert phase_read_measurements[0]["read_capacity_gb_per_s"] is not None
    assert any("reused device benchmark" in note for note in phase_read_measurements[0]["notes"])

    runtime_config = RuntimeConfig(
        {
            "resources": {
                "active_profile": "test_profile",
                "profiles": {"test_profile": {"cpu_cores": 16, "ram_gb": 64, "h5_read_slots": 1, "disk_heavy_slots": 1}},
                "phase_resource_classes": {
                    "h5_metadata": {"cpu_cores": 1, "ram_gb": 2, "h5_read_slots": 1, "disk_heavy_slots": 0}
                },
            }
        }
    )
    summary = build_phase_tuning_summary(
        resources=parse_resources_config(runtime_config=runtime_config),
        tuning_config=PhaseTuningConfig(disk_benchmark_size_mb=1, disk_benchmark_chunk_mb=1),
        observations=[
            {
                "timestamp": "2026-05-04T00:00:10+00:00",
                "stage": "preprocess",
                "phase": "save_rec_metadata",
                "resource_class": "h5_metadata",
                "source_h5_path": "/nas/data.raw.h5",
                "phase_read_h5_path": str(second_scratch_h5_path),
                "wall_time_s": 1.0,
                "total_peak_rss_gb": 0.5,
                "cpu_time_user_s": 0.1,
                "cpu_time_system_s": 0.1,
                "max_threads": 1,
                "disk_read_gb": 0.1,
                "disk_read_gb_per_s": 0.1,
                "disk_write_gb": 0.0,
                "disk_write_gb_per_s": 0.0,
            }
        ],
        selected_stages=["preprocess.save_rec_metadata"],
        run_id="run-a",
        run_root=str(run_root),
        disk_measurements=measurements,
    )

    pressure = summary["active_profile_recommendation"]["disk_bandwidth_pressure"]
    assert any(item["path"] == str(scratch_h5_path) for item in pressure)
    assert not any(item["path"] == str(run_root / "resource_tuning") for item in pressure)


def test_build_phase_tuning_summary_includes_gate_wait_in_requested_slot_demand() -> None:
    runtime_config = RuntimeConfig(
        {
            "resources": {
                "active_profile": "test_profile",
                "profiles": {"test_profile": {"cpu_cores": 16, "ram_gb": 64, "h5_read_slots": 1, "disk_heavy_slots": 1}},
                "phase_resource_classes": {
                    "preprocess_segments": {"cpu_cores": 1, "ram_gb": 2, "h5_read_slots": 1, "disk_heavy_slots": 1}
                },
            }
        }
    )
    resources = parse_resources_config(runtime_config=runtime_config)
    observations = [
        {
            "timestamp": "2026-05-04T00:00:10+00:00",
            "stage": "preprocess",
            "phase": "preprocess_segments",
            "resource_class": "preprocess_segments",
            "source_h5_path": "/src/data.raw.h5",
            "wall_time_s": 5.0,
            "total_peak_rss_gb": 0.5,
            "cpu_time_user_s": 1.0,
            "cpu_time_system_s": 0.1,
            "max_threads": 1,
            "disk_read_gb": 0.5,
            "disk_read_gb_per_s": 0.1,
            "disk_write_gb": 0.5,
            "disk_write_gb_per_s": 0.1,
            "resource_gate_wait_s": 0.0,
        },
        {
            "timestamp": "2026-05-04T00:00:15+00:00",
            "stage": "preprocess",
            "phase": "preprocess_segments",
            "resource_class": "preprocess_segments",
            "source_h5_path": "/src/data.raw.h5",
            "wall_time_s": 5.0,
            "total_peak_rss_gb": 0.5,
            "cpu_time_user_s": 1.0,
            "cpu_time_system_s": 0.1,
            "max_threads": 1,
            "disk_read_gb": 0.5,
            "disk_read_gb_per_s": 0.1,
            "disk_write_gb": 0.5,
            "disk_write_gb_per_s": 0.1,
            "resource_gate_wait_s": 5.0,
        },
    ]

    summary = build_phase_tuning_summary(
        resources=resources,
        tuning_config=PhaseTuningConfig(),
        observations=observations,
        selected_stages=["preprocess.preprocess_segments"],
        run_id="run-a",
        run_root="/out",
        disk_measurements=[
            {
                "path": "/src/data.raw.h5",
                "path_kind": "source_h5",
                "device_id": "src",
                "read_capacity_gb_per_s": 2.0,
                "write_capacity_gb_per_s": None,
            },
            {
                "path": "/out",
                "path_kind": "run_output",
                "device_id": "out",
                "read_capacity_gb_per_s": 2.0,
                "write_capacity_gb_per_s": 2.0,
            },
        ],
    )
    profile_recommendation = summary["active_profile_recommendation"]
    notes = "\n".join(profile_recommendation["notes"])

    assert profile_recommendation["max_active_recommended_h5_read_slot_demand"] == 1
    assert profile_recommendation["max_requested_recommended_h5_read_slot_demand"] == 2
    assert profile_recommendation["max_active_recommended_disk_heavy_slot_demand"] == 1
    assert profile_recommendation["max_requested_recommended_disk_heavy_slot_demand"] == 2
    assert profile_recommendation["recommended_h5_read_slots"] == 2
    assert profile_recommendation["recommended_disk_heavy_slots"] == 2
    assert profile_recommendation["resource_gate_wait_observations"] == 1
    assert profile_recommendation["max_resource_gate_wait_s"] == 5.0
    assert "resource gate waits were observed" in notes


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