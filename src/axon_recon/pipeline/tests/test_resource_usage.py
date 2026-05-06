from __future__ import annotations

import types
from pathlib import Path

from axon_recon.pipeline import resource_usage
from axon_recon.pipeline.resource_usage import (
    PhaseResourceUsage,
    configure_phase_tuning_monitoring,
    format_phase_resource_usage_message,
    parse_iostat_phase_tune_output,
    parse_pidstat_phase_tune_output,
    start_phase_resource_monitor,
)


def test_parse_pidstat_phase_tune_output_aggregates_process_samples() -> None:
    output = """
Linux 5.15.0 (host)  05/05/26  _x86_64_  (36 CPU)

12:00:01      UID       PID    %usr %system  %guest   %wait    %CPU   CPU  Command
12:00:01     1000       101   10.00    5.00    0.00    0.00   15.00     2  python
12:00:01     1000       102   12.00    8.00    0.00    0.00   20.00     3  python

12:00:01      UID       PID  minflt/s  majflt/s     VSZ     RSS   %MEM  Command
12:00:01     1000       101      0.00      0.00 2000000 1048576   1.00  python
12:00:01     1000       102      0.00      0.00 2000000 1048576   1.00  python

12:00:01      UID       PID   kB_rd/s   kB_wr/s kB_ccwr/s iodelay  Command
12:00:01     1000       101 1048576.00 524288.00      0.00       0  python
12:00:01     1000       102 1048576.00 524288.00      0.00       0  python
Average:     1000       102 1048576.00 524288.00      0.00       0  python
"""

    metrics = parse_pidstat_phase_tune_output(output)

    assert metrics["sample_count"] == 1
    assert metrics["peak_cpu_pct"] == 35.0
    assert metrics["avg_cpu_pct"] == 35.0
    assert metrics["peak_rss_gb"] == 2.0
    assert metrics["peak_read_gb_per_s"] == 2.0
    assert metrics["peak_write_gb_per_s"] == 1.0


def test_parse_iostat_phase_tune_output_reports_device_pressure() -> None:
    output = """
Linux 5.15.0 (host)  05/05/26  _x86_64_  (36 CPU)

Device            r/s     w/s   rMB/s   wMB/s  await  aqu-sz  %util
nvme0n1         10.00   20.00  125.50   64.25   7.50    0.25  88.00
sda              1.00    2.00    5.00    3.00  12.00    0.10  45.00
"""

    metrics = parse_iostat_phase_tune_output(output)

    assert metrics["peak_device_read_mb_per_s"] == 125.50
    assert metrics["peak_device_write_mb_per_s"] == 64.25
    assert metrics["peak_device_await_ms"] == 12.00
    assert metrics["peak_device_util_pct"] == 88.00


def test_phase_resource_monitor_reports_pss_separately_from_summed_child_rss(monkeypatch) -> None:
    class FakeProcess:
        def __init__(self, pid: int, *, rss: int, pss: int, children=None) -> None:
            self.pid = int(pid)
            self._rss = int(rss)
            self._pss = int(pss)
            self._children = list(children or [])

        def children(self, recursive: bool = True):
            _ = recursive
            return list(self._children)

        def memory_info(self):
            return types.SimpleNamespace(rss=self._rss)

        def memory_full_info(self):
            return types.SimpleNamespace(rss=self._rss, pss=self._pss)

        def num_threads(self) -> int:
            return 1

        def cpu_times(self):
            return types.SimpleNamespace(user=0.0, system=0.0)

    child_a = FakeProcess(102, rss=4_000, pss=1_000)
    child_b = FakeProcess(103, rss=4_000, pss=1_000)
    parent = FakeProcess(101, rss=1_000, pss=800, children=[child_a, child_b])
    monkeypatch.setattr(
        resource_usage,
        "psutil",
        types.SimpleNamespace(Process=lambda pid: parent),
    )

    monitor = resource_usage.PhaseResourceMonitor(
        include_children=True,
        sample_interval_s=0.05,
        include_gpu=False,
        include_disk_io=False,
        pipeline_thread_count=1,
    )
    monitor._capture_sample(initial=True)
    usage = monitor.stop()

    assert usage.process_peak_rss_gb == 1_000 / float(1024**3)
    assert usage.child_peak_rss_gb == 8_000 / float(1024**3)
    assert usage.total_peak_rss_gb == 9_000 / float(1024**3)
    assert usage.process_peak_pss_gb == 800 / float(1024**3)
    assert usage.child_peak_pss_gb == 2_000 / float(1024**3)
    assert usage.total_peak_pss_gb == 2_800 / float(1024**3)


def test_format_phase_resource_usage_message_includes_inline_phase_tune_recommendation() -> None:
    message = format_phase_resource_usage_message(
        stage_name="reconstruct.build_templates",
        phase_name="build_templates",
        dataset_id="dataset-a",
        recording_id="000031",
        well_id="well000",
        resource_class="template_build",
        status="success",
        resource_usage=PhaseResourceUsage(
            wall_time_s=10.0,
            total_peak_rss_gb=2.0,
            cpu_time_user_s=20.0,
            cpu_time_system_s=1.0,
            max_threads=4,
        ),
        phase_tune_recommendation={
            "current_class_ram_gb": 8.0,
            "recommended_class_ram_gb": 8.0,
            "current_class_cpu_cores": 4,
            "recommended_class_cpu_cores": 4,
            "observations": 1,
            "notes": ["current CPU estimate covers observed pipeline thread demand"],
        },
    )

    assert "Phase resource usage: reconstruct.build_templates.build_templates" in message
    assert "  phase_tune_recommendation:" in message
    assert "    ram_gb=8.000000->8.000000" in message
    assert "    cpu_cores=4->4" in message
    assert "    note=current CPU estimate covers observed pipeline thread demand" in message


def test_phase_tune_enables_monitor_when_regular_resource_usage_is_disabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = int(pid)

        def children(self, recursive: bool = True):
            _ = recursive
            return []

        def memory_info(self):
            return types.SimpleNamespace(rss=1024 * 1024)

        def num_threads(self) -> int:
            return 4

        def cpu_times(self):
            return types.SimpleNamespace(user=1.0, system=0.5)

        def io_counters(self):
            return types.SimpleNamespace(read_bytes=1024, write_bytes=2048)

    monkeypatch.setattr(
        resource_usage,
        "psutil",
        types.SimpleNamespace(Process=lambda pid: FakeProcess(int(pid))),
    )
    configure_phase_tuning_monitoring(enabled=True, system_tools_enabled=False)
    try:
        monitor = start_phase_resource_monitor(
            types.SimpleNamespace(
                enabled=False,
                include_children=True,
                sample_interval_s=0.05,
                include_gpu=False,
                include_disk_io=False,
            ),
            pipeline_thread_count=2,
            run_root=tmp_path,
            run_id="run-a",
            stage_name="stage-a",
            phase_name="phase-a",
            target_label="target-a",
        )
        assert monitor is not None
        usage = monitor.stop()
    finally:
        configure_phase_tuning_monitoring(enabled=False)

    payload = usage.to_dict()
    assert payload["max_threads"] == 2
    assert payload["observed_process_max_threads"] == 4
    assert payload["disk_read_gb"] == 0.0
