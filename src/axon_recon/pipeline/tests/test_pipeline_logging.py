from __future__ import annotations

from contextlib import contextmanager
import json
import logging
from pathlib import Path
import sys
import threading
import types

from axon_recon.pipeline.logging import (
    configure_pipeline_logging,
    finalize_pipeline_logging,
    install_noisy_external_log_filters,
    log_context,
    parse_pipeline_logging_config,
)
from axon_recon.pipeline.logging.multiprocessing import append_text_line
from axon_recon.pipeline.logging.setup import _make_console_handler
from axon_recon.runtime_config import RuntimeConfig


def _write_runtime(
    tmp_path: Path,
    *,
    phase_logs_enabled: bool = True,
    console_enabled: bool = False,
    console_rich: bool = True,
    blank_line_after_phase: bool = False,
    resource_usage_enabled: bool = False,
) -> Path:
    data_path = tmp_path / "data.yml"
    output_root = tmp_path / "outputs"
    data_path.write_text(
        f"output_root: {output_root}\n"
        "use_scratch_root: false\n"
        "datasets: []\n",
        encoding="utf-8",
    )
    runtime_path = tmp_path / "runtime.yml"
    runtime_path.write_text(
        f"data: {data_path}\n"
        "logging:\n"
        "  enabled: true\n"
        "  run_id: test-run\n"
        "  level: INFO\n"
        "  console:\n"
        f"    enabled: {'true' if console_enabled else 'false'}\n"
        f"    rich: {'true' if console_rich else 'false'}\n"
        f"    blank_line_after_phase: {'true' if blank_line_after_phase else 'false'}\n"
        "  structured:\n"
        "    enabled: true\n"
        "    path: logs/pipeline.jsonl\n"
        "  run_log:\n"
        "    enabled: true\n"
        "    path: logs/pipeline.log\n"
        "  error_log:\n"
        "    enabled: true\n"
        "    path: logs/errors.log\n"
        "  dataset_logs:\n"
        "    enabled: true\n"
        "  recording_logs:\n"
        "    enabled: true\n"
        "  well_logs:\n"
        "    enabled: true\n"
        "  phase_logs:\n"
        f"    enabled: {'true' if phase_logs_enabled else 'false'}\n"
        "  resource_usage:\n"
        f"    enabled: {'true' if resource_usage_enabled else 'false'}\n"
        "    level: INFO\n"
        "    include_children: true\n"
        "    sample_interval_s: 0.05\n"
        "    include_gpu: false\n"
        "    include_disk_io: false\n"
        "    write_to_phase_summary: true\n"
        "  summary:\n"
        "    enabled: true\n"
        "    path: logs/summary.json\n",
        encoding="utf-8",
    )
    return runtime_path


def test_pipeline_logging_routes_contextual_records(tmp_path):
    runtime_path = _write_runtime(tmp_path)
    config = configure_pipeline_logging(config_path=runtime_path)
    logger = logging.getLogger("axon_recon.tests.pipeline_logging.routes")

    with log_context(
        dataset_id="dataset-a",
        dataset_index=1,
        recording_id="000123",
        well_id="well001",
        stage="preprocess",
        phase="save_rec_metadata",
    ):
        logger.info("hello well one", extra={"event": "phase_completed", "elapsed_s": 1.25})
    with log_context(dataset_id="dataset-a", recording_id="000123", well_id="well002", stage="preprocess"):
        logger.info("hello well two")

    finalize_pipeline_logging(status="ok")

    jsonl_path = config.run_root / "logs" / "pipeline.jsonl"
    records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["run_id"] == "test-run"
    assert records[0]["dataset_id"] == "dataset-a"
    assert records[0]["well_id"] == "well001"
    assert records[0]["phase"] == "save_rec_metadata"
    assert records[0]["event"] == "phase_completed"

    well_one_log = (
        config.logs_dir
        / "datasets"
        / "dataset-a"
        / "recordings"
        / "000123"
        / "wells"
        / "well001"
        / "well.log"
    )
    well_two_log = (
        config.logs_dir
        / "datasets"
        / "dataset-a"
        / "recordings"
        / "000123"
        / "wells"
        / "well002"
        / "well.log"
    )
    assert "hello well one" in well_one_log.read_text(encoding="utf-8")
    assert "hello well two" not in well_one_log.read_text(encoding="utf-8")
    assert "hello well two" in well_two_log.read_text(encoding="utf-8")

    phase_log = well_one_log.parent / "phases" / "preprocess__save_rec_metadata.log"
    assert "hello well one" in phase_log.read_text(encoding="utf-8")

    summary = json.loads((config.run_root / "logs" / "summary.json").read_text(encoding="utf-8"))
    phase_summary = summary["datasets"]["dataset-a"]["recordings"]["000123"]["wells"]["well001"]["stages"]["preprocess"]["phases"]["save_rec_metadata"]
    assert phase_summary["status"] == "ok"


def test_pipeline_logging_can_disable_phase_logs(tmp_path):
    runtime_path = _write_runtime(tmp_path, phase_logs_enabled=False)
    config = configure_pipeline_logging(config_path=runtime_path)
    logger = logging.getLogger("axon_recon.tests.pipeline_logging.phase_toggle")
    with log_context(dataset_id="dataset-a", recording_id="000123", well_id="well001", stage="preprocess", phase="preprocess_segments"):
        logger.info("phase message", extra={"event": "phase_completed"})
    finalize_pipeline_logging(status="ok")

    phase_log = (
        config.logs_dir
        / "datasets"
        / "dataset-a"
        / "recordings"
        / "000123"
        / "wells"
        / "well001"
        / "phases"
        / "preprocess__preprocess_segments.log"
    )
    assert not phase_log.exists()
    records = [json.loads(line) for line in (config.logs_dir / "pipeline.jsonl").read_text(encoding="utf-8").splitlines()]
    assert records[0]["phase"] == "preprocess_segments"


def test_pipeline_logging_nested_context_preserves_parent_fields(tmp_path):
    runtime_path = _write_runtime(tmp_path)
    config = configure_pipeline_logging(config_path=runtime_path)
    logger = logging.getLogger("axon_recon.tests.pipeline_logging.nested")

    with log_context(dataset_id="dataset-a", recording_id="000123", well_id="well001", stage="preprocess"):
        with log_context(phase="nested_phase"):
            logger.info("nested message")
    finalize_pipeline_logging(status="ok")

    records = [json.loads(line) for line in (config.logs_dir / "pipeline.jsonl").read_text(encoding="utf-8").splitlines()]
    assert records[0]["dataset_id"] == "dataset-a"
    assert records[0]["well_id"] == "well001"
    assert records[0]["stage"] == "preprocess"
    assert records[0]["phase"] == "nested_phase"


def test_pipeline_logging_setup_is_idempotent(tmp_path):
    runtime_path = _write_runtime(tmp_path)
    config = configure_pipeline_logging(config_path=runtime_path)
    first_handlers = [handler for handler in logging.getLogger().handlers if getattr(handler, "_axon_recon_pipeline_handler", False)]
    configure_pipeline_logging(config_path=runtime_path)
    second_handlers = [handler for handler in logging.getLogger().handlers if getattr(handler, "_axon_recon_pipeline_handler", False)]
    assert len(second_handlers) == len(first_handlers)

    logging.getLogger("axon_recon.tests.pipeline_logging.idempotent").info("single line")
    finalize_pipeline_logging(status="ok")
    assert (config.logs_dir / "pipeline.log").read_text(encoding="utf-8").count("single line") == 1


def test_phase_chain_logs_resource_class_and_writes_resource_usage(tmp_path, monkeypatch):
    import axon_recon.pipeline.logging.setup as logging_setup
    import axon_recon.pipeline.resource_usage as resource_usage
    from axon_recon.pipeline.execution import PhaseDescriptor, run_phase_chain

    class _FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = int(pid)

        def children(self, recursive: bool = True):
            _ = recursive
            return []

        def memory_info(self):
            return types.SimpleNamespace(rss=1024)

        def num_threads(self) -> int:
            return 99

        def cpu_times(self):
            return types.SimpleNamespace(user=0.0, system=0.0)

    monkeypatch.setattr(
        resource_usage,
        "psutil",
        types.SimpleNamespace(Process=lambda pid: _FakeProcess(int(pid))),
    )

    runtime_path = _write_runtime(
        tmp_path,
        console_enabled=False,
        blank_line_after_phase=True,
        resource_usage_enabled=True,
    )
    config = configure_pipeline_logging(config_path=runtime_path)
    blank_line_calls: list[str] = []
    monkeypatch.setattr(logging_setup, "emit_pipeline_console_blank_line", lambda: blank_line_calls.append("blank"))

    summary_json = config.run_root / "artifacts" / "phase_summary.json"
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps({"status": "ok"}) + "\n", encoding="utf-8")

    try:
        with log_context(
            dataset_id="dataset-a",
            recording_id="000123",
            well_id="well001",
            stage="spikesort",
        ):
            result = run_phase_chain(
                phases=[
                    PhaseDescriptor(
                        name="sort",
                        runner=lambda: types.SimpleNamespace(summary_json=summary_json),
                        resource_class="cpu_heavy",
                        pipeline_thread_count=3,
                    )
                ],
                logger=logging.getLogger("axon_recon.tests.pipeline_logging.phase_chain"),
                target_label="dataset-a:well001",
            )
            assert result.result is not None
    finally:
        finalize_pipeline_logging(status="ok")

    assert config.console.blank_line_after_phase is True
    assert config.resource_usage.enabled is True

    records = [json.loads(line) for line in (config.logs_dir / "pipeline.jsonl").read_text(encoding="utf-8").splitlines()]
    phase_records = [record for record in records if record.get("phase") == "sort"]
    started = next(record for record in phase_records if record.get("event") == "phase_started")
    completed = next(record for record in phase_records if record.get("event") == "phase_completed")
    usage = next(record for record in phase_records if record.get("event") == "phase_resource_usage")

    assert started["resource_class"] == "cpu_heavy"
    assert completed["resource_class"] == "cpu_heavy"
    assert usage["resource_class"] == "cpu_heavy"
    assert started["message"].startswith("Starting phase: spikesort.sort")
    assert completed["message"].startswith("Finished phase: spikesort.sort")
    assert usage["message"].startswith("Phase resource usage: spikesort.sort")
    assert usage["status"] == "success"
    assert usage["resource_usage"]["wall_time_s"] is not None
    assert usage["resource_usage"]["total_peak_rss_gb"] is not None
    assert usage["resource_usage"]["max_threads"] == 3
    assert usage["resource_usage"]["observed_process_max_threads"] == 99
    assert blank_line_calls == ["blank"]

    summary_payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary_payload["resource_class"] == "cpu_heavy"
    assert summary_payload["resource_usage"]["wall_time_s"] is not None
    assert summary_payload["resource_usage"]["total_peak_rss_gb"] is not None
    assert summary_payload["resource_usage"]["max_threads"] == 3
    assert summary_payload["resource_usage"]["observed_process_max_threads"] == 99


def test_phase_chain_logs_resource_gate_waiting_event_for_queued_worker(tmp_path):
    from axon_recon.pipeline.execution import PhaseDescriptor, run_phase_chain
    from axon_recon.pipeline.resource_budget import ResourceBudgetManager, stage_resource_budget_context
    from axon_recon.pipeline.resources import parse_resources_config

    runtime_path = _write_runtime(
        tmp_path,
        console_enabled=False,
        resource_usage_enabled=False,
    )
    config = configure_pipeline_logging(config_path=runtime_path)
    resources = parse_resources_config(
        runtime_config=RuntimeConfig(
            {
                "resources": {
                    "active_profile": "test_profile",
                    "profiles": {"test_profile": {"capacity": {"h5_read_slots": 1}}},
                    "phase_budgets": {"h5_reader": {"h5_read_slots": 1}},
                }
            }
        )
    )
    manager = ResourceBudgetManager(resources=resources, planned_target_count=2, well_workers=2)
    holder_entered = threading.Event()
    release_holder = threading.Event()
    wait_warning_seen = threading.Event()
    errors: list[BaseException] = []
    logger = logging.getLogger("axon_recon.tests.pipeline_logging.phase_chain_wait")

    class _WaitWarningHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if getattr(record, "event", None) == "phase_resource_gate_waiting":
                wait_warning_seen.set()

    handler = _WaitWarningHandler()
    logger.addHandler(handler)

    def _run_worker(label: str, *, hold_gate: bool = False) -> None:
        def _runner() -> dict[str, str]:
            if hold_gate:
                holder_entered.set()
                assert release_holder.wait(timeout=5)
            return {"status": "ok"}

        try:
            with log_context(
                dataset_id="dataset-a",
                recording_id="000123",
                well_id=label,
                stage="spikesort",
            ):
                run_phase_chain(
                    phases=[
                        PhaseDescriptor(
                            name="sort",
                            runner=_runner,
                            resource_class="h5_reader",
                            pipeline_thread_count=1,
                        )
                    ],
                    logger=logger,
                    target_label=label,
                )
        except BaseException as exc:
            errors.append(exc)

    holder_thread = threading.Thread(target=_run_worker, args=("well000",), kwargs={"hold_gate": True})
    waiter_thread = threading.Thread(target=_run_worker, args=("well001",))
    try:
        with stage_resource_budget_context(manager):
            holder_thread.start()
            assert holder_entered.wait(timeout=5)
            waiter_thread.start()
            assert wait_warning_seen.wait(timeout=5)
            release_holder.set()
            holder_thread.join(timeout=5)
            waiter_thread.join(timeout=5)
            assert not holder_thread.is_alive()
            assert not waiter_thread.is_alive()
    finally:
        release_holder.set()
        logger.removeHandler(handler)
        finalize_pipeline_logging(status="ok" if not errors else "failed")

    assert not errors
    records = [json.loads(line) for line in (config.logs_dir / "pipeline.jsonl").read_text(encoding="utf-8").splitlines()]
    wait_records = [record for record in records if record.get("event") == "phase_resource_gate_waiting"]

    assert len(wait_records) == 1
    wait_record = wait_records[0]
    assert wait_record["level"] == "WARNING"
    assert wait_record["stage"] == "spikesort"
    assert wait_record["phase"] == "sort"
    assert wait_record["resource_class"] == "h5_reader"
    assert wait_record["well_workers"] == 2
    assert wait_record["target_count"] == 2
    assert wait_record["resource_gate"]["waited"] is True
    assert wait_record["resource_gate"]["slot_demands"] == {"h5_read_slots": 1}
    assert wait_record["resource_gate"]["slot_available_at_wait"] == {"h5_read_slots": 0}


def test_append_text_line_recovers_from_stale_unwritable_log(tmp_path, monkeypatch):
    log_path = tmp_path / "logs" / "phase.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("old line\n", encoding="utf-8")

    original_open = Path.open
    state = {"raised": False}

    def _patched_open(self: Path, *args, **kwargs):
        mode = kwargs.get("mode")
        if mode is None and args:
            mode = args[0]
        if self == log_path and mode == "a" and not state["raised"]:
            state["raised"] = True
            raise PermissionError("stale unwritable log")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _patched_open)

    append_text_line(log_path, "new line")

    assert log_path.with_name("phase.log.stale").read_text(encoding="utf-8") == "old line\n"
    assert log_path.read_text(encoding="utf-8") == "new line\n"


def test_make_console_handler_disables_rich_level_prefix(tmp_path, monkeypatch):
    from axon_recon.pipeline.logging.formatters import PipelineHumanFormatter

    runtime_path = _write_runtime(tmp_path, console_enabled=True, console_rich=True)
    rich_module = types.ModuleType("rich")
    rich_logging_module = types.ModuleType("rich.logging")
    created: dict[str, object] = {}

    class FakeRichHandler(logging.Handler):
        def __init__(self, *args, **kwargs):
            super().__init__()
            created["kwargs"] = kwargs

    rich_module.logging = rich_logging_module  # type: ignore[attr-defined]
    rich_logging_module.RichHandler = FakeRichHandler  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rich", rich_module)
    monkeypatch.setitem(sys.modules, "rich.logging", rich_logging_module)

    runtime_config = RuntimeConfig.load(runtime_path)
    config = parse_pipeline_logging_config(runtime_config=runtime_config, config_path=runtime_path)

    handler = _make_console_handler(config)
    record = logging.makeLogRecord(
        {
            "name": "axon_recon.tests.pipeline_logging.console",
            "levelno": logging.INFO,
            "levelname": "INFO",
            "msg": "hello console",
            "args": (),
            "run_id": "test-run",
            "pid": 123,
        }
    )
    rendered = handler.formatter.format(record)

    assert isinstance(handler, FakeRichHandler)
    assert created["kwargs"] == {
        "show_path": False,
        "show_time": False,
        "show_level": True,
        "rich_tracebacks": True,
        "markup": False,
    }
    assert isinstance(handler.formatter, PipelineHumanFormatter)
    assert handler.formatter.include_level is False
    assert "INFO" not in rendered
    assert rendered.endswith("pid=123\nhello console")


def test_make_console_handler_uses_progress_external_write_mode_for_rich_handler(tmp_path, monkeypatch):
    import axon_recon.pipeline.execution.progress as progress_module
    from axon_recon.pipeline.execution.progress import PipelineProgress, ProgressSpec, pipeline_progress_context

    runtime_path = _write_runtime(tmp_path, console_enabled=True, console_rich=True)
    rich_module = types.ModuleType("rich")
    rich_logging_module = types.ModuleType("rich.logging")
    created: dict[str, object] = {}
    emitted: list[str] = []
    redirect_entered: list[str] = []
    external_write_files: list[object | None] = []

    class FakeRichHandler(logging.Handler):
        def __init__(self, *args, **kwargs):
            super().__init__()
            created["kwargs"] = kwargs
            self.console = types.SimpleNamespace(file=object(), print=lambda *args, **kwargs: None)
            created["console_file"] = self.console.file

        def emit(self, record: logging.LogRecord) -> None:
            emitted.append(record.getMessage())

    class _Bar:
        fp = object()

        def update(self, amount: int) -> None:
            _ = amount

        def close(self) -> None:
            return None

    class _FakeTqdm:
        def __call__(self, *args: object, **kwargs: object) -> _Bar:
            _ = args, kwargs
            return _Bar()

        def write(self, message: str, file: object | None = None) -> None:
            _ = message, file

        @contextmanager
        def external_write_mode(self, file: object | None = None):
            external_write_files.append(file)
            yield

    @contextmanager
    def _fake_redirect():
        redirect_entered.append("redirect")
        yield

    rich_module.logging = rich_logging_module  # type: ignore[attr-defined]
    rich_logging_module.RichHandler = FakeRichHandler  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rich", rich_module)
    monkeypatch.setitem(sys.modules, "rich.logging", rich_logging_module)
    monkeypatch.setattr(progress_module, "_tqdm", _FakeTqdm())
    monkeypatch.setattr(progress_module, "_logging_redirect_tqdm", _fake_redirect)

    runtime_config = RuntimeConfig.load(runtime_path)
    config = parse_pipeline_logging_config(runtime_config=runtime_config, config_path=runtime_path)
    handler = _make_console_handler(config)
    progress = PipelineProgress(ProgressSpec(label="test wells", total=1, unit="well", enabled=True))
    record = logging.makeLogRecord({"levelno": logging.INFO, "levelname": "INFO", "msg": "hello progress"})
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    root.handlers = [handler]

    try:
        with progress, pipeline_progress_context(progress):
            handler.emit(record)
    finally:
        root.handlers = original_handlers

    assert getattr(handler, "_axon_recon_pipeline_rich_console_handler", False) is True
    assert redirect_entered == []
    assert external_write_files == [created["console_file"]]
    assert emitted == ["hello progress"]


def test_pipeline_logging_suppresses_noisy_codec_registration_logger():
    logger = logging.getLogger("numcodecs.registry")
    original_level = int(logger.level)
    try:
        logger.setLevel(logging.DEBUG)
        install_noisy_external_log_filters()
        assert int(logger.level) == logging.WARNING
    finally:
        logger.setLevel(original_level)


def test_pipeline_logging_resolves_canonical_scratch_output_root(tmp_path):
    data_path = tmp_path / "data.yml"
    scratch_root = tmp_path / "scratch"
    data_path.write_text(
        f"output_root: {tmp_path / 'published'}\n"
        f"scratch_root: {scratch_root}\n"
        "use_scratch_root: true\n"
        "datasets: []\n",
        encoding="utf-8",
    )
    runtime_path = tmp_path / "runtime.yml"
    runtime_path.write_text(
        f"data: {data_path}\n"
        "logging:\n"
        "  enabled: true\n"
        "  run_id: scratch-test\n"
        "  console:\n"
        "    enabled: false\n"
        "  structured:\n"
        "    enabled: false\n"
        "  run_log:\n"
        "    enabled: false\n"
        "  summary:\n"
        "    enabled: false\n",
        encoding="utf-8",
    )

    config = configure_pipeline_logging(config_path=runtime_path)
    finalize_pipeline_logging(status="ok")

    assert config.run_root == scratch_root / "axon_recon_scratch" / "outputs"