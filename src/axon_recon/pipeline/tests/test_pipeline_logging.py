from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
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
    with log_context(dataset_id="dataset-a", recording_id="000123", well_id="well001", stage="preprocess", phase="concat_segments"):
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
        / "preprocess__concat_segments.log"
    )
    assert not phase_log.exists()
    records = [json.loads(line) for line in (config.logs_dir / "pipeline.jsonl").read_text(encoding="utf-8").splitlines()]
    assert records[0]["phase"] == "concat_segments"


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