from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any

from axon_reconstructor.runtime_config import RuntimeConfig

from axon_recon.pipeline.execution.progress import PipelineProgressStreamHandler

from .config import PipelineLoggingConfig, parse_pipeline_logging_config
from .context import apply_log_context_to_record, set_base_log_context
from .formatters import PipelineHumanFormatter
from .handlers import PipelineJsonlHandler, PipelineRoutingFileHandler
from .summary import PipelineSummaryHandler


_INSTALL_LOCK = threading.RLock()
_ORIGINAL_FACTORY: Any | None = None
_ORIGINAL_EXCEPTHOOK: Any | None = None
_SUMMARY_HANDLER: PipelineSummaryHandler | None = None
_CONFIGURED = False


def _install_record_factory() -> None:
    global _ORIGINAL_FACTORY
    current = logging.getLogRecordFactory()
    if getattr(current, "_axon_recon_pipeline_factory", False):
        return
    _ORIGINAL_FACTORY = current

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = current(*args, **kwargs)
        return apply_log_context_to_record(record)

    factory._axon_recon_pipeline_factory = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(factory)


def _remove_pipeline_handlers(root: logging.Logger) -> None:
    for handler in list(root.handlers):
        if getattr(handler, "_axon_recon_pipeline_handler", False):
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass


def _make_console_handler(config: PipelineLoggingConfig) -> logging.Handler:
    if config.console.rich:
        try:
            from rich.logging import RichHandler  # type: ignore[import-not-found]

            handler: logging.Handler = RichHandler(
                show_path=False,
                show_time=False,
                rich_tracebacks=True,
                markup=False,
            )
        except Exception:
            handler = PipelineProgressStreamHandler()
    else:
        handler = PipelineProgressStreamHandler()
    handler.setLevel(config.console.level)
    handler.setFormatter(PipelineHumanFormatter())
    handler._axon_recon_pipeline_handler = True  # type: ignore[attr-defined]
    return handler


def _install_uncaught_exception_hook(config: PipelineLoggingConfig) -> None:
    global _ORIGINAL_EXCEPTHOOK
    if not bool(config.capture_uncaught_exceptions):
        return
    current_hook = sys.excepthook
    if getattr(current_hook, "_axon_recon_pipeline_excepthook", False):
        return
    _ORIGINAL_EXCEPTHOOK = current_hook

    def excepthook(exc_type: type[BaseException], exc_value: BaseException, traceback: Any) -> None:
        logging.getLogger("axon_recon.pipeline.uncaught").critical(
            "uncaught exception",
            exc_info=(exc_type, exc_value, traceback),
            extra={"event": "run_failed"},
        )
        if _ORIGINAL_EXCEPTHOOK is not None:
            _ORIGINAL_EXCEPTHOOK(exc_type, exc_value, traceback)

    excepthook._axon_recon_pipeline_excepthook = True  # type: ignore[attr-defined]
    sys.excepthook = excepthook


def configure_pipeline_logging(*, config_path: str | Path | None = None) -> PipelineLoggingConfig:
    global _SUMMARY_HANDLER, _CONFIGURED
    with _INSTALL_LOCK:
        runtime_config = RuntimeConfig.load(config_path) if config_path is not None else RuntimeConfig({})
        config = parse_pipeline_logging_config(runtime_config=runtime_config, config_path=config_path)
        _install_record_factory()
        set_base_log_context(run_id=config.run_id)

        root = logging.getLogger()
        _remove_pipeline_handlers(root)
        _SUMMARY_HANDLER = None
        _CONFIGURED = bool(config.enabled)

        logging.captureWarnings(bool(config.enabled and config.capture_warnings))
        root.setLevel(logging.DEBUG)

        if not config.enabled:
            return config
        _install_uncaught_exception_hook(config)

        if config.structured.enabled and config.structured.path is not None:
            root.addHandler(PipelineJsonlHandler(config.structured.path, config.structured.level))
        root.addHandler(PipelineRoutingFileHandler(config))
        if config.summary.enabled:
            _SUMMARY_HANDLER = PipelineSummaryHandler(config)
            root.addHandler(_SUMMARY_HANDLER)
        if config.console.enabled:
            root.addHandler(_make_console_handler(config))

        return config


def finalize_pipeline_logging(*, status: str = "ok") -> None:
    with _INSTALL_LOCK:
        if _SUMMARY_HANDLER is not None:
            _SUMMARY_HANDLER.finalize(status)
        for handler in logging.getLogger().handlers:
            if getattr(handler, "_axon_recon_pipeline_handler", False):
                try:
                    handler.flush()
                except Exception:
                    pass


def pipeline_logging_is_configured() -> bool:
    return bool(_CONFIGURED)