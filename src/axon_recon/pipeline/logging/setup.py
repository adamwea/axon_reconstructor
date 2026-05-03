from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any

from axon_recon.runtime_config import RuntimeConfig

from axon_recon.pipeline.execution.progress import PipelineProgressStreamHandler, progress_external_write_context

from .config import PipelineLoggingConfig, parse_pipeline_logging_config
from .context import apply_log_context_to_record, set_base_log_context
from .formatters import PipelineHumanFormatter
from .handlers import PipelineJsonlHandler, PipelineRoutingFileHandler
from .summary import PipelineSummaryHandler


_INSTALL_LOCK = threading.RLock()
_ORIGINAL_FACTORY: Any | None = None
_ORIGINAL_EXCEPTHOOK: Any | None = None
_SUMMARY_HANDLER: PipelineSummaryHandler | None = None
_CURRENT_CONFIG: PipelineLoggingConfig | None = None
_CONFIGURED = False
_NOISY_EXTERNAL_LOGGER_NAMES: tuple[str, ...] = (
    "numcodecs",
    "numcodecs.registry",
)


def install_noisy_external_log_filters() -> None:
    for logger_name in _NOISY_EXTERNAL_LOGGER_NAMES:
        logger = logging.getLogger(logger_name)
        if int(logger.getEffectiveLevel()) < int(logging.WARNING):
            logger.setLevel(logging.WARNING)


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
    rich_console_handler = False
    if config.console.rich:
        try:
            from rich.logging import RichHandler  # type: ignore[import-not-found]

            class PipelineProgressRichHandler(RichHandler):
                def emit(self, record: logging.LogRecord) -> None:
                    console = getattr(self, "console", None)
                    file = getattr(console, "file", None)
                    with progress_external_write_context(file=file):
                        super().emit(record)

                def emit_blank_line(self) -> None:
                    console = getattr(self, "console", None)
                    if console is None or not hasattr(console, "print"):
                        return
                    file = getattr(console, "file", None)
                    with progress_external_write_context(file=file):
                        console.print("")

            handler: logging.Handler = PipelineProgressRichHandler(
                show_path=False,
                show_time=False,
                show_level=True,
                rich_tracebacks=True,
                markup=False,
            )
            rich_console_handler = True
        except Exception:
            handler = PipelineProgressStreamHandler()
    else:
        handler = PipelineProgressStreamHandler()
    handler.setLevel(config.console.level)
    handler.setFormatter(PipelineHumanFormatter(include_level=not rich_console_handler))
    handler._axon_recon_pipeline_handler = True  # type: ignore[attr-defined]
    handler._axon_recon_pipeline_console_handler = True  # type: ignore[attr-defined]
    if rich_console_handler:
        handler._axon_recon_pipeline_rich_console_handler = True  # type: ignore[attr-defined]
    return handler


def current_pipeline_logging_config() -> PipelineLoggingConfig | None:
    return _CURRENT_CONFIG


def emit_pipeline_console_blank_line() -> None:
    for handler in logging.getLogger().handlers:
        if not getattr(handler, "_axon_recon_pipeline_console_handler", False):
            continue
        try:
            emit_blank_line = getattr(handler, "emit_blank_line", None)
            if callable(emit_blank_line):
                emit_blank_line()
                continue
            console = getattr(handler, "console", None)
            if console is not None and hasattr(console, "print"):
                console.print("")
                continue
            stream = getattr(handler, "stream", None)
            if stream is not None:
                stream.write("\n")
                stream.flush()
        except Exception:
            continue


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
    global _CURRENT_CONFIG, _SUMMARY_HANDLER, _CONFIGURED
    with _INSTALL_LOCK:
        runtime_config = RuntimeConfig.load(config_path) if config_path is not None else RuntimeConfig({})
        config = parse_pipeline_logging_config(runtime_config=runtime_config, config_path=config_path)
        _CURRENT_CONFIG = config
        _install_record_factory()
        set_base_log_context(run_id=config.run_id)

        root = logging.getLogger()
        _remove_pipeline_handlers(root)
        _SUMMARY_HANDLER = None
        _CONFIGURED = bool(config.enabled)

        logging.captureWarnings(bool(config.enabled and config.capture_warnings))
        root.setLevel(logging.DEBUG)
        install_noisy_external_log_filters()

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