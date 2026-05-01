from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from axon_recon.pipeline.execution.logging_context import install_pipeline_log_record_factory
from axon_recon.pipeline.execution.progress import PipelineProgressStreamHandler
from axon_recon.pipeline.logging import pipeline_logging_is_configured

from .checkpointing import parse_mea_style_metadata


def compute_pipeline_log_file(*, well_out_dir: Path, data_file: Path, stream_id: str) -> Path:
    """Compute the Mandar-style per-well log file path.

    MEA_Analysis uses: <output_dir>/<run_id>_<stream_id>_pipeline.log
    We keep the same naming inside the per-well folder.
    """

    run_id = "UnknownRun"
    try:
        md = parse_mea_style_metadata(Path(data_file), stream_id=stream_id)
        run_id = md.get("run_id") or run_id
    except Exception:
        pass

    return Path(well_out_dir) / f"{run_id}_{stream_id}_pipeline.log"


def _has_file_handler(logger: logging.Logger, log_file: Path) -> bool:
    target = str(Path(log_file))
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if str(Path(handler.baseFilename)) == target:
                    return True
            except Exception:
                continue
    return False


def _get_file_handler(logger: logging.Logger, log_file: Path) -> logging.FileHandler | None:
    target = str(Path(log_file))
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if str(Path(handler.baseFilename)) == target:
                    return handler
            except Exception:
                continue
    return None


def _get_console_handler(logger: logging.Logger) -> logging.StreamHandler | None:
    # FileHandler inherits StreamHandler, so explicitly exclude file handlers.
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and (not isinstance(handler, logging.FileHandler)):
            return handler
    return None


def setup_pipeline_logger(
    *,
    log_file: Path,
    logger_name: str,
    verbose: bool = True,
    stream: Optional[object] = None,
) -> logging.Logger:
    """Create/return a MEA_Analysis-like logger.

    - Appends to a per-well log file.
    - Adds a big separator header when the file handler is first attached.
    - Logs to stdout as well.
    - Avoids duplicate handlers if called repeatedly.

    Parameters
    ----------
    log_file:
        Path to the log file.
    logger_name:
        Logger name (use a well-specific name to avoid cross-talk).
    verbose:
        If True uses DEBUG level, else INFO.
    stream:
        Stream to log to (defaults to sys.stdout).
    """

    install_pipeline_log_record_factory()
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    # Keep logger open to DEBUG and let handlers enforce effective verbosity.
    logger.setLevel(logging.DEBUG)

    if pipeline_logging_is_configured():
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
        logger.propagate = True
        return logger

    logger.propagate = False

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s [%(pipeline_target)s]: %(message)s")

    # File handler (append). Only attach once per log file.
    file_level = logging.DEBUG if verbose else logging.INFO
    existing_fh = _get_file_handler(logger, log_file)
    if existing_fh is None:
        fh = logging.FileHandler(log_file, mode="a")
        try:
            fh.stream.write("\n" + "=" * 80 + "\n")
        except Exception:
            pass
        fh.setFormatter(formatter)
        fh.setLevel(file_level)
        logger.addHandler(fh)
    else:
        existing_fh.setFormatter(formatter)
        existing_fh.setLevel(file_level)

    # Stream handler (stdout). Attach at most one.
    stream = stream if stream is not None else sys.stdout
    console_level = logging.DEBUG if verbose else logging.INFO
    existing_ch = _get_console_handler(logger)
    if existing_ch is None:
        ch = PipelineProgressStreamHandler(stream)
        ch.setFormatter(formatter)
        ch.setLevel(console_level)
        logger.addHandler(ch)
    else:
        existing_ch.setFormatter(formatter)
        existing_ch.setLevel(console_level)

    return logger


def build_stage_logger(
    *,
    well_out_dir: Path,
    data_file: Path,
    stream_id: str,
    stage_name: str,
    logger_name_prefix: str = "axon_recon",
    verbose: bool = True,
) -> logging.Logger:
    log_file = compute_pipeline_log_file(well_out_dir=well_out_dir, data_file=data_file, stream_id=stream_id)
    return setup_pipeline_logger(
        log_file=log_file,
        logger_name=f"{logger_name_prefix}.{stream_id}.{stage_name}",
        verbose=bool(verbose),
    )


def _format_stage_fields(*, fields: dict[str, object]) -> str:
    if not fields:
        return ""
    parts: list[str] = []
    for key in sorted(fields.keys()):
        value = fields[key]
        if value is None:
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)


def log_stage_start(logger: logging.Logger, *, stage: str, **fields: object) -> None:
    suffix = _format_stage_fields(fields=fields)
    logger.info("[%s] start%s%s", stage, ": " if suffix else "", suffix, extra={"event": "stage_started"})


def log_stage_complete(logger: logging.Logger, *, stage: str, **fields: object) -> None:
    suffix = _format_stage_fields(fields=fields)
    logger.info("[%s] complete%s%s", stage, ": " if suffix else "", suffix, extra={"event": "stage_completed"})


def log_stage_failure(logger: logging.Logger, *, stage: str, **fields: object) -> None:
    suffix = _format_stage_fields(fields=fields)
    logger.error("[%s] failed%s%s", stage, ": " if suffix else "", suffix, extra={"event": "stage_failed"})
