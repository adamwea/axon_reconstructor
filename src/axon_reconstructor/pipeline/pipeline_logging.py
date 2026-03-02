from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

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

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")

    # File handler (append). Only attach once per log file.
    if not _has_file_handler(logger, log_file):
        fh = logging.FileHandler(log_file, mode="a")
        try:
            fh.stream.write("\n" + "=" * 80 + "\n")
        except Exception:
            pass
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Stream handler (stdout). Attach at most one.
    stream = stream if stream is not None else sys.stdout
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler(stream)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def build_stage_logger(
    *,
    well_out_dir: Path,
    data_file: Path,
    stream_id: str,
    stage_name: str,
    logger_name_prefix: str = "axon_reconstructor",
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
    logger.info("[%s] start%s%s", stage, ": " if suffix else "", suffix)


def log_stage_complete(logger: logging.Logger, *, stage: str, **fields: object) -> None:
    suffix = _format_stage_fields(fields=fields)
    logger.info("[%s] complete%s%s", stage, ": " if suffix else "", suffix)


def log_stage_failure(logger: logging.Logger, *, stage: str, **fields: object) -> None:
    suffix = _format_stage_fields(fields=fields)
    logger.error("[%s] failed%s%s", stage, ": " if suffix else "", suffix)
