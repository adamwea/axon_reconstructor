from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import threading
from typing import Any, Iterator


_LOG_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("axon_recon_pipeline_log_context", default={})
_BASE_CONTEXT: dict[str, Any] = {}
_BASE_CONTEXT_LOCK = threading.RLock()

CONTEXT_FIELDS: tuple[str, ...] = (
    "run_id",
    "dataset_id",
    "dataset_name",
    "dataset_index",
    "recording_id",
    "chip_id",
    "date",
    "assay",
    "well_id",
    "stage",
    "phase",
    "resource_class",
    "source_h5_path",
    "output_path",
)


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def set_base_log_context(**fields: Any) -> None:
    with _BASE_CONTEXT_LOCK:
        _BASE_CONTEXT.clear()
        for key, value in fields.items():
            if key in CONTEXT_FIELDS and value is not None:
                _BASE_CONTEXT[key] = _clean(value)


def current_log_context() -> dict[str, Any]:
    with _BASE_CONTEXT_LOCK:
        out = dict(_BASE_CONTEXT)
    out.update(dict(_LOG_CONTEXT.get({})))
    return out


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    parent = dict(_LOG_CONTEXT.get({}))
    merged = dict(parent)
    for key, value in fields.items():
        if key not in CONTEXT_FIELDS:
            continue
        if value is None:
            continue
        merged[key] = _clean(value)
    token: Token[dict[str, Any]] = _LOG_CONTEXT.set(merged)
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)


def _path_parts(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    try:
        raw = Path(path).expanduser()
    except Exception:
        return {}
    parents = list(raw.parents)
    out: dict[str, str] = {}
    if len(parents) >= 1:
        out["recording_id"] = parents[0].name
    if len(parents) >= 2:
        out["assay"] = parents[1].name
    if len(parents) >= 3:
        out["chip_id"] = parents[2].name
    if len(parents) >= 4:
        out["date"] = parents[3].name
    if len(parents) >= 5:
        out["dataset_name"] = parents[4].name
    return {key: value for key, value in out.items() if value}


def log_context_for_target(target: Any, *, stage: str | None = None, phase: str | None = None) -> Iterator[None]:
    source_path = getattr(target, "source_h5_path", None) or getattr(target, "h5_path", None)
    fields: dict[str, Any] = _path_parts(Path(source_path) if source_path is not None else None)
    dataset_id = getattr(target, "dataset_id", None)
    fields.update(
        {
            "dataset_id": dataset_id,
            "dataset_index": getattr(target, "dataset_index", None),
            "well_id": getattr(target, "stream_id", None),
            "stage": stage,
            "phase": phase,
            "source_h5_path": source_path,
        }
    )
    if fields.get("dataset_name") is None and dataset_id is not None:
        fields["dataset_name"] = str(dataset_id)
    return log_context(**fields)


def _display(value: Any) -> str:
    if value is None:
        return "-"
    token = str(value).strip()
    return token if token else "-"


def _pipeline_target(context: dict[str, Any]) -> str:
    return " ".join(
        (
            f"dataset={_display(context.get('dataset_id'))}",
            f"idx={_display(context.get('dataset_index'))}",
            f"well={_display(context.get('well_id'))}",
        )
    )


def apply_log_context_to_record(record: logging.LogRecord) -> logging.LogRecord:
    context = current_log_context()
    for key in CONTEXT_FIELDS:
        if not hasattr(record, key):
            setattr(record, key, context.get(key))
    if not hasattr(record, "pipeline_target"):
        setattr(record, "pipeline_target", _pipeline_target(context))
    if not hasattr(record, "pipeline_dataset_id"):
        setattr(record, "pipeline_dataset_id", _display(context.get("dataset_id")))
    if not hasattr(record, "pipeline_dataset_index"):
        setattr(record, "pipeline_dataset_index", _display(context.get("dataset_index")))
    if not hasattr(record, "pipeline_well"):
        setattr(record, "pipeline_well", _display(context.get("well_id")))
    if not hasattr(record, "pid"):
        setattr(record, "pid", int(os.getpid()))
    if not hasattr(record, "worker"):
        setattr(record, "worker", threading.current_thread().name)
    return record


def log_event(logger: logging.Logger, event: str, message: str | None = None, **fields: Any) -> None:
    payload = dict(fields)
    payload["event"] = str(event)
    payload.setdefault("event_utc", datetime.now(timezone.utc).isoformat())
    logger.info(message or str(event).replace("_", " ").capitalize(), extra=payload)