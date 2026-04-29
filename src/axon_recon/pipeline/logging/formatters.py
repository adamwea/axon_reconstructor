from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from typing import Any

from .context import CONTEXT_FIELDS


def _utc_timestamp(record: logging.LogRecord) -> str:
    return datetime.fromtimestamp(float(record.created), tz=timezone.utc).isoformat(timespec="milliseconds")


def _record_event(record: logging.LogRecord) -> str | None:
    value = getattr(record, "event", None)
    if value is None:
        value = getattr(record, "pipeline_event", None)
    if value is None:
        return None
    token = str(value).strip()
    return token or None


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return str(value)


class PipelineJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": _utc_timestamp(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": int(record.lineno),
            "pid": int(getattr(record, "pid", record.process)),
            "worker": str(getattr(record, "worker", record.threadName)),
            "process": str(record.processName),
            "thread": str(record.threadName),
        }
        for field in CONTEXT_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = _json_ready(value)
        event = _record_event(record)
        if event is not None:
            payload["event"] = event
        for field in ("elapsed_s", "status", "output_path", "unit_id", "target_count", "phase_count"):
            if hasattr(record, field):
                value = getattr(record, field, None)
                if value is not None:
                    payload[field] = _json_ready(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, sort_keys=True, default=str)


class PipelineHumanFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(float(record.created)).isoformat(timespec="seconds")
        context_parts: list[str] = []
        for field, label in (
            ("run_id", "run"),
            ("dataset_id", "dataset"),
            ("recording_id", "recording"),
            ("well_id", "well"),
            ("stage", "stage"),
            ("phase", "phase"),
        ):
            value = getattr(record, field, None)
            if value is not None and str(value).strip():
                context_parts.append(f"{label}={value}")
        event = _record_event(record)
        if event:
            context_parts.append(f"event={event}")
        context_parts.append(f"pid={int(getattr(record, 'pid', record.process))}")
        context = " ".join(context_parts)
        message = record.getMessage()
        line = f"{timestamp} | {record.levelname:<8} | {context} | {message}"
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        if record.stack_info:
            line = f"{line}\n{self.formatStack(record.stack_info)}"
        return line