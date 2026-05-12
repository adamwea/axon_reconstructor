from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading
from typing import Any

from .config import PipelineLoggingConfig


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _event(record: logging.LogRecord) -> str | None:
    value = getattr(record, "event", None)
    if value is None:
        value = getattr(record, "pipeline_event", None)
    if value is None:
        return None
    token = str(value).strip()
    return token or None


def _node_status(event: str) -> str | None:
    if event.endswith("_started"):
        return "running"
    if event.endswith("_completed"):
        return "ok"
    if event.endswith("_failed"):
        return "error"
    if event.endswith("_skipped"):
        return "skipped"
    return None


def _safe(value: Any, default: str) -> str:
    token = str(value if value is not None else "").strip()
    return token if token else default


class PipelineSummaryHandler(logging.Handler):
    def __init__(self, config: PipelineLoggingConfig) -> None:
        super().__init__(level=logging.DEBUG)
        self.config = config
        self.path = config.summary.path
        self._lock = threading.RLock()
        self._summary: dict[str, Any] = {
            "run_id": str(config.run_id),
            "run_root": str(config.run_root),
            "logs_dir": str(config.logs_dir),
            "started_at": _now(),
            "updated_at": _now(),
            "status": "running",
            "counts": {"warnings": 0, "errors": 0, "events": {}},
            "stages": {},
            "datasets": {},
            "errors": [],
        }
        self._axon_recon_pipeline_handler = True

    def _well_node(self, record: logging.LogRecord) -> dict[str, Any] | None:
        dataset = getattr(record, "dataset_id", None)
        well = getattr(record, "well_id", None)
        if dataset is None or not str(dataset).strip() or well is None or not str(well).strip():
            return None
        dataset_key = _safe(dataset, "unknown_dataset")
        recording_key = _safe(getattr(record, "recording_id", None), "unknown_recording")
        well_key = _safe(well, "unknown_well")
        dataset_node = self._summary["datasets"].setdefault(dataset_key, {"status": "unknown", "recordings": {}})
        recording_node = dataset_node["recordings"].setdefault(recording_key, {"status": "unknown", "wells": {}})
        return recording_node["wells"].setdefault(well_key, {"status": "unknown", "stages": {}})

    def _stage_node(self, record: logging.LogRecord) -> dict[str, Any] | None:
        stage = getattr(record, "stage", None)
        if stage is None or not str(stage).strip():
            return None
        stage_key = _safe(stage, "unknown_stage")
        well_node = self._well_node(record)
        if well_node is None:
            return self._summary["stages"].setdefault(stage_key, {"status": "unknown", "phases": {}})
        return well_node["stages"].setdefault(stage_key, {"status": "unknown", "phases": {}})

    def _update_event(self, record: logging.LogRecord, event: str) -> None:
        counts = self._summary["counts"]["events"]
        counts[event] = int(counts.get(event, 0)) + 1
        status = _node_status(event)
        message = record.getMessage()

        if event == "run_started":
            self._summary["status"] = "running"
        elif event == "run_completed":
            self._summary["status"] = "ok"
        elif event == "run_failed":
            self._summary["status"] = "error"

        if event.startswith("well_") and status is not None:
            well_node = self._well_node(record)
            if well_node is not None:
                well_node["status"] = status
                well_node["updated_at"] = _now()
                if hasattr(record, "elapsed_s"):
                    try:
                        well_node["elapsed_s"] = float(getattr(record, "elapsed_s"))
                    except Exception:
                        pass
                if status == "error":
                    well_node["error"] = message

        stage_node = self._stage_node(record)
        if stage_node is not None and status is not None:
            if event.startswith("phase_"):
                phase = _safe(getattr(record, "phase", None), "unknown_phase")
                node = stage_node["phases"].setdefault(phase, {"status": "unknown"})
            else:
                node = stage_node
            node["status"] = status
            node["updated_at"] = _now()
            if hasattr(record, "elapsed_s"):
                try:
                    node["elapsed_s"] = float(getattr(record, "elapsed_s"))
                except Exception:
                    pass
            if status == "error":
                node["error"] = message

    def _write(self) -> None:
        if self.path is None:
            return
        # Under MPI (size > 1), only rank 0 owns summary.json. Without this
        # guard, every rank races on tmp_path.replace(path): two ranks each
        # write summary.json.tmp, both call replace(), and whoever loses the
        # race raises FileNotFoundError (the winner already consumed the tmp).
        # The guardrail doc spells this out: "Rank 0 owns global summaries
        # unless there is a tested rank-summary merge step." We have no merge
        # step today, so silently skip on non-rank-0 ranks.
        from ..mpi_adapter import current_mpi_context

        mpi_ctx = current_mpi_context()
        if mpi_ctx is not None and int(getattr(mpi_ctx, "size", 1)) > 1 and not bool(
            getattr(mpi_ctx, "is_rank_0", True)
        ):
            return
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(self._summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        tmp_path.replace(path)

    def emit(self, record: logging.LogRecord) -> None:
        if not self.config.summary.enabled:
            return
        try:
            with self._lock:
                self._summary["updated_at"] = _now()
                if int(record.levelno) >= logging.WARNING:
                    self._summary["counts"]["warnings"] = int(self._summary["counts"].get("warnings", 0)) + 1
                if int(record.levelno) >= logging.ERROR:
                    self._summary["counts"]["errors"] = int(self._summary["counts"].get("errors", 0)) + 1
                    errors = self._summary.setdefault("errors", [])
                    errors.append(
                        {
                            "timestamp": _now(),
                            "level": record.levelname,
                            "logger": record.name,
                            "message": record.getMessage(),
                            "dataset_id": getattr(record, "dataset_id", None),
                            "recording_id": getattr(record, "recording_id", None),
                            "well_id": getattr(record, "well_id", None),
                            "stage": getattr(record, "stage", None),
                            "phase": getattr(record, "phase", None),
                        }
                    )
                    del errors[:-100]
                event = _event(record)
                if event is not None:
                    self._update_event(record, event)
                    self._write()
        except Exception:
            self.handleError(record)

    def finalize(self, status: str) -> None:
        with self._lock:
            if status:
                self._summary["status"] = str(status)
            self._summary["completed_at"] = _now()
            self._summary["updated_at"] = _now()
            self._write()

    def close(self) -> None:
        try:
            self.finalize(str(self._summary.get("status", "unknown")))
        finally:
            super().close()