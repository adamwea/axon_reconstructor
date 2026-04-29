from __future__ import annotations

import logging
from pathlib import Path
import re

from .config import PipelineLoggingConfig
from .formatters import PipelineHumanFormatter, PipelineJsonFormatter
from .multiprocessing import append_text_line


_SAFE_PART_RE = re.compile(r"[^A-Za-z0-9_.=-]+")


def _safe_part(value: object, default: str = "unknown") -> str:
    token = str(value if value is not None else "").strip()
    if not token:
        token = default
    token = token.replace("/", "_").replace("\\", "_")
    token = _SAFE_PART_RE.sub("_", token)
    return token.strip("._") or default


def _has_value(record: logging.LogRecord, field: str) -> bool:
    value = getattr(record, field, None)
    return value is not None and str(value).strip() not in {"", "-", "unknown"}


class PipelineJsonlHandler(logging.Handler):
    def __init__(self, path: Path, level: int) -> None:
        super().__init__(level=level)
        self.path = Path(path)
        self.setFormatter(PipelineJsonFormatter())
        self._axon_recon_pipeline_handler = True

    def emit(self, record: logging.LogRecord) -> None:
        try:
            append_text_line(self.path, self.format(record))
        except Exception:
            self.handleError(record)


class PipelineRoutingFileHandler(logging.Handler):
    def __init__(self, config: PipelineLoggingConfig) -> None:
        super().__init__(level=logging.DEBUG)
        self.config = config
        self.setFormatter(PipelineHumanFormatter())
        self._axon_recon_pipeline_handler = True

    def _write(self, path: Path | None, record: logging.LogRecord, min_level: int) -> None:
        if path is None or int(record.levelno) < int(min_level):
            return
        append_text_line(path, self.format(record))

    def _dataset_path(self, record: logging.LogRecord) -> Path | None:
        if not _has_value(record, "dataset_id"):
            return None
        dataset = _safe_part(getattr(record, "dataset_id", None), "unknown_dataset")
        return self.config.logs_dir / "datasets" / dataset / "dataset.log"

    def _recording_path(self, record: logging.LogRecord) -> Path | None:
        if not _has_value(record, "dataset_id") or not _has_value(record, "recording_id"):
            return None
        dataset = _safe_part(getattr(record, "dataset_id", None), "unknown_dataset")
        recording = _safe_part(getattr(record, "recording_id", None), "unknown_recording")
        return self.config.logs_dir / "datasets" / dataset / "recordings" / recording / "recording.log"

    def _well_root(self, record: logging.LogRecord) -> Path | None:
        if not _has_value(record, "dataset_id") or not _has_value(record, "well_id"):
            return None
        dataset = _safe_part(getattr(record, "dataset_id", None), "unknown_dataset")
        recording = _safe_part(getattr(record, "recording_id", None), "unknown_recording")
        well = _safe_part(getattr(record, "well_id", None), "unknown_well")
        return self.config.logs_dir / "datasets" / dataset / "recordings" / recording / "wells" / well

    def _well_path(self, record: logging.LogRecord) -> Path | None:
        root = self._well_root(record)
        if root is None:
            return None
        return root / "well.log"

    def _phase_path(self, record: logging.LogRecord) -> Path | None:
        root = self._well_root(record)
        if root is None or not _has_value(record, "stage") or not _has_value(record, "phase"):
            return None
        stage = _safe_part(getattr(record, "stage", None), "unknown_stage")
        phase = _safe_part(getattr(record, "phase", None), "unknown_phase")
        return root / "phases" / f"{stage}__{phase}.log"

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if self.config.run_log.enabled:
                self._write(self.config.run_log.path, record, self.config.run_log.level)
            if self.config.error_log.enabled:
                self._write(self.config.error_log.path, record, self.config.error_log.level)
            if self.config.dataset_logs.enabled:
                self._write(self._dataset_path(record), record, self.config.dataset_logs.level)
            if self.config.recording_logs.enabled:
                self._write(self._recording_path(record), record, self.config.recording_logs.level)
            if self.config.well_logs.enabled:
                self._write(self._well_path(record), record, self.config.well_logs.level)
            if self.config.phase_logs.enabled:
                self._write(self._phase_path(record), record, self.config.phase_logs.level)
        except Exception:
            self.handleError(record)