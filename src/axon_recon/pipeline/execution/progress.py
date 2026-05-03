from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import contextvars
import logging
import os
import threading
from typing import Any, Iterator


try:  # pragma: no cover - exercised when tqdm is installed in the runtime env.
	from tqdm import tqdm as _tqdm
	from tqdm.contrib.logging import logging_redirect_tqdm as _logging_redirect_tqdm
except Exception:  # pragma: no cover - keep progress optional for minimal installs.
	_tqdm = None
	_logging_redirect_tqdm = None


def _root_uses_rich_handler() -> bool:
	for handler in logging.getLogger().handlers:
		module_name = str(getattr(handler.__class__, "__module__", "") or "")
		class_name = str(getattr(handler.__class__, "__name__", "") or "")
		if class_name == "RichHandler" or module_name.startswith("rich."):
			return True
	return False


@dataclass(frozen=True)
class ProgressSpec:
	label: str
	total: int = 0
	unit: str = "item"
	enabled: bool = True


class PipelineProgress:
	def __init__(self, spec: ProgressSpec):
		self.spec = spec
		self._owner_pid = os.getpid()
		self._lock = threading.RLock()
		self._bar: Any | None = None
		self._redirect_cm: Any | None = None
		self._total = max(0, int(spec.total))
		self._completed = 0

	@property
	def total(self) -> int:
		with self._lock:
			return int(self._total)

	@property
	def completed(self) -> int:
		with self._lock:
			return int(self._completed)

	def is_owner_process(self) -> bool:
		return os.getpid() == self._owner_pid

	def __enter__(self) -> PipelineProgress:
		if not bool(self.spec.enabled) or _tqdm is None or not self.is_owner_process():
			return self
		with self._lock:
			if self._bar is not None:
				return self
			should_redirect_logging = _logging_redirect_tqdm is not None and not _root_uses_rich_handler()
			self._redirect_cm = _logging_redirect_tqdm() if should_redirect_logging else nullcontext()
			self._redirect_cm.__enter__()
			self._bar = _tqdm(
				total=int(self._total),
				desc=str(self.spec.label),
				unit=str(self.spec.unit),
				dynamic_ncols=True,
				leave=True,
				disable=None,
			)
			if _root_uses_rich_handler():
				try:
					_tqdm.write("", file=getattr(self._bar, "fp", None))
				except Exception:
					pass
			if self._completed:
				self._bar.update(int(self._completed))
		return self

	def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
		if not self.is_owner_process():
			return
		with self._lock:
			bar = self._bar
			redirect_cm = self._redirect_cm
			self._bar = None
			self._redirect_cm = None
		if bar is not None:
			bar.close()
		if redirect_cm is not None:
			redirect_cm.__exit__(exc_type, exc, traceback)

	def add_total(self, amount: int) -> None:
		try:
			increment = int(amount)
		except Exception:
			return
		if increment <= 0 or not self.is_owner_process():
			return
		with self._lock:
			self._total += increment
			if self._bar is not None:
				self._bar.total = int(self._total)
				self._bar.refresh()

	def update(self, amount: int = 1) -> None:
		try:
			increment = int(amount)
		except Exception:
			return
		if increment <= 0 or not self.is_owner_process():
			return
		with self._lock:
			self._completed += increment
			if self._bar is not None:
				self._bar.update(increment)


_current_progress: contextvars.ContextVar[PipelineProgress | None] = contextvars.ContextVar(
	"axon_recon_pipeline_progress",
	default=None,
)


@contextmanager
def pipeline_progress_context(progress: PipelineProgress | None) -> Iterator[None]:
	token = _current_progress.set(progress)
	try:
		yield
	finally:
		_current_progress.reset(token)


def current_pipeline_progress() -> PipelineProgress | None:
	return _current_progress.get()


class PipelineProgressStreamHandler(logging.StreamHandler):
	def emit(self, record: logging.LogRecord) -> None:
		progress = current_pipeline_progress()
		if _tqdm is None or progress is None or not progress.is_owner_process():
			super().emit(record)
			return
		try:
			_tqdm.write(self.format(record), file=self.stream)
			self.flush()
		except Exception:
			self.handleError(record)


def add_current_progress_total(amount: int) -> None:
	progress = current_pipeline_progress()
	if progress is not None:
		progress.add_total(int(amount))


def advance_current_progress(amount: int = 1) -> None:
	progress = current_pipeline_progress()
	if progress is not None:
		progress.update(int(amount))