from __future__ import annotations

from contextlib import contextmanager
import contextvars
import logging
from typing import Any, Iterator


_DEFAULT_FIELD = "-"

_dataset_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
	"axon_recon_pipeline_dataset_id",
	default=_DEFAULT_FIELD,
)
_dataset_index_var: contextvars.ContextVar[str] = contextvars.ContextVar(
	"axon_recon_pipeline_dataset_index",
	default=_DEFAULT_FIELD,
)
_well_var: contextvars.ContextVar[str] = contextvars.ContextVar(
	"axon_recon_pipeline_well",
	default=_DEFAULT_FIELD,
)

_FACTORY_SENTINEL = "_axon_recon_pipeline_log_context_factory"


def _coerce_field(value: Any) -> str:
	if value is None:
		return _DEFAULT_FIELD
	text = str(value).strip()
	return text if text else _DEFAULT_FIELD


def format_pipeline_target(*, dataset_id: Any = None, dataset_index: Any = None, well: Any = None) -> str:
	return " ".join(
		(
			f"dataset={_coerce_field(dataset_id)}",
			f"idx={_coerce_field(dataset_index)}",
			f"well={_coerce_field(well)}",
		)
	)


def _current_pipeline_fields() -> dict[str, str]:
	dataset_id = _coerce_field(_dataset_id_var.get())
	dataset_index = _coerce_field(_dataset_index_var.get())
	well = _coerce_field(_well_var.get())
	return {
		"pipeline_dataset": dataset_id,
		"pipeline_dataset_id": dataset_id,
		"pipeline_dataset_index": dataset_index,
		"pipeline_well": well,
		"pipeline_target": format_pipeline_target(
			dataset_id=dataset_id,
			dataset_index=dataset_index,
			well=well,
		),
	}


def apply_pipeline_log_context(record: logging.LogRecord) -> logging.LogRecord:
	for key, value in _current_pipeline_fields().items():
		setattr(record, key, value)
	return record


def install_pipeline_log_record_factory() -> None:
	current_factory = logging.getLogRecordFactory()
	if bool(getattr(current_factory, _FACTORY_SENTINEL, False)):
		return

	def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
		record = current_factory(*args, **kwargs)
		return apply_pipeline_log_context(record)

	setattr(record_factory, _FACTORY_SENTINEL, True)
	logging.setLogRecordFactory(record_factory)


def ensure_pipeline_target_in_format(fmt: str) -> str:
	if "%(pipeline_" in str(fmt):
		return str(fmt)
	message_token = "%(message)s"
	if message_token in str(fmt):
		return str(fmt).replace(message_token, "[%(pipeline_target)s] %(message)s", 1)
	return f"{fmt} [%(pipeline_target)s]"


@contextmanager
def pipeline_log_context(*, dataset_id: Any = None, dataset_index: Any = None, well: Any = None) -> Iterator[None]:
	dataset_token = _dataset_id_var.set(_coerce_field(dataset_id))
	dataset_index_token = _dataset_index_var.set(_coerce_field(dataset_index))
	well_token = _well_var.set(_coerce_field(well))
	try:
		yield
	finally:
		_well_var.reset(well_token)
		_dataset_index_var.reset(dataset_index_token)
		_dataset_id_var.reset(dataset_token)


def pipeline_log_context_for_target(target: Any) -> Iterator[None]:
	return pipeline_log_context(
		dataset_id=getattr(target, "dataset_id", None),
		dataset_index=getattr(target, "dataset_index", None),
		well=getattr(target, "stream_id", None),
	)