from __future__ import annotations

import logging
from typing import Any, Iterator

from axon_recon.pipeline.logging.context import (
	apply_log_context_to_record,
	current_log_context,
	log_context,
	log_context_for_target,
)

_DEFAULT_FIELD = "-"

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
	context = current_log_context()
	dataset_id = _coerce_field(context.get("dataset_id"))
	dataset_index = _coerce_field(context.get("dataset_index"))
	well = _coerce_field(context.get("well_id"))
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
	record = apply_log_context_to_record(record)
	for key, value in _current_pipeline_fields().items():
		setattr(record, key, value)
	return record


def install_pipeline_log_record_factory() -> None:
	current_factory = logging.getLogRecordFactory()
	if bool(getattr(current_factory, _FACTORY_SENTINEL, False)) or bool(
		getattr(current_factory, "_axon_recon_pipeline_factory", False)
	):
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


def pipeline_log_context(
	*,
	dataset_id: Any = None,
	dataset_index: Any = None,
	well: Any = None,
	well_id: Any = None,
	stage: Any = None,
	phase: Any = None,
	recording_id: Any = None,
	dataset_name: Any = None,
	chip_id: Any = None,
	date: Any = None,
	assay: Any = None,
) -> Iterator[None]:
	return log_context(
		dataset_id=dataset_id,
		dataset_index=dataset_index,
		well_id=(well_id if well_id is not None else well),
		stage=stage,
		phase=phase,
		recording_id=recording_id,
		dataset_name=dataset_name,
		chip_id=chip_id,
		date=date,
		assay=assay,
	)


def pipeline_log_context_for_target(target: Any, *, stage: str | None = None, phase: str | None = None) -> Iterator[None]:
	return log_context_for_target(target, stage=stage, phase=phase)