from __future__ import annotations

from pathlib import Path
from typing import Any

from .checkpointing import (
    CheckpointState,
    ProcessingStage,
    compute_checkpoint_file,
    exception_to_error_dict,
    save_checkpoint,
)


def compute_stage_checkpoint_file(*, well_out_dir: Path, h5_path: Path, stream_id: str, stage_name: str) -> Path:
    main_ckpt = compute_checkpoint_file(output_dir=well_out_dir, file_path=h5_path, stream_id=stream_id)
    suffix = f"_{stage_name.strip().lower()}_checkpoint.json"
    name = main_ckpt.name
    if name.endswith("_checkpoint.json"):
        name = name[: -len("_checkpoint.json")] + suffix
    else:
        name = main_ckpt.stem + suffix
    return main_ckpt.with_name(name)


def save_stage_started(
    *,
    checkpoint_file: Path,
    state: CheckpointState,
    stage: ProcessingStage,
    out_dir: Path | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> CheckpointState:
    fields: dict[str, Any] = {}
    if out_dir is not None:
        fields["stage_out_dir"] = str(out_dir)
    if extra_fields:
        fields.update(extra_fields)
    return save_checkpoint(
        checkpoint_file=checkpoint_file,
        state=state,
        stage=stage,
        failed_stage=None,
        error=None,
        extra_fields=fields,
    )


def save_stage_completed(
    *,
    checkpoint_file: Path,
    state: CheckpointState,
    stage: ProcessingStage,
    extra_fields: dict[str, Any] | None = None,
) -> CheckpointState:
    return save_checkpoint(
        checkpoint_file=checkpoint_file,
        state=state,
        stage=stage,
        failed_stage=None,
        error=None,
        extra_fields=extra_fields,
    )


def save_stage_failed(
    *,
    checkpoint_file: Path,
    state: CheckpointState,
    stage: ProcessingStage,
    failed_stage: str | None = None,
    error: Any = None,
    extra_fields: dict[str, Any] | None = None,
) -> CheckpointState:
    if isinstance(error, dict):
        error_payload = error
    else:
        error_payload = exception_to_error_dict(error)

    failed_stage_value = failed_stage if failed_stage is not None else stage.name

    return save_checkpoint(
        checkpoint_file=checkpoint_file,
        state=state,
        stage=stage,
        failed_stage=failed_stage_value,
        error=error_payload,
        extra_fields=extra_fields,
    )
