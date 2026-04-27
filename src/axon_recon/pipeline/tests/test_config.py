from __future__ import annotations

import logging
import os
from pathlib import Path

import axon_recon.pipeline.stages.preprocess.core.copy_src_to_scratch as copy_src_to_scratch_core
from axon_recon.pipeline.stages.preprocess.core.copy_src_to_scratch import (
    _copy_file_if_needed,
    _materialize_dataset_input_in_scratch,
    resolve_copy_src_to_scratch_input_path,
)
from axon_reconstructor.pipeline.scratch_layout import resolve_scratch_layout


def _set_mtime(path: Path, when_ns: int) -> None:
    os.utime(path, ns=(when_ns, when_ns))


def test_copy_file_if_needed_overwrites_read_only_existing_file(tmp_path: Path) -> None:
    src = tmp_path / "input.raw.h5"
    dst = tmp_path / "scratch_inputs" / "input.raw.h5"

    src.write_bytes(b"first-copy")
    _set_mtime(src, 1_700_000_000_000_000_000)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(b"stale-copy")
    dst.chmod(0o444)

    assert _copy_file_if_needed(src=src, dst=dst) is True
    assert dst.read_bytes() == b"first-copy"
    assert int(dst.stat().st_mtime_ns) == int(src.stat().st_mtime_ns)

    src.write_bytes(b"second-copy")
    _set_mtime(src, 1_700_000_100_000_000_000)
    dst.chmod(0o444)

    assert _copy_file_if_needed(src=src, dst=dst) is True
    assert dst.read_bytes() == b"second-copy"
    assert int(dst.stat().st_mtime_ns) == int(src.stat().st_mtime_ns)


def test_copy_file_if_needed_skips_same_size_files_when_mtime_differs_only_within_same_second(tmp_path: Path) -> None:
    src = tmp_path / "input.raw.h5"
    dst = tmp_path / "scratch_inputs" / "input.raw.h5"

    src.write_bytes(b"same-size")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(b"same-size")

    _set_mtime(src, 1_700_000_000_123_456_789)
    _set_mtime(dst, 1_700_000_000_000_000_000)

    assert _copy_file_if_needed(src=src, dst=dst) is False
    assert dst.read_bytes() == b"same-size"


def test_materialize_dataset_input_in_scratch_skips_progress_logs_when_files_are_already_materialized(
    tmp_path: Path,
    caplog,
) -> None:
    source_h5 = tmp_path / "raw_data" / "dataset" / "data.raw.h5"
    source_h5.parent.mkdir(parents=True, exist_ok=True)
    source_h5.write_bytes(b"same-size")
    source_cfg = source_h5.parent / "input.cfg"
    source_cfg.write_text("foo=1\n", encoding="utf-8")

    _set_mtime(source_h5, 1_700_000_000_123_456_789)
    _set_mtime(source_cfg, 1_700_000_050_123_456_789)

    scratch_input_root = tmp_path / "scratch_inputs"
    target_h5 = scratch_input_root / "dataset" / "data.raw.h5"
    target_h5.parent.mkdir(parents=True, exist_ok=True)
    target_h5.write_bytes(b"same-size")
    target_cfg = target_h5.parent / "input.cfg"
    target_cfg.write_text("foo=1\n", encoding="utf-8")

    _set_mtime(target_h5, 1_700_000_000_000_000_000)
    _set_mtime(target_cfg, 1_700_000_050_000_000_000)

    caplog.set_level(logging.INFO, logger="axon_recon.pipeline.config")
    resolved_h5 = _materialize_dataset_input_in_scratch(
        source_h5_path=source_h5,
        scratch_input_root=scratch_input_root,
        dataset_id="dataset-001",
    )

    messages = [record.getMessage() for record in caplog.records]
    assert resolved_h5 == target_h5.resolve()
    assert any("already materialized" in message for message in messages)
    assert not any("Scratch input copy progress" in message for message in messages)


def test_materialize_dataset_input_in_scratch_logs_byte_progress_bar_for_h5_copy(
    tmp_path: Path,
    caplog,
    monkeypatch,
) -> None:
    source_h5 = tmp_path / "raw_data" / "dataset" / "data.raw.h5"
    source_h5.parent.mkdir(parents=True, exist_ok=True)
    source_h5.write_bytes((b"0123456789abcdef" * 64))
    source_cfg = source_h5.parent / "input.cfg"
    source_cfg.write_text("foo=1\n", encoding="utf-8")

    scratch_input_root = tmp_path / "scratch_inputs"

    monkeypatch.setattr(copy_src_to_scratch_core, "_SCRATCH_COPY_CHUNK_BYTES", 64)
    monkeypatch.setattr(copy_src_to_scratch_core, "_SCRATCH_COPY_PROGRESS_MIN_UPDATE_BYTES", 64)
    monkeypatch.setattr(copy_src_to_scratch_core, "_SCRATCH_COPY_PROGRESS_MIN_UPDATE_SECONDS", 0.0)

    caplog.set_level(logging.INFO, logger="axon_recon.pipeline.config")
    resolved_h5 = _materialize_dataset_input_in_scratch(
        source_h5_path=source_h5,
        scratch_input_root=scratch_input_root,
        dataset_id="dataset-002",
    )

    messages = [record.getMessage() for record in caplog.records]
    progress_messages = [message for message in messages if "Scratch input copy progress" in message]

    assert resolved_h5 == (scratch_input_root / "dataset" / "data.raw.h5").resolve()
    assert resolved_h5.read_bytes() == source_h5.read_bytes()
    assert len(progress_messages) >= 2
    assert any("overall=" in message and "rate=" in message and "eta=" in message for message in progress_messages)
    assert any("file_progress=" in message and "dataset-002" in message for message in progress_messages)
    assert any("[" in message and "]" in message for message in progress_messages)
    assert any("100.0%" in message for message in progress_messages)


def test_resolve_copy_src_to_scratch_input_path_reuses_existing_scratch_without_source_stat(
    tmp_path: Path,
) -> None:
    source_h5 = tmp_path / "raw_data" / "dataset" / "data.raw.h5"
    scratch_input_root = tmp_path / "scratch_inputs"
    target_h5 = scratch_input_root / "dataset" / "data.raw.h5"
    target_h5.parent.mkdir(parents=True, exist_ok=True)
    target_h5.write_bytes(b"cached")

    resolved_h5 = resolve_copy_src_to_scratch_input_path(
        source_h5_path=source_h5,
        scratch_input_root=scratch_input_root,
        dataset_id="dataset-003",
        materialize_scratch_inputs=False,
    )

    assert resolved_h5 == target_h5


def test_resolve_scratch_layout_is_idempotent_for_base_and_canonical_paths(tmp_path: Path) -> None:
    scratch_root = tmp_path / "scratch"

    from_root = resolve_scratch_layout(scratch_root)
    assert from_root is not None
    assert from_root.scratch_root == scratch_root.resolve()
    assert from_root.inputs_root == scratch_root.resolve() / "axon_recon_scratch" / "inputs"
    assert from_root.outputs_root == scratch_root.resolve() / "axon_recon_scratch" / "outputs"

    from_canonical_root = resolve_scratch_layout(from_root.canonical_root)
    assert from_canonical_root == from_root

    from_outputs_root = resolve_scratch_layout(from_root.outputs_root)
    assert from_outputs_root == from_root