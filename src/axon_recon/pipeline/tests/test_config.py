from __future__ import annotations

import logging
import os
from pathlib import Path

from axon_recon.pipeline.config import _copy_file_if_needed, _materialize_dataset_input_in_scratch
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