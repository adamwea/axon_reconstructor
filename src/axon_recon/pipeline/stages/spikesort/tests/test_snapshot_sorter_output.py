from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import pytest

from axon_recon.pipeline.stages.spikesort.config import (
    DEFAULT_SPIKESORT_PHASE_SEQUENCE,
    parse_spikesort_stage_config,
)
from axon_recon.pipeline.stages.spikesort.core.snapshot_sorter_output import (
    SNAPSHOT_SUMMARY_FILENAME,
    hash_directory,
    run_restore_sorter_output_from_snapshot,
    run_snapshot_sorter_output_phase,
)
from axon_recon.pipeline.stages.spikesort.orchestrators.snapshot_sorter_output import (
    _run_restore_sorter_output_from_args,
)
from axon_recon.runtime_config import RuntimeConfig


LOGGER = logging.getLogger("test_snapshot_sorter_output")


def _seed_sorter_output(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "params.py").write_text("sample_rate = 30000\n", encoding="utf-8")
    (root / "cluster_KSLabel.tsv").write_text("cluster_id\tKSLabel\n0\tgood\n", encoding="utf-8")
    (root / "cluster_group.tsv").write_text("cluster_id\tgroup\n0\tgood\n", encoding="utf-8")
    sub = root / "subdir"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "spike_times.npy").write_bytes(b"\x00" * 64)


def test_snapshot_sorter_output_creates_byte_identical_copy(tmp_path: Path) -> None:
    sorter_output_dir = tmp_path / "sorter_output"
    snapshot_dir = tmp_path / "sorter_output_snapshot"
    _seed_sorter_output(sorter_output_dir)

    result = run_snapshot_sorter_output_phase(
        sorter_output_dir=sorter_output_dir,
        snapshot_dir=snapshot_dir,
        skip_if_exists=True,
        logger=LOGGER,
    )

    assert result["status"] == "ok"
    assert Path(result["snapshot_dir"]).resolve() == snapshot_dir.resolve()

    src_hashes = hash_directory(sorter_output_dir)
    snapshot_hashes = hash_directory(snapshot_dir)
    # Drop the summary file from the snapshot side before comparing.
    snapshot_hashes_filtered = {
        rel: digest
        for rel, digest in snapshot_hashes.items()
        if rel != SNAPSHOT_SUMMARY_FILENAME
    }
    assert snapshot_hashes_filtered == src_hashes
    # file_count excludes the summary file; snapshot_hashes_filtered does the same.
    assert result["file_count"] == len(snapshot_hashes_filtered)
    summary_payload = json.loads((snapshot_dir / SNAPSHOT_SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary_payload["source_dir"].endswith("sorter_output")
    assert summary_payload["file_count"] == len(snapshot_hashes_filtered)


def test_snapshot_sorter_output_skips_when_existing(tmp_path: Path) -> None:
    sorter_output_dir = tmp_path / "sorter_output"
    snapshot_dir = tmp_path / "sorter_output_snapshot"
    _seed_sorter_output(sorter_output_dir)

    first = run_snapshot_sorter_output_phase(
        sorter_output_dir=sorter_output_dir,
        snapshot_dir=snapshot_dir,
        skip_if_exists=True,
    )
    assert first["status"] == "ok"

    # Mutate the source after the first snapshot — second call must NOT refresh.
    (sorter_output_dir / "params.py").write_text("sample_rate = 99999\n", encoding="utf-8")

    second = run_snapshot_sorter_output_phase(
        sorter_output_dir=sorter_output_dir,
        snapshot_dir=snapshot_dir,
        skip_if_exists=True,
    )
    assert second["status"] == "skipped"
    assert second["reason"] == "snapshot_exists"
    snapshot_params = (snapshot_dir / "params.py").read_text(encoding="utf-8")
    assert snapshot_params == "sample_rate = 30000\n"


def test_snapshot_sorter_output_overwrites_when_skip_disabled(tmp_path: Path) -> None:
    sorter_output_dir = tmp_path / "sorter_output"
    snapshot_dir = tmp_path / "sorter_output_snapshot"
    _seed_sorter_output(sorter_output_dir)

    run_snapshot_sorter_output_phase(
        sorter_output_dir=sorter_output_dir,
        snapshot_dir=snapshot_dir,
        skip_if_exists=True,
    )
    (sorter_output_dir / "params.py").write_text("sample_rate = 99999\n", encoding="utf-8")

    refreshed = run_snapshot_sorter_output_phase(
        sorter_output_dir=sorter_output_dir,
        snapshot_dir=snapshot_dir,
        skip_if_exists=False,
    )
    assert refreshed["status"] == "ok"
    assert (snapshot_dir / "params.py").read_text(encoding="utf-8") == "sample_rate = 99999\n"


def test_snapshot_sorter_output_rejects_missing_source(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "sorter_output_snapshot"
    with pytest.raises(FileNotFoundError):
        run_snapshot_sorter_output_phase(
            sorter_output_dir=tmp_path / "does-not-exist",
            snapshot_dir=snapshot_dir,
            skip_if_exists=True,
        )


def test_restore_sorter_output_round_trips(tmp_path: Path) -> None:
    sorter_output_dir = tmp_path / "sorter_output"
    snapshot_dir = tmp_path / "sorter_output_snapshot"
    _seed_sorter_output(sorter_output_dir)
    run_snapshot_sorter_output_phase(
        sorter_output_dir=sorter_output_dir,
        snapshot_dir=snapshot_dir,
        skip_if_exists=True,
    )
    pre_restore_hashes = hash_directory(sorter_output_dir)

    # Mutate sorter_output (simulates bombcell_label / merge_SLAy applying changes).
    (sorter_output_dir / "cluster_KSLabel.tsv").write_text(
        "cluster_id\tKSLabel\n0\tmua\n", encoding="utf-8"
    )
    (sorter_output_dir / "new_artifact.txt").write_text("introduced", encoding="utf-8")

    result = run_restore_sorter_output_from_snapshot(
        snapshot_dir=snapshot_dir,
        sorter_output_dir=sorter_output_dir,
    )
    assert result["status"] == "ok"

    post_restore_hashes = hash_directory(sorter_output_dir)
    assert post_restore_hashes == pre_restore_hashes
    assert not (sorter_output_dir / "new_artifact.txt").exists()
    assert not (sorter_output_dir / SNAPSHOT_SUMMARY_FILENAME).exists()


def test_restore_sorter_output_requires_summary(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "sorter_output_snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "params.py").write_text("sample_rate = 30000\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        run_restore_sorter_output_from_snapshot(
            snapshot_dir=snapshot_dir,
            sorter_output_dir=tmp_path / "sorter_output",
        )


def test_parse_spikesort_stage_config_snapshot_sorter_output_phase() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "snapshot_sorter_output": {
                            "enabled": True,
                            "relpath": "alt/sorter_snapshot_dir",
                            "skip_if_exists": False,
                        }
                    }
                }
            }
        }
    )
    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.snapshot_sorter_output_enabled is True
    assert parsed.snapshot_sorter_output_relpath == "alt/sorter_snapshot_dir"
    assert parsed.snapshot_sorter_output_skip_if_exists is False
    assert parsed.snapshot_sorter_output_resource_class is None


def test_parse_spikesort_stage_config_snapshot_sorter_output_defaults() -> None:
    cfg = RuntimeConfig({"stages": {"spikesort": {}}})
    parsed = parse_spikesort_stage_config(runtime_config=cfg)

    assert parsed.snapshot_sorter_output_enabled is False
    assert parsed.snapshot_sorter_output_relpath == "sorter_output_snapshot"
    assert parsed.snapshot_sorter_output_skip_if_exists is True
    assert parsed.snapshot_sorter_output_resource_class is None


def test_default_phase_sequence_includes_snapshot_sorter_output() -> None:
    assert "snapshot_sorter_output" in DEFAULT_SPIKESORT_PHASE_SEQUENCE
    summarize_idx = DEFAULT_SPIKESORT_PHASE_SEQUENCE.index("summarize_sort")
    snapshot_idx = DEFAULT_SPIKESORT_PHASE_SEQUENCE.index("snapshot_sorter_output")
    bombcell_idx = DEFAULT_SPIKESORT_PHASE_SEQUENCE.index("bombcell_label")
    assert summarize_idx < snapshot_idx < bombcell_idx


def test_restore_sorter_output_cli_refuses_without_confirm(capsys) -> None:
    args = argparse.Namespace(
        config="/nonexistent/config.yml",
        confirm=False,
        force_restart=False,
        replot=False,
        target_datasets=None,
        limit_segments=None,
        limit_datasets=None,
        limit_wells_per_dataset=None,
        task_allocation_override=None,
    )
    rc = _run_restore_sorter_output_from_args(args)
    assert rc == 2
    out = capsys.readouterr().out
    assert "--confirm" in out
