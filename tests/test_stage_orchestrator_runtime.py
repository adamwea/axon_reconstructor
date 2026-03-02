from __future__ import annotations

from pathlib import Path

from axon_reconstructor.pipeline.scope_config import ScopeConfig, ScopeDatasetSpec, ScopeWellSpec
from axon_reconstructor.pipeline.stage_orchestrator import _build_targets, _expected_sorter_output_dir, run_scope_stage_barriers


def _make_mea_like_path(tmp_path: Path) -> Path:
    p = tmp_path / "ProjectX" / "2026-01-01" / "ChipABC" / "123" / "data.raw.h5"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
    return p


def _build_scope_config(*, tmp_path: Path, stage_order: list[str], fail_fast: bool) -> ScopeConfig:
    h5_path = _make_mea_like_path(tmp_path)
    dataset = ScopeDatasetSpec(
        h5_path=h5_path,
        wells=[ScopeWellSpec(stream_id="well000")],
        dataset_id="ds0",
        enabled=True,
        stage_kwargs={},
    )
    return ScopeConfig(
        mea_output_root=tmp_path / "outputs",
        mea_analysis_repo_root=None,
        sorter="kilosort4",
        docker_image=None,
        n_jobs=1,
        chunk_duration=None,
        force_restart=False,
        per_well_parallelism=1,
        fail_fast=bool(fail_fast),
        stage_order=list(stage_order),
        stage_kwargs={},
        datasets=[dataset],
    )


def test_transition_gates_block_without_spikesort_artifacts(tmp_path: Path) -> None:
    cfg = _build_scope_config(
        tmp_path=tmp_path,
        stage_order=["unit_match", "merge_update"],
        fail_fast=False,
    )

    summary = run_scope_stage_barriers(config=cfg, dry_run=False)

    unit_match = summary["stages"][0]
    merge_update = summary["stages"][1]

    assert unit_match["stage"] == "unit_match"
    assert unit_match["failed"] == 1
    assert unit_match["results"][0]["status"] == "error"

    assert merge_update["stage"] == "merge_update"
    assert merge_update["failed"] == 1
    assert merge_update["results"][0]["status"] == "error"


def test_transition_gates_pass_with_spikesort_artifacts(tmp_path: Path) -> None:
    cfg = _build_scope_config(
        tmp_path=tmp_path,
        stage_order=["unit_match", "merge_update"],
        fail_fast=True,
    )

    target = _build_targets(cfg)[0]
    sorter_output_dir = _expected_sorter_output_dir(config=cfg, target=target)
    sorter_output_dir.mkdir(parents=True, exist_ok=True)

    summary = run_scope_stage_barriers(config=cfg, dry_run=False)

    unit_match = summary["stages"][0]
    merge_update = summary["stages"][1]

    assert unit_match["failed"] == 0
    assert unit_match["results"][0]["status"] == "ok"
    assert merge_update["failed"] == 0
    assert merge_update["results"][0]["status"] == "ok"


def test_fail_fast_stops_later_stages(monkeypatch, tmp_path: Path) -> None:
    cfg = _build_scope_config(
        tmp_path=tmp_path,
        stage_order=["preprocess", "analysis"],
        fail_fast=True,
    )

    def _fake_run_single_stage_target(*, stage, config, target):
        if stage == "preprocess":
            raise RuntimeError("forced preprocess failure")
        return {
            "status": "ok",
            "stage": stage,
            "dataset_id": target.dataset_id,
            "h5_path": str(target.h5_path),
            "stream_id": target.stream_id,
        }

    import axon_reconstructor.pipeline.stage_orchestrator as orchestrator

    monkeypatch.setattr(orchestrator, "_run_single_stage_target", _fake_run_single_stage_target)
    summary = run_scope_stage_barriers(config=cfg, dry_run=False)

    assert len(summary["stages"]) == 1
    assert summary["stages"][0]["stage"] == "preprocess"
    assert summary["stages"][0]["failed"] == 1


def test_non_fail_fast_continues_to_later_stages(monkeypatch, tmp_path: Path) -> None:
    cfg = _build_scope_config(
        tmp_path=tmp_path,
        stage_order=["preprocess", "analysis"],
        fail_fast=False,
    )

    def _fake_run_single_stage_target(*, stage, config, target):
        if stage == "preprocess":
            raise RuntimeError("forced preprocess failure")
        return {
            "status": "ok",
            "stage": stage,
            "dataset_id": target.dataset_id,
            "h5_path": str(target.h5_path),
            "stream_id": target.stream_id,
        }

    import axon_reconstructor.pipeline.stage_orchestrator as orchestrator

    monkeypatch.setattr(orchestrator, "_run_single_stage_target", _fake_run_single_stage_target)
    summary = run_scope_stage_barriers(config=cfg, dry_run=False)

    assert len(summary["stages"]) == 2
    assert summary["stages"][0]["stage"] == "preprocess"
    assert summary["stages"][0]["failed"] == 1
    assert summary["stages"][1]["stage"] == "analysis"
    assert summary["stages"][1]["failed"] == 0
