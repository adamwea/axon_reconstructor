from __future__ import annotations

import argparse
from pathlib import Path

from axon_reconstructor.cli import _cmd_stage
from axon_reconstructor.pipeline.scope_config import ScopeConfig, ScopeDatasetSpec, ScopeWellSpec
from axon_reconstructor.pipeline.pipeline_driver import StageExecutionResult, run_scope_stage_barriers


def _make_mea_like_path(tmp_path: Path) -> Path:
    p = tmp_path / "ProjectX" / "2026-01-01" / "ChipABC" / "123" / "data.raw.h5"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
    return p


def test_scope_orchestrator_dispatches_via_shared_stage_executor(monkeypatch, tmp_path: Path) -> None:
    h5_path = _make_mea_like_path(tmp_path)
    cfg = ScopeConfig(
        mea_output_root=tmp_path / "outputs",
        scratch_output_root=None,
        sorter="kilosort4",
        docker_image=None,
        n_jobs=2,
        chunk_duration="1s",
        force_restart=False,
        per_well_parallelism=1,
        fail_fast=True,
        stage_order=["preprocess"],
        stage_kwargs={"preprocess": {"custom_flag": True}},
        datasets=[
            ScopeDatasetSpec(
                h5_path=h5_path,
                wells=[ScopeWellSpec(stream_id="well000")],
                dataset_id="ds0",
                enabled=True,
                stage_kwargs={},
            )
        ],
    )

    calls: list[tuple[str, dict]] = []

    def _fake_execute_stage(*, stage, context, stage_kwargs, logger):
        calls.append(
            (
                stage,
                {
                    "h5_path": str(context.h5_path),
                    "stream_id": context.stream_id,
                    "n_jobs": context.n_jobs,
                    "stage_kwargs": dict(stage_kwargs or {}),
                },
            )
        )
        return StageExecutionResult(stage=stage, artifacts={"n_common_electrodes": 123})

    import axon_reconstructor.pipeline.pipeline_driver as stage_driver

    monkeypatch.setattr(stage_driver, "execute_stage", _fake_execute_stage)

    summary = run_scope_stage_barriers(config=cfg, dry_run=False)

    assert len(calls) == 1
    stage, details = calls[0]
    assert stage == "preprocess"
    assert details["stream_id"] == "well000"
    assert details["n_jobs"] == 2
    assert details["stage_kwargs"]["custom_flag"] is True

    results = summary["stages"][0]["results"]
    assert results[0]["n_common_electrodes"] == 123


def test_stage_cli_dispatches_via_shared_stage_executor(monkeypatch, tmp_path: Path) -> None:
    h5_path = _make_mea_like_path(tmp_path)
    output_root = tmp_path / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    calls: list[tuple[str, dict]] = []

    def _fake_execute_stage(*, stage, context, stage_kwargs, logger):
        calls.append(
            (
                stage,
                {
                    "h5_path": str(context.h5_path),
                    "stream_id": context.stream_id,
                    "n_jobs": context.n_jobs,
                    "force_restart": context.force_restart,
                    "stage_kwargs": dict(stage_kwargs or {}),
                },
            )
        )
        return StageExecutionResult(stage=stage, artifacts={"n_common_electrodes": 77})

    import axon_reconstructor.pipeline.pipeline_driver as stage_driver

    monkeypatch.setattr(stage_driver, "execute_stage", _fake_execute_stage)

    args = argparse.Namespace(
        env_file=None,
        stage="preprocess",
        stage_kwargs_file=None,
        stage_kwargs=None,
        debug=False,
        force_restart=False,
        force_replot=False,
        n_jobs=4,
        sorter="kilosort4",
        docker_image=None,
        chunk_duration=None,
        debug_max_units=None,
        debug_max_segments=None,
        h5_path=h5_path,
        stream_id="well001",
        mea_output_root=output_root,
        unit_ids=None,
        unit_limit=None,
        prefer_curated_waveforms_panels=None,
        botm_enable=None,
        botm_n_events=None,
        botm_n_noise_windows=None,
        botm_seed=None,
        botm_prior_signal=None,
        botm_match_fraction_threshold=None,
        botm_sorter=None,
    )

    code = _cmd_stage(args)

    assert code == 0
    assert len(calls) == 1
    stage, details = calls[0]
    assert stage == "preprocess"
    assert details["stream_id"] == "well001"
    assert details["n_jobs"] == 2
    assert details["force_restart"] is False


def test_stage_cli_respects_use_scratch_root_false_from_data_config(monkeypatch, tmp_path: Path) -> None:
    h5_path = _make_mea_like_path(tmp_path)
    final_root = (tmp_path / "outputs_final").resolve()
    scratch_root = (tmp_path / "outputs_scratch").resolve()

    data_cfg = tmp_path / "debug.data.yml"
    data_cfg.write_text(
        f"""
output_root: {final_root}
scratch_root: {scratch_root}
use_scratch_root: false
datasets:
  - raw_data_h5_path: {h5_path}
    include_in_runtime: true
    wells:
      - well_id: well001
""".strip()
        + "\n",
        encoding="utf-8",
    )

    runtime_cfg = tmp_path / "debug.runtime.yml"
    runtime_cfg.write_text(
        f"""
data: {data_cfg}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    calls: list[tuple[str, dict]] = []

    def _fake_execute_stage(*, stage, context, stage_kwargs, logger):
        calls.append(
            (
                stage,
                {
                    "mea_output_root": str(context.mea_output_root),
                    "final_output_root": (str(context.final_output_root) if context.final_output_root is not None else None),
                    "scratch_output_root": (
                        str(context.scratch_output_root) if context.scratch_output_root is not None else None
                    ),
                },
            )
        )
        return StageExecutionResult(stage=stage, artifacts={"n_common_electrodes": 11})

    import axon_reconstructor.pipeline.pipeline_driver as stage_driver

    monkeypatch.setattr(stage_driver, "execute_stage", _fake_execute_stage)

    args = argparse.Namespace(
        env_file=None,
        config=runtime_cfg,
        stage="preprocess",
        stage_kwargs_file=None,
        stage_kwargs=None,
        debug=False,
        force_restart=False,
        force_replot=False,
        n_jobs=None,
        sorter="kilosort4",
        docker_image=None,
        chunk_duration=None,
        debug_max_units=None,
        debug_max_segments=None,
        h5_path=None,
        stream_id=None,
        mea_output_root=None,
        scratch_output_root=None,
        unit_ids=None,
        unit_limit=None,
        prefer_curated_waveforms_panels=None,
        botm_enable=None,
        botm_n_events=None,
        botm_n_noise_windows=None,
        botm_seed=None,
        botm_prior_signal=None,
        botm_match_fraction_threshold=None,
        botm_sorter=None,
        resume_from=None,
    )

    code = _cmd_stage(args)

    assert code == 0
    assert len(calls) == 1
    _stage, details = calls[0]
    assert details["mea_output_root"] == str(final_root)
    assert details["final_output_root"] == str(final_root)
    assert details["scratch_output_root"] is None
