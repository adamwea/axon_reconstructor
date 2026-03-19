from __future__ import annotations

import argparse
from pathlib import Path

from axon_reconstructor.cli import _cmd_stage
from axon_reconstructor.pipeline.pipeline_driver import StageExecutionResult


def _make_mea_like_path(tmp_path: Path) -> Path:
    p = tmp_path / "ProjectX" / "2026-01-01" / "ChipABC" / "123" / "data.raw.h5"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
    return p


def _write_runtime_cfg(tmp_path: Path, *, stage: str, max_workers: int, max_stage_workers: int, well_workers: int) -> Path:
    cfg = (
        "resources:\n"
        f"  max_workers: {int(max_workers)}\n"
        "stages:\n"
        f"  {stage}:\n"
        "    resources:\n"
        f"      max_stage_workers: {int(max_stage_workers)}\n"
        f"      well_workers: {int(well_workers)}\n"
    )
    p = tmp_path / "runtime.yml"
    p.write_text(cfg, encoding="utf-8")
    return p


def _base_args(*, stage: str, config: Path, h5_path: Path, output_root: Path, stage_kwargs: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        env_file=None,
        config=str(config),
        stage=stage,
        stage_kwargs_file=None,
        stage_kwargs=stage_kwargs,
        debug=False,
        force_restart=False,
        force_replot=False,
        n_jobs=None,
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
        recon_templates_variant_name=None,
        recon_variant_name=None,
        recon_top_n_density_requested=None,
        recon_write_top_density_grid=None,
        recon_show_density_scale_debug_text=None,
        recon_show_density_scale_global_debug_text=None,
        recon_show_density_scale_local_debug_text=None,
        recon_replot_top_density_grid_only=None,
    )


def test_preprocess_derives_n_jobs_from_stage_and_well_workers(monkeypatch, tmp_path: Path) -> None:
    h5_path = _make_mea_like_path(tmp_path)
    output_root = tmp_path / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    cfg_path = _write_runtime_cfg(
        tmp_path,
        stage="preprocess",
        max_workers=24,
        max_stage_workers=12,
        well_workers=5,
    )

    calls: list[tuple[int, dict]] = []

    def _fake_execute_stage(*, stage, context, stage_kwargs, logger):
        calls.append((int(context.n_jobs), dict(stage_kwargs or {})))
        return StageExecutionResult(stage=stage, artifacts={"n_common_electrodes": 7})

    import axon_reconstructor.pipeline.pipeline_driver as stage_driver

    monkeypatch.setattr(stage_driver, "execute_stage", _fake_execute_stage)

    args = _base_args(
        stage="preprocess",
        config=cfg_path,
        h5_path=h5_path,
        output_root=output_root,
        stage_kwargs='{"n_jobs": 99}',
    )
    code = _cmd_stage(args)

    assert code == 0
    assert len(calls) == 1
    context_n_jobs, kwargs = calls[0]
    assert context_n_jobs == 2
    assert int(kwargs.get("n_jobs")) == 2


def test_reconstruct_derives_unit_workers_from_stage_and_well_workers(monkeypatch, tmp_path: Path) -> None:
    h5_path = _make_mea_like_path(tmp_path)
    output_root = tmp_path / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    cfg_path = _write_runtime_cfg(
        tmp_path,
        stage="reconstruct",
        max_workers=24,
        max_stage_workers=8,
        well_workers=4,
    )

    calls: list[dict] = []

    def _fake_execute_stage(*, stage, context, stage_kwargs, logger):
        calls.append({
            "stage": stage,
            "n_jobs": int(context.n_jobs),
            "stage_kwargs": dict(stage_kwargs or {}),
        })
        return StageExecutionResult(stage=stage, artifacts={"reconstruction_out_dir": str(output_root / "dummy")})

    import axon_reconstructor.pipeline.pipeline_driver as stage_driver

    monkeypatch.setattr(stage_driver, "execute_stage", _fake_execute_stage)

    args = _base_args(
        stage="reconstruct",
        config=cfg_path,
        h5_path=h5_path,
        output_root=output_root,
        stage_kwargs='{"unit_workers": 10}',
    )
    code = _cmd_stage(args)

    assert code == 0
    assert len(calls) == 1
    call = calls[0]
    assert call["stage"] == "reconstruct"
    assert call["n_jobs"] == 2
    assert int(call["stage_kwargs"].get("unit_workers")) == 2
