from __future__ import annotations

from pathlib import Path

from axon_reconstructor.pipeline.checkpointing import ProcessingStage, load_checkpoint
from axon_reconstructor.pipeline.stage_checkpointing import (
    compute_stage_checkpoint_file,
    save_stage_completed,
    save_stage_failed,
    save_stage_started,
)


def _make_mea_like_path(tmp_path: Path) -> Path:
    p = tmp_path / "ProjectX" / "2026-01-01" / "ChipABC" / "123" / "data.raw.h5"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
    return p


def test_compute_stage_checkpoint_file_suffix(tmp_path: Path) -> None:
    file_path = _make_mea_like_path(tmp_path)
    output_dir = tmp_path / "outputs" / "well000"
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_file = compute_stage_checkpoint_file(
        well_out_dir=output_dir,
        h5_path=file_path,
        stream_id="well000",
        stage_name="waveforms",
    )

    assert ckpt_file.name.endswith("_waveforms_checkpoint.json")


def test_stage_checkpoint_roundtrip_started_then_completed(tmp_path: Path) -> None:
    file_path = _make_mea_like_path(tmp_path)
    output_dir = tmp_path / "outputs" / "well000"
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_file = compute_stage_checkpoint_file(
        well_out_dir=output_dir,
        h5_path=file_path,
        stream_id="well000",
        stage_name="analysis",
    )

    state = load_checkpoint(
        checkpoint_file=ckpt_file,
        force_restart=False,
        output_dir=output_dir,
        file_path=file_path,
        stream_id="well000",
    )

    state = save_stage_started(
        checkpoint_file=ckpt_file,
        state=state,
        stage=ProcessingStage.REPORTS,
        out_dir=output_dir / "analysis_outputs",
        extra_fields={"analysis_out_dir": "abc"},
    )

    state = save_stage_completed(
        checkpoint_file=ckpt_file,
        state=state,
        stage=ProcessingStage.REPORTS_COMPLETE,
        extra_fields={"analysis_summary_json": "summary.json"},
    )

    loaded = load_checkpoint(
        checkpoint_file=ckpt_file,
        force_restart=False,
        output_dir=output_dir,
        file_path=file_path,
        stream_id="well000",
    )

    assert loaded.stage == ProcessingStage.REPORTS_COMPLETE.value
    assert loaded.failed_stage is None
    assert loaded.extras.get("stage_out_dir") == str(output_dir / "analysis_outputs")
    assert loaded.extras.get("analysis_summary_json") == "summary.json"


def test_save_stage_failed_serializes_exception_and_defaults_failed_stage(tmp_path: Path) -> None:
    file_path = _make_mea_like_path(tmp_path)
    output_dir = tmp_path / "outputs" / "well000"
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_file = compute_stage_checkpoint_file(
        well_out_dir=output_dir,
        h5_path=file_path,
        stream_id="well000",
        stage_name="waveforms",
    )

    state = load_checkpoint(
        checkpoint_file=ckpt_file,
        force_restart=False,
        output_dir=output_dir,
        file_path=file_path,
        stream_id="well000",
    )

    failed = save_stage_failed(
        checkpoint_file=ckpt_file,
        state=state,
        stage=ProcessingStage.ANALYZER,
        error=RuntimeError("boom"),
    )

    assert failed.stage == ProcessingStage.ANALYZER.value
    assert failed.failed_stage == "ANALYZER"
    assert isinstance(failed.error, dict)
    assert failed.error.get("type") == "RuntimeError"
    assert "boom" in str(failed.error.get("message"))
