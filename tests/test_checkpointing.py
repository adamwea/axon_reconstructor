from __future__ import annotations

from pathlib import Path

from axon_reconstructor.pipeline.checkpointing import (
    ProcessingStage,
    compute_checkpoint_file,
    load_checkpoint,
    save_checkpoint,
)


def _make_mea_like_path(tmp_path: Path) -> Path:
    # Path structure chosen to match MEA_Analysis's relative_pattern parsing:
    #   .../<project>/<date>/<chip>/<run_id>/data.raw.h5
    p = tmp_path / "ProjectX" / "2026-01-01" / "ChipABC" / "123" / "data.raw.h5"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
    return p


def test_checkpoint_roundtrip_and_force_restart(tmp_path: Path) -> None:
    file_path = _make_mea_like_path(tmp_path)
    output_dir = tmp_path / "outputs" / "some" / "well000"
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_file = compute_checkpoint_file(output_dir=output_dir, file_path=file_path, stream_id="well000")

    state = load_checkpoint(
        checkpoint_file=ckpt_file,
        force_restart=False,
        output_dir=output_dir,
        file_path=file_path,
        stream_id="well000",
    )
    assert state.stage == ProcessingStage.NOT_STARTED.value

    save_checkpoint(
        checkpoint_file=ckpt_file,
        state=state,
        stage=ProcessingStage.PREPROCESSING,
        extra_fields={"custom_field": "hello"},
    )

    loaded = load_checkpoint(
        checkpoint_file=ckpt_file,
        force_restart=False,
        output_dir=output_dir,
        file_path=file_path,
        stream_id="well000",
    )
    assert loaded.stage == ProcessingStage.PREPROCESSING.value
    assert loaded.extras.get("custom_field") == "hello"

    restarted = load_checkpoint(
        checkpoint_file=ckpt_file,
        force_restart=True,
        output_dir=output_dir,
        file_path=file_path,
        stream_id="well000",
    )
    assert restarted.stage == ProcessingStage.NOT_STARTED.value
    assert restarted.extras == {}
