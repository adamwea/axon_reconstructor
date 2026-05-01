from __future__ import annotations

from pathlib import Path

import pytest

import axon_recon.pipeline.runner as pipeline_runner
from axon_recon.pipeline.execution.context import ExecutionTarget
from axon_recon.pipeline.execution.results import TargetStageResult
from axon_recon.pipeline.stages.preprocess.models.results import PreprocessResult
from axon_recon.pipeline.publish import remap_path_string_to_final
from axon_recon.runtime_config import RuntimeConfig


def _build_target(tmp_path: Path) -> ExecutionTarget:
    scratch_root = tmp_path / "scratch"
    final_root = tmp_path / "final"
    scratch_root.mkdir(parents=True, exist_ok=True)
    final_root.mkdir(parents=True, exist_ok=True)
    return ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test.h5",
        h5_path=tmp_path / "test.h5",
        stream_id="well001",
        mea_output_root=scratch_root,
        final_output_root=final_root,
        scratch_output_root=scratch_root,
    )


def _build_preprocess_item(target: ExecutionTarget) -> TargetStageResult:
    preprocess_out_dir = target.scratch_output_root / "dataset" / target.stream_id / "stg1_preprocess_outputs"
    well_out_dir = preprocess_out_dir.parent
    summary_json = preprocess_out_dir / "preprocess_summary.json"
    result = PreprocessResult(
        well_out_dir=well_out_dir,
        preprocess_out_dir=preprocess_out_dir,
        summary_json=summary_json,
        outputs={"preprocessed_recording_dir": str(preprocess_out_dir / "preprocessed_recording")},
    )
    return TargetStageResult(target=target, status="ok", result=result, error=None)


def test_publish_preprocess_skipped_when_policy_disabled(monkeypatch, tmp_path: Path) -> None:
    target = _build_target(tmp_path)
    item = _build_preprocess_item(target)

    calls: list[dict[str, object]] = []

    def _fake_publish_path_to_final(*, path, active_root, final_root, mode):
        calls.append({"path": path, "active_root": active_root, "final_root": final_root, "mode": mode})

    monkeypatch.setattr(pipeline_runner, "publish_path_to_final", _fake_publish_path_to_final)

    updated = pipeline_runner._publish_preprocess_target_result(
        item,
        policy=pipeline_runner.PublishPolicy(publish_outputs=False, wipe_scratch_roots=False),
    )

    assert updated is item
    assert calls == []


@pytest.mark.parametrize(
    ("wipe_scratch_roots", "expected_mode"),
    [
        (False, "copy"),
        (True, "move"),
    ],
)
def test_publish_preprocess_mode_and_path_remap(
    monkeypatch, tmp_path: Path, wipe_scratch_roots: bool, expected_mode: str
) -> None:
    target = _build_target(tmp_path)
    item = _build_preprocess_item(target)

    calls: list[dict[str, object]] = []

    def _fake_publish_path_to_final(*, path, active_root, final_root, mode):
        calls.append({"path": path, "active_root": active_root, "final_root": final_root, "mode": mode})

    monkeypatch.setattr(pipeline_runner, "publish_path_to_final", _fake_publish_path_to_final)

    updated = pipeline_runner._publish_preprocess_target_result(
        item,
        policy=pipeline_runner.PublishPolicy(publish_outputs=True, wipe_scratch_roots=wipe_scratch_roots),
    )

    assert len(calls) == 1
    assert calls[0]["mode"] == expected_mode

    expected_preprocess_out = Path(
        remap_path_string_to_final(
            raw=item.result.preprocess_out_dir,
            active_root=target.scratch_output_root,
            final_root=target.final_output_root,
        )
    ).expanduser().resolve()
    expected_output_path = remap_path_string_to_final(
        raw=item.result.outputs["preprocessed_recording_dir"],
        active_root=target.scratch_output_root,
        final_root=target.final_output_root,
    )

    assert updated is not item
    assert isinstance(updated.result, PreprocessResult)
    assert updated.result.preprocess_out_dir == expected_preprocess_out
    assert updated.result.outputs["preprocessed_recording_dir"] == expected_output_path


def test_resolve_publish_policy_forces_wipe_off_when_publish_disabled() -> None:
    runtime_cfg = RuntimeConfig(
        {
            "pipeline": {
                "publish_outputs": False,
                "wipe_scratch_roots": True,
            }
        }
    )
    data_cfg = RuntimeConfig({})

    policy = pipeline_runner._resolve_publish_policy(runtime_config=runtime_cfg, data_config=data_cfg)

    assert policy.publish_outputs is False
    assert policy.wipe_scratch_roots is False


def test_resolve_publish_policy_reads_top_level_values_from_data_config() -> None:
    runtime_cfg = RuntimeConfig({})
    data_cfg = RuntimeConfig(
        {
            "publish_outputs": True,
            "wipe_scratch_roots": True,
        }
    )

    policy = pipeline_runner._resolve_publish_policy(runtime_config=runtime_cfg, data_config=data_cfg)

    assert policy.publish_outputs is True
    assert policy.wipe_scratch_roots is True


def test_resolve_publish_policy_prefers_data_config_over_runtime_config() -> None:
    runtime_cfg = RuntimeConfig(
        {
            "pipeline": {
                "publish_outputs": False,
                "wipe_scratch_roots": False,
            }
        }
    )
    data_cfg = RuntimeConfig(
        {
            "publish_outputs": True,
            "wipe_scratch_roots": True,
        }
    )

    policy = pipeline_runner._resolve_publish_policy(runtime_config=runtime_cfg, data_config=data_cfg)

    assert policy.publish_outputs is True
    assert policy.wipe_scratch_roots is True
