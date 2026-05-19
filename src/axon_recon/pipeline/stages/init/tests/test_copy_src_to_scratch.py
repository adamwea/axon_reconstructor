"""Unit + integration tests for the init-stage `copy_src_to_scratch` phase.

Covers:
  - `InitInputs` construction from an `ExecutionTarget` + `InitStageConfig` (via
    `build_init_inputs_for_target`).
  - The phase orchestrator (`run_init_copy_src_to_scratch_phase`) writes a
    well-formed summary_json at the expected path under the well's
    init_outputs/ tree.
  - End-to-end run of `run_init_stage` over a fake target completes
    successfully and yields the expected `MultiTargetStageResult`.
  - Validation: the orchestrator raises when `requires_use_scratch_root: true`
    but the target wasn't materialized into scratch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from axon_recon.pipeline.execution.context import ExecutionTarget

from ..config import InitStageConfig
from ..models.inputs import (
	InitCopySrcToScratchPhaseConfig,
	InitInputs,
	InitPhasesConfig,
)
from ..phases.copy_src_to_scratch import run_init_copy_src_to_scratch_phase
from ..runner import build_init_inputs_for_target, run_init_stage


def _make_target(tmp_path: Path, *, copied_to_scratch: bool) -> ExecutionTarget:
	"""Build an ExecutionTarget that mimics what select_execution_targets produces.

	`copied_to_scratch` is encoded in the *path layout*, not as a direct flag:
	when True, h5_path points at a scratch_inputs/-style location while
	source_h5_path stays at raw_data/. `build_init_inputs_for_target` infers
	the bool by comparing the two paths.
	"""

	source_h5_path = tmp_path / "raw_data" / "260326" / "M08073" / "AxonTracking" / "000208" / "data.raw.h5"
	source_h5_path.parent.mkdir(parents=True, exist_ok=True)
	source_h5_path.write_text("raw\n", encoding="utf-8")
	if copied_to_scratch:
		scratch_h5_path = tmp_path / "scratch_inputs" / "260326" / "M08073" / "AxonTracking" / "000208" / "data.raw.h5"
		scratch_h5_path.parent.mkdir(parents=True, exist_ok=True)
		scratch_h5_path.write_text("scratch\n", encoding="utf-8")
		h5_path = scratch_h5_path
	else:
		h5_path = source_h5_path
	return ExecutionTarget(
		dataset_index=0,
		dataset_id="dataset_000:M08073/000208",
		h5_path=h5_path,
		source_h5_path=source_h5_path,
		stream_id="well000",
		mea_output_root=tmp_path / "analyzed_data",
	)


def test_build_init_inputs_for_target_infers_copied_to_scratch_true(tmp_path: Path) -> None:
	target = _make_target(tmp_path, copied_to_scratch=True)
	stage_config = InitStageConfig(
		enabled=True,
		phase_sequence=("copy_src_to_scratch",),
		phases=InitPhasesConfig(
			copy_src_to_scratch=InitCopySrcToScratchPhaseConfig(enabled=True),
		),
	)
	inputs = build_init_inputs_for_target(target=target, stage_config=stage_config)
	assert isinstance(inputs, InitInputs)
	assert inputs.h5_path == target.h5_path
	assert inputs.source_h5_path == target.source_h5_path
	assert inputs.stream_id == "well000"
	assert inputs.copied_to_scratch is True
	assert inputs.output_rel_root == "init_outputs"
	assert inputs.phase_sequence == ("copy_src_to_scratch",)


def test_build_init_inputs_for_target_infers_copied_to_scratch_false(tmp_path: Path) -> None:
	target = _make_target(tmp_path, copied_to_scratch=False)
	stage_config = InitStageConfig(
		enabled=True,
		phase_sequence=("copy_src_to_scratch",),
	)
	inputs = build_init_inputs_for_target(target=target, stage_config=stage_config)
	assert inputs.copied_to_scratch is False
	assert inputs.h5_path == inputs.source_h5_path


def test_run_init_copy_src_to_scratch_phase_writes_summary_json(tmp_path: Path) -> None:
	"""Orchestrator writes a well-formed summary_json marker at the expected path."""

	target = _make_target(tmp_path, copied_to_scratch=True)
	stage_config = InitStageConfig(
		enabled=True,
		phase_sequence=("copy_src_to_scratch",),
		phases=InitPhasesConfig(
			copy_src_to_scratch=InitCopySrcToScratchPhaseConfig(
				enabled=True,
				requires_use_scratch_root=True,
			),
		),
	)
	inputs = build_init_inputs_for_target(target=target, stage_config=stage_config)
	payload = run_init_copy_src_to_scratch_phase(inputs)

	assert payload["phase"] == "copy_src_to_scratch"
	assert payload["status"] == "ok"
	assert payload["copied_to_scratch"] is True
	assert payload["requires_use_scratch_root"] is True
	assert payload["source_h5_path"] == str(target.source_h5_path)
	assert payload["resolved_h5_path"] == str(target.h5_path)
	summary_path = Path(str(payload["summary_json"]))
	assert summary_path.is_file()
	# Path layout: <mea_output_root>/<rel_pattern>/<well>/init_outputs/context/copy_src_to_scratch_summary.json
	assert summary_path.name == "copy_src_to_scratch_summary.json"
	assert summary_path.parent.name == "context"
	assert summary_path.parent.parent.name == "init_outputs"
	assert summary_path.parent.parent.parent.name == "well000"
	# Round-trip via json: marker is valid JSON with expected keys.
	loaded = json.loads(summary_path.read_text(encoding="utf-8"))
	assert loaded["phase"] == "copy_src_to_scratch"
	assert loaded["status"] == "ok"
	assert loaded["copied_to_scratch"] is True


def test_run_init_copy_src_to_scratch_phase_raises_when_scratch_required_but_not_copied(tmp_path: Path) -> None:
	"""requires_use_scratch_root=true but copied_to_scratch=false → RuntimeError."""

	target = _make_target(tmp_path, copied_to_scratch=False)
	stage_config = InitStageConfig(
		enabled=True,
		phase_sequence=("copy_src_to_scratch",),
		phases=InitPhasesConfig(
			copy_src_to_scratch=InitCopySrcToScratchPhaseConfig(
				enabled=True,
				requires_use_scratch_root=True,
			),
		),
	)
	inputs = build_init_inputs_for_target(target=target, stage_config=stage_config)
	with pytest.raises(RuntimeError, match="requires scratch input materialization"):
		run_init_copy_src_to_scratch_phase(inputs)


def test_run_init_stage_dispatches_copy_phase_for_each_target(tmp_path: Path) -> None:
	"""End-to-end: run_init_stage iterates phase_sequence × targets and accumulates results."""

	target_a = _make_target(tmp_path, copied_to_scratch=True)
	# Two distinct wells under the same dataset; reuse the same source h5.
	target_b = ExecutionTarget(
		dataset_index=target_a.dataset_index,
		dataset_id=target_a.dataset_id,
		h5_path=target_a.h5_path,
		source_h5_path=target_a.source_h5_path,
		stream_id="well001",
		mea_output_root=target_a.mea_output_root,
	)
	stage_config = InitStageConfig(
		enabled=True,
		phase_sequence=("copy_src_to_scratch",),
		phases=InitPhasesConfig(
			copy_src_to_scratch=InitCopySrcToScratchPhaseConfig(
				enabled=True,
				requires_use_scratch_root=True,
			),
		),
	)
	result = run_init_stage(stage_config, targets=[target_a, target_b])
	assert result.stage == "init"
	assert result.total_targets == 2
	assert result.succeeded_targets == 2
	assert result.failed_targets == 0
	# Both target results carry the per-target payload.
	stream_ids = {item.target.stream_id for item in result.target_results}
	assert stream_ids == {"well000", "well001"}
	for item in result.target_results:
		assert item.status == "ok"
		assert item.result["phase"] == "copy_src_to_scratch"
		assert Path(item.result["summary_json"]).is_file()


def test_run_init_stage_captures_per_target_failure(tmp_path: Path) -> None:
	"""A target that fails validation surfaces as `status='error'` without crashing the stage."""

	# copied_to_scratch=False but requires_use_scratch_root=True → orchestrator raises.
	target = _make_target(tmp_path, copied_to_scratch=False)
	stage_config = InitStageConfig(
		enabled=True,
		phase_sequence=("copy_src_to_scratch",),
		phases=InitPhasesConfig(
			copy_src_to_scratch=InitCopySrcToScratchPhaseConfig(
				enabled=True,
				requires_use_scratch_root=True,
			),
		),
	)
	result = run_init_stage(stage_config, targets=[target])
	assert result.total_targets == 1
	assert result.succeeded_targets == 0
	assert result.failed_targets == 1
	assert result.target_results[0].status == "error"
	assert "requires scratch input materialization" in str(result.target_results[0].error)
