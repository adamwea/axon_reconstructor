"""Unit + integration tests for the cleanup-stage `wipe_src_scratch` phase.

Covers:
  - `CleanupInputs` construction from an `ExecutionTarget` + `CleanupStageConfig`
    (via `build_cleanup_inputs_for_target`).
  - The phase orchestrator (`run_cleanup_wipe_src_scratch_phase`) writes a
    well-formed summary_json at the expected path under the well's
    cleanup_outputs/ tree AND deletes the scratch-staged h5 + .cfg sidecars.
  - End-to-end run of `run_cleanup_stage` over a fake target completes
    successfully and yields the expected `MultiTargetStageResult`.
  - Dry-run mode reports the would-remove paths without deleting them.
  - Validation: the orchestrator raises when `requires_use_scratch_root: true`
    but the target wasn't materialized into scratch.
  - copied_to_scratch=False short-circuits to status=skipped with no deletes.

The tests intentionally exercise the orchestrator directly (not through
`select_execution_targets`) so they're hermetic: tmp_path holds both the
mock source and scratch trees.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from axon_recon.pipeline.execution.context import ExecutionTarget

from ..config import CleanupStageConfig
from ..models.inputs import (
	CleanupInputs,
	CleanupPhasesConfig,
	CleanupWipeSrcScratchPhaseConfig,
)
from ..phases.wipe_src_scratch import run_cleanup_wipe_src_scratch_phase
from ..runner import build_cleanup_inputs_for_target, run_cleanup_stage


def _make_target(tmp_path: Path, *, copied_to_scratch: bool) -> ExecutionTarget:
	"""Build an ExecutionTarget that mimics what select_execution_targets produces.

	`copied_to_scratch` is encoded in the *path layout*, not as a direct flag:
	when True, h5_path points at a scratch_inputs/-style location while
	source_h5_path stays at raw_data/. `build_cleanup_inputs_for_target`
	infers the bool by comparing the two paths.
	"""

	source_h5_path = tmp_path / "raw_data" / "260326" / "M08073" / "AxonTracking" / "000208" / "data.raw.h5"
	source_h5_path.parent.mkdir(parents=True, exist_ok=True)
	source_h5_path.write_text("raw\n", encoding="utf-8")
	if copied_to_scratch:
		scratch_h5_path = tmp_path / "scratch_inputs" / "260326" / "M08073" / "AxonTracking" / "000208" / "data.raw.h5"
		scratch_h5_path.parent.mkdir(parents=True, exist_ok=True)
		scratch_h5_path.write_text("scratch\n", encoding="utf-8")
		# A couple of sidecar configs are part of the wipe candidate set.
		(scratch_h5_path.parent / "data.cfg").write_text("foo=1\n", encoding="utf-8")
		(scratch_h5_path.parent / "extra.cfg").write_text("bar=2\n", encoding="utf-8")
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


def test_build_cleanup_inputs_for_target_infers_copied_to_scratch_true(tmp_path: Path) -> None:
	target = _make_target(tmp_path, copied_to_scratch=True)
	stage_config = CleanupStageConfig(
		enabled=True,
		phase_sequence=("wipe_src_scratch",),
		phases=CleanupPhasesConfig(
			wipe_src_scratch=CleanupWipeSrcScratchPhaseConfig(enabled=True),
		),
	)
	inputs = build_cleanup_inputs_for_target(target=target, stage_config=stage_config)
	assert isinstance(inputs, CleanupInputs)
	assert inputs.h5_path == target.h5_path
	assert inputs.source_h5_path == target.source_h5_path
	assert inputs.stream_id == "well000"
	assert inputs.copied_to_scratch is True
	assert inputs.output_rel_root == "cleanup_outputs"
	assert inputs.phase_sequence == ("wipe_src_scratch",)


def test_build_cleanup_inputs_for_target_infers_copied_to_scratch_false(tmp_path: Path) -> None:
	target = _make_target(tmp_path, copied_to_scratch=False)
	stage_config = CleanupStageConfig(
		enabled=True,
		phase_sequence=("wipe_src_scratch",),
	)
	inputs = build_cleanup_inputs_for_target(target=target, stage_config=stage_config)
	assert inputs.copied_to_scratch is False
	assert inputs.h5_path == inputs.source_h5_path


def test_run_cleanup_wipe_src_scratch_phase_removes_scratch_input_files(tmp_path: Path) -> None:
	"""Orchestrator deletes the scratch h5 + .cfg sidecars and writes summary_json."""

	target = _make_target(tmp_path, copied_to_scratch=True)
	stage_config = CleanupStageConfig(
		enabled=True,
		phase_sequence=("wipe_src_scratch",),
		phases=CleanupPhasesConfig(
			wipe_src_scratch=CleanupWipeSrcScratchPhaseConfig(
				enabled=True,
				dry_run=False,
				requires_use_scratch_root=True,
			),
		),
	)
	inputs = build_cleanup_inputs_for_target(target=target, stage_config=stage_config)

	scratch_h5_path = target.h5_path
	scratch_cfg_a = target.h5_path.parent / "data.cfg"
	scratch_cfg_b = target.h5_path.parent / "extra.cfg"
	assert scratch_h5_path.exists()
	assert scratch_cfg_a.exists()
	assert scratch_cfg_b.exists()

	payload = run_cleanup_wipe_src_scratch_phase(inputs)

	assert payload["phase"] == "wipe_src_scratch"
	assert payload["dry_run"] is False
	assert payload["status"] == "ok"
	assert payload["copied_to_scratch"] is True
	assert payload["requires_use_scratch_root"] is True
	assert payload["source_h5_path"] == str(target.source_h5_path)
	assert payload["resolved_h5_path"] == str(target.h5_path)
	assert str(scratch_h5_path) in list(payload["removed_paths"])
	assert str(scratch_cfg_a) in list(payload["removed_paths"])
	assert str(scratch_cfg_b) in list(payload["removed_paths"])
	assert list(payload["would_remove_paths"]) == []
	# Side effects: the scratch files are gone.
	assert not scratch_h5_path.exists()
	assert not scratch_cfg_a.exists()
	assert not scratch_cfg_b.exists()
	# Source files are NOT touched.
	assert target.source_h5_path.exists()

	summary_path = Path(str(payload["summary_json"]))
	assert summary_path.is_file()
	# Path layout: <mea_output_root>/<rel_pattern>/<well>/cleanup_outputs/context/wipe_src_scratch_summary.json
	assert summary_path.name == "wipe_src_scratch_summary.json"
	assert summary_path.parent.name == "context"
	assert summary_path.parent.parent.name == "cleanup_outputs"
	assert summary_path.parent.parent.parent.name == "well000"
	loaded = json.loads(summary_path.read_text(encoding="utf-8"))
	assert loaded["phase"] == "wipe_src_scratch"
	assert loaded["status"] == "ok"


def test_run_cleanup_wipe_src_scratch_phase_dry_run_reports_paths_without_deleting(tmp_path: Path) -> None:
	"""dry_run: orchestrator reports would-remove paths and leaves files in place."""

	target = _make_target(tmp_path, copied_to_scratch=True)
	stage_config = CleanupStageConfig(
		enabled=True,
		phase_sequence=("wipe_src_scratch",),
		phases=CleanupPhasesConfig(
			wipe_src_scratch=CleanupWipeSrcScratchPhaseConfig(
				enabled=True,
				dry_run=True,
				requires_use_scratch_root=True,
			),
		),
	)
	inputs = build_cleanup_inputs_for_target(target=target, stage_config=stage_config)

	scratch_h5_path = target.h5_path
	scratch_cfg_a = target.h5_path.parent / "data.cfg"

	payload = run_cleanup_wipe_src_scratch_phase(inputs)

	assert payload["phase"] == "wipe_src_scratch"
	assert payload["dry_run"] is True
	assert payload["status"] == "dry_run"
	assert list(payload["removed_paths"]) == []
	assert str(scratch_h5_path) in list(payload["would_remove_paths"])
	assert str(scratch_cfg_a) in list(payload["would_remove_paths"])
	# Files survive dry-run.
	assert scratch_h5_path.exists()
	assert scratch_cfg_a.exists()


def test_run_cleanup_wipe_src_scratch_phase_skipped_when_not_copied_to_scratch(tmp_path: Path) -> None:
	"""copied_to_scratch=False short-circuits to status=skipped, no deletes."""

	target = _make_target(tmp_path, copied_to_scratch=False)
	stage_config = CleanupStageConfig(
		enabled=True,
		phase_sequence=("wipe_src_scratch",),
		phases=CleanupPhasesConfig(
			wipe_src_scratch=CleanupWipeSrcScratchPhaseConfig(
				enabled=True,
				# requires_use_scratch_root=False here so the orchestrator skips
				# instead of raising — the alternative path is exercised below.
				requires_use_scratch_root=False,
			),
		),
	)
	inputs = build_cleanup_inputs_for_target(target=target, stage_config=stage_config)

	payload = run_cleanup_wipe_src_scratch_phase(inputs)

	assert payload["phase"] == "wipe_src_scratch"
	assert payload["status"] == "skipped"
	assert payload.get("reason") == "selected_target_did_not_use_scratch_input_root"
	assert list(payload["removed_paths"]) == []
	# Source file untouched.
	assert target.source_h5_path.exists()


def test_run_cleanup_wipe_src_scratch_phase_raises_when_scratch_required_but_not_copied(tmp_path: Path) -> None:
	"""requires_use_scratch_root=true but copied_to_scratch=false → RuntimeError."""

	target = _make_target(tmp_path, copied_to_scratch=False)
	stage_config = CleanupStageConfig(
		enabled=True,
		phase_sequence=("wipe_src_scratch",),
		phases=CleanupPhasesConfig(
			wipe_src_scratch=CleanupWipeSrcScratchPhaseConfig(
				enabled=True,
				requires_use_scratch_root=True,
			),
		),
	)
	inputs = build_cleanup_inputs_for_target(target=target, stage_config=stage_config)
	with pytest.raises(RuntimeError, match="requires scratch input materialization"):
		run_cleanup_wipe_src_scratch_phase(inputs)


def test_run_cleanup_stage_dispatches_wipe_phase_for_each_target(tmp_path: Path) -> None:
	"""End-to-end: run_cleanup_stage iterates phase_sequence × targets."""

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
	stage_config = CleanupStageConfig(
		enabled=True,
		phase_sequence=("wipe_src_scratch",),
		phases=CleanupPhasesConfig(
			wipe_src_scratch=CleanupWipeSrcScratchPhaseConfig(
				enabled=True,
				dry_run=True,  # keep test hermetic — don't depend on the actual file existing post-delete in the 2nd iteration
				requires_use_scratch_root=True,
			),
		),
	)
	result = run_cleanup_stage(stage_config, targets=[target_a, target_b])
	assert result.stage == "cleanup"
	assert result.total_targets == 2
	assert result.succeeded_targets == 2
	assert result.failed_targets == 0
	stream_ids = {item.target.stream_id for item in result.target_results}
	assert stream_ids == {"well000", "well001"}
	for item in result.target_results:
		assert item.status == "ok"
		assert item.result["phase"] == "wipe_src_scratch"
		assert Path(item.result["summary_json"]).is_file()


def test_run_cleanup_stage_captures_per_target_failure(tmp_path: Path) -> None:
	"""A target that fails validation surfaces as `status='error'` without crashing the stage."""

	# copied_to_scratch=False but requires_use_scratch_root=True → orchestrator raises.
	target = _make_target(tmp_path, copied_to_scratch=False)
	stage_config = CleanupStageConfig(
		enabled=True,
		phase_sequence=("wipe_src_scratch",),
		phases=CleanupPhasesConfig(
			wipe_src_scratch=CleanupWipeSrcScratchPhaseConfig(
				enabled=True,
				requires_use_scratch_root=True,
			),
		),
	)
	result = run_cleanup_stage(stage_config, targets=[target])
	assert result.total_targets == 1
	assert result.succeeded_targets == 0
	assert result.failed_targets == 1
	assert result.target_results[0].status == "error"
	assert "requires scratch input materialization" in str(result.target_results[0].error)
