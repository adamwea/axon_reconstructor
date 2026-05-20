from __future__ import annotations

from axon_recon.pipeline.execution.results import (
	MultiTargetStageResult,
	stage_aggregate_exit_code,
)


def _mk(total: int, succeeded: int) -> MultiTargetStageResult:
	return MultiTargetStageResult(
		stage="test.stage",
		total_targets=total,
		succeeded_targets=succeeded,
		failed_targets=total - succeeded,
		target_results=[],
	)


def test_all_succeeded_returns_zero() -> None:
	assert stage_aggregate_exit_code(_mk(total=3, succeeded=3)) == 0


def test_partial_success_returns_zero() -> None:
	# afterok chains should proceed when at least one target produced output.
	assert stage_aggregate_exit_code(_mk(total=3, succeeded=1)) == 0


def test_all_failed_returns_two() -> None:
	# afterok chains should NOT proceed when nothing succeeded.
	assert stage_aggregate_exit_code(_mk(total=3, succeeded=0)) == 2


def test_zero_targets_returns_zero() -> None:
	# Nothing to do — phase plan empty or YAML disabled all phases.
	assert stage_aggregate_exit_code(_mk(total=0, succeeded=0)) == 0
