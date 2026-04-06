from __future__ import annotations

from pathlib import Path
from typing import Any

from .models.inputs import SpikesortInputs
from .models.results import SpikesortMergeResult, SpikesortResult
from .runner import run_spikesort_merge_stage, run_spikesort_stage


def run_spikesort(inputs: SpikesortInputs) -> SpikesortResult:
	return run_spikesort_stage(inputs)


def run_spikesort_merge(
	*,
	h5_path: Path,
	stream_id: str,
	mea_output_root: Path,
	output_rel_root: str,
	stage_config: Any,
	force_restart: bool,
) -> SpikesortMergeResult:
	return run_spikesort_merge_stage(
		h5_path=h5_path,
		stream_id=stream_id,
		mea_output_root=mea_output_root,
		output_rel_root=output_rel_root,
		stage_config=stage_config,
		force_restart=force_restart,
	)
