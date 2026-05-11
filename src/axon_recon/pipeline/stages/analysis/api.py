from __future__ import annotations

from pathlib import Path
from typing import Any

from .models.results import AnalysisResult
from .orchestrators import run_analysis_compute_metrics


def run_analysis_metrics(
	*,
	dataset_index: int,
	dataset_id: str | None,
	h5_path: Path,
	stream_id: str,
	mea_output_root: Path,
	output_rel_root: str,
	stage_config: Any,
	force_restart: bool,
) -> AnalysisResult:
	return run_analysis_compute_metrics(
		dataset_index=dataset_index,
		dataset_id=dataset_id,
		h5_path=h5_path,
		stream_id=stream_id,
		mea_output_root=mea_output_root,
		output_rel_root=output_rel_root,
		stage_config=stage_config,
		force_restart=force_restart,
	)
