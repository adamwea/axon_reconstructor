from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AnalysisInputs:
	h5_path: Path
	stream_id: str
	mea_output_root: Path
	probe_pitch_um: float | None = None

	output_rel_root: str = "analysis_outputs"
	unit_ids: list[Any] | None = None
	unit_limit: int | None = None

	force_restart: bool = False
	force_replot: bool = False
	n_jobs: int = 1

	# Parsed from stages.analysis.outputs.metrics as a pass-through tree.
	metrics: dict[str, Any] = field(default_factory=dict)
	deferred_warnings: list[str] = field(default_factory=list)
