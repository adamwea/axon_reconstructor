from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AnalysisResult:
	well_out_dir: Path
	analysis_out_dir: Path
	manifest_json: Path
	outputs: dict[str, str] = field(default_factory=dict)
