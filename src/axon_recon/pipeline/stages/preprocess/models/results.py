from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PreprocessResult:
	well_out_dir: Path
	preprocess_out_dir: Path
	summary_json: Path
	outputs: dict[str, str] = field(default_factory=dict)
