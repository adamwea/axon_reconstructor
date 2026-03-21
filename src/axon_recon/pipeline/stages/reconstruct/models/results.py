from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UnitReconstructionResult:
	unit_id: Any
	status: str
	outputs: dict[str, str] = field(default_factory=dict)
	error: str | None = None


@dataclass(frozen=True)
class ReconstructionResult:
	well_out_dir: Path
	reconstruction_out_dir: Path
	summary_json: Path
	units: list[UnitReconstructionResult]

