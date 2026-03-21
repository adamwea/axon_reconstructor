from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UnitTemplatesResult:
	unit_id: Any
	status: str
	outputs: dict[str, str] = field(default_factory=dict)
	error: str | None = None


@dataclass(frozen=True)
class TemplatesResult:
	well_out_dir: Path
	templates_out_dir: Path
	summary_json: Path
	units: list[UnitTemplatesResult]
