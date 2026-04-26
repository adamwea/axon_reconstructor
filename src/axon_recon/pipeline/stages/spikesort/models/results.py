from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SpikesortResult:
	well_out_dir: Path
	spikesort_out_dir: Path
	summary_json: Path
	outputs: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SpikesortMergeResult:
	well_out_dir: Path
	merge_out_dir: Path
	summary_json: Path
	outputs: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SpikesortBombcellResult:
	well_out_dir: Path
	bombcell_out_dir: Path
	summary_json: Path | None = None
	outputs: dict[str, str] = field(default_factory=dict)
