from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models.inputs import PerUnitOutputsConfig


def read_json(path: Path) -> Any:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def write_json(path: Path, payload: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)


def jsonable(x: Any) -> Any:
	try:
		import numpy as np  # type: ignore[import-not-found]

		if isinstance(x, (np.integer, np.floating)):
			return x.item()
	except Exception:
		pass
	if isinstance(x, Path):
		return str(x)
	return x


def as_list(x: Any) -> list[Any]:
	if x is None:
		return []
	if isinstance(x, (list, tuple)):
		return list(x)
	if isinstance(x, (str, bytes)):
		return [x]
	try:
		import numpy as np  # type: ignore[import-not-found]

		if isinstance(x, np.ndarray):
			return x.ravel().tolist()
	except Exception:
		pass
	try:
		return list(x)
	except Exception:
		return [x]


def as_int_list(x: Any) -> list[int]:
	out: list[int] = []
	for value in as_list(x):
		try:
			out.append(int(value))
		except Exception:
			continue
	return out


def as_float_list(x: Any) -> list[float]:
	out: list[float] = []
	for value in as_list(x):
		try:
			out.append(float(value))
		except Exception:
			continue
	return out


def format_unit_reldir(unit_reldir: str, unit_id: Any) -> Path:
	try:
		unit_id_int = int(unit_id)
		rendered = str(unit_reldir).format(unit_id=unit_id_int)
	except Exception:
		rendered = str(unit_reldir).format(unit_id=unit_id)
	return Path(rendered)


def resolve_unit_output_paths(
	*,
	reconstruction_out_dir: Path,
	unit_id: Any,
	per_unit_outputs: PerUnitOutputsConfig,
) -> dict[str, Path]:
	unit_rel = format_unit_reldir(per_unit_outputs.unit_reldir, unit_id)
	unit_dir = reconstruction_out_dir / unit_rel

	return {
		"unit_dir": unit_dir,
		"unit_summary_json": unit_dir / "unit_reconstruction_summary.json",
		"branches_raw_json": unit_dir / Path(str(per_unit_outputs.branches_raw_relpath)).expanduser(),
		"branches_json": unit_dir / Path(str(per_unit_outputs.branches_relpath)).expanduser(),
		"heuristics_json": unit_dir / Path(str(per_unit_outputs.heuristics_relpath)).expanduser(),
		"gtr_pkl": unit_dir / Path(str(per_unit_outputs.gtr_pkl_relpath)).expanduser(),
		"gtr_json": unit_dir / Path(str(per_unit_outputs.gtr_json_relpath)).expanduser(),
	}

