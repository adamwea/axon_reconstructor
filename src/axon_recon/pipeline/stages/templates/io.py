from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models.inputs import PerUnitTemplatesOutputsConfig


def read_json(path: Path) -> Any:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def write_json(path: Path, payload: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)


def format_unit_reldir(unit_reldir: str, unit_id: Any) -> Path:
	try:
		unit_id_int = int(unit_id)
		rendered = str(unit_reldir).format(unit_id=unit_id_int)
	except Exception:
		rendered = str(unit_reldir).format(unit_id=unit_id)
	return Path(rendered)


def _render_template_paths(unit_dir: Path, relpath: str) -> tuple[Path, Path]:
	raw = Path(str(relpath)).expanduser()
	if raw.suffix.lower() in {".png", ".svg"}:
		base = raw.with_suffix("")
	else:
		base = raw
	return unit_dir / base.with_suffix(".png"), unit_dir / base.with_suffix(".svg")


def resolve_unit_output_paths(
	*,
	templates_out_dir: Path,
	unit_id: Any,
	per_unit_outputs: PerUnitTemplatesOutputsConfig,
) -> dict[str, Path]:
	unit_rel = format_unit_reldir(per_unit_outputs.unit_reldir, unit_id)
	unit_dir = templates_out_dir / unit_rel
	template_png, template_svg = _render_template_paths(unit_dir, per_unit_outputs.template.relpath)

	return {
		"unit_dir": unit_dir,
		"unit_summary_json": unit_dir / "unit_templates_summary.json",
		"template_png": template_png,
		"template_svg": template_svg,
	}
