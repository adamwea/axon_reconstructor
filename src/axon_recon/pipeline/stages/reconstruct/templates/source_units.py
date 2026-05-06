from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote

from .io import read_json, write_json


ANALYZER_SOURCE_UNITS_RELPATH = Path("context/analyzer_source_units")


def unit_key(value: Any) -> str:
    try:
        return str(int(value))
    except Exception:
        return str(value)


def extract_analyzer_unit_ids(analyzer: Any) -> list[Any] | None:
    sorting = getattr(analyzer, "sorting", None)
    if sorting is None:
        return None
    get_unit_ids = getattr(sorting, "get_unit_ids", None)
    if callable(get_unit_ids):
        try:
            return list(get_unit_ids())
        except Exception:
            pass
    try:
        unit_ids = sorting.unit_ids
    except Exception:
        return None
    if unit_ids is None:
        return None
    try:
        return list(unit_ids)
    except Exception:
        return None


def resolve_analyzer_source_units_path(*, templates_out_dir: Path, source_name: str) -> Path:
    token = quote(str(source_name), safe="") or "source"
    return templates_out_dir / ANALYZER_SOURCE_UNITS_RELPATH / f"{token}.json"


def write_analyzer_source_units(
    *,
    templates_out_dir: Path,
    source_name: str,
    unit_ids: list[Any] | tuple[Any, ...],
    source_kind: str | None = None,
) -> Path:
    units = list(unit_ids)
    path = resolve_analyzer_source_units_path(
        templates_out_dir=templates_out_dir,
        source_name=str(source_name),
    )
    write_json(
        path,
        {
            "artifact_version": 1,
            "source_name": str(source_name),
            "source_kind": (None if source_kind is None else str(source_kind)),
            "unit_ids": units,
            "unit_count": int(len(units)),
            "membership_source": "analyzer.sorting.unit_ids",
        },
    )
    return path


def load_analyzer_source_units(*, templates_out_dir: Path, source_name: str) -> dict[str, Any] | None:
    path = resolve_analyzer_source_units_path(
        templates_out_dir=templates_out_dir,
        source_name=str(source_name),
    )
    if not path.exists():
        return None
    payload = read_json(path)
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("unit_ids", None), list):
        return None
    return payload