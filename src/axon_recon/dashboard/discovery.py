"""Walk the runtime target scope to find every `<well>/analysis_outputs/manifest.json`.

Reuses `select_execution_targets` so the dashboard sees the exact same well set
that `axon-recon stages analysis` would compute for the same scope flags.
Read-only — tolerates missing manifests (returns only the ones present on disk).

Slice 3 of `dashboard_ui_refinement_plan.md` adds `discover_available()`
that surfaces the SUPERSET of available tables + columns across loaded
manifests. The dashboard's dropdown population uses this so new
analysis-phase slices that add tables / columns are picked up
automatically without app.py changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from axon_recon.pipeline.config import (
	PipelineRuntimeBundle,
	load_pipeline_runtime_bundle,
	select_execution_targets,
)
from axon_recon.pipeline.output_paths import compute_mea_analysis_output_dir


_DEFAULT_OUTPUT_REL_ROOT = "analysis_outputs"
_DEFAULT_MANIFEST_RELPATH = "manifest.json"


def _analysis_output_rel_root(runtime_config: Any) -> str:
	value = runtime_config.get("stages.analysis.output_rel_root", None)
	text = str(value or _DEFAULT_OUTPUT_REL_ROOT).strip()
	return text.lstrip("/") or _DEFAULT_OUTPUT_REL_ROOT


def _manifest_relpath(runtime_config: Any) -> str:
	value = runtime_config.get("stages.analysis.manifest_relpath", None)
	text = str(value or _DEFAULT_MANIFEST_RELPATH).strip()
	return text.lstrip("/") or _DEFAULT_MANIFEST_RELPATH


def iter_manifest_paths(
	bundle: PipelineRuntimeBundle,
	*,
	target_datasets: list[int] | None = None,
	limit_wells: int | None = None,
	limit_datasets: int | None = None,
	limit_wells_per_dataset: int | None = None,
) -> list[Path]:
	"""Return existing manifest.json paths within the runtime target scope.

	Missing manifests (no analysis stage run for that well yet) are skipped
	silently — the dashboard surfaces what's on disk and doesn't try to
	re-run the stage.
	"""
	targets = select_execution_targets(
		bundle=bundle,
		limit_datasets=limit_datasets,
		target_datasets=target_datasets,
		limit_wells=limit_wells,
		limit_wells_per_dataset=limit_wells_per_dataset,
	)
	output_rel_root = _analysis_output_rel_root(bundle.runtime_config)
	manifest_relpath = _manifest_relpath(bundle.runtime_config)

	found: list[Path] = []
	for target in targets:
		well_out_dir = compute_mea_analysis_output_dir(
			output_root=target.mea_output_root,
			data_file=target.h5_path,
			well=target.stream_id,
		)
		manifest = well_out_dir / output_rel_root / manifest_relpath
		if manifest.is_file():
			found.append(manifest)
	return found


def iter_manifest_paths_from_config(
	*,
	config_path: str,
	target_datasets: list[int] | None = None,
	limit_wells: int | None = None,
	limit_datasets: int | None = None,
	limit_wells_per_dataset: int | None = None,
) -> list[Path]:
	"""Convenience wrapper that loads the bundle then delegates."""
	bundle = load_pipeline_runtime_bundle(config_path=str(config_path))
	return iter_manifest_paths(
		bundle,
		target_datasets=target_datasets,
		limit_wells=limit_wells,
		limit_datasets=limit_datasets,
		limit_wells_per_dataset=limit_wells_per_dataset,
	)


@dataclass(frozen=True)
class DataDiscovery:
	"""Snapshot of what tables / columns are available across loaded
	manifests. Surfaced for slice 3 of dashboard_ui_refinement_plan to
	let the dashboard discover its data surface without hardcoding
	table names.
	"""

	tables: frozenset[str] = field(default_factory=frozenset)
	columns_by_table: dict[str, frozenset[str]] = field(default_factory=dict)
	numeric_columns_by_table: dict[str, frozenset[str]] = field(default_factory=dict)
	categorical_columns_by_table: dict[str, frozenset[str]] = field(default_factory=dict)
	manifest_count: int = 0

	def numeric_columns(self, table: str) -> list[str]:
		"""Return numeric columns for a table in stable insertion order
		when possible (frozenset's hash order is implementation-defined,
		so the result is sorted for determinism)."""
		return sorted(self.numeric_columns_by_table.get(table, frozenset()))

	def categorical_columns(self, table: str) -> list[str]:
		return sorted(self.categorical_columns_by_table.get(table, frozenset()))


def _read_manifest(path: Path) -> dict[str, Any] | None:
	"""Read a manifest.json defensively — tolerate corrupt / missing JSON."""
	try:
		with Path(path).open("r", encoding="utf-8") as fh:
			payload = json.load(fh)
	except (OSError, json.JSONDecodeError):
		return None
	return payload if isinstance(payload, dict) else None


def discover_available(manifest_paths: list[Path] | tuple[Path, ...]) -> DataDiscovery:
	"""Build a DataDiscovery snapshot from the supplied manifests.

	Walks every manifest's ``tables`` map and loads each table just far
	enough to extract column names + dtypes via
	``pd.read_parquet(..., engine=...).dtypes``. Errors (missing parquet,
	unreadable JSON) are silently skipped — the dashboard surfaces what's
	on disk and doesn't fail on partial state.

	Columns are categorized via ``dtype.kind``:
	- numeric: ``i`` / ``u`` / ``f`` (int / uint / float).
	- categorical: everything else (object / string / category / bool / dt).
	"""

	tables: set[str] = set()
	cols_by_table: dict[str, set[str]] = {}
	num_by_table: dict[str, set[str]] = {}
	cat_by_table: dict[str, set[str]] = {}
	manifest_count = 0

	for manifest_path in manifest_paths:
		mp = Path(manifest_path)
		manifest = _read_manifest(mp)
		if manifest is None:
			continue
		manifest_count += 1
		tables_map = manifest.get("tables", {})
		if not isinstance(tables_map, dict):
			continue
		manifest_dir = mp.resolve().parent
		for table_name, table_relpath in tables_map.items():
			if not isinstance(table_relpath, str) or not table_relpath:
				continue
			table_path = (manifest_dir / table_relpath).resolve()
			if not table_path.is_file():
				continue
			try:
				dtypes = pd.read_parquet(table_path).dtypes
			except (OSError, ValueError, ImportError):
				continue
			name = str(table_name)
			tables.add(name)
			cols = cols_by_table.setdefault(name, set())
			numerics = num_by_table.setdefault(name, set())
			categoricals = cat_by_table.setdefault(name, set())
			for col, dtype in dtypes.items():
				col_str = str(col)
				cols.add(col_str)
				if dtype.kind in ("i", "u", "f"):
					numerics.add(col_str)
				else:
					categoricals.add(col_str)
	return DataDiscovery(
		tables=frozenset(tables),
		columns_by_table={k: frozenset(v) for k, v in cols_by_table.items()},
		numeric_columns_by_table={k: frozenset(v) for k, v in num_by_table.items()},
		categorical_columns_by_table={k: frozenset(v) for k, v in cat_by_table.items()},
		manifest_count=manifest_count,
	)


__all__ = [
	"DataDiscovery",
	"discover_available",
	"iter_manifest_paths",
	"iter_manifest_paths_from_config",
]
