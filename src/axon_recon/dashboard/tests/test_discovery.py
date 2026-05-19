from __future__ import annotations

import json
from pathlib import Path

import pytest

from ..discovery import (
	DataDiscovery,
	discover_available,
	iter_manifest_paths_from_config,
)


def _write_yaml(path: Path, payload: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(payload, encoding="utf-8")


def _write_manifest(well_dir: Path, *, well_id: str, identity: dict | None = None) -> Path:
	analysis_dir = well_dir / "analysis_outputs"
	analysis_dir.mkdir(parents=True, exist_ok=True)
	payload = {
		"schema_version": "axon_analysis_v1",
		"well_id": well_id,
	}
	if identity:
		payload.update(identity)
	manifest = analysis_dir / "manifest.json"
	manifest.write_text(json.dumps(payload), encoding="utf-8")
	return manifest


def _make_fixture_tree(root: Path, *, datasets: list[dict]) -> tuple[Path, Path]:
	"""Create a minimal runtime+data yaml + on-disk well outputs.

	`datasets` is a list of dicts shaped like:
	  {"raw_h5": ..., "include": True, "wells": [{"well_id": str, "include": bool, "make_manifest": bool}]}
	"""
	# Outputs live under scratch/axon_recon_scratch/outputs/
	scratch_root = root / "scratch"
	outputs_root = scratch_root / "axon_recon_scratch" / "outputs"
	outputs_root.mkdir(parents=True, exist_ok=True)
	output_root = root / "ben_shalom_nas_analysis"
	output_root.mkdir(parents=True, exist_ok=True)

	# Build data.yml
	lines = [
		f"output_root: {output_root}",
		f"scratch_root: {scratch_root}",
		"use_scratch_root: true",
		"datasets:",
	]
	for entry in datasets:
		raw_h5 = entry["raw_h5"]
		# create directories so compute_mea_analysis_output_dir works
		Path(raw_h5).parent.mkdir(parents=True, exist_ok=True)
		Path(raw_h5).write_bytes(b"")
		lines.append(f"  - raw_data_h5_path: {raw_h5}")
		lines.append(f"    include_in_runtime: {str(bool(entry.get('include', True))).lower()}")
		lines.append("    wells:")
		for well in entry.get("wells", []):
			lines.append(f"      - well_id: {well['well_id']}")
			lines.append(
				f"        include_in_runtime: {str(bool(well.get('include', True))).lower()}"
			)
	data_yml = root / "debug.data.yml"
	_write_yaml(data_yml, "\n".join(lines) + "\n")

	# Build runtime.yml referencing the data.yml
	_write_yaml(
		root / "debug.runtime.yml",
		"data: ./debug.data.yml\n",
	)

	# Create on-disk analysis_outputs/manifest.json for wells that opt in.
	for dataset_index, entry in enumerate(datasets):
		raw_h5 = Path(entry["raw_h5"])
		# Mirror compute_mea_analysis_output_dir: scratch/outputs/<parts-after-raw_data>/<well_id>/
		# The fixture h5 lives under {root}/raw/<...>/data.raw.h5 so the analogous
		# output dir is outputs_root / <...> / <well_id>.
		# Using a relative scheme like compute_mea_analysis_output_dir would: we
		# replicate it by computing the path manually here for the test fixture.
		from axon_recon.pipeline.output_paths import compute_mea_analysis_output_dir

		# scratch outputs root mirrors output_root layout
		scratch_outputs_root = outputs_root
		for well in entry.get("wells", []):
			if not well.get("make_manifest", False):
				continue
			well_out_dir = compute_mea_analysis_output_dir(
				output_root=scratch_outputs_root,
				data_file=raw_h5,
				well=well["well_id"],
			)
			well_out_dir.mkdir(parents=True, exist_ok=True)
			_write_manifest(
				well_out_dir,
				well_id=well["well_id"],
				identity={"dataset_index": dataset_index},
			)

	return data_yml, root / "debug.runtime.yml"


def test_iter_manifest_paths_returns_only_wells_with_manifests(tmp_path: Path) -> None:
	root = tmp_path / "fixture"
	root.mkdir()
	_, runtime_yml = _make_fixture_tree(
		root,
		datasets=[
			{
				"raw_h5": str(root / "raw" / "P" / "260101" / "Chip" / "AxonTracking" / "000001" / "data.raw.h5"),
				"include": True,
				"wells": [
					{"well_id": "well000", "include": True, "make_manifest": True},
					{"well_id": "well001", "include": True, "make_manifest": False},
				],
			},
		],
	)
	manifests = iter_manifest_paths_from_config(config_path=str(runtime_yml))
	assert len(manifests) == 1
	assert manifests[0].is_file()
	assert manifests[0].name == "manifest.json"


def test_iter_manifest_paths_respects_target_datasets(tmp_path: Path) -> None:
	root = tmp_path / "fixture"
	root.mkdir()
	_, runtime_yml = _make_fixture_tree(
		root,
		datasets=[
			{
				"raw_h5": str(root / "raw" / "P" / "260101" / "Chip" / "AxonTracking" / "000001" / "data.raw.h5"),
				"include": True,
				"wells": [{"well_id": "well000", "include": True, "make_manifest": True}],
			},
			{
				"raw_h5": str(root / "raw" / "P" / "260102" / "Chip" / "AxonTracking" / "000002" / "data.raw.h5"),
				"include": True,
				"wells": [{"well_id": "well000", "include": True, "make_manifest": True}],
			},
		],
	)
	manifests = iter_manifest_paths_from_config(config_path=str(runtime_yml), target_datasets=[1])
	# Only dataset 1's manifest should be discovered.
	assert len(manifests) == 1
	assert "260102" in str(manifests[0])


def test_iter_manifest_paths_respects_limit_wells(tmp_path: Path) -> None:
	root = tmp_path / "fixture"
	root.mkdir()
	_, runtime_yml = _make_fixture_tree(
		root,
		datasets=[
			{
				"raw_h5": str(root / "raw" / "P" / "260101" / "Chip" / "AxonTracking" / "000001" / "data.raw.h5"),
				"include": True,
				"wells": [
					{"well_id": "well000", "include": True, "make_manifest": True},
					{"well_id": "well001", "include": True, "make_manifest": True},
					{"well_id": "well002", "include": True, "make_manifest": True},
				],
			},
		],
	)
	manifests = iter_manifest_paths_from_config(config_path=str(runtime_yml), limit_wells=2)
	assert len(manifests) == 2


def test_iter_manifest_paths_skips_missing_manifests(tmp_path: Path) -> None:
	root = tmp_path / "fixture"
	root.mkdir()
	_, runtime_yml = _make_fixture_tree(
		root,
		datasets=[
			{
				"raw_h5": str(root / "raw" / "P" / "260101" / "Chip" / "AxonTracking" / "000001" / "data.raw.h5"),
				"include": True,
				"wells": [
					{"well_id": "well000", "include": True, "make_manifest": False},
					{"well_id": "well001", "include": True, "make_manifest": False},
				],
			},
		],
	)
	manifests = iter_manifest_paths_from_config(config_path=str(runtime_yml))
	assert manifests == []


# --- Slice 3 of dashboard_ui_refinement_plan: discover_available ---


def _write_parquet(path: Path, df) -> None:
	import pandas as pd

	path.parent.mkdir(parents=True, exist_ok=True)
	df.to_parquet(path, index=False)


def _make_manifest_with_tables(
	well_dir: Path,
	*,
	well_id: str,
	tables: dict[str, "pd.DataFrame"],
) -> Path:
	import json as _json

	analysis_dir = well_dir / "analysis_outputs"
	analysis_dir.mkdir(parents=True, exist_ok=True)
	tables_meta: dict[str, str] = {}
	for table_name, df in tables.items():
		rel = f"tables/{table_name}.parquet"
		_write_parquet(analysis_dir / rel, df)
		tables_meta[table_name] = rel
	manifest = analysis_dir / "manifest.json"
	manifest.write_text(
		_json.dumps({"well_id": well_id, "tables": tables_meta}),
		encoding="utf-8",
	)
	return manifest


def test_discover_available_empty_input() -> None:
	out = discover_available([])
	assert isinstance(out, DataDiscovery)
	assert out.tables == frozenset()
	assert out.manifest_count == 0


def test_discover_available_aggregates_tables_and_columns(tmp_path: Path) -> None:
	import pandas as pd

	w0 = tmp_path / "w0"
	manifest0 = _make_manifest_with_tables(
		w0,
		well_id="well000",
		tables={
			"units": pd.DataFrame(
				{
					"unit_id": [1, 2, 3],
					"branch_count": [4.0, 5.0, 6.0],
					"genotype": ["wt", "wt", "ko"],
				}
			),
			"well_summary": pd.DataFrame(
				{"n_units": [3], "median_isi_ms": [10.5]}
			),
		},
	)
	w1 = tmp_path / "w1"
	manifest1 = _make_manifest_with_tables(
		w1,
		well_id="well001",
		tables={
			"units": pd.DataFrame(
				{
					"unit_id": [1, 2],
					"branch_count": [4.0, 5.0],
					# different column - tests aggregation
					"isolation": [0.9, 0.85],
				}
			),
		},
	)

	out = discover_available([manifest0, manifest1])
	assert out.tables == frozenset({"units", "well_summary"})
	assert out.manifest_count == 2

	# units columns are the union across manifests.
	units_cols = out.columns_by_table["units"]
	assert "unit_id" in units_cols
	assert "branch_count" in units_cols
	assert "genotype" in units_cols
	assert "isolation" in units_cols

	# Numeric vs categorical classification.
	assert "branch_count" in out.numeric_columns_by_table["units"]
	assert "isolation" in out.numeric_columns_by_table["units"]
	assert "genotype" in out.categorical_columns_by_table["units"]
	assert "genotype" not in out.numeric_columns_by_table["units"]


def test_discover_available_skips_missing_parquet_files(tmp_path: Path) -> None:
	import json as _json
	import pandas as pd

	# Manifest references a parquet that doesn't exist on disk — skipped
	# silently.
	analysis_dir = tmp_path / "well_x" / "analysis_outputs"
	analysis_dir.mkdir(parents=True, exist_ok=True)
	manifest = analysis_dir / "manifest.json"
	manifest.write_text(
		_json.dumps(
			{"well_id": "well_x", "tables": {"units": "tables/does_not_exist.parquet"}}
		),
		encoding="utf-8",
	)
	out = discover_available([manifest])
	# Manifest counted (it was valid JSON), but no tables surface.
	assert out.manifest_count == 1
	assert out.tables == frozenset()


def test_discover_available_numeric_columns_helper(tmp_path: Path) -> None:
	import pandas as pd

	w0 = tmp_path / "w0"
	manifest = _make_manifest_with_tables(
		w0,
		well_id="w0",
		tables={
			"units": pd.DataFrame(
				{
					"a": [1.0, 2.0],
					"b": ["x", "y"],
					"c": [10, 20],
				}
			),
		},
	)
	out = discover_available([manifest])
	# Sorted for determinism.
	assert out.numeric_columns("units") == ["a", "c"]
	assert out.categorical_columns("units") == ["b"]
	# Unknown table returns empty lists.
	assert out.numeric_columns("nonexistent") == []
