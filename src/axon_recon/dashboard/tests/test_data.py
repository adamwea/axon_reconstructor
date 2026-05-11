from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..data import load_all


def _make_well(
	tmp_path: Path,
	*,
	well_id: str,
	units_rows: list[dict] | None,
	summary_row: dict | None,
	manifest_identity: dict | None = None,
) -> Path:
	well_dir = tmp_path / well_id
	analysis_dir = well_dir / "analysis_outputs"
	tables_dir = analysis_dir / "tables"
	tables_dir.mkdir(parents=True, exist_ok=True)

	tables_section: dict[str, str] = {}
	if units_rows is not None:
		units_path = tables_dir / "units.parquet"
		pd.DataFrame(units_rows).to_parquet(units_path, engine="pyarrow", index=False)
		tables_section["units"] = "tables/units.parquet"
	if summary_row is not None:
		summary_path = tables_dir / "well_summary.parquet"
		pd.DataFrame([summary_row]).to_parquet(summary_path, engine="pyarrow", index=False)
		tables_section["well_summary"] = "tables/well_summary.parquet"

	manifest = {
		"schema_version": "axon_analysis_v1",
		"well_id": well_id,
		"tables": tables_section,
	}
	if manifest_identity:
		manifest.update(manifest_identity)
	(analysis_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
	return analysis_dir / "manifest.json"


def test_load_all_concatenates_units_and_well_summary(tmp_path: Path) -> None:
	m1 = _make_well(
		tmp_path,
		well_id="well000",
		units_rows=[{"unit_id": 1, "branch_count": 1.0}, {"unit_id": 2, "branch_count": 3.0}],
		summary_row={"unit_count_total": 2, "mean_branch_count": 2.0},
		manifest_identity={"project": "P", "DIV": 12, "well_attributes": {"genotype": "WT"}},
	)
	m2 = _make_well(
		tmp_path,
		well_id="well001",
		units_rows=[{"unit_id": 1, "branch_count": 5.0}],
		summary_row={"unit_count_total": 1, "mean_branch_count": 5.0},
		manifest_identity={"project": "P", "DIV": 18, "well_attributes": {"genotype": "KO"}},
	)
	tables = load_all([m1, m2])
	units = tables["units"]
	assert len(units) == 3
	# Identity columns are stamped onto every row from the manifest.
	assert set(units["project"]) == {"P"}
	assert set(units["genotype"]) == {"WT", "KO"}
	assert set(units["well_id"]) == {"well000", "well001"}

	ws = tables["well_summary"]
	assert len(ws) == 2


def test_load_all_handles_missing_tables_entry(tmp_path: Path) -> None:
	m = _make_well(tmp_path, well_id="well000", units_rows=None, summary_row=None)
	tables = load_all([m])
	# Both expected tables exist as empty DataFrames.
	assert "units" in tables and tables["units"].empty
	assert "well_summary" in tables and tables["well_summary"].empty


def test_load_all_skips_missing_parquet_files(tmp_path: Path) -> None:
	# Create a manifest with tables entries but delete the parquet file.
	m = _make_well(
		tmp_path,
		well_id="well000",
		units_rows=[{"unit_id": 1}],
		summary_row=None,
	)
	(m.parent / "tables" / "units.parquet").unlink()
	tables = load_all([m])
	assert tables["units"].empty


def test_load_all_empty_manifest_list_returns_empty_defaults(tmp_path: Path) -> None:
	tables = load_all([])
	assert set(tables.keys()) == {"units", "well_summary"}
	assert tables["units"].empty
	assert tables["well_summary"].empty


def test_load_all_handles_bad_json_manifest(tmp_path: Path) -> None:
	manifest_dir = tmp_path / "broken" / "analysis_outputs"
	manifest_dir.mkdir(parents=True, exist_ok=True)
	bad = manifest_dir / "manifest.json"
	bad.write_text("not valid json", encoding="utf-8")
	tables = load_all([bad])
	assert tables["units"].empty
	assert tables["well_summary"].empty
