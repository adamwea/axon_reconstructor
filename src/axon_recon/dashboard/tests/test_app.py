from __future__ import annotations

import dash
import pandas as pd

from ..app import (
	ID_FILTER_BOMBCELL,
	ID_FILTER_CHIP,
	ID_FILTER_DIV_RANGE,
	ID_FILTER_GENOTYPE,
	ID_FILTER_MEDIA,
	ID_FILTER_MIN_NUM_BRANCHES,
	ID_FILTER_MIN_NUM_SPIKES,
	ID_FILTER_MIN_RECON_QUALITY,
	ID_FILTER_PLATING,
	ID_FILTER_PROJECT,
	ID_FILTER_REQUIRE_RECON_OK,
	ID_FILTER_SCAN_TYPE,
	ID_FILTER_WELL,
	ID_HIST_COLOR,
	ID_HIST_X_AXIS,
	ID_HISTOGRAM,
	ID_UNITS_TABLE,
	build_app,
)


def _walk_components(tree, out):
	out.append(tree)
	for child in getattr(tree, "children", None) or []:
		if isinstance(child, list):
			for sub in child:
				_walk_components(sub, out)
		elif child is not None:
			_walk_components(child, out)


def _component_ids(app: dash.Dash) -> set[str]:
	ids: set[str] = set()
	collected: list = []
	_walk_components(app.layout, collected)
	for component in collected:
		ident = getattr(component, "id", None)
		if isinstance(ident, str):
			ids.add(ident)
	return ids


def _make_units_df():
	return pd.DataFrame(
		{
			"project": ["P", "P"],
			"chip_id": ["c1", "c2"],
			"well_id": ["w0", "w1"],
			"scan_type": ["AxonTracking", "AxonTracking"],
			"genotype": ["WT", "KO"],
			"media": ["DMEM", "DMEM"],
			"plating_density": [80000, 80000],
			"DIV": [12, 18],
			"recon_status": ["ok", "ok"],
			"bombcell_label": ["good", "non_soma_good"],
			"num_spikes": [100, 200],
			"num_branches": [3, 2],
			"branch_count": [3.0, 2.0],
			"total_branch_length_um": [120.0, 80.0],
			"template_density": [0.4, 0.5],
			"recon_density": [0.2, 0.3],
		}
	)


def test_build_app_returns_dash_with_expected_ids() -> None:
	app = build_app(_make_units_df(), pd.DataFrame())
	assert isinstance(app, dash.Dash)
	ids = _component_ids(app)
	expected = {
		ID_FILTER_PROJECT,
		ID_FILTER_CHIP,
		ID_FILTER_WELL,
		ID_FILTER_SCAN_TYPE,
		ID_FILTER_GENOTYPE,
		ID_FILTER_MEDIA,
		ID_FILTER_PLATING,
		ID_FILTER_DIV_RANGE,
		ID_FILTER_BOMBCELL,
		ID_FILTER_MIN_NUM_SPIKES,
		ID_FILTER_MIN_NUM_BRANCHES,
		ID_FILTER_MIN_RECON_QUALITY,
		ID_FILTER_REQUIRE_RECON_OK,
		ID_HIST_X_AXIS,
		ID_HIST_COLOR,
		ID_HISTOGRAM,
		ID_UNITS_TABLE,
	}
	missing = expected - ids
	assert not missing, f"missing component ids: {sorted(missing)}"


def test_build_app_handles_empty_units_df() -> None:
	app = build_app(pd.DataFrame(), pd.DataFrame())
	assert isinstance(app, dash.Dash)
	ids = _component_ids(app)
	# Layout still includes the filters even when no rows are present.
	assert ID_FILTER_PROJECT in ids
	assert ID_HISTOGRAM in ids
	assert ID_UNITS_TABLE in ids


def test_build_app_title_is_set() -> None:
	app = build_app(_make_units_df(), pd.DataFrame())
	assert app.title == "axon-recon dashboard"
