from __future__ import annotations

import dash
import numpy as np
import pandas as pd

from ..app import (
	ID_BOX_COLOR,
	ID_BOX_CORRECTION,
	ID_BOX_GROUP_COL,
	ID_BOX_PLOT,
	ID_BOX_SHOW_SIGNIFICANCE,
	ID_BOX_TEST,
	ID_BOX_VALUE_COL,
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
	ID_MAIN_TABS,
	ID_UNITS_TABLE,
	build_app,
	build_box_plot,
)


def _walk_components(tree, out):
	out.append(tree)
	children = getattr(tree, "children", None)
	if children is None:
		return
	if isinstance(children, (list, tuple)):
		for child in children:
			if child is not None and not isinstance(child, (str, int, float, bool)):
				_walk_components(child, out)
		return
	if isinstance(children, (str, int, float, bool)):
		return
	# Single non-list child (e.g. dcc.Tab(children=html.Div(...))).
	_walk_components(children, out)


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


# ---------- slice 5: box plot + significance brackets ----------


def test_build_app_exposes_box_plot_component_ids() -> None:
	app = build_app(_make_units_df(), pd.DataFrame())
	ids = _component_ids(app)
	expected = {
		ID_MAIN_TABS,
		ID_BOX_VALUE_COL,
		ID_BOX_GROUP_COL,
		ID_BOX_COLOR,
		ID_BOX_TEST,
		ID_BOX_CORRECTION,
		ID_BOX_SHOW_SIGNIFICANCE,
		ID_BOX_PLOT,
	}
	missing = expected - ids
	assert not missing, f"missing component ids: {sorted(missing)}"


def _two_group_units_df() -> pd.DataFrame:
	rng = np.random.default_rng(0)
	wt_vals = rng.normal(loc=1.0, scale=0.1, size=20)
	ko_vals = rng.normal(loc=5.0, scale=0.1, size=20)
	return pd.DataFrame(
		{
			"recon_status": ["ok"] * 40,
			"bombcell_label": ["good"] * 40,
			"genotype": ["WT"] * 20 + ["KO"] * 20,
			"branch_count": list(wt_vals) + list(ko_vals),
			"total_branch_length_um": list(wt_vals * 100) + list(ko_vals * 100),
			"template_density": list(wt_vals * 0.1) + list(ko_vals * 0.1),
			"recon_density": list(wt_vals * 0.05) + list(ko_vals * 0.05),
			"num_spikes": [100] * 40,
			"num_branches": [1] * 20 + [5] * 20,
		}
	)


def test_build_box_plot_renders_significance_for_separated_groups() -> None:
	df = _two_group_units_df()
	fig = build_box_plot(
		df,
		value_col="branch_count",
		group_col="genotype",
		test="mann_whitney",
		correction="none",
		show_significance=True,
	)
	# Box plot trace exists + a significance bracket shape was added.
	assert any(getattr(trace, "type", None) == "box" for trace in fig.data)
	assert len(fig.layout.shapes) >= 1
	assert len(fig.layout.annotations) >= 1
	assert fig.layout.annotations[0].text == "***"


def test_build_box_plot_handles_empty_df_without_raising() -> None:
	fig = build_box_plot(pd.DataFrame(), value_col="branch_count", group_col="genotype", show_significance=True)
	# Empty placeholder figure, no traces with real data.
	assert fig is not None


def test_build_box_plot_kruskal_wallis_adds_omnibus_annotation() -> None:
	df = _two_group_units_df()
	fig = build_box_plot(
		df,
		value_col="branch_count",
		group_col="genotype",
		test="kruskal_wallis",
		correction="none",
		show_significance=True,
	)
	# Omnibus mode annotates the figure with the K-W p-value summary.
	annotation_texts = [ann.text for ann in fig.layout.annotations]
	assert any("Kruskal-Wallis" in str(text) for text in annotation_texts)


def test_build_box_plot_show_significance_off_drops_brackets() -> None:
	df = _two_group_units_df()
	fig = build_box_plot(
		df,
		value_col="branch_count",
		group_col="genotype",
		test="mann_whitney",
		correction="none",
		show_significance=False,
	)
	assert len(fig.layout.shapes) == 0
	assert len(fig.layout.annotations) == 0


def test_build_box_plot_missing_group_column_returns_empty_figure() -> None:
	df = pd.DataFrame({"branch_count": [1.0, 2.0, 3.0]})
	fig = build_box_plot(df, value_col="branch_count", group_col="missing", show_significance=True)
	# Falls back to an empty box figure rather than raising.
	assert fig is not None


# ---------- slice 6: scatter + facet + download endpoints ----------


def test_build_app_exposes_scatter_component_ids() -> None:
	from ..app import (
		ID_DOWNLOAD_BOX_PDF,
		ID_DOWNLOAD_BOX_PNG,
		ID_DOWNLOAD_BOX_SVG,
		ID_DOWNLOAD_CSV,
		ID_DOWNLOAD_HIST_PDF,
		ID_DOWNLOAD_HIST_PNG,
		ID_DOWNLOAD_HIST_SVG,
		ID_DOWNLOAD_SCATTER_PDF,
		ID_DOWNLOAD_SCATTER_PNG,
		ID_DOWNLOAD_SCATTER_SVG,
		ID_DOWNLOAD_SPEC_JSON,
		ID_SCATTER_COLOR,
		ID_SCATTER_FACET_COL,
		ID_SCATTER_FACET_ROW,
		ID_SCATTER_PLOT,
		ID_SCATTER_X,
		ID_SCATTER_Y,
	)

	app = build_app(_two_group_units_df(), pd.DataFrame())
	ids = _component_ids(app)
	expected = {
		ID_SCATTER_X,
		ID_SCATTER_Y,
		ID_SCATTER_COLOR,
		ID_SCATTER_FACET_COL,
		ID_SCATTER_FACET_ROW,
		ID_SCATTER_PLOT,
		ID_DOWNLOAD_HIST_PNG,
		ID_DOWNLOAD_HIST_SVG,
		ID_DOWNLOAD_HIST_PDF,
		ID_DOWNLOAD_BOX_PNG,
		ID_DOWNLOAD_BOX_SVG,
		ID_DOWNLOAD_BOX_PDF,
		ID_DOWNLOAD_SCATTER_PNG,
		ID_DOWNLOAD_SCATTER_SVG,
		ID_DOWNLOAD_SCATTER_PDF,
		ID_DOWNLOAD_CSV,
		ID_DOWNLOAD_SPEC_JSON,
	}
	missing = expected - ids
	assert not missing, f"missing component ids: {sorted(missing)}"


def test_build_scatter_renders_with_color_and_facet_col() -> None:
	from ..app import build_scatter

	df = _two_group_units_df()
	df = df.copy()
	df["DIV"] = [12, 18] * (len(df) // 2)
	fig = build_scatter(
		df,
		x_col="branch_count",
		y_col="total_branch_length_um",
		color_col="genotype",
		facet_col="DIV",
	)
	# Color split + DIV facets → multiple traces (one per genotype × DIV cell).
	assert len(fig.data) >= 2


def test_build_scatter_empty_df_returns_empty_figure() -> None:
	from ..app import build_scatter

	fig = build_scatter(pd.DataFrame(), x_col="x", y_col="y")
	assert fig is not None


def test_build_scatter_missing_columns_returns_empty_figure() -> None:
	from ..app import build_scatter

	fig = build_scatter(pd.DataFrame({"x": [1, 2]}), x_col="x", y_col="missing")
	assert fig is not None


def test_image_exports_produce_non_zero_content_for_each_format() -> None:
	"""Direct kaleido export shapes used by the download endpoints."""
	from ..app import _IMAGE_MIME_BY_FORMAT, build_scatter

	df = _two_group_units_df()
	fig = build_scatter(df, x_col="branch_count", y_col="total_branch_length_um")
	for fmt, expected_mime in _IMAGE_MIME_BY_FORMAT.items():
		blob = fig.to_image(format=fmt)
		assert isinstance(blob, (bytes, bytearray))
		assert len(blob) > 100, f"{fmt} export was suspiciously small: {len(blob)} bytes"
		if fmt == "svg":
			assert b"<svg" in blob
		if fmt == "pdf":
			assert blob.startswith(b"%PDF")
		if fmt == "png":
			assert blob.startswith(b"\x89PNG")
		assert expected_mime in {"image/png", "image/svg+xml", "application/pdf"}
