from __future__ import annotations

from typing import Any

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


def test_build_box_plot_points_mode_off_keeps_outliers_only() -> None:
	df = _two_group_units_df()
	fig = build_box_plot(
		df,
		value_col="branch_count",
		group_col="genotype",
		show_significance=False,
		points_mode="off",
	)
	modes = {getattr(t, "boxpoints", None) for t in fig.data}
	assert "all" not in modes


def test_build_box_plot_points_mode_jittered_side_uses_negative_pointpos() -> None:
	df = _two_group_units_df()
	fig = build_box_plot(
		df,
		value_col="branch_count",
		group_col="genotype",
		show_significance=False,
		points_mode="jittered_side",
	)
	box_traces = [t for t in fig.data if getattr(t, "type", None) == "box"]
	assert box_traces, "expected at least one box trace"
	assert any(getattr(t, "boxpoints", None) == "all" for t in box_traces)
	# `points="all"` plus negative pointpos = classic offset-to-the-left look.
	pos_vals = [getattr(t, "pointpos", None) for t in box_traces]
	assert any(pos is not None and float(pos) < 0 for pos in pos_vals)


def test_build_box_plot_points_mode_over_box_centers_points() -> None:
	df = _two_group_units_df()
	fig = build_box_plot(
		df,
		value_col="branch_count",
		group_col="genotype",
		show_significance=False,
		points_mode="over_box",
	)
	box_traces = [t for t in fig.data if getattr(t, "type", None) == "box"]
	assert box_traces
	assert all(float(getattr(t, "pointpos", 0)) == 0.0 for t in box_traces)


def test_build_box_plot_numeric_group_uses_category_axis_in_numeric_order() -> None:
	"""Brackets line up with boxes only when numeric DIV is treated as category.

	With type='category' + numeric category_orders, DIV values appear in
	numeric order rather than at their numeric positions on the axis. Empty
	groups (none in the df) don't render.
	"""
	df = pd.DataFrame(
		{
			"branch_count": [1, 2, 3, 4, 10, 11, 12, 13],
			"DIV": [22, 22, 22, 22, 6, 6, 6, 6],
		}
	)
	fig = build_box_plot(
		df,
		value_col="branch_count",
		group_col="DIV",
		show_significance=False,
		points_mode="off",
	)
	xaxis = fig.layout.xaxis
	assert xaxis.type == "category"
	# Category order is numeric ascending — 6 comes before 22 even though
	# 22 appears first in the dataframe.
	categoryarray = list(xaxis.categoryarray or [])
	assert categoryarray == [6, 22]


def test_build_box_plot_secondary_grouping_renders_side_by_side_per_primary() -> None:
	"""When color_col is set, every (primary, secondary) pair gets its own
	box trace, all sharing the same primary-axis categories."""
	df = pd.DataFrame(
		{
			"branch_count": [1, 2, 3, 4, 5, 6, 7, 8],
			"DIV": [6, 6, 22, 22, 6, 6, 22, 22],
			"plating_density": ["low", "low", "low", "low", "high", "high", "high", "high"],
		}
	)
	fig = build_box_plot(
		df,
		value_col="branch_count",
		group_col="DIV",
		color_col="plating_density",
		show_significance=False,
		points_mode="off",
	)
	box_traces = [t for t in fig.data if getattr(t, "type", None) == "box"]
	# px.box with `color=` produces one trace per unique color value.
	trace_names = {getattr(t, "name", None) for t in box_traces}
	assert trace_names == {"low", "high"}
	# Primary axis is forced categorical with numeric-sorted DIV order.
	xaxis = fig.layout.xaxis
	assert xaxis.type == "category"
	assert list(xaxis.categoryarray or []) == [6, 22]


def test_build_box_plot_secondary_grouping_sorts_numeric_categories_in_legend_order() -> None:
	"""Numeric secondary values (e.g. plating density) should be sorted
	ascending so legend / box order matches viewer expectations."""
	df = pd.DataFrame(
		{
			"branch_count": [1, 2, 3, 4],
			"DIV": [6, 6, 22, 22],
			"plating_density": [40000, 20000, 40000, 20000],
		}
	)
	fig = build_box_plot(
		df,
		value_col="branch_count",
		group_col="DIV",
		color_col="plating_density",
		show_significance=False,
		points_mode="off",
	)
	box_traces = [t for t in fig.data if getattr(t, "type", None) == "box"]
	# Trace names match the unique color values in legend order — plotly
	# follows the category_orders we set, so 20000 comes before 40000.
	trace_names = [getattr(t, "name", None) for t in box_traces]
	assert trace_names == ["20000", "40000"]


def test_build_box_plot_no_secondary_grouping_falls_back_to_single_color() -> None:
	"""Sanity: when color_col is the explicit (none) sentinel, we get a
	single box-trace-per-primary-category (no secondary)."""
	from axon_recon.dashboard.app import _BOX_COLOR_NONE

	df = pd.DataFrame({"branch_count": [1, 2, 3, 4], "DIV": [6, 6, 22, 22]})
	fig = build_box_plot(
		df,
		value_col="branch_count",
		group_col="DIV",
		color_col=_BOX_COLOR_NONE,
		show_significance=False,
		points_mode="off",
	)
	box_traces = [t for t in fig.data if getattr(t, "type", None) == "box"]
	# Single trace, no color subdivision.
	assert len(box_traces) == 1


def test_build_box_plot_exclude_nulls_drops_nan_rows_before_plotting() -> None:
	df = pd.DataFrame(
		{
			"branch_count": [1.0, float("nan"), 3.0, float("nan")],
			"genotype": ["A", "A", "B", "B"],
		}
	)
	fig_with_nulls = build_box_plot(
		df,
		value_col="branch_count",
		group_col="genotype",
		show_significance=False,
		points_mode="off",
		exclude_nulls=False,
	)
	fig_without_nulls = build_box_plot(
		df,
		value_col="branch_count",
		group_col="genotype",
		show_significance=False,
		points_mode="off",
		exclude_nulls=True,
	)
	# Both figures should render two groups (A and B each have a non-NaN value).
	# We don't get a precise row count back, but the y-array on each trace
	# loses NaN entries when exclude_nulls=True.
	def _yvals(fig: Any) -> list[float]:
		out: list[float] = []
		for trace in fig.data:
			y = getattr(trace, "y", None)
			if y is None:
				continue
			out.extend([v for v in list(y) if v is not None])
		return out
	assert any(v != v for v in _yvals(fig_with_nulls))  # NaN present
	assert all(v == v for v in _yvals(fig_without_nulls))  # all finite


def test_build_scatter_jitter_perturbs_numeric_axes() -> None:
	"""With jitter on, numeric x/y values differ from the raw df values."""
	df = _two_group_units_df()
	fig_off = _build_scatter_no_jitter(df)
	fig_on = _build_scatter_with_jitter(df)
	# `customdata` isn't set; compare raw trace x/y instead.
	# At least one numeric axis must differ between the two figures.
	off_x = [list(t.x) for t in fig_off.data if t.x is not None]
	on_x = [list(t.x) for t in fig_on.data if t.x is not None]
	# Same number of points, but values shifted.
	assert off_x and on_x
	assert any(off != on for off, on in zip(off_x, on_x))


def _build_scatter_no_jitter(df):
	from ..app import build_scatter

	return build_scatter(
		df,
		x_col="branch_count",
		y_col="total_branch_length_um",
		color_col="genotype",
		jitter=False,
	)


def _build_scatter_with_jitter(df):
	from ..app import build_scatter

	return build_scatter(
		df,
		x_col="branch_count",
		y_col="total_branch_length_um",
		color_col="genotype",
		jitter=True,
	)


def test_build_scatter_jitter_leaves_categorical_axes_untouched() -> None:
	"""Jitter only perturbs numeric columns; string-category x stays as labels."""
	from ..app import build_scatter

	df = _two_group_units_df()
	fig = build_scatter(
		df,
		x_col="genotype",  # string-categorical x
		y_col="branch_count",
		jitter=True,
	)
	# Plotly keeps the category labels intact on the x-axis trace.
	for trace in fig.data:
		if trace.x is None:
			continue
		assert set(trace.x).issubset({"WT", "KO"})


def test_categorical_columns_includes_recon_status_and_string_cols() -> None:
	"""The previous filter excluded numeric-sounding columns; now everything
	shows up so users can pick any column for color/group/facet dropdowns."""
	from ..app import _categorical_columns, _numeric_columns

	df = _two_group_units_df()
	# String-categorical columns appear first.
	cats = _categorical_columns(df)
	assert "recon_status" in cats
	assert "bombcell_label" in cats
	assert "genotype" in cats
	# Numeric columns also surface (at the end of the list, but available).
	assert "branch_count" in cats
	# Numeric helper surfaces every numeric column.
	nums = _numeric_columns(df)
	assert "branch_count" in nums
	assert "total_branch_length_um" in nums
	# Names that used to be filtered out (substring "count" / "length" /
	# "amplitude" / "density") must still be selectable.
	for col in ("branch_count", "total_branch_length_um", "template_density"):
		assert col in nums


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
