"""Dash app builder for the slice-4/slice-5 dashboard.

Layout:
  - Left rail with modular inclusion/exclusion filters (plan §4).
  - Main pane tabs:
      - Histogram (axis + color dropdowns) — slice 4
      - Box plot with optional significance brackets — slice 5
      - dash_ag_grid table view of the filtered units table.

The build is kept testable: `build_app(units_df, well_summary_df)` returns a
`dash.Dash` whose `layout` is fully assembled and whose callbacks operate on
the bound data via closures. No NAS access, no global state.
"""

from __future__ import annotations

from typing import Any

import dash
import dash_ag_grid as dag
import pandas as pd
import plotly.express as px
from dash import Input, Output, dcc, html

from . import filters as filter_helpers
from . import significance as significance_helpers


# Component IDs (also used by tests).
ID_FILTER_PROJECT = "filter-project"
ID_FILTER_CHIP = "filter-chip"
ID_FILTER_WELL = "filter-well"
ID_FILTER_SCAN_TYPE = "filter-scan-type"
ID_FILTER_GENOTYPE = "filter-genotype"
ID_FILTER_MEDIA = "filter-media"
ID_FILTER_PLATING = "filter-plating"
ID_FILTER_DIV_RANGE = "filter-div-range"
ID_FILTER_BOMBCELL = "filter-bombcell"
ID_FILTER_MIN_NUM_SPIKES = "filter-min-num-spikes"
ID_FILTER_MIN_NUM_BRANCHES = "filter-min-num-branches"
ID_FILTER_MIN_RECON_QUALITY = "filter-min-recon-quality"
ID_FILTER_REQUIRE_RECON_OK = "filter-require-recon-ok"
ID_HIST_X_AXIS = "histogram-x-axis"
ID_HIST_COLOR = "histogram-color"
ID_HISTOGRAM = "histogram-graph"
ID_UNITS_TABLE = "units-table"
ID_MAIN_TABS = "main-tabs"
ID_BOX_VALUE_COL = "box-value-col"
ID_BOX_GROUP_COL = "box-group-col"
ID_BOX_COLOR = "box-color"
ID_BOX_TEST = "box-test"
ID_BOX_CORRECTION = "box-correction"
ID_BOX_SHOW_SIGNIFICANCE = "box-show-significance"
ID_BOX_PLOT = "box-plot-graph"

_DEFAULT_BOMBCELL_ALLOWLIST: tuple[str | None, ...] = ("good", "non_soma_good")
_HISTOGRAM_NUMERIC_DEFAULT = "branch_count"
_HISTOGRAM_COLOR_DEFAULT = "(none)"
_BOX_COLOR_NONE = "(none)"
_BOX_TEST_DEFAULT = "mann_whitney"
_BOX_CORRECTION_DEFAULT = "bh"


def _unique_sorted(series: pd.Series) -> list[Any]:
	values = [v for v in series.dropna().unique().tolist() if v is not None]
	try:
		return sorted(values)
	except TypeError:
		return sorted(values, key=str)


def _numeric_columns(df: pd.DataFrame) -> list[str]:
	if df is None or df.empty:
		return [_HISTOGRAM_NUMERIC_DEFAULT]
	cols = [str(c) for c in df.select_dtypes(include="number").columns.tolist()]
	return cols or [_HISTOGRAM_NUMERIC_DEFAULT]


def _categorical_columns(df: pd.DataFrame) -> list[str]:
	if df is None or df.empty:
		return []
	exclude_substr = ("location", "amplitude", "ptp", "delay", "density", "count", "length")
	cols: list[str] = []
	for col in df.columns:
		if df[col].dtype.kind in ("O", "b"):
			# Skip identity-like or list-y obvious blobs.
			lower = str(col).lower()
			if not any(token in lower for token in exclude_substr):
				cols.append(str(col))
	return cols


def _multiselect_options(series: pd.Series) -> list[dict[str, Any]]:
	return [{"label": str(value), "value": value} for value in _unique_sorted(series)]


def _bombcell_options(series: pd.Series) -> list[dict[str, Any]]:
	labels = _unique_sorted(series)
	options = [{"label": str(value), "value": value} for value in labels]
	# Always offer an explicit None choice so missing-bombcell wells can be included.
	options.append({"label": "<unlabeled>", "value": "__null__"})
	return options


def _decode_bombcell_selection(values: list[Any] | None) -> list[Any] | None:
	if values is None:
		return None
	out: list[Any] = []
	for value in values:
		out.append(None if value == "__null__" else value)
	return out


def _div_range_extents(df: pd.DataFrame) -> tuple[float, float]:
	if df is None or df.empty or "DIV" not in df.columns:
		return (0.0, 1.0)
	series = pd.to_numeric(df["DIV"], errors="coerce").dropna()
	if series.empty:
		return (0.0, 1.0)
	lo = float(series.min())
	hi = float(series.max())
	if lo == hi:
		hi = lo + 1.0
	return (lo, hi)


def _build_layout(units_df: pd.DataFrame, well_summary_df: pd.DataFrame) -> html.Div:
	div_lo, div_hi = _div_range_extents(units_df)
	hist_axis_options = [{"label": c, "value": c} for c in _numeric_columns(units_df)]
	color_options = [{"label": _HISTOGRAM_COLOR_DEFAULT, "value": _HISTOGRAM_COLOR_DEFAULT}] + [
		{"label": c, "value": c} for c in _categorical_columns(units_df)
	]
	column_defs = [{"field": c, "headerName": c} for c in (units_df.columns if not units_df.empty else [])]

	left_rail = html.Div(
		[
			html.H3("Filters"),
			html.Label("Project"),
			dcc.Dropdown(
				id=ID_FILTER_PROJECT,
				options=_multiselect_options(units_df.get("project", pd.Series(dtype=object))),
				multi=True,
			),
			html.Label("Chip"),
			dcc.Dropdown(
				id=ID_FILTER_CHIP,
				options=_multiselect_options(units_df.get("chip_id", pd.Series(dtype=object))),
				multi=True,
			),
			html.Label("Well"),
			dcc.Dropdown(
				id=ID_FILTER_WELL,
				options=_multiselect_options(units_df.get("well_id", pd.Series(dtype=object))),
				multi=True,
			),
			html.Label("Scan type"),
			dcc.Dropdown(
				id=ID_FILTER_SCAN_TYPE,
				options=_multiselect_options(units_df.get("scan_type", pd.Series(dtype=object))),
				multi=True,
			),
			html.Label("Genotype"),
			dcc.Dropdown(
				id=ID_FILTER_GENOTYPE,
				options=_multiselect_options(units_df.get("genotype", pd.Series(dtype=object))),
				multi=True,
			),
			html.Label("Media"),
			dcc.Dropdown(
				id=ID_FILTER_MEDIA,
				options=_multiselect_options(units_df.get("media", pd.Series(dtype=object))),
				multi=True,
			),
			html.Label("Plating density"),
			dcc.Dropdown(
				id=ID_FILTER_PLATING,
				options=_multiselect_options(units_df.get("plating_density", pd.Series(dtype=object))),
				multi=True,
			),
			html.Label("DIV range"),
			dcc.RangeSlider(
				id=ID_FILTER_DIV_RANGE,
				min=div_lo,
				max=div_hi,
				value=[div_lo, div_hi],
				marks=None,
				tooltip={"placement": "bottom", "always_visible": False},
				step=1,
			),
			html.Label("Bombcell label allowlist"),
			dcc.Dropdown(
				id=ID_FILTER_BOMBCELL,
				options=_bombcell_options(units_df.get("bombcell_label", pd.Series(dtype=object))),
				value=[v for v in _DEFAULT_BOMBCELL_ALLOWLIST if v in (units_df.get("bombcell_label", pd.Series()).tolist() if not units_df.empty else [])],
				multi=True,
			),
			html.Label("Min num_spikes"),
			dcc.Input(id=ID_FILTER_MIN_NUM_SPIKES, type="number", value=0, step=1),
			html.Label("Min num_branches"),
			dcc.Input(id=ID_FILTER_MIN_NUM_BRANCHES, type="number", value=0, step=1),
			html.Label("Min recon_quality_score"),
			dcc.Input(id=ID_FILTER_MIN_RECON_QUALITY, type="number", value=None, placeholder="(no filter)"),
			dcc.Checklist(
				id=ID_FILTER_REQUIRE_RECON_OK,
				options=[{"label": "Require recon_status='ok'", "value": "on"}],
				value=["on"],
			),
		],
		style={
			"width": "280px",
			"padding": "1rem",
			"boxSizing": "border-box",
			"borderRight": "1px solid #ddd",
			"overflowY": "auto",
			"height": "100vh",
		},
	)

	cat_cols = _categorical_columns(units_df)
	box_color_options = [{"label": _BOX_COLOR_NONE, "value": _BOX_COLOR_NONE}] + [
		{"label": c, "value": c} for c in cat_cols
	]
	box_group_options = [{"label": c, "value": c} for c in cat_cols]
	default_group = "genotype" if "genotype" in cat_cols else (cat_cols[0] if cat_cols else None)

	histogram_tab = dcc.Tab(
		label="Histogram",
		value="histogram",
		children=html.Div(
			[
				html.Div(
					[
						html.Label("X axis (numeric column)"),
						dcc.Dropdown(
							id=ID_HIST_X_AXIS,
							options=hist_axis_options,
							value=hist_axis_options[0]["value"] if hist_axis_options else None,
							clearable=False,
						),
						html.Label("Color by (categorical)"),
						dcc.Dropdown(
							id=ID_HIST_COLOR,
							options=color_options,
							value=_HISTOGRAM_COLOR_DEFAULT,
							clearable=False,
						),
					],
					style={"display": "flex", "gap": "1rem", "flexWrap": "wrap"},
				),
				dcc.Graph(id=ID_HISTOGRAM),
			],
			style={"paddingTop": "0.5rem"},
		),
	)

	box_tab = dcc.Tab(
		label="Box plot",
		value="box",
		children=html.Div(
			[
				html.Div(
					[
						html.Label("Value (numeric column)"),
						dcc.Dropdown(
							id=ID_BOX_VALUE_COL,
							options=hist_axis_options,
							value=hist_axis_options[0]["value"] if hist_axis_options else None,
							clearable=False,
						),
						html.Label("Group (categorical column)"),
						dcc.Dropdown(
							id=ID_BOX_GROUP_COL,
							options=box_group_options,
							value=default_group,
							clearable=False,
						),
						html.Label("Color (categorical or none)"),
						dcc.Dropdown(
							id=ID_BOX_COLOR,
							options=box_color_options,
							value=_BOX_COLOR_NONE,
							clearable=False,
						),
					],
					style={"display": "flex", "gap": "1rem", "flexWrap": "wrap"},
				),
				html.Div(
					[
						html.Label("Test"),
						dcc.RadioItems(
							id=ID_BOX_TEST,
							options=[
								{"label": "Mann-Whitney U", "value": "mann_whitney"},
								{"label": "Welch's t-test", "value": "welch_t"},
								{"label": "Tukey HSD", "value": "tukey_hsd"},
								{"label": "Kruskal-Wallis (omnibus)", "value": "kruskal_wallis"},
							],
							value=_BOX_TEST_DEFAULT,
							inline=True,
						),
						html.Label("Multiple-comparison correction"),
						dcc.RadioItems(
							id=ID_BOX_CORRECTION,
							options=[
								{"label": "None", "value": "none"},
								{"label": "Bonferroni", "value": "bonferroni"},
								{"label": "Holm", "value": "holm"},
								{"label": "Benjamini-Hochberg (FDR)", "value": "bh"},
							],
							value=_BOX_CORRECTION_DEFAULT,
							inline=True,
						),
						dcc.Checklist(
							id=ID_BOX_SHOW_SIGNIFICANCE,
							options=[{"label": "Show significance brackets", "value": "on"}],
							value=["on"],
						),
					],
					style={"display": "flex", "flexDirection": "column", "gap": "0.5rem", "marginTop": "0.5rem"},
				),
				dcc.Graph(id=ID_BOX_PLOT),
			],
			style={"paddingTop": "0.5rem"},
		),
	)

	main_pane = html.Div(
		[
			dcc.Tabs(
				id=ID_MAIN_TABS,
				value="histogram",
				children=[histogram_tab, box_tab],
			),
			html.H3("Filtered units"),
			dag.AgGrid(
				id=ID_UNITS_TABLE,
				columnDefs=column_defs,
				rowData=[],
				defaultColDef={"sortable": True, "filter": True, "resizable": True, "minWidth": 90},
				style={"height": "55vh"},
			),
		],
		style={"flex": "1 1 auto", "padding": "1rem", "overflow": "auto", "height": "100vh"},
	)

	return html.Div(
		[left_rail, main_pane],
		style={"display": "flex", "flexDirection": "row", "height": "100vh", "fontFamily": "sans-serif"},
	)


def _build_filter_spec_from_state(
	*,
	require_recon_ok: list[str] | None,
	bombcell: list[Any] | None,
	min_num_spikes: Any,
	min_num_branches: Any,
	min_recon_quality: Any,
	project: list[Any] | None,
	chip: list[Any] | None,
	well: list[Any] | None,
	scan_type: list[Any] | None,
	genotype: list[Any] | None,
	media: list[Any] | None,
	plating: list[Any] | None,
	div_range: list[float] | None,
) -> dict[str, Any]:
	return {
		"require_recon_ok": bool(require_recon_ok and "on" in require_recon_ok),
		"bombcell_allowlist": _decode_bombcell_selection(bombcell),
		"min_num_spikes": min_num_spikes,
		"min_num_branches": min_num_branches,
		"min_recon_quality_score": min_recon_quality,
		"project": project,
		"chip_id": chip,
		"well_id": well,
		"scan_type": scan_type,
		"genotype": genotype,
		"media": media,
		"plating_density": plating,
		"div_lo": div_range[0] if div_range else None,
		"div_hi": div_range[1] if div_range else None,
	}


def build_app(units_df: pd.DataFrame, well_summary_df: pd.DataFrame) -> dash.Dash:
	"""Build a Dash app over the loaded units + well_summary tables.

	The data is captured in callback closures; the app contains no global
	mutable state, so multiple instances can coexist (e.g., in tests).
	"""
	app = dash.Dash(__name__, suppress_callback_exceptions=True)
	app.title = "axon-recon dashboard"
	app.layout = _build_layout(units_df, well_summary_df)

	@app.callback(
		Output(ID_HISTOGRAM, "figure"),
		Output(ID_UNITS_TABLE, "rowData"),
		Output(ID_UNITS_TABLE, "columnDefs"),
		Input(ID_FILTER_REQUIRE_RECON_OK, "value"),
		Input(ID_FILTER_BOMBCELL, "value"),
		Input(ID_FILTER_MIN_NUM_SPIKES, "value"),
		Input(ID_FILTER_MIN_NUM_BRANCHES, "value"),
		Input(ID_FILTER_MIN_RECON_QUALITY, "value"),
		Input(ID_FILTER_PROJECT, "value"),
		Input(ID_FILTER_CHIP, "value"),
		Input(ID_FILTER_WELL, "value"),
		Input(ID_FILTER_SCAN_TYPE, "value"),
		Input(ID_FILTER_GENOTYPE, "value"),
		Input(ID_FILTER_MEDIA, "value"),
		Input(ID_FILTER_PLATING, "value"),
		Input(ID_FILTER_DIV_RANGE, "value"),
		Input(ID_HIST_X_AXIS, "value"),
		Input(ID_HIST_COLOR, "value"),
	)
	def _update(
		require_recon_ok,
		bombcell,
		min_num_spikes,
		min_num_branches,
		min_recon_quality,
		project,
		chip,
		well,
		scan_type,
		genotype,
		media,
		plating,
		div_range,
		hist_x,
		hist_color,
	):
		spec = _build_filter_spec_from_state(
			require_recon_ok=require_recon_ok,
			bombcell=bombcell,
			min_num_spikes=min_num_spikes,
			min_num_branches=min_num_branches,
			min_recon_quality=min_recon_quality,
			project=project,
			chip=chip,
			well=well,
			scan_type=scan_type,
			genotype=genotype,
			media=media,
			plating=plating,
			div_range=div_range,
		)
		filtered = filter_helpers.apply_filter_spec(units_df, spec)
		fig = _build_histogram(filtered, x_column=hist_x, color_column=hist_color)
		column_defs = [{"field": c, "headerName": c} for c in filtered.columns]
		# AgGrid rowData must be records (list[dict]).
		row_data = filtered.to_dict("records") if not filtered.empty else []
		return fig, row_data, column_defs

	@app.callback(
		Output(ID_BOX_PLOT, "figure"),
		Input(ID_FILTER_REQUIRE_RECON_OK, "value"),
		Input(ID_FILTER_BOMBCELL, "value"),
		Input(ID_FILTER_MIN_NUM_SPIKES, "value"),
		Input(ID_FILTER_MIN_NUM_BRANCHES, "value"),
		Input(ID_FILTER_MIN_RECON_QUALITY, "value"),
		Input(ID_FILTER_PROJECT, "value"),
		Input(ID_FILTER_CHIP, "value"),
		Input(ID_FILTER_WELL, "value"),
		Input(ID_FILTER_SCAN_TYPE, "value"),
		Input(ID_FILTER_GENOTYPE, "value"),
		Input(ID_FILTER_MEDIA, "value"),
		Input(ID_FILTER_PLATING, "value"),
		Input(ID_FILTER_DIV_RANGE, "value"),
		Input(ID_BOX_VALUE_COL, "value"),
		Input(ID_BOX_GROUP_COL, "value"),
		Input(ID_BOX_COLOR, "value"),
		Input(ID_BOX_TEST, "value"),
		Input(ID_BOX_CORRECTION, "value"),
		Input(ID_BOX_SHOW_SIGNIFICANCE, "value"),
	)
	def _update_box(
		require_recon_ok,
		bombcell,
		min_num_spikes,
		min_num_branches,
		min_recon_quality,
		project,
		chip,
		well,
		scan_type,
		genotype,
		media,
		plating,
		div_range,
		box_value_col,
		box_group_col,
		box_color,
		box_test,
		box_correction,
		show_significance,
	):
		spec = _build_filter_spec_from_state(
			require_recon_ok=require_recon_ok,
			bombcell=bombcell,
			min_num_spikes=min_num_spikes,
			min_num_branches=min_num_branches,
			min_recon_quality=min_recon_quality,
			project=project,
			chip=chip,
			well=well,
			scan_type=scan_type,
			genotype=genotype,
			media=media,
			plating=plating,
			div_range=div_range,
		)
		filtered = filter_helpers.apply_filter_spec(units_df, spec)
		return build_box_plot(
			filtered,
			value_col=box_value_col,
			group_col=box_group_col,
			color_col=box_color,
			test=box_test,
			correction=box_correction,
			show_significance=bool(show_significance and "on" in show_significance),
		)

	return app


def _build_histogram(df: pd.DataFrame, *, x_column: Any, color_column: Any) -> Any:
	"""Compose a Plotly histogram figure; empty df yields an empty figure."""
	if df is None or df.empty or not x_column:
		return px.histogram(pd.DataFrame({"_": []}), x="_")
	x = str(x_column)
	if x not in df.columns:
		return px.histogram(pd.DataFrame({"_": []}), x="_")
	color = None
	if color_column and color_column != _HISTOGRAM_COLOR_DEFAULT and str(color_column) in df.columns:
		color = str(color_column)
	return px.histogram(df, x=x, color=color, barmode="overlay" if color else "relative")


def build_box_plot(
	df: pd.DataFrame,
	*,
	value_col: Any,
	group_col: Any,
	color_col: Any = _BOX_COLOR_NONE,
	test: Any = _BOX_TEST_DEFAULT,
	correction: Any = _BOX_CORRECTION_DEFAULT,
	show_significance: bool = True,
) -> Any:
	"""Build a Plotly box-plot figure with optional significance brackets.

	Empty / missing-column inputs return an empty Plotly figure (rather than
	raising) so the Dash callback can render something on every fired update.
	"""
	if df is None or df.empty or not value_col or not group_col:
		return px.box(pd.DataFrame({"_": []}), y="_")
	value = str(value_col)
	group = str(group_col)
	if value not in df.columns or group not in df.columns:
		return px.box(pd.DataFrame({"_": []}), y="_")
	color = None
	if color_col and color_col != _BOX_COLOR_NONE and str(color_col) in df.columns:
		color = str(color_col)
	fig = px.box(df, x=group, y=value, color=color, points="outliers")
	if not show_significance:
		return fig
	test_key = str(test or _BOX_TEST_DEFAULT).strip().lower()
	if test_key == "kruskal_wallis":
		p_value = significance_helpers.kruskal_wallis_omnibus(df, group_col=group, value_col=value)
		if p_value is not None:
			stars = significance_helpers.asterisks_for_p(p_value)
			label = stars if stars else "n.s."
			fig.add_annotation(
				xref="paper",
				yref="paper",
				x=0.99,
				y=1.04,
				xanchor="right",
				yanchor="bottom",
				text=f"Kruskal-Wallis p={p_value:.3g} ({label})",
				showarrow=False,
				font={"size": 11, "color": "#333"},
			)
		return fig
	raw_pvalues = significance_helpers.compute_pairwise_pvalues(
		df, group_col=group, value_col=value, test=test_key
	)
	corrected = significance_helpers.apply_correction(
		raw_pvalues, method=str(correction or _BOX_CORRECTION_DEFAULT)
	)
	return significance_helpers.significance_brackets(fig, corrected)
