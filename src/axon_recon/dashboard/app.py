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
from dash import Input, Output, State, dcc, html

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
ID_FILTER_TREATMENT = "filter-treatment"
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
ID_BOX_SHOW_POINTS = "box-show-points"
ID_BOX_PLOT = "box-plot-graph"
ID_SCATTER_X = "scatter-x"
ID_SCATTER_Y = "scatter-y"
ID_SCATTER_COLOR = "scatter-color"
ID_SCATTER_FACET_COL = "scatter-facet-col"
ID_SCATTER_FACET_ROW = "scatter-facet-row"
ID_SCATTER_JITTER = "scatter-jitter"
ID_SCATTER_PLOT = "scatter-plot-graph"

# Download component ids — each plot has its own group + a shared CSV / spec.
ID_DOWNLOAD_HIST_PNG = "download-hist-png"
ID_DOWNLOAD_HIST_SVG = "download-hist-svg"
ID_DOWNLOAD_HIST_PDF = "download-hist-pdf"
ID_DOWNLOAD_BOX_PNG = "download-box-png"
ID_DOWNLOAD_BOX_SVG = "download-box-svg"
ID_DOWNLOAD_BOX_PDF = "download-box-pdf"
ID_DOWNLOAD_SCATTER_PNG = "download-scatter-png"
ID_DOWNLOAD_SCATTER_SVG = "download-scatter-svg"
ID_DOWNLOAD_SCATTER_PDF = "download-scatter-pdf"
ID_DOWNLOAD_CSV = "download-csv"
ID_DOWNLOAD_SPEC_JSON = "download-spec-json"
ID_DOWNLOAD_HIST_PNG_TARGET = "download-hist-png-target"
ID_DOWNLOAD_HIST_SVG_TARGET = "download-hist-svg-target"
ID_DOWNLOAD_HIST_PDF_TARGET = "download-hist-pdf-target"
ID_DOWNLOAD_BOX_PNG_TARGET = "download-box-png-target"
ID_DOWNLOAD_BOX_SVG_TARGET = "download-box-svg-target"
ID_DOWNLOAD_BOX_PDF_TARGET = "download-box-pdf-target"
ID_DOWNLOAD_SCATTER_PNG_TARGET = "download-scatter-png-target"
ID_DOWNLOAD_SCATTER_SVG_TARGET = "download-scatter-svg-target"
ID_DOWNLOAD_SCATTER_PDF_TARGET = "download-scatter-pdf-target"
ID_DOWNLOAD_CSV_TARGET = "download-csv-target"
ID_DOWNLOAD_SPEC_TARGET = "download-spec-target"

_DEFAULT_BOMBCELL_ALLOWLIST: tuple[str | None, ...] = ("good", "non_soma_good")
_HISTOGRAM_NUMERIC_DEFAULT = "branch_count"
_HISTOGRAM_COLOR_DEFAULT = "(none)"
_BOX_COLOR_NONE = "(none)"
_BOX_TEST_DEFAULT = "mann_whitney"
_BOX_CORRECTION_DEFAULT = "bh"
_FACET_NONE = "(none)"
_SCATTER_COLOR_NONE = "(none)"

_IMAGE_MIME_BY_FORMAT: dict[str, str] = {
	"png": "image/png",
	"svg": "image/svg+xml",
	"pdf": "application/pdf",
}


def _unique_sorted(series: pd.Series) -> list[Any]:
	values = [v for v in series.dropna().unique().tolist() if v is not None]
	try:
		return sorted(values)
	except TypeError:
		return sorted(values, key=str)


def _all_columns(df: pd.DataFrame) -> list[str]:
	"""Return every column on `df`, preserving on-disk order."""
	if df is None or df.empty:
		return [_HISTOGRAM_NUMERIC_DEFAULT]
	return [str(c) for c in df.columns.tolist()]


def _numeric_columns(df: pd.DataFrame) -> list[str]:
	"""Return numeric columns first, then everything else.

	Plotly's `px.histogram` / `px.box` / `px.scatter` all accept categorical
	inputs (just rendered differently), so dropdowns expose every column.
	Numeric ones are surfaced first so the default selection still lands on
	a sensible axis.
	"""
	if df is None or df.empty:
		return [_HISTOGRAM_NUMERIC_DEFAULT]
	numeric = [str(c) for c in df.select_dtypes(include="number").columns.tolist()]
	numeric_set = set(numeric)
	other = [str(c) for c in df.columns.tolist() if str(c) not in numeric_set]
	return numeric + other if numeric or other else [_HISTOGRAM_NUMERIC_DEFAULT]


def _categorical_columns(df: pd.DataFrame) -> list[str]:
	"""Return non-numeric columns first, then numerics.

	The dashboard's dropdowns for color / group / facet accept any column
	(plotly maps numerics to a continuous color scale or bins for facets);
	non-numeric columns are surfaced first so the default lands on something
	categorical when one exists.
	"""
	if df is None or df.empty:
		return []
	non_numeric = [
		str(col)
		for col in df.columns
		if df[col].dtype.kind not in ("i", "u", "f")
	]
	numeric = [
		str(col)
		for col in df.columns
		if df[col].dtype.kind in ("i", "u", "f")
	]
	return non_numeric + numeric


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


def _download_button_row(
	*,
	prefix: str,
	button_ids: tuple[str, str, str],
	target_ids: tuple[str, str, str],
) -> html.Div:
	"""Render a small PNG/SVG/PDF download button group + dcc.Download targets."""
	png_btn, svg_btn, pdf_btn = button_ids
	png_target, svg_target, pdf_target = target_ids
	return html.Div(
		[
			html.Button(f"{prefix} PNG", id=png_btn, n_clicks=0),
			html.Button(f"{prefix} SVG", id=svg_btn, n_clicks=0),
			html.Button(f"{prefix} PDF", id=pdf_btn, n_clicks=0),
			dcc.Download(id=png_target),
			dcc.Download(id=svg_target),
			dcc.Download(id=pdf_target),
		],
		style={"display": "flex", "gap": "0.5rem", "marginTop": "0.5rem"},
	)


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
			html.Label("Treatment"),
			dcc.Dropdown(
				id=ID_FILTER_TREATMENT,
				options=_multiselect_options(units_df.get("treatment", pd.Series(dtype=object))),
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
	# Scatter uses numeric for x/y; color and facets can include DIV as well.
	numeric_options = hist_axis_options
	scatter_color_options = [{"label": _SCATTER_COLOR_NONE, "value": _SCATTER_COLOR_NONE}] + [
		{"label": c, "value": c} for c in cat_cols
	]
	facet_options = [{"label": _FACET_NONE, "value": _FACET_NONE}] + [
		{"label": c, "value": c} for c in cat_cols + (["DIV"] if "DIV" in units_df.columns else [])
	]
	default_scatter_x = "branch_count" if "branch_count" in units_df.columns else (
		numeric_options[0]["value"] if numeric_options else None
	)
	default_scatter_y = "total_branch_length_um" if "total_branch_length_um" in units_df.columns else default_scatter_x

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
				_download_button_row(
					prefix="Histogram",
					button_ids=(ID_DOWNLOAD_HIST_PNG, ID_DOWNLOAD_HIST_SVG, ID_DOWNLOAD_HIST_PDF),
					target_ids=(ID_DOWNLOAD_HIST_PNG_TARGET, ID_DOWNLOAD_HIST_SVG_TARGET, ID_DOWNLOAD_HIST_PDF_TARGET),
				),
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
						dcc.Checklist(
							id=ID_BOX_SHOW_POINTS,
							options=[{"label": "Overlay individual points (jittered)", "value": "on"}],
							value=[],
						),
					],
					style={"display": "flex", "flexDirection": "column", "gap": "0.5rem", "marginTop": "0.5rem"},
				),
				dcc.Graph(id=ID_BOX_PLOT),
				_download_button_row(
					prefix="Box plot",
					button_ids=(ID_DOWNLOAD_BOX_PNG, ID_DOWNLOAD_BOX_SVG, ID_DOWNLOAD_BOX_PDF),
					target_ids=(ID_DOWNLOAD_BOX_PNG_TARGET, ID_DOWNLOAD_BOX_SVG_TARGET, ID_DOWNLOAD_BOX_PDF_TARGET),
				),
			],
			style={"paddingTop": "0.5rem"},
		),
	)

	scatter_tab = dcc.Tab(
		label="Scatter",
		value="scatter",
		children=html.Div(
			[
				html.Div(
					[
						html.Label("X"),
						dcc.Dropdown(
							id=ID_SCATTER_X,
							options=numeric_options,
							value=default_scatter_x,
							clearable=False,
						),
						html.Label("Y"),
						dcc.Dropdown(
							id=ID_SCATTER_Y,
							options=numeric_options,
							value=default_scatter_y,
							clearable=False,
						),
						html.Label("Color"),
						dcc.Dropdown(
							id=ID_SCATTER_COLOR,
							options=scatter_color_options,
							value=_SCATTER_COLOR_NONE,
							clearable=False,
						),
						html.Label("Facet column"),
						dcc.Dropdown(
							id=ID_SCATTER_FACET_COL,
							options=facet_options,
							value=_FACET_NONE,
							clearable=False,
						),
						html.Label("Facet row"),
						dcc.Dropdown(
							id=ID_SCATTER_FACET_ROW,
							options=facet_options,
							value=_FACET_NONE,
							clearable=False,
						),
					],
					style={"display": "flex", "gap": "1rem", "flexWrap": "wrap"},
				),
				dcc.Checklist(
					id=ID_SCATTER_JITTER,
					options=[{"label": "Jitter points (adds small Gaussian noise to numeric axes)", "value": "on"}],
					value=[],
					style={"marginTop": "0.5rem"},
				),
				dcc.Graph(id=ID_SCATTER_PLOT),
				_download_button_row(
					prefix="Scatter",
					button_ids=(ID_DOWNLOAD_SCATTER_PNG, ID_DOWNLOAD_SCATTER_SVG, ID_DOWNLOAD_SCATTER_PDF),
					target_ids=(ID_DOWNLOAD_SCATTER_PNG_TARGET, ID_DOWNLOAD_SCATTER_SVG_TARGET, ID_DOWNLOAD_SCATTER_PDF_TARGET),
				),
			],
			style={"paddingTop": "0.5rem"},
		),
	)

	main_pane = html.Div(
		[
			dcc.Tabs(
				id=ID_MAIN_TABS,
				value="histogram",
				children=[histogram_tab, box_tab, scatter_tab],
			),
			html.Div(
				[
					html.H3("Filtered units"),
					html.Div(
						[
							html.Button("Download filtered CSV", id=ID_DOWNLOAD_CSV, n_clicks=0),
							dcc.Download(id=ID_DOWNLOAD_CSV_TARGET),
							html.Button("Download filter+plot spec (JSON)", id=ID_DOWNLOAD_SPEC_JSON, n_clicks=0),
							dcc.Download(id=ID_DOWNLOAD_SPEC_TARGET),
						],
						style={"display": "flex", "gap": "0.5rem", "marginBottom": "0.5rem"},
					),
				]
			),
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
	treatment: list[Any] | None,
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
		"treatment": treatment,
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
		Input(ID_FILTER_TREATMENT, "value"),
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
		treatment,
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
			treatment=treatment,
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
		Input(ID_FILTER_TREATMENT, "value"),
		Input(ID_FILTER_DIV_RANGE, "value"),
		Input(ID_BOX_VALUE_COL, "value"),
		Input(ID_BOX_GROUP_COL, "value"),
		Input(ID_BOX_COLOR, "value"),
		Input(ID_BOX_TEST, "value"),
		Input(ID_BOX_CORRECTION, "value"),
		Input(ID_BOX_SHOW_SIGNIFICANCE, "value"),
		Input(ID_BOX_SHOW_POINTS, "value"),
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
		treatment,
		div_range,
		box_value_col,
		box_group_col,
		box_color,
		box_test,
		box_correction,
		show_significance,
		show_points,
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
			treatment=treatment,
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
			show_points=bool(show_points and "on" in show_points),
		)

	@app.callback(
		Output(ID_SCATTER_PLOT, "figure"),
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
		Input(ID_FILTER_TREATMENT, "value"),
		Input(ID_FILTER_DIV_RANGE, "value"),
		Input(ID_SCATTER_X, "value"),
		Input(ID_SCATTER_Y, "value"),
		Input(ID_SCATTER_COLOR, "value"),
		Input(ID_SCATTER_FACET_COL, "value"),
		Input(ID_SCATTER_FACET_ROW, "value"),
		Input(ID_SCATTER_JITTER, "value"),
	)
	def _update_scatter(
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
		treatment,
		div_range,
		scatter_x,
		scatter_y,
		scatter_color,
		scatter_facet_col,
		scatter_facet_row,
		scatter_jitter,
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
			treatment=treatment,
			div_range=div_range,
		)
		filtered = filter_helpers.apply_filter_spec(units_df, spec)
		return build_scatter(
			filtered,
			x_col=scatter_x,
			y_col=scatter_y,
			color_col=scatter_color,
			facet_col=scatter_facet_col,
			facet_row=scatter_facet_row,
			jitter=bool(scatter_jitter and "on" in scatter_jitter),
		)

	_register_image_download(
		app,
		button_id=ID_DOWNLOAD_HIST_PNG,
		target_id=ID_DOWNLOAD_HIST_PNG_TARGET,
		format="png",
		filename_stem="histogram",
		fig_inputs=(ID_HIST_X_AXIS, ID_HIST_COLOR),
		fig_builder=lambda filtered, *fig_args: _build_histogram(filtered, x_column=fig_args[0], color_column=fig_args[1]),
		units_df=units_df,
	)
	_register_image_download(
		app,
		button_id=ID_DOWNLOAD_HIST_SVG,
		target_id=ID_DOWNLOAD_HIST_SVG_TARGET,
		format="svg",
		filename_stem="histogram",
		fig_inputs=(ID_HIST_X_AXIS, ID_HIST_COLOR),
		fig_builder=lambda filtered, *fig_args: _build_histogram(filtered, x_column=fig_args[0], color_column=fig_args[1]),
		units_df=units_df,
	)
	_register_image_download(
		app,
		button_id=ID_DOWNLOAD_HIST_PDF,
		target_id=ID_DOWNLOAD_HIST_PDF_TARGET,
		format="pdf",
		filename_stem="histogram",
		fig_inputs=(ID_HIST_X_AXIS, ID_HIST_COLOR),
		fig_builder=lambda filtered, *fig_args: _build_histogram(filtered, x_column=fig_args[0], color_column=fig_args[1]),
		units_df=units_df,
	)

	def _build_box_from_state(filtered, *args):
		value_col, group_col, color_col, test, correction, show_sig, show_pts = args
		return build_box_plot(
			filtered,
			value_col=value_col,
			group_col=group_col,
			color_col=color_col,
			test=test,
			correction=correction,
			show_significance=bool(show_sig and "on" in show_sig),
			show_points=bool(show_pts and "on" in show_pts),
		)

	for fmt, btn_id, target_id in (
		("png", ID_DOWNLOAD_BOX_PNG, ID_DOWNLOAD_BOX_PNG_TARGET),
		("svg", ID_DOWNLOAD_BOX_SVG, ID_DOWNLOAD_BOX_SVG_TARGET),
		("pdf", ID_DOWNLOAD_BOX_PDF, ID_DOWNLOAD_BOX_PDF_TARGET),
	):
		_register_image_download(
			app,
			button_id=btn_id,
			target_id=target_id,
			format=fmt,
			filename_stem="box_plot",
			fig_inputs=(
				ID_BOX_VALUE_COL,
				ID_BOX_GROUP_COL,
				ID_BOX_COLOR,
				ID_BOX_TEST,
				ID_BOX_CORRECTION,
				ID_BOX_SHOW_SIGNIFICANCE,
				ID_BOX_SHOW_POINTS,
			),
			fig_builder=_build_box_from_state,
			units_df=units_df,
		)

	def _build_scatter_from_state(filtered, *args):
		x_col, y_col, color_col, facet_col, facet_row, jitter = args
		return build_scatter(
			filtered,
			x_col=x_col,
			y_col=y_col,
			color_col=color_col,
			facet_col=facet_col,
			facet_row=facet_row,
			jitter=bool(jitter and "on" in jitter),
		)

	for fmt, btn_id, target_id in (
		("png", ID_DOWNLOAD_SCATTER_PNG, ID_DOWNLOAD_SCATTER_PNG_TARGET),
		("svg", ID_DOWNLOAD_SCATTER_SVG, ID_DOWNLOAD_SCATTER_SVG_TARGET),
		("pdf", ID_DOWNLOAD_SCATTER_PDF, ID_DOWNLOAD_SCATTER_PDF_TARGET),
	):
		_register_image_download(
			app,
			button_id=btn_id,
			target_id=target_id,
			format=fmt,
			filename_stem="scatter",
			fig_inputs=(
				ID_SCATTER_X,
				ID_SCATTER_Y,
				ID_SCATTER_COLOR,
				ID_SCATTER_FACET_COL,
				ID_SCATTER_FACET_ROW,
				ID_SCATTER_JITTER,
			),
			fig_builder=_build_scatter_from_state,
			units_df=units_df,
		)

	@app.callback(
		Output(ID_DOWNLOAD_CSV_TARGET, "data"),
		Input(ID_DOWNLOAD_CSV, "n_clicks"),
		State(ID_FILTER_REQUIRE_RECON_OK, "value"),
		State(ID_FILTER_BOMBCELL, "value"),
		State(ID_FILTER_MIN_NUM_SPIKES, "value"),
		State(ID_FILTER_MIN_NUM_BRANCHES, "value"),
		State(ID_FILTER_MIN_RECON_QUALITY, "value"),
		State(ID_FILTER_PROJECT, "value"),
		State(ID_FILTER_CHIP, "value"),
		State(ID_FILTER_WELL, "value"),
		State(ID_FILTER_SCAN_TYPE, "value"),
		State(ID_FILTER_GENOTYPE, "value"),
		State(ID_FILTER_MEDIA, "value"),
		State(ID_FILTER_PLATING, "value"),
		State(ID_FILTER_TREATMENT, "value"),
		State(ID_FILTER_DIV_RANGE, "value"),
		prevent_initial_call=True,
	)
	def _download_csv(
		n_clicks,
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
		treatment,
		div_range,
	):
		if not n_clicks:
			raise dash.exceptions.PreventUpdate
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
			treatment=treatment,
			div_range=div_range,
		)
		filtered = filter_helpers.apply_filter_spec(units_df, spec)
		return dcc.send_data_frame(filtered.to_csv, "axon_dashboard_units.csv", index=False)

	@app.callback(
		Output(ID_DOWNLOAD_SPEC_TARGET, "data"),
		Input(ID_DOWNLOAD_SPEC_JSON, "n_clicks"),
		State(ID_FILTER_REQUIRE_RECON_OK, "value"),
		State(ID_FILTER_BOMBCELL, "value"),
		State(ID_FILTER_MIN_NUM_SPIKES, "value"),
		State(ID_FILTER_MIN_NUM_BRANCHES, "value"),
		State(ID_FILTER_MIN_RECON_QUALITY, "value"),
		State(ID_FILTER_PROJECT, "value"),
		State(ID_FILTER_CHIP, "value"),
		State(ID_FILTER_WELL, "value"),
		State(ID_FILTER_SCAN_TYPE, "value"),
		State(ID_FILTER_GENOTYPE, "value"),
		State(ID_FILTER_MEDIA, "value"),
		State(ID_FILTER_PLATING, "value"),
		State(ID_FILTER_TREATMENT, "value"),
		State(ID_FILTER_DIV_RANGE, "value"),
		State(ID_HIST_X_AXIS, "value"),
		State(ID_HIST_COLOR, "value"),
		State(ID_BOX_VALUE_COL, "value"),
		State(ID_BOX_GROUP_COL, "value"),
		State(ID_BOX_COLOR, "value"),
		State(ID_BOX_TEST, "value"),
		State(ID_BOX_CORRECTION, "value"),
		State(ID_BOX_SHOW_SIGNIFICANCE, "value"),
		State(ID_BOX_SHOW_POINTS, "value"),
		State(ID_SCATTER_X, "value"),
		State(ID_SCATTER_Y, "value"),
		State(ID_SCATTER_COLOR, "value"),
		State(ID_SCATTER_FACET_COL, "value"),
		State(ID_SCATTER_FACET_ROW, "value"),
		State(ID_SCATTER_JITTER, "value"),
		State(ID_MAIN_TABS, "value"),
		prevent_initial_call=True,
	)
	def _download_spec(
		n_clicks,
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
		treatment,
		div_range,
		hist_x,
		hist_color,
		box_value_col,
		box_group_col,
		box_color,
		box_test,
		box_correction,
		show_significance,
		show_points,
		scatter_x,
		scatter_y,
		scatter_color,
		scatter_facet_col,
		scatter_facet_row,
		scatter_jitter,
		active_tab,
	):
		if not n_clicks:
			raise dash.exceptions.PreventUpdate
		filter_spec = _build_filter_spec_from_state(
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
			treatment=treatment,
			div_range=div_range,
		)
		plot_spec = {
			"active_tab": active_tab,
			"histogram": {"x_axis": hist_x, "color": hist_color},
			"box": {
				"value_col": box_value_col,
				"group_col": box_group_col,
				"color": box_color,
				"test": box_test,
				"correction": box_correction,
				"show_significance": bool(show_significance and "on" in show_significance),
				"show_points": bool(show_points and "on" in show_points),
			},
			"scatter": {
				"x": scatter_x,
				"y": scatter_y,
				"color": scatter_color,
				"facet_col": scatter_facet_col,
				"facet_row": scatter_facet_row,
				"jitter": bool(scatter_jitter and "on" in scatter_jitter),
			},
		}
		text = filter_helpers.filter_spec_to_json(filter_spec, plot_spec=plot_spec)
		return dcc.send_string(text, "axon_dashboard_spec.json")

	return app


def _register_image_download(
	app: dash.Dash,
	*,
	button_id: str,
	target_id: str,
	format: str,
	filename_stem: str,
	fig_inputs: tuple[str, ...],
	fig_builder,
	units_df: pd.DataFrame,
) -> None:
	"""Wire a single PNG/SVG/PDF download button into the app.

	The callback recomputes the filtered DataFrame + figure on click rather
	than reading from a hidden store so the export always matches the
	current filter/plot inputs.
	"""
	mime = _IMAGE_MIME_BY_FORMAT.get(format)
	if mime is None:
		raise ValueError(f"Unsupported image format: {format!r}")
	filter_state_ids = (
		ID_FILTER_REQUIRE_RECON_OK,
		ID_FILTER_BOMBCELL,
		ID_FILTER_MIN_NUM_SPIKES,
		ID_FILTER_MIN_NUM_BRANCHES,
		ID_FILTER_MIN_RECON_QUALITY,
		ID_FILTER_PROJECT,
		ID_FILTER_CHIP,
		ID_FILTER_WELL,
		ID_FILTER_SCAN_TYPE,
		ID_FILTER_GENOTYPE,
		ID_FILTER_MEDIA,
		ID_FILTER_PLATING,
		ID_FILTER_DIV_RANGE,
	)
	filter_states = [State(_id, "value") for _id in filter_state_ids]
	fig_states = [State(_id, "value") for _id in fig_inputs]

	@app.callback(
		Output(target_id, "data"),
		Input(button_id, "n_clicks"),
		*filter_states,
		*fig_states,
		prevent_initial_call=True,
	)
	def _callback(n_clicks, *all_args):
		if not n_clicks:
			raise dash.exceptions.PreventUpdate
		filter_values = all_args[: len(filter_state_ids)]
		fig_values = all_args[len(filter_state_ids) :]
		spec = _build_filter_spec_from_state(
			require_recon_ok=filter_values[0],
			bombcell=filter_values[1],
			min_num_spikes=filter_values[2],
			min_num_branches=filter_values[3],
			min_recon_quality=filter_values[4],
			project=filter_values[5],
			chip=filter_values[6],
			well=filter_values[7],
			scan_type=filter_values[8],
			genotype=filter_values[9],
			media=filter_values[10],
			plating=filter_values[11],
			div_range=filter_values[12],
		)
		filtered = filter_helpers.apply_filter_spec(units_df, spec)
		fig = fig_builder(filtered, *fig_values)
		image_bytes = fig.to_image(format=format)
		return dcc.send_bytes(image_bytes, f"{filename_stem}.{format}", type=mime)


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
	show_points: bool = False,
) -> Any:
	"""Build a Plotly box-plot figure with optional significance brackets.

	Empty / missing-column inputs return an empty Plotly figure (rather than
	raising) so the Dash callback can render something on every fired update.
	When `show_points` is True, every observation is overlaid on the box as
	a jittered point (Plotly's `points="all"` mode).
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
	points_mode = "all" if show_points else "outliers"
	fig = px.box(df, x=group, y=value, color=color, points=points_mode)
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


def build_scatter(
	df: pd.DataFrame,
	*,
	x_col: Any,
	y_col: Any,
	color_col: Any = _SCATTER_COLOR_NONE,
	facet_col: Any = _FACET_NONE,
	facet_row: Any = _FACET_NONE,
	jitter: bool = False,
) -> Any:
	"""Plotly scatter figure with optional color + facet + jitter support.

	When `jitter` is True, numeric x and/or y columns get small Gaussian
	noise added (stdev scaled to 1% of the per-column range, min 1e-6).
	Categorical axes are left untouched — plotly already stacks string
	categories on integer positions and we don't want to silently change
	their dtype.

	Empty / missing-column inputs return an empty Plotly figure rather than
	raising, so the Dash callback can render something on every update.
	"""
	if df is None or df.empty or not x_col or not y_col:
		return px.scatter(pd.DataFrame({"_": []}), x="_", y="_")
	x = str(x_col)
	y = str(y_col)
	if x not in df.columns or y not in df.columns:
		return px.scatter(pd.DataFrame({"_": []}), x="_", y="_")
	color = None
	if color_col and color_col != _SCATTER_COLOR_NONE and str(color_col) in df.columns:
		color = str(color_col)
	fc = None
	if facet_col and facet_col != _FACET_NONE and str(facet_col) in df.columns:
		fc = str(facet_col)
	fr = None
	if facet_row and facet_row != _FACET_NONE and str(facet_row) in df.columns:
		fr = str(facet_row)
	plot_df = _apply_scatter_jitter(df, x_col=x, y_col=y) if jitter else df
	return px.scatter(plot_df, x=x, y=y, color=color, facet_col=fc, facet_row=fr, opacity=0.7)


def _apply_scatter_jitter(df: pd.DataFrame, *, x_col: str, y_col: str) -> pd.DataFrame:
	"""Return a copy of `df` with small Gaussian noise added to numeric x/y.

	Visualization-only — does not mutate `df`. Non-numeric columns are
	passed through unchanged (plotly handles categorical stacking on its
	own). The noise stdev scales with the column's observed range so the
	jitter remains visible at any data magnitude.
	"""
	import numpy as np

	out = df.copy()
	rng = np.random.default_rng(0)
	for col in (x_col, y_col):
		if col not in out.columns:
			continue
		if out[col].dtype.kind not in ("i", "u", "f"):
			continue
		values = pd.to_numeric(out[col], errors="coerce")
		finite = values.dropna()
		if finite.empty:
			continue
		span = float(finite.max() - finite.min())
		scale = max(span * 0.01, 1e-6)
		out[col] = values + rng.normal(0.0, scale, size=len(values))
	return out
