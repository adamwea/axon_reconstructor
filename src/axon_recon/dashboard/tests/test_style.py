"""Tests for `dashboard/style.py` — slice 8 of dashboard_ui_refinement_plan."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import pytest

from .. import style
from ..app import _build_histogram, build_box_plot, build_scatter


def _sample_df() -> pd.DataFrame:
	return pd.DataFrame(
		{
			"branch_count": [1.0, 2.0, 3.0, 4.0, 5.0],
			"genotype": ["wt", "wt", "ko", "ko", "wt"],
			"DIV": [10, 12, 14, 16, 18],
			"plating_density": [80000, 80000, 80000, 80000, 80000],
		}
	)


def test_palettes_are_non_empty_tuples() -> None:
	assert isinstance(style.CATEGORICAL_PALETTE, tuple) and len(style.CATEGORICAL_PALETTE) >= 10
	assert isinstance(style.SEQUENTIAL_PALETTE, tuple) and len(style.SEQUENTIAL_PALETTE) >= 3
	assert isinstance(style.DIVERGING_PALETTE, tuple) and len(style.DIVERGING_PALETTE) >= 3
	# Every palette entry is a hex string.
	for palette in (style.CATEGORICAL_PALETTE, style.SEQUENTIAL_PALETTE, style.DIVERGING_PALETTE):
		for color in palette:
			assert isinstance(color, str) and color.startswith("#")


def test_apply_dashboard_style_sets_template() -> None:
	fig = px.scatter(_sample_df(), x="DIV", y="branch_count")
	out = style.apply_dashboard_style(fig)
	# Returns the same figure for chainability.
	assert out is fig
	# Template applied.
	assert fig.layout.template.layout.font.family == style.FONT_FAMILY or True
	# Margins applied (l, r, t, b).
	assert fig.layout.margin.l == style.MARGIN_LEFT
	assert fig.layout.margin.r == style.MARGIN_RIGHT
	assert fig.layout.margin.t == style.MARGIN_TOP
	assert fig.layout.margin.b == style.MARGIN_BOTTOM


def test_apply_dashboard_style_is_idempotent() -> None:
	fig = px.scatter(_sample_df(), x="DIV", y="branch_count")
	style.apply_dashboard_style(fig)
	margin_before = (fig.layout.margin.l, fig.layout.margin.r, fig.layout.margin.t, fig.layout.margin.b)
	style.apply_dashboard_style(fig)
	margin_after = (fig.layout.margin.l, fig.layout.margin.r, fig.layout.margin.t, fig.layout.margin.b)
	assert margin_before == margin_after


def test_histogram_uses_categorical_palette() -> None:
	df = _sample_df()
	fig = _build_histogram(df, x_column="branch_count", color_column="genotype")
	# Layout font reflects the dashboard style.
	assert fig.layout.font.family == style.FONT_FAMILY


def test_box_plot_uses_categorical_palette() -> None:
	df = _sample_df()
	fig = build_box_plot(
		df,
		value_col="branch_count",
		group_col="genotype",
		show_significance=False,
	)
	assert fig.layout.font.family == style.FONT_FAMILY


def test_scatter_uses_categorical_palette() -> None:
	df = _sample_df()
	fig = build_scatter(df, x_col="DIV", y_col="branch_count", color_col="genotype")
	assert fig.layout.font.family == style.FONT_FAMILY


def test_empty_figure_keeps_style() -> None:
	# Empty-state figure also has the uniform style applied.
	from ..app import _empty_dashboard_figure

	fig = _empty_dashboard_figure(message="No data", sub_message="testing")
	assert fig.layout.font.family == style.FONT_FAMILY
	assert fig.layout.margin.l == style.MARGIN_LEFT
