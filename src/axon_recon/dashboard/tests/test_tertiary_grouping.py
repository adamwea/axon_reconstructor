"""Slice 7 of dashboard_ui_refinement_plan.md: tertiary grouping for
box + bar plots. Two render modes:

- `small_multiples` (default per current_state.md PRE-OVERNIGHT CLEARANCE
  item #1): tertiary column drives plotly's `facet_col` — one subplot
  per tertiary value.
- `hierarchical_labels`: compound x-axis labels ("group | tertiary")
  preserve a single subplot but show the tertiary breakdown.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ..app import aggregate_by_group, build_bar_plot, build_box_plot


def _three_dim_df() -> pd.DataFrame:
	"""Synthetic frame with primary (DIV), secondary (genotype), tertiary (media)
	dimensions. 24 rows = 2 DIVs × 2 genotypes × 2 media × 3 reps."""

	rows = []
	for div in (6, 12):
		for genotype in ("wt", "ko"):
			for media in ("media_a", "media_b"):
				# 3 reps with predictable values so tests can sanity-check.
				for rep in range(3):
					rows.append(
						{
							"value": 10.0 * div + (1.0 if genotype == "ko" else 0.0)
							+ (0.1 if media == "media_b" else 0.0) + 0.001 * rep,
							"DIV": div,
							"genotype": genotype,
							"media": media,
						}
					)
	return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Box plot tertiary — small_multiples mode (default)
# ---------------------------------------------------------------------


def test_box_tertiary_small_multiples_creates_facet() -> None:
	df = _three_dim_df()
	fig = build_box_plot(
		df,
		value_col="value",
		group_col="DIV",
		color_col="genotype",
		tertiary_group_col="media",
		# tertiary_render_mode defaults to small_multiples
	)
	# Plotly creates one subplot per tertiary value when facet_col is set.
	# Each subplot has its own xaxis (xaxis, xaxis2, ...).
	xaxis_keys = [k for k in fig.layout if str(k).startswith("xaxis")]
	assert len(xaxis_keys) >= 2  # 2 facets for media_a + media_b


def test_box_tertiary_unset_no_facet() -> None:
	df = _three_dim_df()
	fig = build_box_plot(
		df,
		value_col="value",
		group_col="DIV",
		color_col="genotype",
		# no tertiary_group_col
	)
	xaxis_keys = [k for k in fig.layout if str(k).startswith("xaxis")]
	assert len(xaxis_keys) == 1


def test_box_tertiary_missing_column_treated_as_unset() -> None:
	df = _three_dim_df()
	fig = build_box_plot(
		df,
		value_col="value",
		group_col="DIV",
		tertiary_group_col="not_a_real_column",
	)
	# Missing column → fall back to no tertiary → no faceting.
	xaxis_keys = [k for k in fig.layout if str(k).startswith("xaxis")]
	assert len(xaxis_keys) == 1


# ---------------------------------------------------------------------
# Box plot tertiary — hierarchical_labels mode
# ---------------------------------------------------------------------


def test_box_tertiary_hierarchical_labels_compound_x_axis() -> None:
	df = _three_dim_df()
	fig = build_box_plot(
		df,
		value_col="value",
		group_col="DIV",
		tertiary_group_col="media",
		tertiary_render_mode="hierarchical_labels",
	)
	# Single subplot (no faceting) but x-axis categories are compound.
	xaxis_keys = [k for k in fig.layout if str(k).startswith("xaxis")]
	assert len(xaxis_keys) == 1
	x_array = list(fig.layout.xaxis.categoryarray)
	# Compound labels of the form "<DIV> | <media>".
	assert all("|" in label for label in x_array)
	# 2 DIVs × 2 media = 4 compound categories.
	assert len(x_array) == 4


def test_box_tertiary_hierarchical_labels_skips_significance_brackets() -> None:
	"""Per slice 7 design: in hierarchical_labels mode the pairwise
	stat-test brackets are skipped to keep v1 ergonomics tractable."""

	df = _three_dim_df()
	fig = build_box_plot(
		df,
		value_col="value",
		group_col="DIV",
		tertiary_group_col="media",
		tertiary_render_mode="hierarchical_labels",
		test="mann_whitney",
		show_significance=True,
	)
	# Significance brackets render as shapes (lines) + annotations (p-text).
	# In hierarchical_labels mode, the bracket-rendering helper is skipped,
	# so no significance shapes are present.
	shapes = list(fig.layout.shapes or [])
	# All shapes (if any) should NOT be significance brackets — those have
	# a specific structure. The simplest check: in small_multiples mode the
	# brackets DO render; here they DON'T.
	bracket_like = [
		s for s in shapes
		if getattr(s, "type", None) == "line" and "y" in str(getattr(s, "yref", ""))
	]
	# Default test path doesn't render brackets in hierarchical_labels mode.
	# This is the stricter assertion; if brackets WERE rendered they'd be lines.
	assert bracket_like == []


def test_box_tertiary_invalid_render_mode_falls_back_to_small_multiples() -> None:
	df = _three_dim_df()
	fig = build_box_plot(
		df,
		value_col="value",
		group_col="DIV",
		tertiary_group_col="media",
		tertiary_render_mode="some_unknown_mode",
	)
	# Should fall back to small_multiples → facets.
	xaxis_keys = [k for k in fig.layout if str(k).startswith("xaxis")]
	assert len(xaxis_keys) >= 2


# ---------------------------------------------------------------------
# Bar plot tertiary — small_multiples + hierarchical_labels
# ---------------------------------------------------------------------


def test_bar_tertiary_small_multiples_creates_facet() -> None:
	df = _three_dim_df()
	fig = build_bar_plot(
		df,
		value_col="value",
		group_col="DIV",
		color_col="genotype",
		tertiary_group_col="media",
	)
	xaxis_keys = [k for k in fig.layout if str(k).startswith("xaxis")]
	assert len(xaxis_keys) >= 2


def test_bar_tertiary_hierarchical_labels_compound_x() -> None:
	df = _three_dim_df()
	fig = build_bar_plot(
		df,
		value_col="value",
		group_col="DIV",
		tertiary_group_col="media",
		tertiary_render_mode="hierarchical_labels",
	)
	xaxis_keys = [k for k in fig.layout if str(k).startswith("xaxis")]
	assert len(xaxis_keys) == 1


def test_bar_tertiary_unset_no_facet() -> None:
	df = _three_dim_df()
	fig = build_bar_plot(df, value_col="value", group_col="DIV")
	xaxis_keys = [k for k in fig.layout if str(k).startswith("xaxis")]
	assert len(xaxis_keys) == 1


# ---------------------------------------------------------------------
# aggregate_by_group tertiary support
# ---------------------------------------------------------------------


def test_aggregate_by_group_with_tertiary_col() -> None:
	df = _three_dim_df()
	out = aggregate_by_group(
		df,
		value_col="value",
		group_col="DIV",
		tertiary_col="media",
		aggregate="mean",
		error="std",
	)
	# 2 DIVs × 2 media = 4 rows.
	assert len(out) == 4
	assert set(out.columns) >= {"DIV", "media", "value", "error"}


def test_aggregate_by_group_tertiary_missing_column_ignored() -> None:
	df = _three_dim_df()
	out = aggregate_by_group(
		df,
		value_col="value",
		group_col="DIV",
		tertiary_col="not_a_column",
		aggregate="mean",
		error="none",
	)
	# Tertiary col absent → no extra grouping. 2 DIVs = 2 rows.
	assert len(out) == 2


def test_aggregate_by_group_color_and_tertiary_compose() -> None:
	df = _three_dim_df()
	out = aggregate_by_group(
		df,
		value_col="value",
		group_col="DIV",
		color_col="genotype",
		tertiary_col="media",
		aggregate="mean",
		error="none",
	)
	# 2 DIVs × 2 genotypes × 2 media = 8 rows.
	assert len(out) == 8


# ---------------------------------------------------------------------
# Tertiary equals primary or secondary — defensive: ignored.
# ---------------------------------------------------------------------


def test_box_tertiary_same_as_primary_ignored() -> None:
	df = _three_dim_df()
	fig = build_box_plot(
		df,
		value_col="value",
		group_col="DIV",
		tertiary_group_col="DIV",  # same as primary — should be ignored
	)
	xaxis_keys = [k for k in fig.layout if str(k).startswith("xaxis")]
	assert len(xaxis_keys) == 1


def test_box_tertiary_same_as_secondary_ignored() -> None:
	df = _three_dim_df()
	fig = build_box_plot(
		df,
		value_col="value",
		group_col="DIV",
		color_col="genotype",
		tertiary_group_col="genotype",  # same as secondary — should be ignored
	)
	xaxis_keys = [k for k in fig.layout if str(k).startswith("xaxis")]
	assert len(xaxis_keys) == 1
