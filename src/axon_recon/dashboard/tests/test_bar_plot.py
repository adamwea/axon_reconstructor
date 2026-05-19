"""Slice 6 of dashboard_ui_refinement_plan: box↔bar toggle render tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ..app import aggregate_by_group, build_bar_plot


def _two_group_df() -> pd.DataFrame:
	return pd.DataFrame(
		{
			"value": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 100.0, 200.0],
			"genotype": ["wt", "wt", "wt", "ko", "ko", "ko", "wt", "ko"],
		}
	)


# --- aggregate_by_group ---


def test_aggregate_mean_std() -> None:
	df = _two_group_df()
	out = aggregate_by_group(df, value_col="value", group_col="genotype", aggregate="mean", error="std")
	out_sorted = out.sort_values("genotype").reset_index(drop=True)
	# wt: [1, 2, 3, 100]; ko: [10, 20, 30, 200].
	assert out_sorted.loc[out_sorted["genotype"] == "ko", "value"].iloc[0] == pytest.approx(65.0)
	assert out_sorted.loc[out_sorted["genotype"] == "wt", "value"].iloc[0] == pytest.approx(26.5)
	# Std with ddof=1.
	wt_std = float(np.std([1.0, 2.0, 3.0, 100.0], ddof=1))
	assert out_sorted.loc[out_sorted["genotype"] == "wt", "error"].iloc[0] == pytest.approx(wt_std)


def test_aggregate_median_no_error() -> None:
	df = _two_group_df()
	out = aggregate_by_group(df, value_col="value", group_col="genotype", aggregate="median", error="none")
	out_sorted = out.sort_values("genotype").reset_index(drop=True)
	assert out_sorted.loc[out_sorted["genotype"] == "ko", "value"].iloc[0] == pytest.approx(25.0)
	assert out_sorted.loc[out_sorted["genotype"] == "wt", "value"].iloc[0] == pytest.approx(2.5)
	# Error column present but zero.
	assert (out_sorted["error"] == 0.0).all()


def test_aggregate_sem_smaller_than_std() -> None:
	df = _two_group_df()
	out_std = aggregate_by_group(df, value_col="value", group_col="genotype", aggregate="mean", error="std")
	out_sem = aggregate_by_group(df, value_col="value", group_col="genotype", aggregate="mean", error="sem")
	# SEM = std / sqrt(N) — must be smaller than std for N > 1.
	wt_std = float(out_std.loc[out_std["genotype"] == "wt", "error"].iloc[0])
	wt_sem = float(out_sem.loc[out_sem["genotype"] == "wt", "error"].iloc[0])
	assert wt_sem < wt_std


def test_aggregate_ci95_is_196_times_sem() -> None:
	df = _two_group_df()
	out_sem = aggregate_by_group(df, value_col="value", group_col="genotype", aggregate="mean", error="sem")
	out_ci = aggregate_by_group(df, value_col="value", group_col="genotype", aggregate="mean", error="ci95")
	wt_sem = float(out_sem.loc[out_sem["genotype"] == "wt", "error"].iloc[0])
	wt_ci = float(out_ci.loc[out_ci["genotype"] == "wt", "error"].iloc[0])
	assert wt_ci == pytest.approx(1.96 * wt_sem)


def test_aggregate_with_color_col_groups_on_combination() -> None:
	df = pd.DataFrame(
		{
			"value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
			"genotype": ["wt", "wt", "wt", "ko", "ko", "ko"],
			"media": ["a", "a", "b", "a", "b", "b"],
		}
	)
	out = aggregate_by_group(
		df,
		value_col="value",
		group_col="genotype",
		color_col="media",
		aggregate="mean",
		error="std",
	)
	# 4 unique (genotype, media) combinations: wt-a, wt-b, ko-a, ko-b.
	assert len(out) == 4


def test_aggregate_drops_non_numeric_rows() -> None:
	df = pd.DataFrame(
		{
			"value": ["1.0", "2.0", "bogus", "4.0"],
			"genotype": ["wt", "wt", "ko", "ko"],
		}
	)
	out = aggregate_by_group(df, value_col="value", group_col="genotype", aggregate="mean", error="std")
	# ko: only "4.0" is numeric → mean 4.0; wt: 1.5.
	out_sorted = out.sort_values("genotype").reset_index(drop=True)
	assert out_sorted.loc[out_sorted["genotype"] == "ko", "value"].iloc[0] == pytest.approx(4.0)
	assert out_sorted.loc[out_sorted["genotype"] == "wt", "value"].iloc[0] == pytest.approx(1.5)


def test_aggregate_empty_df_returns_empty_frame() -> None:
	out = aggregate_by_group(pd.DataFrame(), value_col="value", group_col="genotype")
	assert isinstance(out, pd.DataFrame)
	assert len(out) == 0


# --- build_bar_plot ---


def test_bar_plot_renders_with_error_bars() -> None:
	df = _two_group_df()
	fig = build_bar_plot(df, value_col="value", group_col="genotype", error="std")
	# Two bars (wt + ko). Plotly renders each color as a separate trace
	# when color is unset; single trace here.
	assert len(fig.data) >= 1
	# error_y was forwarded — each trace has error_y configured.
	traces_with_error = [t for t in fig.data if getattr(t, "error_y", None) is not None and getattr(t.error_y, "array", None) is not None]
	assert len(traces_with_error) >= 1


def test_bar_plot_empty_df_renders_empty_state() -> None:
	fig = build_bar_plot(pd.DataFrame(), value_col="value", group_col="genotype")
	texts = [str(getattr(ann, "text", "")) for ann in fig.layout.annotations]
	assert any("No data available" in t for t in texts)


def test_bar_plot_missing_columns_renders_empty_state() -> None:
	df = pd.DataFrame({"present": [1, 2, 3]})
	fig = build_bar_plot(df, value_col="absent_value", group_col="absent_group")
	texts = [str(getattr(ann, "text", "")) for ann in fig.layout.annotations]
	assert any("not present" in t for t in texts)


def test_bar_plot_log_transform_drops_nonpositive_rows() -> None:
	df = pd.DataFrame(
		{
			"value": [1.0, 10.0, 100.0, -1.0, 0.0],
			"genotype": ["wt"] * 5,
		}
	)
	fig = build_bar_plot(df, value_col="value", group_col="genotype", log_transform=True)
	# Surviving 3 positive rows → mean(log10) finite + bar present.
	assert len(fig.data) >= 1


def test_bar_plot_log_transform_all_nonpositive_renders_empty_state() -> None:
	df = pd.DataFrame(
		{
			"value": [0.0, -1.0, -10.0],
			"genotype": ["wt"] * 3,
		}
	)
	fig = build_bar_plot(df, value_col="value", group_col="genotype", log_transform=True)
	texts = [str(getattr(ann, "text", "")) for ann in fig.layout.annotations]
	assert any("Log transform requires positive values" in t for t in texts)
