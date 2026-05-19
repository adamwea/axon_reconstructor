"""Slice 4 of dashboard_ui_refinement_plan: PlotConfig scaffold tests."""

from __future__ import annotations

import pandas as pd
import pytest

from ..plots import PlotConfig, apply_filters_to_dataframe


def test_plot_config_default_mode_is_histogram() -> None:
	cfg = PlotConfig()
	assert cfg.mode == "histogram"
	assert cfg.data_source == "units"


def test_plot_config_is_frozen() -> None:
	cfg = PlotConfig()
	with pytest.raises(Exception):
		cfg.mode = "box"  # type: ignore[misc]


def test_plot_config_box_plot_defaults() -> None:
	cfg = PlotConfig(mode="box", group_column="genotype", x_column="branch_count")
	assert cfg.mode == "box"
	assert cfg.group_column == "genotype"
	assert cfg.x_column == "branch_count"
	assert cfg.significance_test == "mann_whitney"
	assert cfg.points_mode == "off"


def test_plot_config_scatter_defaults() -> None:
	cfg = PlotConfig(mode="scatter", x_column="x", y_column="y", color_column="genotype")
	assert cfg.log_x is False
	assert cfg.log_y is False
	assert cfg.jitter is False
	assert cfg.opacity == 0.7


def test_plot_config_bar_mode_future_slice6_defaults() -> None:
	# Slice 6 will use bar mode for the box↔bar toggle. The defaults
	# are ready for that slice to plug in.
	cfg = PlotConfig(mode="bar", group_column="genotype", x_column="branch_count")
	assert cfg.bar_aggregate == "mean"
	assert cfg.bar_error == "std"


def test_plot_config_tertiary_grouping_future_slice7() -> None:
	# Slice 7 will use tertiary_group_column. The field is present so
	# the API is stable when slice 7 ships.
	cfg = PlotConfig(
		mode="box",
		group_column="DIV",
		secondary_group_column="plating_density",
		tertiary_group_column="media",
	)
	assert cfg.tertiary_group_column == "media"


def test_apply_filters_passes_dataframe_through() -> None:
	# Slice 4 scaffold: filter application currently no-ops. The
	# function exists as a future-facing seam — slices that migrate
	# filter logic out of app.py's callbacks will fill it in.
	df = pd.DataFrame({"x": [1, 2, 3]})
	out = apply_filters_to_dataframe(df, PlotConfig())
	assert out is df


def test_plot_config_log_transform_field() -> None:
	# log_transform stays separate from log_x/log_y because the box
	# plot's existing knob uses single-axis log on `value_col` rather
	# than per-axis. Three orthogonal toggles.
	cfg = PlotConfig(mode="box", log_transform=True)
	assert cfg.log_transform is True
	assert cfg.log_x is False
	assert cfg.log_y is False
