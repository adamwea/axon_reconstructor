"""Tests for the --no-plot override and resolve_plots_enabled helper."""

from __future__ import annotations

import pytest

from axon_recon.pipeline.config import (
	get_no_plot_override,
	resolve_plots_enabled,
	set_no_plot_override,
)


@pytest.fixture(autouse=True)
def _reset_no_plot_override():
	"""Ensure each test starts and ends with a clean override state."""
	set_no_plot_override(None)
	yield
	set_no_plot_override(None)


def test_no_plot_override_defaults_to_none() -> None:
	assert get_no_plot_override() is None


def test_set_no_plot_override_to_true() -> None:
	set_no_plot_override(True)
	assert get_no_plot_override() is True


def test_set_no_plot_override_clears_with_none() -> None:
	set_no_plot_override(True)
	set_no_plot_override(None)
	assert get_no_plot_override() is None


def test_resolve_plots_enabled_no_override_honors_yaml_true() -> None:
	assert resolve_plots_enabled(True) is True


def test_resolve_plots_enabled_no_override_honors_yaml_false() -> None:
	assert resolve_plots_enabled(False) is False


def test_resolve_plots_enabled_no_override_yaml_unset_uses_default() -> None:
	assert resolve_plots_enabled(None, default=True) is True
	assert resolve_plots_enabled(None, default=False) is False


def test_resolve_plots_enabled_override_true_forces_false_regardless_of_yaml() -> None:
	"""--no-plot wins even when YAML says plots_enabled: true."""
	set_no_plot_override(True)
	assert resolve_plots_enabled(True) is False
	assert resolve_plots_enabled(False) is False
	assert resolve_plots_enabled(None, default=True) is False
