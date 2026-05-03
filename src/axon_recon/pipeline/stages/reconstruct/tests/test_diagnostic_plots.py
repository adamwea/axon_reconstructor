from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def _raise_save_failure(**_kwargs):
	raise RuntimeError("boom")


def test_write_unit_channel_selection_diagnostic_figure_closes_on_save_error(monkeypatch) -> None:
	import matplotlib

	matplotlib.use("Agg", force=True)
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]

	from axon_recon.pipeline.stages.reconstruct.core import diagnostic_plots as diagnostic_plots_module

	class _FakeAV:
		def plot_amplitude_map(self, template, locations, log, ax, cmap, colorbar, colorbar_orientation):
			_ = (template, locations, log, cmap, colorbar, colorbar_orientation)
			ax.plot([0.0, 1.0], [0.0, 1.0])

		def plot_peak_latency_map(self, template, locations, fs, log, ax, colorbar, colorbar_orientation):
			_ = (template, locations, fs, log, colorbar, colorbar_orientation)
			ax.plot([0.0, 1.0], [1.0, 0.0])

	monkeypatch.setattr(diagnostic_plots_module, "_plot_channel_selection_panels", lambda **kwargs: kwargs["fig"].add_subplot(1, 1, 1))
	monkeypatch.setattr(diagnostic_plots_module, "_save_figure", _raise_save_failure)
	gtr = SimpleNamespace(
		template=np.asarray([[-1.0, -2.0], [-0.5, -1.0]], dtype=float),
		locations=np.asarray([[0.0, 0.0], [17.5, 0.0]], dtype=float),
		fs=10_000.0,
	)
	before = tuple(plt.get_fignums())
	with pytest.raises(RuntimeError, match="boom"):
		diagnostic_plots_module.write_unit_channel_selection_diagnostic_figure(
			av=_FakeAV(),
			output_png=None,
			output_svg=SimpleNamespace(parent=SimpleNamespace(mkdir=lambda **_kwargs: None)),
			template_ch_by_t=gtr.template,
			locs_xy=gtr.locations,
			gtr=gtr,
		)
	assert tuple(plt.get_fignums()) == before


def test_write_unit_axon_reconstruction_diagnostic_figure_closes_on_save_error(monkeypatch) -> None:
	import matplotlib

	matplotlib.use("Agg", force=True)
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]

	from axon_recon.pipeline.stages.reconstruct.core import diagnostic_plots as diagnostic_plots_module

	class _FakeGtr:
		graph = SimpleNamespace(edges=SimpleNamespace(data=lambda: [(0, 1, {"heur": 1.0})]))
		_node_heuristic = np.asarray([1.0, 2.0], dtype=float)

		def _plot_nodes(self, *, cmap_nodes, node_searched_labels, ax):
			_ = (cmap_nodes, node_searched_labels)
			ax.plot([0.0, 1.0], [0.0, 1.0])

		def _plot_edges(self, *, cmap_edges, ax):
			_ = cmap_edges
			ax.plot([0.0, 1.0], [1.0, 0.0])

		def plot_raw_branches(self, *, cmap, plot_bp, plot_neighbors, plot_full_template, ax):
			_ = (cmap, plot_bp, plot_neighbors, plot_full_template)
			ax.plot([0.0, 1.0], [0.5, 0.5])

	monkeypatch.setattr(diagnostic_plots_module, "_plot_branch_velocity_panel", lambda **kwargs: kwargs["ax"].plot([0.0, 1.0], [0.0, 1.0]))
	monkeypatch.setattr(diagnostic_plots_module, "_save_figure", _raise_save_failure)
	before = tuple(plt.get_fignums())
	with pytest.raises(RuntimeError, match="boom"):
		diagnostic_plots_module.write_unit_axon_reconstruction_diagnostic_figure(
			output_png=None,
			output_svg=SimpleNamespace(parent=SimpleNamespace(mkdir=lambda **_kwargs: None)),
			gtr=_FakeGtr(),
		)
	assert tuple(plt.get_fignums()) == before