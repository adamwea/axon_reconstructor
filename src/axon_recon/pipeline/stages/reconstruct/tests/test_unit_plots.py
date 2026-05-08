from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from axon_recon.pipeline.shared.plotting import SharedHeatmapConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import CircleReconConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import CircleReconDisplayConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import CircleReconOutputConfig
from axon_recon.pipeline.stages.reconstruct.core.unit_plots import write_unit_amplitude_map_png
from axon_recon.pipeline.stages.reconstruct.core.unit_plots import write_unit_circle_recon_plot


def test_write_unit_amplitude_map_png_writes_file(tmp_path: Path) -> None:
	template_ch_by_t = np.array(
		[
			[-5.0, -10.0, -3.0],
			[-2.0, -4.0, -1.0],
			[-1.0, -6.0, -2.0],
		],
		dtype=float,
	)
	locs_xy = np.array(
		[
			[0.0, 0.0],
			[17.5, 0.0],
			[0.0, 17.5],
		],
		dtype=float,
	)
	out = tmp_path / "amplitude_map.png"

	write_unit_amplitude_map_png(
		output_png=out,
		template_ch_by_t=template_ch_by_t,
		locs_xy=locs_xy,
	)

	assert out.exists()
	assert out.stat().st_size > 0


def test_write_unit_circle_recon_plot_writes_png(tmp_path: Path) -> None:
	template_ch_by_t = np.array(
		[
			[-5.0, -10.0, -3.0, -1.0],
			[-2.0, -4.0, -1.0, -0.5],
			[-1.0, -6.0, -2.0, -0.5],
		],
		dtype=float,
	)
	locs_xy = np.array(
		[
			[0.0, 0.0],
			[17.5, 0.0],
			[0.0, 17.5],
		],
		dtype=float,
	)

	class _GraphMock:
		def nodes(self):
			return [0, 1, 2]

	class _GtrMock:
		def __init__(self):
			self._paths_raw = [[0, 1, 2]]
			self.branches = [{"branch_index": 0, "channels": [0, 1, 2]}]
			self.graph = _GraphMock()

	gtr = _GtrMock()
	out_png = tmp_path / "circle_recon.png"
	out_svg = tmp_path / "circle_recon.svg"

	cfg = CircleReconConfig(
		display=CircleReconDisplayConfig(
			base="template_circles",
			channel_scope="nodes_and_branches",
			force_center_soma=True,
			branch_scope="raw",
			unique_color_per_branch=True,
			show_branch_labels=True,
			color_scheme="tab20",
			node_border_linewidth=0.35,
			edge_linewidth=0.8,
		),
		output=CircleReconOutputConfig(write_png=True, write_svg=False, relpath="circle_recon", dpi=200.0),
	)

	write_unit_circle_recon_plot(
		output_png=out_png,
		output_svg=out_svg,
		template_ch_by_t=template_ch_by_t,
		locs_xy=locs_xy,
		gtr=gtr,
		circle_config=cfg,
		unit_id=1,
	)

	assert out_png.exists()
	assert out_png.stat().st_size > 0


def test_write_unit_amplitude_map_png_uses_shared_heatmap_config(tmp_path: Path) -> None:
	template_ch_by_t = np.array(
		[
			[-5.0, -10.0, -3.0],
			[-2.0, -4.0, -1.0],
			[-1.0, -6.0, -2.0],
		],
		dtype=float,
	)
	locs_xy = np.array(
		[
			[0.0, 0.0],
			[17.5, 0.0],
			[0.0, 17.5],
		],
		dtype=float,
	)
	out = tmp_path / "amplitude_map_cfg.png"

	cfg = SharedHeatmapConfig(
		background="white",
		show_colorbar=True,
		colorbar_location="bottomleft",
		show_ticks=(1, "dynamic_high"),
		high_color="red",
		mid_color="white",
		low_color="blue",
		scale="linear",
	)

	write_unit_amplitude_map_png(
		output_png=out,
		template_ch_by_t=template_ch_by_t,
		locs_xy=locs_xy,
		heatmap_config=cfg,
	)

	assert out.exists()
	assert out.stat().st_size > 0


def test_write_unit_amplitude_map_png_inverts_y_axis_when_enabled(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	template_ch_by_t = np.array(
		[
			[-5.0, -10.0, -3.0],
			[-2.0, -4.0, -1.0],
			[-1.0, -6.0, -2.0],
		],
		dtype=float,
	)
	locs_xy = np.array(
		[
			[0.0, 0.0],
			[17.5, 0.0],
			[0.0, 17.5],
		],
		dtype=float,
	)
	out = tmp_path / "amplitude_map_inverted.png"
	seen = {"count": 0}
	orig_invert_yaxis = matplotlib.axes.Axes.invert_yaxis

	def _spy_invert_yaxis(self, *args, **kwargs):
		seen["count"] += 1
		return orig_invert_yaxis(self, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "invert_yaxis", _spy_invert_yaxis)

	write_unit_amplitude_map_png(
		output_png=out,
		template_ch_by_t=template_ch_by_t,
		locs_xy=locs_xy,
		heatmap_config=SharedHeatmapConfig(invert_y_axis=True),
	)

	assert out.exists()
	assert seen["count"] >= 1


def test_write_unit_circle_recon_plot_branches_only_scope_uses_raw_and_remaps(monkeypatch, caplog) -> None:
	template_ch_by_t = np.array(
		[
			[-5.0, -10.0, -3.0, -1.0],
			[-2.0, -4.0, -1.0, -0.5],
			[-1.0, -6.0, -2.0, -0.5],
			[-1.0, -2.0, -1.0, -0.2],
		],
		dtype=float,
	)
	locs_xy = np.array(
		[
			[0.0, 0.0],
			[17.5, 0.0],
			[35.0, 0.0],
			[52.5, 0.0],
		],
		dtype=float,
	)

	class _GraphMock:
		def nodes(self):
			return [0, 2]

	class _GtrMock:
		def __init__(self):
			self._paths_raw = [[3, 1, 0]]
			self._paths_clean = [[0, 2]]
			self.branches = [{"branch_index": 7, "channels": [0, 2]}]
			self.graph = _GraphMock()

	gtr = _GtrMock()
	captured: dict[str, Any] = {}

	def _fake_render_template_circles_plot(**kwargs):
		captured["template"] = np.asarray(kwargs["template"], dtype=float)
		captured["locations_xy"] = np.asarray(kwargs["locations_xy"], dtype=float)
		captured["config"] = kwargs["config"]
		captured["branch_morphology"] = kwargs.get("branch_morphology")
		captured["branch_cfg"] = kwargs.get("branch_cfg")
		return {"template_circles_v2_png": "noop.png"}

	monkeypatch.setattr(
		"axon_recon.pipeline.stages.reconstruct.templates.core.render.render_template_circles_plot_v2",
		_fake_render_template_circles_plot,
	)

	cfg = CircleReconConfig(
		display=CircleReconDisplayConfig(
			base="template_circles",
			channel_scope="branches_only",
			zoom_padding_percent=12.0,
			invert_y_axis=False,
			force_center_soma=False,
			branch_scope="raw",
			unique_color_per_branch=False,
			show_branch_labels=True,
			color_scheme="tab10",
			node_border_linewidth=0.5,
			edge_linewidth=1.25,
		),
		output=CircleReconOutputConfig(write_png=False, write_svg=False, relpath="circle_recon", dpi=200.0),
	)
	caplog.set_level("INFO", logger="axon_recon.reconstruct")

	write_unit_circle_recon_plot(
		output_png=Path("/tmp/noop.png"),
		output_svg=Path("/tmp/noop.svg"),
		template_ch_by_t=template_ch_by_t,
		locs_xy=locs_xy,
		gtr=gtr,
		circle_config=cfg,
		unit_id=1,
	)

	assert captured["template"].shape == (3, 4)
	assert captured["locations_xy"].shape == (3, 2)
	np.testing.assert_allclose(captured["template"][0, :], template_ch_by_t[0, :])
	np.testing.assert_allclose(captured["template"][1, :], template_ch_by_t[1, :])
	np.testing.assert_allclose(captured["template"][2, :], template_ch_by_t[3, :])

	branch_payload = captured["branch_morphology"]
	assert isinstance(branch_payload, dict)
	branches = branch_payload.get("branches")
	assert branches == [{"branch_index": 0, "channels": [0, 1, 2], "label": 0, "color": "#1f77b4"}]

	branch_cfg = captured["branch_cfg"]
	assert bool(branch_cfg.enabled) is True
	assert bool(branch_cfg.unique_color_per_branch) is False
	assert bool(branch_cfg.show_branch_labels) is True
	assert str(branch_cfg.color_scheme) == "tab10"
	assert float(branch_cfg.node_border_linewidth) == 0.5
	assert float(branch_cfg.edge_linewidth) == 1.25
	render_cfg = captured["config"]
	assert bool(render_cfg.invert_y_axis) is False
	assert "branch_scope=raw" in caplog.text
	assert "source=gtr._paths_raw" in caplog.text
	assert "selected_branch_class=list" in caplog.text


def test_write_unit_circle_recon_plot_nodes_only_scope_uses_clean_payload(monkeypatch, caplog) -> None:
	template_ch_by_t = np.array(
		[
			[-5.0, -10.0, -3.0, -1.0],
			[-2.0, -4.0, -1.0, -0.5],
			[-1.0, -6.0, -2.0, -0.5],
			[-1.0, -2.0, -1.0, -0.2],
		],
		dtype=float,
	)
	locs_xy = np.array(
		[
			[0.0, 0.0],
			[17.5, 0.0],
			[35.0, 0.0],
			[52.5, 0.0],
		],
		dtype=float,
	)

	class _GraphMock:
		def nodes(self):
			return [0, 2]

	class _GtrMock:
		def __init__(self):
			self._paths_raw = [[3, 1, 0]]
			self._paths_clean = [[0, 2]]
			self.branches = [{"branch_index": 7, "channels": [0, 2]}]
			self.graph = _GraphMock()

	gtr = _GtrMock()
	captured: dict[str, Any] = {}

	def _fake_render_template_circles_plot(**kwargs):
		captured["template"] = np.asarray(kwargs["template"], dtype=float)
		captured["locations_xy"] = np.asarray(kwargs["locations_xy"], dtype=float)
		captured["branch_morphology"] = kwargs.get("branch_morphology")
		return {"template_circles_png": "noop.png"}

	monkeypatch.setattr(
		"axon_recon.pipeline.stages.reconstruct.templates.core.render.render_template_circles_plot_v2",
		_fake_render_template_circles_plot,
	)

	cfg = CircleReconConfig(
		display=CircleReconDisplayConfig(
			base="template_circles",
			channel_scope="nodes_only",
			force_center_soma=True,
			branch_scope="clean",
		),
		output=CircleReconOutputConfig(write_png=False, write_svg=False, relpath="circle_recon", dpi=200.0),
	)
	caplog.set_level("INFO", logger="axon_recon.reconstruct")

	write_unit_circle_recon_plot(
		output_png=Path("/tmp/noop.png"),
		output_svg=Path("/tmp/noop.svg"),
		template_ch_by_t=template_ch_by_t,
		locs_xy=locs_xy,
		gtr=gtr,
		circle_config=cfg,
		unit_id=1,
	)

	assert captured["template"].shape == (2, 4)
	assert captured["locations_xy"].shape == (2, 2)
	np.testing.assert_allclose(captured["template"][0, :], template_ch_by_t[0, :])
	np.testing.assert_allclose(captured["template"][1, :], template_ch_by_t[2, :])

	branch_payload = captured["branch_morphology"]
	assert isinstance(branch_payload, dict)
	branches = branch_payload.get("branches")
	assert len(branches) == 1
	assert branches[0]["branch_index"] == 7
	assert branches[0]["channels"] == [0, 1]
	assert branches[0]["label"] == 7
	assert isinstance(branches[0]["color"], str)
	assert branches[0]["color"].startswith("#")
	assert "branch_scope=clean" in caplog.text
	assert "source=gtr.branches" in caplog.text
	assert "selected_branch_class=dict" in caplog.text


def test_write_unit_circle_recon_plot_selected_channels_scope_uses_filtered_channels(monkeypatch, caplog) -> None:
	template_ch_by_t = np.array(
		[
			[-5.0, -10.0, -3.0, -1.0],
			[-2.0, -4.0, -1.0, -0.5],
			[-1.0, -6.0, -2.0, -0.5],
			[-1.0, -2.0, -1.0, -0.2],
		],
		dtype=float,
	)
	locs_xy = np.array(
		[
			[0.0, 0.0],
			[17.5, 0.0],
			[35.0, 0.0],
			[52.5, 0.0],
		],
		dtype=float,
	)

	class _GraphMock:
		def nodes(self):
			return [0, 1, 2]

	class _GtrMock:
		def __init__(self):
			self._paths_raw = [[3, 2, 0]]
			self._paths_clean = [[0, 2]]
			self.branches = [{"branch_index": 4, "channels": [0, 2]}]
			self.selected_channels = [0, 2, 3]
			self.graph = _GraphMock()

	gtr = _GtrMock()
	captured: dict[str, Any] = {}

	def _fake_render_template_circles_plot(**kwargs):
		captured["template"] = np.asarray(kwargs["template"], dtype=float)
		captured["locations_xy"] = np.asarray(kwargs["locations_xy"], dtype=float)
		captured["branch_morphology"] = kwargs.get("branch_morphology")
		captured["branch_cfg"] = kwargs.get("branch_cfg")
		captured["config"] = kwargs.get("config")
		return {"template_circles_v2_png": "noop.png"}

	monkeypatch.setattr(
		"axon_recon.pipeline.stages.reconstruct.templates.core.render.render_template_circles_plot_v2",
		_fake_render_template_circles_plot,
	)

	cfg = CircleReconConfig(
		display=CircleReconDisplayConfig(
			base="template_circles",
			channel_scope="selected_channels",
			force_center_soma=False,
			branch_scope="raw",
			show_branch_legend=True,
		),
		output=CircleReconOutputConfig(write_png=False, write_svg=False, relpath="circle_recon", dpi=200.0),
	)
	caplog.set_level("INFO", logger="axon_recon.reconstruct")

	write_unit_circle_recon_plot(
		output_png=Path("/tmp/noop.png"),
		output_svg=Path("/tmp/noop.svg"),
		template_ch_by_t=template_ch_by_t,
		locs_xy=locs_xy,
		gtr=gtr,
		circle_config=cfg,
		unit_id=1,
	)

	assert captured["template"].shape == (3, 4)
	assert captured["locations_xy"].shape == (3, 2)
	np.testing.assert_allclose(captured["template"][0, :], template_ch_by_t[0, :])
	np.testing.assert_allclose(captured["template"][1, :], template_ch_by_t[2, :])
	np.testing.assert_allclose(captured["template"][2, :], template_ch_by_t[3, :])
	assert bool(captured["branch_cfg"].show_branch_legend) is True
	assert captured["branch_morphology"] == {
		"branches": [{"branch_index": 0, "channels": [0, 1, 2], "label": 0, "color": "#1f77b4"}]
	}
	assert "branch_scope=raw" in caplog.text
	assert "source=gtr._paths_raw" in caplog.text


def test_write_unit_circle_recon_plot_clean_scope_falls_back_to_paths_clean(monkeypatch, caplog) -> None:
	template_ch_by_t = np.array(
		[
			[-5.0, -10.0, -3.0, -1.0],
			[-2.0, -4.0, -1.0, -0.5],
			[-1.0, -6.0, -2.0, -0.5],
			[-1.0, -2.0, -1.0, -0.2],
		],
		dtype=float,
	)
	locs_xy = np.array(
		[
			[0.0, 0.0],
			[17.5, 0.0],
			[35.0, 0.0],
			[52.5, 0.0],
		],
		dtype=float,
	)

	class _GraphMock:
		def nodes(self):
			return [0, 1, 2, 3]

	class _GtrMock:
		def __init__(self):
			self._paths_raw = [[3, 1, 0], [3, 2]]
			self._paths_clean = [[0, 2]]
			self.branches = None
			self.graph = _GraphMock()

	gtr = _GtrMock()
	captured: dict[str, Any] = {}

	def _fake_render_template_circles_plot(**kwargs):
		captured["branch_morphology"] = kwargs.get("branch_morphology")
		return {"template_circles_png": "noop.png"}

	monkeypatch.setattr(
		"axon_recon.pipeline.stages.reconstruct.templates.core.render.render_template_circles_plot_v2",
		_fake_render_template_circles_plot,
	)

	cfg = CircleReconConfig(
		display=CircleReconDisplayConfig(
			base="template_circles",
			channel_scope="nodes_and_branches",
			force_center_soma=True,
			branch_scope="clean",
		),
		output=CircleReconOutputConfig(write_png=False, write_svg=False, relpath="circle_recon", dpi=200.0),
	)
	caplog.set_level("INFO", logger="axon_recon.reconstruct")

	write_unit_circle_recon_plot(
		output_png=Path("/tmp/noop.png"),
		output_svg=Path("/tmp/noop.svg"),
		template_ch_by_t=template_ch_by_t,
		locs_xy=locs_xy,
		gtr=gtr,
		circle_config=cfg,
		unit_id=1,
	)

	branch_payload = captured["branch_morphology"]
	assert isinstance(branch_payload, dict)
	assert branch_payload.get("branches") == [{"branch_index": 0, "channels": [0, 2], "label": 0, "color": "#1f77b4"}]
	assert "branch_scope=clean" in caplog.text
	assert "source=gtr._paths_clean" in caplog.text
	assert "selected_branch_class=list" in caplog.text


def test_write_unit_circle_recon_plot_base_amplitude_map_dispatches(monkeypatch) -> None:
	template_ch_by_t = np.array(
		[
			[-5.0, -10.0, -3.0, -1.0],
			[-2.0, -4.0, -1.0, -0.5],
			[-1.0, -6.0, -2.0, -0.5],
			[-1.0, -2.0, -1.0, -0.2],
		],
		dtype=float,
	)
	locs_xy = np.array(
		[
			[0.0, 0.0],
			[17.5, 0.0],
			[35.0, 0.0],
			[52.5, 0.0],
		],
		dtype=float,
	)

	class _GraphMock:
		def nodes(self):
			return [0, 2]

	class _GtrMock:
		def __init__(self):
			self._paths_raw = [[3, 1, 0]]
			self.branches = [{"branch_index": 7, "channels": [0, 2]}]
			self.graph = _GraphMock()

	gtr = _GtrMock()
	captured: dict[str, Any] = {}

	def _fake_render_footprint_amplitude_map(**kwargs):
		captured["kwargs"] = kwargs
		return {"footprint_amplitude_map_png": "noop.png"}

	def _should_not_call(**kwargs):
		raise AssertionError("unexpected renderer called")

	monkeypatch.setattr(
		"axon_recon.pipeline.stages.reconstruct.templates.core.render.render_footprint_amplitude_map",
		_fake_render_footprint_amplitude_map,
	)
	monkeypatch.setattr(
		"axon_recon.pipeline.stages.reconstruct.templates.core.render.render_footprint_latency_map",
		_should_not_call,
	)
	monkeypatch.setattr(
		"axon_recon.pipeline.stages.reconstruct.templates.core.render.render_template_circles_plot_v2",
		_should_not_call,
	)

	cfg = CircleReconConfig(
		display=CircleReconDisplayConfig(
			base="amplitude_map",
			channel_scope="branches_only",
			invert_y_axis=False,
			force_center_soma=True,
			branch_scope="raw",
			unique_color_per_branch=True,
			show_branch_labels=True,
			color_scheme="Set1",
			node_outline_color="white",
				node_outline_linewidth=2.5,
			branch_outline_color="white",
				branch_outline_linewidth=1.5,
			node_border_linewidth=0.55,
			edge_linewidth=1.5,
		),
		output=CircleReconOutputConfig(write_png=True, write_svg=False, relpath="maps/circle_recon_amp", dpi=250.0),
	)

	out = write_unit_circle_recon_plot(
		output_png=Path("/tmp/noop.png"),
		output_svg=Path("/tmp/noop.svg"),
		template_ch_by_t=template_ch_by_t,
		locs_xy=locs_xy,
		gtr=gtr,
		circle_config=cfg,
		unit_id=1,
	)

	assert out == {"footprint_amplitude_map_png": "noop.png"}
	kwargs = captured["kwargs"]
	np.testing.assert_allclose(np.asarray(kwargs["template"], dtype=float), template_ch_by_t[[0, 1, 3], :])
	np.testing.assert_allclose(np.asarray(kwargs["locations_xy"], dtype=float), locs_xy[[0, 1, 3], :])
	assert kwargs["branch_morphology"] == {
		"branches": [{"branch_index": 0, "channels": [0, 1, 2], "label": 0, "color": "#e41a1c"}]
	}
	branch_cfg = kwargs["branch_cfg"]
	assert bool(branch_cfg.enabled) is True
	assert str(branch_cfg.node_outline_color) == "white"
	assert float(branch_cfg.node_outline_linewidth) == 2.5
	assert str(branch_cfg.branch_outline_color) == "white"
	assert float(branch_cfg.branch_outline_linewidth) == 1.5
	assert float(branch_cfg.node_border_linewidth) == 0.55
	assert float(branch_cfg.edge_linewidth) == 1.5
	assert bool(kwargs["config"].invert_y_axis) is False
	assert bool(kwargs["config"].write_png) is True
	assert bool(kwargs["config"].write_svg) is False
	assert str(kwargs["config"].relpath) == "maps/circle_recon_amp"


def test_write_unit_circle_recon_plot_base_latency_map_dispatches(monkeypatch) -> None:
	template_ch_by_t = np.array(
		[
			[-5.0, -10.0, -3.0, -1.0],
			[-2.0, -4.0, -1.0, -0.5],
			[-1.0, -6.0, -2.0, -0.5],
			[-1.0, -2.0, -1.0, -0.2],
		],
		dtype=float,
	)
	locs_xy = np.array(
		[
			[0.0, 0.0],
			[17.5, 0.0],
			[35.0, 0.0],
			[52.5, 0.0],
		],
		dtype=float,
	)

	class _GraphMock:
		def nodes(self):
			return [0, 2]

	class _GtrMock:
		def __init__(self):
			self._paths_raw = [[3, 1, 0]]
			self.branches = [{"branch_index": 7, "channels": [0, 2]}]
			self.graph = _GraphMock()

	gtr = _GtrMock()
	captured: dict[str, Any] = {}

	def _fake_render_footprint_latency_map(**kwargs):
		captured["kwargs"] = kwargs
		return {"footprint_latency_map_png": "noop.png"}

	def _should_not_call(**kwargs):
		raise AssertionError("unexpected renderer called")

	monkeypatch.setattr(
		"axon_recon.pipeline.stages.reconstruct.templates.core.render.render_footprint_latency_map",
		_fake_render_footprint_latency_map,
	)
	monkeypatch.setattr(
		"axon_recon.pipeline.stages.reconstruct.templates.core.render.render_footprint_amplitude_map",
		_should_not_call,
	)
	monkeypatch.setattr(
		"axon_recon.pipeline.stages.reconstruct.templates.core.render.render_template_circles_plot_v2",
		_should_not_call,
	)

	cfg = CircleReconConfig(
		display=CircleReconDisplayConfig(
			base="latency_map",
			channel_scope="nodes_only",
			invert_y_axis=False,
			force_center_soma=True,
			branch_scope="clean",
			unique_color_per_branch=False,
			show_branch_labels=False,
			color_scheme="tab10",
			node_border_linewidth=0.5,
			edge_linewidth=1.25,
		),
		output=CircleReconOutputConfig(write_png=True, write_svg=True, relpath="maps/circle_recon_lat", dpi=250.0),
	)

	out = write_unit_circle_recon_plot(
		output_png=Path("/tmp/noop.png"),
		output_svg=Path("/tmp/noop.svg"),
		template_ch_by_t=template_ch_by_t,
		locs_xy=locs_xy,
		gtr=gtr,
		circle_config=cfg,
		unit_id=1,
	)

	assert out == {"footprint_latency_map_png": "noop.png"}
	kwargs = captured["kwargs"]
	np.testing.assert_allclose(np.asarray(kwargs["template"], dtype=float), template_ch_by_t[[0, 2], :])
	np.testing.assert_allclose(np.asarray(kwargs["locations_xy"], dtype=float), locs_xy[[0, 2], :])
	assert kwargs["branch_morphology"] == {
		"branches": [{"branch_index": 7, "channels": [0, 1], "label": 7, "color": "#1f77b4"}]
	}
	branch_cfg = kwargs["branch_cfg"]
	assert bool(branch_cfg.enabled) is True
	assert bool(branch_cfg.unique_color_per_branch) is False
	assert bool(branch_cfg.show_branch_labels) is False
	assert str(branch_cfg.color_scheme) == "tab10"
	assert float(branch_cfg.node_border_linewidth) == 0.5
	assert float(branch_cfg.edge_linewidth) == 1.25
	assert bool(kwargs["config"].invert_y_axis) is False
	assert bool(kwargs["config"].write_png) is True
	assert bool(kwargs["config"].write_svg) is True
	assert str(kwargs["config"].relpath) == "maps/circle_recon_lat"
