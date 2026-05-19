"""Slice 4 of analysis_propagation_video_plan: render core tests.

Mocks `axon_velocity.plotting.play_template_map` and
`matplotlib.animation.PillowWriter` so the tests run without
axon_velocity installed. A real-data smoke is the user's domain (run
inside shifter or `[full]`-installed env).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from axon_recon.pipeline.stages.analysis.core.propagation_video_inputs import (
	PropagationVideoInputs,
)
from axon_recon.pipeline.stages.analysis.core.propagation_video_render import (
	PropagationVideoRenderUnavailable,
	render_unit_propagation_video,
)


def _scaffold_unit_inputs(tmp_path: Path) -> PropagationVideoInputs:
	"""Write minimal template + locations + GTR pickle to disk and
	return a PropagationVideoInputs pointing at them."""

	unit_dir = tmp_path / "recon_outputs" / "cache" / "templates" / "merged" / "unit_5"
	unit_dir.mkdir(parents=True, exist_ok=True)
	gtr_dir = tmp_path / "recon_outputs" / "units" / "0005"
	gtr_dir.mkdir(parents=True, exist_ok=True)

	# (n_channels=8, n_samples=82): smallest synthetic shape that
	# matches axon_velocity's expected (channels, samples) layout.
	template = np.random.RandomState(0).randn(8, 82).astype(float)
	np.save(unit_dir / "merged_template.npy", template)
	# (n_channels, 2): 2D channel positions.
	locations = np.column_stack([
		np.arange(8, dtype=float) * 17.5,
		np.zeros(8, dtype=float),
	])
	np.save(unit_dir / "merged_channel_locations.npy", locations)
	# Minimal GTR-shaped pickle.
	gtr_obj = SimpleNamespace(
		branches=[{"channels": [0, 1, 2]}, {"channels": [3, 4, 5]}],
		locations=locations,
	)
	with (gtr_dir / "gtr.pkl").open("wb") as fh:
		pickle.dump(gtr_obj, fh)
	(gtr_dir / "gtr.json").write_text("{}", encoding="utf-8")

	return PropagationVideoInputs(
		dataset_index=0,
		well_id="well000",
		unit_id=5,
		well_out_dir=tmp_path,
		recon_output_dir=tmp_path / "recon_outputs",
		merged_template_npy=unit_dir / "merged_template.npy",
		merged_locations_npy=unit_dir / "merged_channel_locations.npy",
		merged_meta_json=unit_dir / "unit_templates_summary.json",
		gtr_pkl=gtr_dir / "gtr.pkl",
		gtr_json=gtr_dir / "gtr.json",
	)


def _mock_play_template_map() -> Any:
	"""Mock that returns a MagicMock animation whose `.save(path, writer=...)`
	just touches the file."""
	mock = MagicMock()

	def _factory(template, locations, *, gtr=None, cmap, skip_frames, ax):
		anim = MagicMock()
		# Save creates the output file so the idempotency check can find it.
		def _save(path, writer=None):
			Path(path).parent.mkdir(parents=True, exist_ok=True)
			Path(path).write_bytes(b"GIF89a-mock")
		anim.save = _save
		mock.last_call_args = {
			"template_shape": tuple(np.asarray(template).shape),
			"locations_shape": tuple(np.asarray(locations).shape),
			"cmap": str(cmap),
			"skip_frames": int(skip_frames),
		}
		return anim

	mock.side_effect = _factory
	return mock


def _mock_pillow_writer_cls() -> Any:
	return MagicMock(return_value=MagicMock())


def test_render_writes_gif_and_returns_ok(tmp_path: Path) -> None:
	inputs = _scaffold_unit_inputs(tmp_path)
	out_path = tmp_path / "video.gif"

	mock_play = _mock_play_template_map()
	result = render_unit_propagation_video(
		inputs=inputs,
		out_path=out_path,
		_play_template_map_override=mock_play,
		_pillow_writer_override=_mock_pillow_writer_cls(),
	)

	assert result["status"] == "ok"
	assert result["reason"] == "rendered"
	assert result["out_path"] == str(out_path)
	assert result["unit_id"] == 5
	assert out_path.is_file()
	# Mock recorded the play_template_map call signature.
	assert mock_play.last_call_args["template_shape"] == (8, 82)
	assert mock_play.last_call_args["locations_shape"] == (8, 2)
	assert mock_play.last_call_args["cmap"] == "coolwarm"


def test_render_idempotent_when_output_exists(tmp_path: Path) -> None:
	inputs = _scaffold_unit_inputs(tmp_path)
	out_path = tmp_path / "video.gif"
	out_path.write_bytes(b"existing")

	# Mock should NOT be called when output already exists.
	mock_play = _mock_play_template_map()
	result = render_unit_propagation_video(
		inputs=inputs,
		out_path=out_path,
		_play_template_map_override=mock_play,
		_pillow_writer_override=_mock_pillow_writer_cls(),
	)

	assert result["status"] == "skipped"
	assert result["reason"] == "output_exists"
	# Existing file is untouched.
	assert out_path.read_bytes() == b"existing"


def test_force_restart_overwrites_existing_output(tmp_path: Path) -> None:
	inputs = _scaffold_unit_inputs(tmp_path)
	out_path = tmp_path / "video.gif"
	out_path.write_bytes(b"existing")

	result = render_unit_propagation_video(
		inputs=inputs,
		out_path=out_path,
		force_restart=True,
		_play_template_map_override=_mock_play_template_map(),
		_pillow_writer_override=_mock_pillow_writer_cls(),
	)

	assert result["status"] == "ok"
	# Mock wrote new bytes overwriting the existing file.
	assert out_path.read_bytes() == b"GIF89a-mock"


def test_render_raises_when_axon_velocity_missing(tmp_path: Path, monkeypatch) -> None:
	"""Default code path (no override) imports axon_velocity. When that
	import fails, the renderer raises PropagationVideoRenderUnavailable
	with an actionable message."""

	inputs = _scaffold_unit_inputs(tmp_path)
	out_path = tmp_path / "video.gif"

	# Stash None for axon_velocity.plotting so the soft-import path
	# raises.
	monkeypatch.setitem(sys.modules, "axon_velocity", None)
	monkeypatch.setitem(sys.modules, "axon_velocity.plotting", None)
	with pytest.raises(PropagationVideoRenderUnavailable) as exc_info:
		render_unit_propagation_video(
			inputs=inputs,
			out_path=out_path,
		)
	assert "axon_velocity" in str(exc_info.value)


def test_render_passes_skip_frames_and_fps_through(tmp_path: Path) -> None:
	inputs = _scaffold_unit_inputs(tmp_path)
	out_path = tmp_path / "video.gif"

	mock_play = _mock_play_template_map()
	mock_writer_cls = MagicMock(return_value=MagicMock())
	render_unit_propagation_video(
		inputs=inputs,
		out_path=out_path,
		skip_frames=4,
		fps=10,
		_play_template_map_override=mock_play,
		_pillow_writer_override=mock_writer_cls,
	)

	assert mock_play.last_call_args["skip_frames"] == 4
	# PillowWriter constructor received fps=10.
	mock_writer_cls.assert_called_once()
	_, kwargs = mock_writer_cls.call_args
	assert kwargs.get("fps") == 10
