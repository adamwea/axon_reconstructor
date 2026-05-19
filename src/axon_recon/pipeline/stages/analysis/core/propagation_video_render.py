"""Per-unit propagation-video render core.

Slice 4 of `analysis_propagation_video_plan.md` (minimal v1 port).
Wraps `axon_velocity.plotting.play_template_map` to produce a GIF of
extracellular template propagation along the unit's reconstructed
branches. Soft-imports `axon_velocity` + `matplotlib.animation` at
call time so the analysis stage loads cleanly even when these are
unavailable in the active env (e.g. base axon_recon conda env without
the `[full]` extra).

Slice-1 archeology audit
(`dev/notes/refs/propagation_video_audit.md`) identified the v1
implementation at retired `axon_reconstructor/pipeline/reconstruction/
plotting.py:1655-1810`. This minimal port covers the central
play_template_map → PillowWriter chain; v1's crop / clip-quantile /
colorbar / time-counter elaborations are intentionally deferred to
follow-up slices once the user has verified the basic output against
the slice-8 HARD-gate visual diagnostic.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .propagation_video_inputs import PropagationVideoInputs


_DEFAULT_CMAP = "coolwarm"
_DEFAULT_SKIP_FRAMES = 2
_DEFAULT_FPS = 20
_DEFAULT_FIGSIZE = (7.2, 6.2)


class PropagationVideoRenderUnavailable(RuntimeError):
	"""Raised when `axon_velocity` isn't importable — slice-1 plan calls
	this out as a soft dep so the analysis stage's other phases can run
	without it.
	"""


def _require_axon_velocity() -> Any:
	try:
		from axon_velocity.plotting import play_template_map  # type: ignore[import-not-found]
	except Exception as exc:
		raise PropagationVideoRenderUnavailable(
			"propagation_video: `axon_velocity` is not importable. Install "
			"`axon_velocity==0.1.2` (already in pyproject.toml's [full] "
			"extra) or run inside the shifter image."
		) from exc
	return play_template_map


def _require_pillow_writer() -> Any:
	try:
		from matplotlib.animation import PillowWriter
	except Exception as exc:
		raise PropagationVideoRenderUnavailable(
			"propagation_video: matplotlib.animation.PillowWriter is "
			"unavailable. Install matplotlib's animation extras."
		) from exc
	return PillowWriter


def _load_inputs(inputs: PropagationVideoInputs) -> tuple[np.ndarray, np.ndarray, Any]:
	"""Materialize template + locations + GTR from disk.

	Returns ``(template, locations_xy, gtr)``. ``gtr`` is the pickled
	``axon_velocity.GraphTreeRepresentation`` (or whatever the recon
	stage's axon_velocity_gtrs phase wrote).
	"""

	template = np.asarray(np.load(inputs.merged_template_npy), dtype=float)
	locations_xy = np.asarray(np.load(inputs.merged_locations_npy), dtype=float)
	with inputs.gtr_pkl.open("rb") as fh:
		gtr = pickle.load(fh)
	return template, locations_xy, gtr


def render_unit_propagation_video(
	*,
	inputs: PropagationVideoInputs,
	out_path: Path,
	cmap: str = _DEFAULT_CMAP,
	skip_frames: int = _DEFAULT_SKIP_FRAMES,
	fps: int = _DEFAULT_FPS,
	figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
	force_restart: bool = False,
	# Test-only seam: allows tests to inject a mock animator without
	# requiring matplotlib + axon_velocity in the env.
	_play_template_map_override: Any = None,
	_pillow_writer_override: Any = None,
) -> dict[str, Any]:
	"""Render one unit's propagation video to ``out_path``.

	Idempotent: when ``out_path`` already exists AND ``force_restart``
	is False, returns a ``status: skipped, reason: output_exists``
	marker without re-rendering.

	Returns a dict that the orchestrator (slice 7) bakes into the
	per-target summary marker.
	"""

	out_path = Path(out_path)
	if out_path.is_file() and not force_restart:
		return {
			"status": "skipped",
			"reason": "output_exists",
			"out_path": str(out_path),
			"unit_id": int(inputs.unit_id),
		}

	play_template_map = _play_template_map_override or _require_axon_velocity()
	pillow_writer_cls = _pillow_writer_override or _require_pillow_writer()

	# Defer matplotlib import to call time so the analysis-stage module
	# loads without it in non-render contexts (e.g. dry-run / scaffold).
	import matplotlib

	matplotlib.use("Agg")  # headless write-to-file
	import matplotlib.pyplot as plt

	template, locations_xy, gtr = _load_inputs(inputs)

	fig = plt.figure(figsize=tuple(figsize))
	ax = fig.add_subplot(111)
	try:
		anim = play_template_map(
			template,
			locations_xy,
			gtr=gtr,
			cmap=cmap,
			skip_frames=int(skip_frames),
			ax=ax,
		)
		out_path.parent.mkdir(parents=True, exist_ok=True)
		writer = pillow_writer_cls(fps=int(fps))
		anim.save(str(out_path), writer=writer)
	finally:
		plt.close(fig)

	return {
		"status": "ok",
		"reason": "rendered",
		"out_path": str(out_path),
		"unit_id": int(inputs.unit_id),
		"frames": int(template.shape[1] // max(1, int(skip_frames))) if template.ndim >= 2 else 0,
		"cmap": str(cmap),
		"fps": int(fps),
		"skip_frames": int(skip_frames),
	}


__all__ = [
	"PropagationVideoRenderUnavailable",
	"render_unit_propagation_video",
]
