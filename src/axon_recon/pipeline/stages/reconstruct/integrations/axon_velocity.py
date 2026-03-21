from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path
import sys
from typing import Any


def import_axon_velocity(repo_root: Path | None = None) -> Any:
	try:
		import axon_velocity as av  # type: ignore[import-not-found]

		return av
	except Exception as first_error:
		fallback = Path(repo_root) if repo_root is not None else Path("/home/adamm/dev/pkgs/axon_velocity")
		if (fallback / "axon_velocity").exists():
			if str(fallback) not in sys.path:
				sys.path.insert(0, str(fallback))
			import axon_velocity as av  # type: ignore[import-not-found]

			return av
		raise RuntimeError(
			f"Reconstruction requires axon_velocity; attempted fallback path: {fallback}"
		) from first_error


def _filter_kwargs_for_callable(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
	try:
		sig = inspect.signature(fn)
		allowed = set(sig.parameters.keys())
		return {k: v for k, v in kwargs.items() if k in allowed}
	except Exception:
		return dict(kwargs)


def get_default_graph_velocity_params(av: Any) -> dict[str, Any]:
	try:
		fn = getattr(av, "get_default_graph_velocity_params", None)
		if callable(fn):
			out = fn()
			return dict(out) if isinstance(out, dict) else {}
	except Exception:
		pass

	try:
		sub = getattr(av, "axon_velocity", None)
		fn_sub = getattr(sub, "get_default_graph_velocity_params", None)
		if callable(fn_sub):
			out = fn_sub()
			return dict(out) if isinstance(out, dict) else {}
	except Exception:
		pass

	try:
		cls = getattr(av, "GraphAxonTracking", None)
		defaults = getattr(cls, "default_params", None)
		if isinstance(defaults, dict):
			return deepcopy(defaults)
	except Exception:
		pass

	return {}


def compute_graph_tracking(
	*,
	av: Any,
	template_ch_by_t: Any,
	locs_xy: Any,
	sampling_frequency_hz: float,
	params: dict[str, Any],
) -> Any:
	effective = get_default_graph_velocity_params(av)
	effective.update(dict(params))
	filtered = _filter_kwargs_for_callable(av.compute_graph_propagation_velocity, effective)
	return av.compute_graph_propagation_velocity(template_ch_by_t, locs_xy, float(sampling_frequency_hz), **filtered)

