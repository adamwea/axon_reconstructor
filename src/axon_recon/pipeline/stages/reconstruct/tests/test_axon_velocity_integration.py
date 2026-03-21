from __future__ import annotations

from axon_recon.pipeline.stages.reconstruct.integrations.axon_velocity import _filter_kwargs_for_callable


def test_filter_kwargs_for_callable() -> None:
	def fn(a: int, b: int) -> int:
		return a + b

	filtered = _filter_kwargs_for_callable(fn, {"a": 1, "b": 2, "c": 3})
	assert filtered == {"a": 1, "b": 2}

