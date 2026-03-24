from __future__ import annotations

import numpy as np

from axon_recon.pipeline.stages.templates.integrations.spikeinterface_extract import (
	build_unit_source_payload,
)


class _MockTemplatesExtension:
	def __init__(self, template: np.ndarray) -> None:
		self._template = np.asarray(template, dtype=float)

	def get_unit_template(self, *, unit_id: int) -> np.ndarray:
		_ = unit_id
		return self._template


class _MockTemplatesExtensionArrayOnly:
	def __init__(self, templates: np.ndarray) -> None:
		self._templates = np.asarray(templates, dtype=float)

	def get_templates(self) -> np.ndarray:
		return self._templates


class _MockSparsity:
	def __init__(self, mapping: dict[object, list[int]]) -> None:
		self.unit_id_to_channel_indices = mapping


class _MockRecording:
	def __init__(self) -> None:
		self._locations = np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=float)
		self._electrode_ids = [100, 101, 102]
		self._channel_ids = [0, 1, 2]

	def get_channel_locations(self) -> np.ndarray:
		return self._locations

	def get_property_keys(self) -> list[str]:
		return ["electrode_id"]

	def get_property(self, key: str):
		if key == "electrode_id":
			return self._electrode_ids
		return None

	def get_channel_ids(self) -> list[int]:
		return self._channel_ids

	def get_sampling_frequency(self) -> float:
		return 10_000.0


class _MockSorting:
	def __init__(self, unit_ids: list[int]) -> None:
		self.unit_ids = unit_ids

	def get_num_segments(self) -> int:
		return 2

	def get_unit_spike_train(self, *, unit_id: int, segment_index: int):
		_ = unit_id
		if segment_index == 0:
			return np.asarray([1, 2, 3], dtype=int)
		return np.asarray([4, 5], dtype=int)


class _MockAnalyzer:
	def __init__(self, *, templates_ext, has_templates: bool = True) -> None:
		self.recording = _MockRecording()
		self.sorting = _MockSorting(unit_ids=[94])
		self.sparsity = _MockSparsity(mapping={"94": [2, 0]})
		self._templates_ext = templates_ext
		self._has_templates = bool(has_templates)

	def has_extension(self, name: str) -> bool:
		if name == "templates":
			return self._has_templates
		return False

	def get_extension(self, name: str):
		if name == "templates":
			return self._templates_ext
		raise KeyError(name)

	def compute(self, names, verbose: bool = False, n_jobs: int = 1) -> None:
		_ = names, verbose, n_jobs
		# Simulate unavailable template computation path.
		self._has_templates = False


def test_build_unit_source_payload_contract_with_get_unit_template() -> None:
	template_time_by_ch = np.asarray(
		[
			[1.0, 3.0],
			[2.0, 4.0],
			[0.0, 0.0],
			[0.0, 0.0],
		],
		dtype=float,
	)
	analyzer = _MockAnalyzer(templates_ext=_MockTemplatesExtension(template_time_by_ch), has_templates=True)

	payload = build_unit_source_payload(analyzer=analyzer, unit_id=94)
	assert payload is not None
	t_ch_by_t, locs_xy, electrode_ids, channel_ids, waveform_count, sampling_rate_hz, top_wf, top_id, top_count = payload

	assert t_ch_by_t.shape == (2, 4)
	np.testing.assert_allclose(locs_xy, np.asarray([[20.0, 0.0], [0.0, 0.0]], dtype=float))
	assert electrode_ids == [102, 100]
	assert channel_ids == [2, 0]
	assert waveform_count == 5
	assert sampling_rate_hz == 10_000.0
	assert top_wf is None
	assert top_id in {100, 102}
	assert top_count is None


def test_build_unit_source_payload_contract_with_get_templates_fallback() -> None:
	templates = np.asarray(
		[
			[
				[1.0, 2.0, 0.0],
				[0.5, 1.0, 0.0],
			],
		],
		dtype=float,
	)
	analyzer = _MockAnalyzer(templates_ext=_MockTemplatesExtensionArrayOnly(templates), has_templates=True)

	payload = build_unit_source_payload(analyzer=analyzer, unit_id=94)
	assert payload is not None
	t_ch_by_t, _, _, _, _, _, top_wf, top_id, top_count = payload
	assert t_ch_by_t.shape[0] >= 1
	assert top_wf is None
	assert top_id is not None
	assert top_count is None


def test_build_unit_source_payload_returns_none_when_templates_unavailable() -> None:
	analyzer = _MockAnalyzer(templates_ext=_MockTemplatesExtension(np.zeros((2, 2), dtype=float)), has_templates=False)
	payload = build_unit_source_payload(analyzer=analyzer, unit_id=94)
	assert payload is None
