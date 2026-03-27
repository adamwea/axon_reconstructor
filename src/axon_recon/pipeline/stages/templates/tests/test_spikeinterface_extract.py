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


class _MockWaveformsExtension:
	def __init__(self, waveforms: np.ndarray, *, ms_before: float = 1.0, ms_after: float = 2.0) -> None:
		self._waveforms = np.asarray(waveforms, dtype=float)
		self.params = {"waveforms": {"ms_before": float(ms_before), "ms_after": float(ms_after)}}

	def get_waveforms_one_unit(self, *, unit_id: int, force_dense: bool = False) -> np.ndarray:
		_ = unit_id, force_dense
		return np.asarray(self._waveforms, dtype=float)

	def set_waveforms(self, waveforms: np.ndarray) -> None:
		self._waveforms = np.asarray(waveforms, dtype=float)


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
	def __init__(
		self,
		*,
		templates_ext,
		has_templates: bool = True,
		waveforms: np.ndarray | None = None,
		full_waveforms: np.ndarray | None = None,
	) -> None:
		self.recording = _MockRecording()
		self.sorting = _MockSorting(unit_ids=[94])
		self.sparsity = _MockSparsity(mapping={"94": [2, 0]})
		self._templates_ext = templates_ext
		self._has_templates = bool(has_templates)
		self._full_waveforms = None if full_waveforms is None else np.asarray(full_waveforms, dtype=float)
		self.last_compute_extension_params = None
		if waveforms is None:
			self._waveforms_ext = None
		else:
			self._waveforms_ext = _MockWaveformsExtension(np.asarray(waveforms, dtype=float))

	def has_extension(self, name: str) -> bool:
		if name == "templates":
			return self._has_templates
		if name == "waveforms":
			return self._waveforms_ext is not None
		return False

	def get_extension(self, name: str):
		if name == "templates":
			return self._templates_ext
		if name == "waveforms" and self._waveforms_ext is not None:
			return self._waveforms_ext
		raise KeyError(name)

	def compute(self, names, extension_params=None, verbose: bool = False, n_jobs: int = 1) -> None:
		_ = names, verbose, n_jobs
		self.last_compute_extension_params = extension_params
		name_list = [str(n) for n in (names or [])]
		if "random_spikes" in name_list and "waveforms" in name_list and self._full_waveforms is not None:
			max_spikes = None
			if isinstance(extension_params, dict):
				rs = extension_params.get("random_spikes", {})
				if isinstance(rs, dict):
					max_spikes = rs.get("max_spikes_per_unit", None)
			if max_spikes is None:
				n = int(self._full_waveforms.shape[0])
			else:
				n = max(1, min(int(max_spikes), int(self._full_waveforms.shape[0])))
			if self._waveforms_ext is None:
				self._waveforms_ext = _MockWaveformsExtension(self._full_waveforms[:n, :, :])
			else:
				self._waveforms_ext.set_waveforms(self._full_waveforms[:n, :, :])
			return

		# Simulate unavailable template computation path for templates-only compute calls.
		if "templates" in name_list:
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


def test_build_unit_source_payload_expands_to_all_waveforms_when_unlimited() -> None:
	template_time_by_ch = np.asarray(
		[
			[1.0, 3.0],
			[2.0, 4.0],
			[0.0, 0.0],
			[0.0, 0.0],
		],
		dtype=float,
	)
	full_waveforms = np.arange(5 * 4 * 2, dtype=float).reshape(5, 4, 2)
	limited_waveforms = full_waveforms[:3, :, :]
	analyzer = _MockAnalyzer(
		templates_ext=_MockTemplatesExtension(template_time_by_ch),
		has_templates=True,
		waveforms=limited_waveforms,
		full_waveforms=full_waveforms,
	)

	payload = build_unit_source_payload(
		analyzer=analyzer,
		unit_id=94,
		max_spikes_per_unit=None,
	)
	assert payload is not None
	_, _, _, _, waveform_count, _, top_wf, _, top_count = payload
	assert waveform_count == 5
	assert top_wf is not None
	assert int(top_wf.shape[0]) == 5
	assert top_count == 5


def test_build_unit_source_payload_honors_positive_waveform_cap() -> None:
	template_time_by_ch = np.asarray(
		[
			[1.0, 3.0],
			[2.0, 4.0],
			[0.0, 0.0],
			[0.0, 0.0],
		],
		dtype=float,
	)
	full_waveforms = np.arange(6 * 4 * 2, dtype=float).reshape(6, 4, 2)
	limited_waveforms = full_waveforms[:3, :, :]
	analyzer = _MockAnalyzer(
		templates_ext=_MockTemplatesExtension(template_time_by_ch),
		has_templates=True,
		waveforms=limited_waveforms,
		full_waveforms=full_waveforms,
	)

	payload = build_unit_source_payload(
		analyzer=analyzer,
		unit_id=94,
		max_spikes_per_unit=2,
	)
	assert payload is not None
	_, _, _, _, _, _, top_wf, _, top_count = payload
	assert top_wf is not None
	assert int(top_wf.shape[0]) == 2
	assert top_count == 2


def test_build_unit_source_payload_forwards_waveform_window_on_recompute() -> None:
	template_time_by_ch = np.asarray(
		[
			[1.0, 3.0],
			[2.0, 4.0],
			[0.0, 0.0],
			[0.0, 0.0],
		],
		dtype=float,
	)
	full_waveforms = np.arange(6 * 4 * 2, dtype=float).reshape(6, 4, 2)
	limited_waveforms = full_waveforms[:2, :, :]
	analyzer = _MockAnalyzer(
		templates_ext=_MockTemplatesExtension(template_time_by_ch),
		has_templates=True,
		waveforms=limited_waveforms,
		full_waveforms=full_waveforms,
	)

	payload = build_unit_source_payload(
		analyzer=analyzer,
		unit_id=94,
		max_spikes_per_unit=None,
		waveform_ms_before=1.5,
		waveform_ms_after=2.5,
	)
	assert payload is not None
	_, _, _, _, waveform_count, _, top_wf, _, top_count = payload
	assert top_wf is not None
	assert int(top_wf.shape[0]) == int(waveform_count)
	assert top_count == int(waveform_count)

	params = analyzer.last_compute_extension_params
	assert isinstance(params, dict)
	assert "random_spikes" in params
	assert params["random_spikes"].get("max_spikes_per_unit") == int(waveform_count)
	assert params.get("waveforms", {}).get("ms_before") == 1.5
	assert params.get("waveforms", {}).get("ms_after") == 2.5
