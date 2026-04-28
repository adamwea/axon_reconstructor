from __future__ import annotations

import json
import logging
import sys
import types
import numpy as np
import pytest

from axon_recon.pipeline.stages.templates.integrations.spikeinterface_extract import (
	build_unit_source_payload,
	discover_cached_spikeinterface_analyzer_source_names,
	load_spikeinterface_analyzers,
)
from axon_recon.pipeline.stages.templates.models.inputs import AnalyzerPreparationPolicyConfig


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
		self.last_compute_kwargs = None
		self.compute_call_count = 0
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

	def compute(self, names, extension_params=None, verbose: bool = False, n_jobs: int = 1, **kwargs) -> None:
		_ = names, verbose, n_jobs
		self.compute_call_count += 1
		self.last_compute_extension_params = extension_params
		self.last_compute_kwargs = dict(kwargs)
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
	assert params["random_spikes"].get("max_spikes_per_unit") is None
	assert params.get("waveforms", {}).get("ms_before") == 1.5
	assert params.get("waveforms", {}).get("ms_after") == 2.5


def test_build_unit_source_payload_forwards_random_spikes_policy_on_recompute() -> None:
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
		random_spikes_method="all",
		random_seed=123,
	)
	assert payload is not None

	params = analyzer.last_compute_extension_params
	assert isinstance(params, dict)
	assert params["random_spikes"].get("method") == "all"
	assert "seed" not in params["random_spikes"]
	assert "max_spikes_per_unit" not in params["random_spikes"]


def test_build_unit_source_payload_forwards_grouped_analyzer_controls_on_recompute() -> None:
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
		max_spikes_per_unit=4,
		min_spikes_per_unit=2,
		waveform_ms_before=1.25,
		waveform_ms_after=2.75,
		waveform_dtype="float32",
		random_spikes_method="percentage",
		random_spikes_percentage=0.75,
		random_seed=123,
		log_before_after_spike_counts=True,
		margin_size=7,
		compute_n_jobs=4,
		compute_chunk_duration="1s",
	)
	assert payload is not None

	params = analyzer.last_compute_extension_params
	assert isinstance(params, dict)
	assert params["random_spikes"].get("method") == "percentage"
	assert params["random_spikes"].get("percentage") == 0.75
	assert params["random_spikes"].get("max_spikes_per_unit") == 4
	assert params["random_spikes"].get("min_spikes_per_unit") == 2
	assert params["random_spikes"].get("seed") == 123
	assert params["random_spikes"].get("log_before_after_spike_counts") is True
	assert params["random_spikes"].get("margin_size") == 7
	assert params.get("waveforms", {}).get("ms_before") == 1.25
	assert params.get("waveforms", {}).get("ms_after") == 2.75
	assert params.get("waveforms", {}).get("dtype") == "float32"
	assert analyzer.last_compute_kwargs == {"chunk_duration": "1s"}


def test_build_unit_source_payload_retries_without_compat_only_kwargs() -> None:
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

	class _CompatFallbackAnalyzer(_MockAnalyzer):
		def __init__(self) -> None:
			super().__init__(
				templates_ext=_MockTemplatesExtension(template_time_by_ch),
				has_templates=True,
				waveforms=limited_waveforms,
				full_waveforms=full_waveforms,
			)
			self.compute_attempts: list[tuple[dict[str, object] | None, dict[str, object]]] = []

		def compute(self, names, extension_params=None, verbose: bool = False, n_jobs: int = 1, **kwargs) -> None:
			_ = names, verbose, n_jobs
			raw_params = extension_params if isinstance(extension_params, dict) else None
			params_copy = None if raw_params is None else json.loads(json.dumps(raw_params))
			self.compute_attempts.append((params_copy, dict(kwargs)))
			random_spikes = {} if raw_params is None else dict(raw_params.get("random_spikes", {}))
			waveforms = {} if raw_params is None else dict(raw_params.get("waveforms", {}))
			if (
				"log_before_after_spike_counts" in random_spikes
				or "margin_size" in random_spikes
				or "dtype" in waveforms
				or "chunk_duration" in kwargs
			):
				raise TypeError("unsupported compatibility kwargs")
			return super().compute(names, extension_params=extension_params, verbose=verbose, n_jobs=n_jobs, **kwargs)

	analyzer = _CompatFallbackAnalyzer()
	payload = build_unit_source_payload(
		analyzer=analyzer,
		unit_id=94,
		max_spikes_per_unit=None,
		waveform_ms_before=1.25,
		waveform_ms_after=2.75,
		waveform_dtype="float32",
		log_before_after_spike_counts=True,
		margin_size=7,
		compute_chunk_duration="1s",
	)

	assert payload is not None
	_, _, _, _, waveform_count, _, top_wf, _, top_count = payload
	assert int(waveform_count) == 5
	assert top_wf is not None
	assert int(top_wf.shape[0]) == 5
	assert top_count == 5
	assert len(analyzer.compute_attempts) == 2

	first_params, first_kwargs = analyzer.compute_attempts[0]
	assert first_params is not None
	assert first_params["random_spikes"].get("log_before_after_spike_counts") is True
	assert first_params["random_spikes"].get("margin_size") == 7
	assert first_params.get("waveforms", {}).get("dtype") == "float32"
	assert first_kwargs == {"chunk_duration": "1s"}

	second_params, second_kwargs = analyzer.compute_attempts[1]
	assert second_params is not None
	assert "log_before_after_spike_counts" not in second_params["random_spikes"]
	assert "margin_size" not in second_params["random_spikes"]
	assert "dtype" not in second_params.get("waveforms", {})
	assert second_kwargs == {}


def test_build_unit_source_payload_skips_recompute_when_waveforms_are_prepared() -> None:
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
	analyzer = _MockAnalyzer(
		templates_ext=_MockTemplatesExtension(template_time_by_ch),
		has_templates=True,
		waveforms=full_waveforms,
		full_waveforms=full_waveforms,
	)
	analyzer._axon_recon_prepared_waveforms_signature = (None, 1.5, 2.5)

	payload = build_unit_source_payload(
		analyzer=analyzer,
		unit_id=94,
		max_spikes_per_unit=None,
		waveform_ms_before=1.5,
		waveform_ms_after=2.5,
	)
	assert payload is not None
	_, _, _, _, waveform_count, _, top_wf, _, top_count = payload
	assert int(waveform_count) == 5
	assert top_wf is not None
	assert int(top_wf.shape[0]) == 5
	assert top_count == 5
	assert analyzer.compute_call_count == 0
	assert analyzer.last_compute_extension_params is None


def test_load_spikeinterface_analyzers_honors_explicit_source_paths(tmp_path, monkeypatch) -> None:
	well_out_dir = tmp_path / "well001"
	concat_dir = well_out_dir / "custom_concat"
	segments_dir = well_out_dir / "custom_segments"
	seg_a = segments_dir / "segA"
	seg_b = segments_dir / "segB"
	concat_dir.mkdir(parents=True, exist_ok=True)
	seg_a.mkdir(parents=True, exist_ok=True)
	seg_b.mkdir(parents=True, exist_ok=True)

	loaded_paths: list[str] = []

	def _fake_load_sorting_analyzer(path):
		loaded_paths.append(str(path))
		return {"path": str(path)}

	fake_full = types.ModuleType("spikeinterface.full")
	fake_full.load_sorting_analyzer = _fake_load_sorting_analyzer  # type: ignore[attr-defined]
	fake_root = types.ModuleType("spikeinterface")
	fake_root.full = fake_full  # type: ignore[attr-defined]

	monkeypatch.setitem(sys.modules, "spikeinterface", fake_root)
	monkeypatch.setitem(sys.modules, "spikeinterface.full", fake_full)

	analyzers = load_spikeinterface_analyzers(
		well_out_dir=well_out_dir,
		concat_analyzer_relpath="/custom_concat",
		preproc_seg_sources_reldir="/custom_segments",
		include_concat=True,
		include_segments=True,
	)

	assert len(analyzers) == 3
	assert str(concat_dir) in loaded_paths
	assert str(seg_a) in loaded_paths
	assert str(seg_b) in loaded_paths


def test_load_spikeinterface_analyzers_filters_requested_source_names(tmp_path, monkeypatch) -> None:
	well_out_dir = tmp_path / "well001"
	concat_dir = well_out_dir / "custom_concat"
	segments_dir = well_out_dir / "custom_segments"
	seg_a = segments_dir / "segA"
	seg_b = segments_dir / "segB"
	concat_dir.mkdir(parents=True, exist_ok=True)
	seg_a.mkdir(parents=True, exist_ok=True)
	seg_b.mkdir(parents=True, exist_ok=True)

	loaded_paths: list[str] = []

	def _fake_load_sorting_analyzer(path):
		loaded_paths.append(str(path))
		return {"path": str(path)}

	fake_full = types.ModuleType("spikeinterface.full")
	fake_full.load_sorting_analyzer = _fake_load_sorting_analyzer  # type: ignore[attr-defined]
	fake_root = types.ModuleType("spikeinterface")
	fake_root.full = fake_full  # type: ignore[attr-defined]

	monkeypatch.setitem(sys.modules, "spikeinterface", fake_root)
	monkeypatch.setitem(sys.modules, "spikeinterface.full", fake_full)

	analyzers = load_spikeinterface_analyzers(
		well_out_dir=well_out_dir,
		concat_analyzer_relpath="/custom_concat",
		preproc_seg_sources_reldir="/custom_segments",
		include_concat=True,
		include_segments=True,
		requested_source_names=["segB"],
	)

	assert [name for name, _ in analyzers] == ["segB"]
	assert loaded_paths == [str(seg_b)]


def test_load_spikeinterface_analyzers_persists_loaded_analyzers_to_cache(tmp_path, monkeypatch) -> None:
	well_out_dir = tmp_path / "well001"
	concat_dir = well_out_dir / "custom_concat"
	segments_dir = well_out_dir / "custom_segments"
	seg_a = segments_dir / "segA"
	seg_b = segments_dir / "segB"
	cache_dir = well_out_dir / "templates_outputs" / "cache" / "analyzers"
	concat_dir.mkdir(parents=True, exist_ok=True)
	seg_a.mkdir(parents=True, exist_ok=True)
	seg_b.mkdir(parents=True, exist_ok=True)

	save_calls: list[tuple[str, str, str]] = []

	class _FakeAnalyzer:
		def __init__(self, source_path: str) -> None:
			self.source_path = str(source_path)

		def has_extension(self, name: str) -> bool:
			_ = name
			return True

		def compute(self, names, extension_params=None, verbose: bool = False, n_jobs: int = 1, **kwargs) -> None:
			_ = names, extension_params, verbose, n_jobs, kwargs

		def save_as(self, format="memory", folder=None, backend_options=None):
			_ = backend_options
			save_calls.append((self.source_path, str(folder), str(format)))
			folder.mkdir(parents=True, exist_ok=True)
			return self

	def _fake_load_sorting_analyzer(path):
		return _FakeAnalyzer(str(path))

	fake_full = types.ModuleType("spikeinterface.full")
	fake_full.load_sorting_analyzer = _fake_load_sorting_analyzer  # type: ignore[attr-defined]
	fake_root = types.ModuleType("spikeinterface")
	fake_root.full = fake_full  # type: ignore[attr-defined]

	monkeypatch.setitem(sys.modules, "spikeinterface", fake_root)
	monkeypatch.setitem(sys.modules, "spikeinterface.full", fake_full)

	analyzers = load_spikeinterface_analyzers(
		well_out_dir=well_out_dir,
		concat_analyzer_relpath="/custom_concat",
		preproc_seg_sources_reldir="/custom_segments",
		analyzer_cache_dir=cache_dir,
		include_concat=True,
		include_segments=True,
	)

	assert len(analyzers) == 3
	assert len(save_calls) == 3
	assert (str(concat_dir), str(cache_dir / "concat"), "binary_folder") in save_calls
	assert (str(seg_a), str(cache_dir / "segA"), "binary_folder") in save_calls
	assert (str(seg_b), str(cache_dir / "segB"), "binary_folder") in save_calls


def test_load_spikeinterface_analyzers_builds_dense_segment_analyzers_from_preprocessed_sources(tmp_path, monkeypatch, caplog) -> None:
	well_out_dir = tmp_path / "well000"
	concat_dir = well_out_dir / "custom_concat"
	segments_dir = well_out_dir / "custom_segments"
	seg_a = segments_dir / "000_recA"
	seg_b = segments_dir / "001_recB"
	concat_dir.mkdir(parents=True, exist_ok=True)
	seg_a.mkdir(parents=True, exist_ok=True)
	seg_b.mkdir(parents=True, exist_ok=True)

	epochs = [
		{"segment_index": 0, "rec_name": "recA", "start_sample": 0, "end_sample": 100},
		{"segment_index": 1, "rec_name": "recB", "start_sample": 100, "end_sample": 200},
	]
	(well_out_dir / "concatenation_stitch_epochs_well000.json").write_text(json.dumps(epochs), encoding="utf-8")

	class _FakeConcatSorting:
		def get_unit_ids(self):
			return [94]

		def get_num_segments(self):
			return 1

		def get_sampling_frequency(self):
			return 10_000.0

		def get_unit_spike_train(self, unit_id: int, segment_index: int = 0):
			_ = unit_id, segment_index
			return np.asarray([10, 20, 110, 120], dtype=int)

	class _FakeRecording:
		def get_sampling_frequency(self):
			return 10_000.0

	class _FakeConcatAnalyzer:
		def __init__(self) -> None:
			self.sorting = _FakeConcatSorting()
			self.recording = _FakeRecording()

	class _FakeNumpySorting:
		def __init__(self, mapping: dict[int, np.ndarray], fs: float) -> None:
			self._mapping = {int(k): np.asarray(v, dtype=int) for k, v in mapping.items()}
			self._fs = float(fs)

		@property
		def unit_ids(self):
			return [u for u, spikes in sorted(self._mapping.items()) if int(spikes.size) > 0]

		def remove_empty_units(self):
			self._mapping = {u: spikes for u, spikes in self._mapping.items() if int(spikes.size) > 0}
			return self

		def get_unit_spike_train(self, unit_id: int, segment_index: int = 0):
			_ = segment_index
			return np.asarray(self._mapping.get(int(unit_id), np.asarray([], dtype=int)), dtype=int)

		@staticmethod
		def from_unit_dict(unit_trains: dict[int, np.ndarray], sampling_frequency: float):
			return _FakeNumpySorting(mapping=unit_trains, fs=sampling_frequency)

		@staticmethod
		def from_times_labels(times_list, labels_list, sampling_frequency: float):
			times = np.asarray(times_list[0], dtype=int)
			labels = np.asarray(labels_list[0], dtype=int)
			mapping: dict[int, list[int]] = {}
			for t, lab in zip(times.tolist(), labels.tolist(), strict=False):
				mapping.setdefault(int(lab), []).append(int(t))
			return _FakeNumpySorting(
				mapping={int(k): np.asarray(v, dtype=int) for k, v in mapping.items()},
				fs=sampling_frequency,
			)

	create_calls: list[dict[str, object]] = []

	class _FakeBuiltAnalyzer:
		def __init__(self, sorting, recording):
			self.sorting = sorting
			self.recording = recording

		def compute(self, names, extension_params=None, verbose: bool = False, n_jobs: int = 1, **kwargs):
			_ = names, extension_params, verbose, n_jobs, kwargs

	concat_analyzer = _FakeConcatAnalyzer()

	def _fake_load_sorting_analyzer(path):
		if str(path) == str(concat_dir):
			return concat_analyzer
		raise RuntimeError("not an analyzer")

	def _fake_create_sorting_analyzer(sorting, recording, format="memory", return_in_uV=True, **kwargs):
		create_calls.append(
			{
				"format": format,
				"return_in_uV": return_in_uV,
				"units": list(getattr(sorting, "unit_ids", [])),
				**kwargs,
			}
		)
		return _FakeBuiltAnalyzer(sorting=sorting, recording=recording)

	def _fake_remove_excess_spikes(sorting, recording):
		_ = recording
		return sorting

	def _fake_load_extractor(path):
		_ = path
		return _FakeRecording()

	fake_full = types.ModuleType("spikeinterface.full")
	fake_full.load_sorting_analyzer = _fake_load_sorting_analyzer  # type: ignore[attr-defined]
	fake_full.create_sorting_analyzer = _fake_create_sorting_analyzer  # type: ignore[attr-defined]
	fake_full.remove_excess_spikes = _fake_remove_excess_spikes  # type: ignore[attr-defined]
	fake_full.load_extractor = _fake_load_extractor  # type: ignore[attr-defined]
	fake_full.load = _fake_load_extractor  # type: ignore[attr-defined]

	fake_core = types.ModuleType("spikeinterface.core")
	fake_core.NumpySorting = _FakeNumpySorting  # type: ignore[attr-defined]

	fake_root = types.ModuleType("spikeinterface")
	fake_root.full = fake_full  # type: ignore[attr-defined]
	fake_root.core = fake_core  # type: ignore[attr-defined]

	monkeypatch.setitem(sys.modules, "spikeinterface", fake_root)
	monkeypatch.setitem(sys.modules, "spikeinterface.full", fake_full)
	monkeypatch.setitem(sys.modules, "spikeinterface.core", fake_core)

	with caplog.at_level(logging.INFO, logger="axon_recon.templates.spikeinterface"):
		analyzers = load_spikeinterface_analyzers(
			well_out_dir=well_out_dir,
			concat_analyzer_relpath="/custom_concat",
			preproc_seg_sources_reldir="/custom_segments",
			stream_id="well000",
			include_concat=True,
			include_segments=True,
			segments_policy=AnalyzerPreparationPolicyConfig(compute_sparsity=False, sparsity_mode="dense"),
		)

	names = [name for name, _ in analyzers]
	assert "concat" in names
	assert "000_recA" in names
	assert "001_recB" in names
	assert len(create_calls) == 2
	assert all(call["format"] == "memory" for call in create_calls)
	assert all(call["sparse"] is False for call in create_calls)
	messages = [rec.getMessage() for rec in caplog.records]
	assert any("Discovered preprocessed segment recording sources: count=2" in msg for msg in messages)
	assert any("Loaded preprocessed segment recording source: segment=000_recA" in msg for msg in messages)
	assert any("Registered preprocessed segment recording with concat spikes: segment=000_recA" in msg for msg in messages)
	assert any("Generating segment analyzer: segment=000_recA" in msg for msg in messages)
	assert any("Computing segment analyzer extensions: segment=000_recA" in msg for msg in messages)
	assert not any("Segment directory is not a loadable analyzer" in msg for msg in messages)


def test_load_spikeinterface_analyzers_builds_segments_from_concat_sorting_without_concat_analyzer(
	tmp_path,
	monkeypatch,
	caplog,
) -> None:
	well_out_dir = tmp_path / "well000"
	sorting_dir = well_out_dir / "custom_sorting"
	segments_dir = well_out_dir / "custom_segments"
	seg_a = segments_dir / "000_recA"
	sorting_dir.mkdir(parents=True, exist_ok=True)
	seg_a.mkdir(parents=True, exist_ok=True)

	epochs = [{"segment_index": 0, "rec_name": "recA", "start_sample": 0, "end_sample": 100}]
	(well_out_dir / "concatenation_stitch_epochs_well000.json").write_text(json.dumps(epochs), encoding="utf-8")

	class _FakeConcatSorting:
		def get_unit_ids(self):
			return [94]

		def get_num_segments(self):
			return 1

		def get_sampling_frequency(self):
			return 10_000.0

		def get_unit_spike_train(self, unit_id: int, segment_index: int = 0):
			_ = unit_id, segment_index
			return np.asarray([10, 20, 110, 120], dtype=int)

	class _FakeRecording:
		def get_sampling_frequency(self):
			return 10_000.0

	class _FakeNumpySorting:
		def __init__(self, mapping: dict[int, np.ndarray], fs: float) -> None:
			self._mapping = {int(k): np.asarray(v, dtype=int) for k, v in mapping.items()}
			self._fs = float(fs)

		@property
		def unit_ids(self):
			return [u for u, spikes in sorted(self._mapping.items()) if int(spikes.size) > 0]

		def remove_empty_units(self):
			self._mapping = {u: spikes for u, spikes in self._mapping.items() if int(spikes.size) > 0}
			return self

		@staticmethod
		def from_unit_dict(unit_trains: dict[int, np.ndarray], sampling_frequency: float):
			return _FakeNumpySorting(mapping=unit_trains, fs=sampling_frequency)

		@staticmethod
		def from_times_labels(times_list, labels_list, sampling_frequency: float):
			times = np.asarray(times_list[0], dtype=int)
			labels = np.asarray(labels_list[0], dtype=int)
			mapping: dict[int, list[int]] = {}
			for t, lab in zip(times.tolist(), labels.tolist(), strict=False):
				mapping.setdefault(int(lab), []).append(int(t))
			return _FakeNumpySorting(
				mapping={int(k): np.asarray(v, dtype=int) for k, v in mapping.items()},
				fs=sampling_frequency,
			)

	create_calls: list[dict[str, object]] = []
	fake_sorting = _FakeConcatSorting()

	class _FakeBuiltAnalyzer:
		def __init__(self, sorting, recording):
			self.sorting = sorting
			self.recording = recording

		def compute(self, names, extension_params=None, verbose: bool = False, n_jobs: int = 1, **kwargs):
			_ = names, extension_params, verbose, n_jobs, kwargs

	def _fake_load_sorting_analyzer(path):
		_ = path
		raise AssertionError("concat analyzer should not be loaded for segment-only registration")

	def _fake_load_sorting(path):
		if str(path) == str(sorting_dir):
			return fake_sorting
		raise RuntimeError("unexpected sorting path")

	def _fake_create_sorting_analyzer(sorting, recording, format="memory", return_in_uV=True, **kwargs):
		create_calls.append(
			{
				"format": format,
				"return_in_uV": return_in_uV,
				"units": list(getattr(sorting, "unit_ids", [])),
				**kwargs,
			}
		)
		return _FakeBuiltAnalyzer(sorting=sorting, recording=recording)

	def _fake_remove_excess_spikes(sorting, recording):
		_ = recording
		return sorting

	def _fake_load_extractor(path):
		if str(path) == str(seg_a):
			return _FakeRecording()
		raise RuntimeError("unexpected extractor path")

	fake_full = types.ModuleType("spikeinterface.full")
	fake_full.load_sorting_analyzer = _fake_load_sorting_analyzer  # type: ignore[attr-defined]
	fake_full.load_sorting = _fake_load_sorting  # type: ignore[attr-defined]
	fake_full.create_sorting_analyzer = _fake_create_sorting_analyzer  # type: ignore[attr-defined]
	fake_full.remove_excess_spikes = _fake_remove_excess_spikes  # type: ignore[attr-defined]
	fake_full.load_extractor = _fake_load_extractor  # type: ignore[attr-defined]
	fake_full.load = _fake_load_extractor  # type: ignore[attr-defined]

	fake_core = types.ModuleType("spikeinterface.core")
	fake_core.NumpySorting = _FakeNumpySorting  # type: ignore[attr-defined]

	fake_root = types.ModuleType("spikeinterface")
	fake_root.full = fake_full  # type: ignore[attr-defined]
	fake_root.core = fake_core  # type: ignore[attr-defined]

	monkeypatch.setitem(sys.modules, "spikeinterface", fake_root)
	monkeypatch.setitem(sys.modules, "spikeinterface.full", fake_full)
	monkeypatch.setitem(sys.modules, "spikeinterface.core", fake_core)

	with caplog.at_level(logging.INFO, logger="axon_recon.templates.spikeinterface"):
		analyzers = load_spikeinterface_analyzers(
			well_out_dir=well_out_dir,
			concat_sorting_relpath="/custom_sorting",
			preproc_seg_sources_reldir="/custom_segments",
			stream_id="well000",
			include_concat=False,
			include_segments=True,
			segments_policy=AnalyzerPreparationPolicyConfig(compute_sparsity=False, sparsity_mode="dense"),
		)

	names = [name for name, _ in analyzers]
	assert names == ["000_recA"]
	assert len(create_calls) == 1
	assert create_calls[0]["units"] == [94]
	assert create_calls[0]["sparse"] is False
	messages = [rec.getMessage() for rec in caplog.records]
	assert any("Loading concat sorting for segment registration:" in msg for msg in messages)
	assert any("Registered preprocessed segment recording with concat spikes: segment=000_recA" in msg for msg in messages)
	assert not any("Concat analyzer selection:" in msg for msg in messages)


def test_load_spikeinterface_analyzers_builds_dense_concat_from_sorting_and_preprocessed_concat(tmp_path, monkeypatch) -> None:
	well_out_dir = tmp_path / "well001"
	sorting_dir = well_out_dir / "custom_sorting"
	preprocessed_concat_dir = well_out_dir / "custom_preprocessed_concat"
	sorting_dir.mkdir(parents=True, exist_ok=True)
	preprocessed_concat_dir.mkdir(parents=True, exist_ok=True)

	create_calls: list[dict[str, object]] = []

	class _FakeSorting:
		pass

	class _FakeRecording:
		pass

	class _FakeAnalyzer:
		def __init__(self, sorting, recording) -> None:
			self.sorting = sorting
			self.recording = recording

	fake_sorting = _FakeSorting()
	fake_recording = _FakeRecording()

	def _fake_load_sorting_analyzer(path):
		_ = path
		raise RuntimeError("no concat analyzer present")

	def _fake_load_sorting(path):
		if str(path) == str(sorting_dir):
			return fake_sorting
		raise RuntimeError("unexpected sorting path")

	def _fake_load_extractor(path):
		if str(path) == str(preprocessed_concat_dir):
			return fake_recording
		raise RuntimeError("unexpected extractor path")

	def _fake_create_sorting_analyzer(sorting, recording, format="memory", return_in_uV=True, **kwargs):
		create_calls.append(
			{
				"sorting": sorting,
				"recording": recording,
				"format": str(format),
				"return_in_uV": bool(return_in_uV),
				**kwargs,
			}
		)
		return _FakeAnalyzer(sorting=sorting, recording=recording)

	fake_full = types.ModuleType("spikeinterface.full")
	fake_full.load_sorting_analyzer = _fake_load_sorting_analyzer  # type: ignore[attr-defined]
	fake_full.load_sorting = _fake_load_sorting  # type: ignore[attr-defined]
	fake_full.load_extractor = _fake_load_extractor  # type: ignore[attr-defined]
	fake_full.load = _fake_load_extractor  # type: ignore[attr-defined]
	fake_full.create_sorting_analyzer = _fake_create_sorting_analyzer  # type: ignore[attr-defined]

	fake_root = types.ModuleType("spikeinterface")
	fake_root.full = fake_full  # type: ignore[attr-defined]

	monkeypatch.setitem(sys.modules, "spikeinterface", fake_root)
	monkeypatch.setitem(sys.modules, "spikeinterface.full", fake_full)

	analyzers = load_spikeinterface_analyzers(
		well_out_dir=well_out_dir,
		concat_sorting_relpath="/custom_sorting",
		preprocessed_concat_reldir="/custom_preprocessed_concat",
		include_concat=True,
		include_segments=False,
		concat_policy=AnalyzerPreparationPolicyConfig(compute_sparsity=False, sparsity_mode="dense"),
	)

	assert len(analyzers) == 1
	assert analyzers[0][0] == "concat"
	assert len(create_calls) == 1
	assert create_calls[0]["sorting"] is fake_sorting
	assert create_calls[0]["recording"] is fake_recording
	assert create_calls[0]["format"] == "memory"
	assert create_calls[0]["return_in_uV"] is True
	assert create_calls[0]["sparse"] is False


def test_load_spikeinterface_analyzers_rebuilds_concat_without_reusing_existing_analyzer(tmp_path, monkeypatch, caplog) -> None:
	well_out_dir = tmp_path / "well001"
	concat_dir = well_out_dir / "custom_concat"
	sorting_dir = well_out_dir / "custom_sorting"
	preprocessed_concat_dir = well_out_dir / "custom_preprocessed_concat"
	concat_dir.mkdir(parents=True, exist_ok=True)
	sorting_dir.mkdir(parents=True, exist_ok=True)
	preprocessed_concat_dir.mkdir(parents=True, exist_ok=True)

	load_sorting_analyzer_calls: list[str] = []
	create_calls: list[dict[str, object]] = []

	class _FakeSorting:
		pass

	class _FakeRecording:
		pass

	class _FakeAnalyzer:
		def __init__(self, sorting, recording) -> None:
			self.sorting = sorting
			self.recording = recording

		def has_extension(self, name: str) -> bool:
			_ = name
			return True

		def compute(self, names, extension_params=None, verbose: bool = False, n_jobs: int = 1, **kwargs) -> None:
			_ = names, extension_params, verbose, n_jobs, kwargs

	fake_sorting = _FakeSorting()
	fake_recording = _FakeRecording()

	def _fake_load_sorting_analyzer(path):
		load_sorting_analyzer_calls.append(str(path))
		if str(path) == str(concat_dir):
			raise AssertionError("existing concat analyzer should not be loaded when reuse is disabled")
		raise RuntimeError("unexpected analyzer path")

	def _fake_load_sorting(path):
		if str(path) == str(sorting_dir):
			return fake_sorting
		raise RuntimeError("unexpected sorting path")

	def _fake_load_extractor(path):
		if str(path) == str(preprocessed_concat_dir):
			return fake_recording
		raise RuntimeError("unexpected extractor path")

	def _fake_create_sorting_analyzer(sorting, recording, format="memory", return_in_uV=True, **kwargs):
		create_calls.append(
			{
				"sorting": sorting,
				"recording": recording,
				"format": str(format),
				"return_in_uV": bool(return_in_uV),
				**kwargs,
			}
		)
		return _FakeAnalyzer(sorting=sorting, recording=recording)

	fake_full = types.ModuleType("spikeinterface.full")
	fake_full.load_sorting_analyzer = _fake_load_sorting_analyzer  # type: ignore[attr-defined]
	fake_full.load_sorting = _fake_load_sorting  # type: ignore[attr-defined]
	fake_full.load_extractor = _fake_load_extractor  # type: ignore[attr-defined]
	fake_full.load = _fake_load_extractor  # type: ignore[attr-defined]
	fake_full.create_sorting_analyzer = _fake_create_sorting_analyzer  # type: ignore[attr-defined]

	fake_root = types.ModuleType("spikeinterface")
	fake_root.full = fake_full  # type: ignore[attr-defined]

	monkeypatch.setitem(sys.modules, "spikeinterface", fake_root)
	monkeypatch.setitem(sys.modules, "spikeinterface.full", fake_full)

	with caplog.at_level(logging.INFO, logger="axon_recon.templates.spikeinterface"):
		analyzers = load_spikeinterface_analyzers(
			well_out_dir=well_out_dir,
			concat_analyzer_relpath="/custom_concat",
			concat_sorting_relpath="/custom_sorting",
			preprocessed_concat_reldir="/custom_preprocessed_concat",
			include_concat=True,
			include_segments=False,
			concat_use_existing_analyzer=False,
			concat_build_if_missing=True,
			concat_policy=AnalyzerPreparationPolicyConfig(random_spikes_method="all"),
		)

	assert len(analyzers) == 1
	assert analyzers[0][0] == "concat"
	assert len(create_calls) == 1
	assert create_calls[0]["sorting"] is fake_sorting
	assert create_calls[0]["recording"] is fake_recording
	assert create_calls[0]["format"] == "memory"
	assert create_calls[0]["return_in_uV"] is True
	assert create_calls[0]["sparse"] is True
	assert load_sorting_analyzer_calls == []
	messages = [rec.getMessage() for rec in caplog.records]
	assert any("Concat analyzer reuse disabled; skipping existing concat analyzer load" in msg for msg in messages)
	assert any("Loading concat sorting for analyzer build:" in msg for msg in messages)
	assert any("Loaded preprocessed concat recording for analyzer build:" in msg for msg in messages)
	assert any("Generating concat analyzer: creation=" in msg for msg in messages)


def test_load_spikeinterface_analyzers_persists_cache_with_configured_subdirs(tmp_path, monkeypatch) -> None:
	well_out_dir = tmp_path / "well001"
	concat_dir = well_out_dir / "custom_concat"
	segments_dir = well_out_dir / "custom_segments"
	seg_a = segments_dir / "segA"
	seg_b = segments_dir / "segB"
	cache_dir = well_out_dir / "templates_outputs" / "cache" / "analyzers"
	concat_dir.mkdir(parents=True, exist_ok=True)
	seg_a.mkdir(parents=True, exist_ok=True)
	seg_b.mkdir(parents=True, exist_ok=True)

	save_calls: list[tuple[str, str, str]] = []

	class _FakeAnalyzer:
		def __init__(self, source_path: str) -> None:
			self.source_path = str(source_path)

		def has_extension(self, name: str) -> bool:
			_ = name
			return True

		def compute(self, names, extension_params=None, verbose: bool = False, n_jobs: int = 1, **kwargs) -> None:
			_ = names, extension_params, verbose, n_jobs, kwargs

		def save_as(self, format="memory", folder=None, backend_options=None):
			_ = backend_options
			save_calls.append((self.source_path, str(folder), str(format)))
			folder.mkdir(parents=True, exist_ok=True)
			return self

	def _fake_load_sorting_analyzer(path):
		return _FakeAnalyzer(str(path))

	fake_full = types.ModuleType("spikeinterface.full")
	fake_full.load_sorting_analyzer = _fake_load_sorting_analyzer  # type: ignore[attr-defined]
	fake_root = types.ModuleType("spikeinterface")
	fake_root.full = fake_full  # type: ignore[attr-defined]

	monkeypatch.setitem(sys.modules, "spikeinterface", fake_root)
	monkeypatch.setitem(sys.modules, "spikeinterface.full", fake_full)

	analyzers = load_spikeinterface_analyzers(
		well_out_dir=well_out_dir,
		concat_analyzer_relpath="/custom_concat",
		preproc_seg_sources_reldir="/custom_segments",
		analyzer_cache_dir=cache_dir,
		analyzer_cache_concat_subdir="concat_custom",
		analyzer_cache_segments_subdir="segments_custom",
		include_concat=True,
		include_segments=True,
	)

	assert len(analyzers) == 3
	assert len(save_calls) == 3
	assert (str(concat_dir), str(cache_dir / "concat_custom"), "binary_folder") in save_calls
	assert (str(seg_a), str(cache_dir / "segments_custom" / "segA"), "binary_folder") in save_calls
	assert (str(seg_b), str(cache_dir / "segments_custom" / "segB"), "binary_folder") in save_calls


def test_discover_cached_spikeinterface_analyzer_source_names_skips_segments_container(tmp_path) -> None:
	cache_dir = tmp_path / "templates_outputs" / "cache" / "analyzers"
	(cache_dir / "concat").mkdir(parents=True, exist_ok=True)
	(cache_dir / "segments" / "000_rec0000").mkdir(parents=True, exist_ok=True)
	(cache_dir / "segments" / "001_rec0001").mkdir(parents=True, exist_ok=True)

	source_names = discover_cached_spikeinterface_analyzer_source_names(
		analyzer_cache_dir=cache_dir,
		analyzer_cache_concat_subdir="concat",
		analyzer_cache_segments_subdir="segments",
		include_concat=True,
		include_segments=True,
	)

	assert source_names == ["concat", "000_rec0000", "001_rec0001"]


def test_load_spikeinterface_analyzers_raises_when_segments_required_but_missing(tmp_path, monkeypatch) -> None:
	well_out_dir = tmp_path / "well001"
	concat_dir = well_out_dir / "custom_concat"
	concat_dir.mkdir(parents=True, exist_ok=True)

	class _FakeAnalyzer:
		def has_extension(self, name: str) -> bool:
			_ = name
			return True

		def compute(self, names, extension_params=None, verbose: bool = False, n_jobs: int = 1, **kwargs) -> None:
			_ = names, extension_params, verbose, n_jobs, kwargs

	def _fake_load_sorting_analyzer(path):
		if str(path) == str(concat_dir):
			return _FakeAnalyzer()
		raise RuntimeError("unexpected analyzer path")

	fake_full = types.ModuleType("spikeinterface.full")
	fake_full.load_sorting_analyzer = _fake_load_sorting_analyzer  # type: ignore[attr-defined]
	fake_root = types.ModuleType("spikeinterface")
	fake_root.full = fake_full  # type: ignore[attr-defined]

	monkeypatch.setitem(sys.modules, "spikeinterface", fake_root)
	monkeypatch.setitem(sys.modules, "spikeinterface.full", fake_full)

	with pytest.raises(FileNotFoundError, match="require_segments=True"):
		load_spikeinterface_analyzers(
			well_out_dir=well_out_dir,
			concat_analyzer_relpath="/custom_concat",
			preproc_seg_sources_reldir="/missing_segments",
			include_concat=True,
			include_segments=True,
			require_segments=True,
		)
