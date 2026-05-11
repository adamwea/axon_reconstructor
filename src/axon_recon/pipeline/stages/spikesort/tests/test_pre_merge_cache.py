from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from axon_recon.pipeline.stages.spikesort.core.pre_merge_cache import (
	CACHE_FORMAT_VERSION,
	META_FILENAME,
	TEMPLATES_FILENAME,
	PreMergeCache,
	read_pre_merge_cache,
	write_pre_merge_cache,
)


def _make_fake_analyzer(
	*,
	unit_ids: np.ndarray,
	templates: np.ndarray,
	unit_locations: np.ndarray,
	spike_counts: dict,
	channel_ids: np.ndarray,
	channel_locations: np.ndarray,
	ms_before: float = 1.0,
	ms_after: float = 2.0,
	sampling_frequency: float = 10_000.0,
) -> SimpleNamespace:
	templates_ext = SimpleNamespace(
		params={"ms_before": ms_before, "ms_after": ms_after, "operators": ["average"]},
		get_templates=lambda: templates,
	)
	unit_locations_ext = SimpleNamespace(get_data=lambda: unit_locations)

	def get_extension(name: str):
		if name == "templates":
			return templates_ext
		if name == "unit_locations":
			return unit_locations_ext
		return None

	sorting = SimpleNamespace(count_num_spikes_per_unit=lambda: dict(spike_counts))
	return SimpleNamespace(
		get_extension=get_extension,
		unit_ids=unit_ids,
		channel_ids=channel_ids,
		get_channel_locations=lambda: channel_locations,
		sampling_frequency=sampling_frequency,
		sorting=sorting,
	)


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
	unit_ids = np.array([0, 1, 2], dtype=np.int64)
	templates = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
	unit_locations = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 20.0]], dtype=np.float32)
	channel_ids = np.array([100, 101, 102, 103, 104], dtype=np.int64)
	channel_locations = np.arange(5 * 2, dtype=np.float32).reshape(5, 2)
	spike_counts = {0: 100, 1: 200, 2: 50}

	analyzer = _make_fake_analyzer(
		unit_ids=unit_ids,
		templates=templates,
		unit_locations=unit_locations,
		spike_counts=spike_counts,
		channel_ids=channel_ids,
		channel_locations=channel_locations,
	)

	info = write_pre_merge_cache(analyzer=analyzer, cache_dir=tmp_path)
	assert Path(info["templates_path"]).exists()
	assert Path(info["meta_path"]).exists()
	assert info["n_units"] == 3

	meta = json.loads((tmp_path / META_FILENAME).read_text())
	assert meta["format_version"] == CACHE_FORMAT_VERSION
	assert meta["ms_before"] == 1.0
	assert meta["ms_after"] == 2.0
	assert meta["sampling_frequency"] == 10_000.0
	assert meta["n_units"] == 3
	assert meta["n_samples"] == 4
	assert meta["n_channels"] == 5

	cache = read_pre_merge_cache(tmp_path)
	np.testing.assert_array_equal(cache.unit_ids, unit_ids)
	np.testing.assert_array_equal(cache.templates, templates)
	np.testing.assert_array_equal(cache.unit_locations, unit_locations)
	np.testing.assert_array_equal(cache.channel_ids, channel_ids)
	np.testing.assert_array_equal(cache.channel_locations, channel_locations)
	np.testing.assert_array_equal(cache.spike_counts, np.array([100, 200, 50], dtype=np.int64))
	assert cache.ms_before == 1.0
	assert cache.ms_after == 2.0
	assert cache.sampling_frequency == 10_000.0
	assert cache.n_units == 3
	assert cache.n_samples == 4
	assert cache.n_channels == 5


def test_index_by_unit_id_uses_strings(tmp_path: Path) -> None:
	unit_ids = np.array([7, 42, 100], dtype=np.int64)
	templates = np.zeros((3, 2, 3), dtype=np.float32)
	unit_locations = np.zeros((3, 2), dtype=np.float32)
	channel_ids = np.array([0, 1, 2], dtype=np.int64)
	channel_locations = np.zeros((3, 2), dtype=np.float32)
	spike_counts = {7: 1, 42: 1, 100: 1}

	analyzer = _make_fake_analyzer(
		unit_ids=unit_ids,
		templates=templates,
		unit_locations=unit_locations,
		spike_counts=spike_counts,
		channel_ids=channel_ids,
		channel_locations=channel_locations,
	)
	write_pre_merge_cache(analyzer=analyzer, cache_dir=tmp_path)
	cache = read_pre_merge_cache(tmp_path)
	idx = cache.index_by_unit_id()
	assert idx == {"7": 0, "42": 1, "100": 2}


def test_format_version_mismatch_raises(tmp_path: Path) -> None:
	unit_ids = np.array([0], dtype=np.int64)
	templates = np.zeros((1, 2, 2), dtype=np.float32)
	unit_locations = np.zeros((1, 2), dtype=np.float32)
	channel_ids = np.array([0, 1], dtype=np.int64)
	channel_locations = np.zeros((2, 2), dtype=np.float32)
	analyzer = _make_fake_analyzer(
		unit_ids=unit_ids,
		templates=templates,
		unit_locations=unit_locations,
		spike_counts={0: 1},
		channel_ids=channel_ids,
		channel_locations=channel_locations,
	)
	write_pre_merge_cache(analyzer=analyzer, cache_dir=tmp_path)
	meta_path = tmp_path / META_FILENAME
	meta = json.loads(meta_path.read_text())
	meta["format_version"] = "999"
	meta_path.write_text(json.dumps(meta) + "\n", encoding="utf-8")
	with pytest.raises(RuntimeError, match="unsupported format_version"):
		read_pre_merge_cache(tmp_path)


def test_missing_files_raises(tmp_path: Path) -> None:
	with pytest.raises(FileNotFoundError, match="meta missing"):
		read_pre_merge_cache(tmp_path)


def test_missing_templates_extension_raises(tmp_path: Path) -> None:
	analyzer = SimpleNamespace(
		get_extension=lambda name: None,
		unit_ids=np.array([0]),
		channel_ids=np.array([0]),
		get_channel_locations=lambda: np.zeros((1, 2), dtype=np.float32),
		sampling_frequency=10_000.0,
		sorting=SimpleNamespace(count_num_spikes_per_unit=lambda: {}),
	)
	with pytest.raises(RuntimeError, match="templates extension"):
		write_pre_merge_cache(analyzer=analyzer, cache_dir=tmp_path)


def test_missing_unit_locations_extension_raises(tmp_path: Path) -> None:
	templates_ext = SimpleNamespace(
		params={"ms_before": 1.0, "ms_after": 2.0},
		get_templates=lambda: np.zeros((1, 2, 3), dtype=np.float32),
	)

	def get_extension(name: str):
		if name == "templates":
			return templates_ext
		return None

	analyzer = SimpleNamespace(
		get_extension=get_extension,
		unit_ids=np.array([0]),
		channel_ids=np.array([0, 1, 2]),
		get_channel_locations=lambda: np.zeros((3, 2), dtype=np.float32),
		sampling_frequency=10_000.0,
		sorting=SimpleNamespace(count_num_spikes_per_unit=lambda: {0: 1}),
	)
	with pytest.raises(RuntimeError, match="unit_locations extension"):
		write_pre_merge_cache(analyzer=analyzer, cache_dir=tmp_path)


def test_pre_merge_cache_dataclass_constructor() -> None:
	# Smoke-test that PreMergeCache itself can be constructed and queried in-memory.
	cache = PreMergeCache(
		unit_ids=np.array([0, 1]),
		templates=np.zeros((2, 3, 4), dtype=np.float32),
		unit_locations=np.zeros((2, 2), dtype=np.float32),
		spike_counts=np.array([10, 20], dtype=np.int64),
		channel_ids=np.array([0, 1, 2, 3]),
		channel_locations=np.zeros((4, 2), dtype=np.float32),
		ms_before=1.0,
		ms_after=2.0,
		sampling_frequency=10_000.0,
	)
	assert cache.n_units == 2
	assert cache.n_samples == 3
	assert cache.n_channels == 4
	assert cache.index_by_unit_id() == {"0": 0, "1": 1}
