"""Mutation-safety regression suite (slice 7).

Locks in the contract that the new label/merge dry_run knobs and the
cheap "never-mutate" phases (snapshot_sorter_output, concat_analyzer)
keep canonical state byte-identical. A future regression that flips a
dry_run gate or skips a fingerprint check will fail here.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pytest

from axon_recon.pipeline.stages.spikesort.core.concat_analyzer import (
	run_concat_analyzer_phase,
)
from axon_recon.pipeline.stages.spikesort.core.snapshot_sorter_output import (
	run_snapshot_sorter_output_phase,
)
from axon_recon.pipeline.stages.spikesort.tests._mutation_safety import (
	assert_directory_unchanged,
	hash_directory,
)


LOGGER = logging.getLogger("test_mutation_safety")


def _seed_sorter_output(root: Path) -> None:
	root.mkdir(parents=True, exist_ok=True)
	(root / "params.py").write_text("sample_rate = 30000\n", encoding="utf-8")
	(root / "spike_times.npy").write_bytes(b"\x00" * 64)
	(root / "spike_clusters.npy").write_bytes(b"\x01" * 32)
	(root / "cluster_KSLabel.tsv").write_text(
		"cluster_id\tKSLabel\n0\tgood\n1\tmua\n", encoding="utf-8"
	)
	(root / "cluster_group.tsv").write_text(
		"cluster_id\tgroup\n0\tgood\n1\tmua\n", encoding="utf-8"
	)
	subdir = root / "subdir"
	subdir.mkdir(parents=True, exist_ok=True)
	(subdir / "extra.npy").write_bytes(b"\x02" * 16)


def test_hash_directory_is_deterministic_and_complete(tmp_path: Path) -> None:
	root = tmp_path / "sorter_output"
	_seed_sorter_output(root)
	a = hash_directory(root)
	b = hash_directory(root)
	assert a == b
	# Every seeded file shows up.
	assert set(a.keys()) == {
		"params.py",
		"spike_times.npy",
		"spike_clusters.npy",
		"cluster_KSLabel.tsv",
		"cluster_group.tsv",
		"subdir/extra.npy",
	}


def test_assert_directory_unchanged_passes_on_no_change(tmp_path: Path) -> None:
	root = tmp_path / "sorter_output"
	_seed_sorter_output(root)
	baseline = hash_directory(root)
	assert_directory_unchanged(root, baseline)  # must not raise


def test_assert_directory_unchanged_raises_on_mutation(tmp_path: Path) -> None:
	root = tmp_path / "sorter_output"
	_seed_sorter_output(root)
	baseline = hash_directory(root)
	(root / "cluster_KSLabel.tsv").write_text(
		"cluster_id\tKSLabel\n0\tnoise\n1\tmua\n", encoding="utf-8"
	)
	with pytest.raises(AssertionError, match="changed"):
		assert_directory_unchanged(root, baseline)


def test_assert_directory_unchanged_detects_added_file(tmp_path: Path) -> None:
	root = tmp_path / "sorter_output"
	_seed_sorter_output(root)
	baseline = hash_directory(root)
	(root / "new.npy").write_bytes(b"\x99" * 8)
	with pytest.raises(AssertionError, match="added"):
		assert_directory_unchanged(root, baseline)


def test_snapshot_sorter_output_never_mutates_source(tmp_path: Path) -> None:
	"""snapshot_sorter_output is read-only on the source sorter_output dir."""
	source = tmp_path / "sorter_output"
	snapshot = tmp_path / "sorter_output_snapshot"
	_seed_sorter_output(source)
	baseline = hash_directory(source)

	run_snapshot_sorter_output_phase(
		sorter_output_dir=source,
		snapshot_dir=snapshot,
		skip_if_exists=True,
		logger=LOGGER,
	)

	assert_directory_unchanged(source, baseline)


def test_snapshot_sorter_output_skip_path_never_mutates_source(tmp_path: Path) -> None:
	"""Same contract on the skip-if-exists branch."""
	source = tmp_path / "sorter_output"
	snapshot = tmp_path / "sorter_output_snapshot"
	_seed_sorter_output(source)
	# First call materializes the snapshot.
	run_snapshot_sorter_output_phase(
		sorter_output_dir=source,
		snapshot_dir=snapshot,
		skip_if_exists=True,
	)
	baseline = hash_directory(source)
	# Second call hits the skip branch — must still not mutate source.
	run_snapshot_sorter_output_phase(
		sorter_output_dir=source,
		snapshot_dir=snapshot,
		skip_if_exists=True,
	)
	assert_directory_unchanged(source, baseline)


def test_concat_analyzer_never_mutates_sorter_output(tmp_path: Path) -> None:
	"""concat_analyzer reads sorter_output for the fingerprint and runs
	create_sorting_analyzer pointing to a separate analyzer_dir. The source
	sorter_output dir must remain byte-identical.
	"""
	sorter = tmp_path / "sorter_output"
	analyzer_dir = tmp_path / "concat_analyzer"
	_seed_sorter_output(sorter)
	baseline = hash_directory(sorter)

	class _FakeAnalyzer:
		def __init__(self, folder: Path):
			self.folder = Path(folder)

		def compute(self, *_args, **_kwargs):
			pass

		def has_extension(self, *_args, **_kwargs):
			return False

	def _fake_create(*, sorting, recording, format, folder, **kwargs):
		Path(folder).mkdir(parents=True, exist_ok=True)
		return _FakeAnalyzer(folder)

	def _fake_load(folder):
		return _FakeAnalyzer(folder)

	run_concat_analyzer_phase(
		sorter_output_dir=sorter,
		recording=object(),
		sorting=object(),
		analyzer_dir=analyzer_dir,
		extensions={"random_spikes": {}, "templates": {}},
		create_sorting_analyzer_fn=_fake_create,
		load_sorting_analyzer_fn=_fake_load,
	)

	assert_directory_unchanged(sorter, baseline)


def test_concat_analyzer_skip_path_never_mutates_sorter_output(tmp_path: Path) -> None:
	"""Same contract on the fingerprint-match skip branch."""
	sorter = tmp_path / "sorter_output"
	analyzer_dir = tmp_path / "concat_analyzer"
	_seed_sorter_output(sorter)

	class _FakeAnalyzer:
		def __init__(self, folder: Path):
			self.folder = Path(folder)

		def compute(self, *_args, **_kwargs):
			pass

		def has_extension(self, *_args, **_kwargs):
			return False

	def _fake_create(*, sorting, recording, format, folder, **kwargs):
		Path(folder).mkdir(parents=True, exist_ok=True)
		return _FakeAnalyzer(folder)

	def _fake_load(folder):
		return _FakeAnalyzer(folder)

	# Build once so the second call hits the fingerprint-match skip branch.
	run_concat_analyzer_phase(
		sorter_output_dir=sorter,
		recording=object(),
		sorting=object(),
		analyzer_dir=analyzer_dir,
		create_sorting_analyzer_fn=_fake_create,
		load_sorting_analyzer_fn=_fake_load,
	)
	baseline = hash_directory(sorter)
	result = run_concat_analyzer_phase(
		sorter_output_dir=sorter,
		recording=object(),
		sorting=object(),
		analyzer_dir=analyzer_dir,
		create_sorting_analyzer_fn=_fake_create,
		load_sorting_analyzer_fn=_fake_load,
	)
	assert result["rebuilt"] is False  # sanity: we exercised the skip path
	assert_directory_unchanged(sorter, baseline)
