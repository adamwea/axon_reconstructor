from __future__ import annotations

from pathlib import Path

import numpy as np

from ..labels_io import (
	lookup_label,
	lookup_spike_count,
	read_bombcell_labels,
	read_per_cluster_spike_counts,
)


def _make_sorter_output_dir(tmp_path: Path, *, nested: bool = False) -> Path:
	root = tmp_path / "well000" / "spikesort_outputs"
	if nested:
		out = root / "sorter_output" / "sorter_output"
	else:
		out = root / "sorter_output"
	out.mkdir(parents=True, exist_ok=True)
	return out


def test_read_bombcell_labels_from_cluster_group_tsv(tmp_path: Path) -> None:
	sorter_out = _make_sorter_output_dir(tmp_path)
	(sorter_out / "cluster_group.tsv").write_text(
		"cluster_id\tKSLabel\tlabel\tlabel_reason\n"
		"0\tmua\tmua\t\n"
		"1\tgood\tgood\t\n"
		"2\tnoise\tnon_soma_good\t\n",
		encoding="utf-8",
	)
	labels = read_bombcell_labels(sorter_out.parent.parent / "spikesort_outputs")
	assert labels == {0: "mua", 1: "good", 2: "non_soma_good"}


def test_read_bombcell_labels_handles_nested_sorter_output(tmp_path: Path) -> None:
	# spikesort_outputs/sorter_output/sorter_output/cluster_group.tsv
	sorter_out = _make_sorter_output_dir(tmp_path, nested=True)
	(sorter_out / "cluster_group.tsv").write_text(
		"cluster_id\tKSLabel\tlabel\tlabel_reason\n7\tgood\tgood\t\n",
		encoding="utf-8",
	)
	# sorter_out is .../spikesort_outputs/sorter_output/sorter_output
	# Two .parent hops lift back to spikesort_outputs.
	root = sorter_out.parent.parent
	labels = read_bombcell_labels(root)
	assert labels == {7: "good"}


def test_read_bombcell_labels_falls_back_to_kslabel(tmp_path: Path) -> None:
	sorter_out = _make_sorter_output_dir(tmp_path)
	# Only cluster_KSLabel.tsv is present.
	(sorter_out / "cluster_KSLabel.tsv").write_text(
		"cluster_id\tKSLabel\n10\tmua\n11\tgood\n",
		encoding="utf-8",
	)
	labels = read_bombcell_labels(sorter_out.parent.parent / "spikesort_outputs")
	assert labels == {10: "mua", 11: "good"}


def test_read_bombcell_labels_missing_dir_returns_empty(tmp_path: Path) -> None:
	# spikesort_outputs/ doesn't exist at all.
	assert read_bombcell_labels(tmp_path / "no_such_dir") == {}


def test_read_bombcell_labels_header_only_returns_empty(tmp_path: Path) -> None:
	sorter_out = _make_sorter_output_dir(tmp_path)
	(sorter_out / "cluster_group.tsv").write_text(
		"cluster_id\tKSLabel\tlabel\tlabel_reason\n",
		encoding="utf-8",
	)
	labels = read_bombcell_labels(sorter_out.parent.parent / "spikesort_outputs")
	assert labels == {}


def test_read_bombcell_labels_skips_rows_with_unparseable_cluster_id(tmp_path: Path) -> None:
	sorter_out = _make_sorter_output_dir(tmp_path)
	(sorter_out / "cluster_group.tsv").write_text(
		"cluster_id\tKSLabel\tlabel\tlabel_reason\n"
		"\tmua\tmua\t\n"  # empty cluster_id
		"bogus\tmua\tmua\t\n"  # non-int cluster_id
		"5\tgood\tgood\t\n",
		encoding="utf-8",
	)
	labels = read_bombcell_labels(sorter_out.parent.parent / "spikesort_outputs")
	assert labels == {5: "good"}


def test_read_per_cluster_spike_counts_aggregates(tmp_path: Path) -> None:
	sorter_out = _make_sorter_output_dir(tmp_path)
	np.save(sorter_out / "spike_clusters.npy", np.array([0, 0, 1, 1, 1, 5, 5, 5, 5], dtype=np.int64))
	counts = read_per_cluster_spike_counts(sorter_out.parent.parent / "spikesort_outputs")
	assert counts == {0: 2, 1: 3, 5: 4}


def test_read_per_cluster_spike_counts_missing_npy_returns_empty(tmp_path: Path) -> None:
	sorter_out = _make_sorter_output_dir(tmp_path)
	# No spike_clusters.npy at all.
	counts = read_per_cluster_spike_counts(sorter_out.parent.parent / "spikesort_outputs")
	assert counts == {}


def test_lookup_label_and_count_handle_unit_id_types() -> None:
	labels = {0: "mua", 1: "good"}
	counts = {0: 100, 1: 250}
	assert lookup_label(labels, 1) == "good"
	assert lookup_label(labels, "1") == "good"
	assert lookup_label(labels, 999) is None
	assert lookup_label(labels, None) is None
	assert lookup_spike_count(counts, 0) == 100
	assert lookup_spike_count(counts, "1") == 250
	assert lookup_spike_count(counts, 999) is None
	assert lookup_spike_count(counts, None) is None
