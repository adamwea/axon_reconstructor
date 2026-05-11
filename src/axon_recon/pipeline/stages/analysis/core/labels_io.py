"""Read-only readers for spikesort cluster labels and per-cluster spike counts.

Reads under `<well>/spikesort_outputs/sorter_output/`. Tolerates the doubly
nested `sorter_output/sorter_output/` layout produced by SpikeInterface +
Kilosort, and tolerates missing files (returns empty dicts).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _candidate_sorter_output_dirs(spikesort_outputs_dir: Path) -> list[Path]:
	"""Return likely sorter-output directories, in priority order."""
	root = Path(spikesort_outputs_dir)
	primary = root / "sorter_output"
	candidates: list[Path] = []
	if primary.is_dir():
		candidates.append(primary)
		nested = primary / "sorter_output"
		if nested.is_dir():
			candidates.append(nested)
	return candidates


def _find_tsv(spikesort_outputs_dir: Path, name: str) -> Path | None:
	for candidate in _candidate_sorter_output_dirs(spikesort_outputs_dir):
		path = candidate / name
		if path.is_file():
			return path
	return None


def read_bombcell_labels(spikesort_outputs_dir: Path) -> dict[int, str]:
	"""Return `cluster_id -> bombcell_label` from `cluster_group.tsv`.

	Falls back to `cluster_KSLabel.tsv` (column `KSLabel`) when
	`cluster_group.tsv` is absent. Returns an empty dict when neither file
	exists, when the file has no rows, or when parsing fails — callers must
	tolerate `None` labels per the §4 filter contract.
	"""
	primary = _find_tsv(spikesort_outputs_dir, "cluster_group.tsv")
	if primary is not None:
		return _parse_label_tsv(primary, column="label")
	fallback = _find_tsv(spikesort_outputs_dir, "cluster_KSLabel.tsv")
	if fallback is not None:
		return _parse_label_tsv(fallback, column="KSLabel")
	return {}


def _parse_label_tsv(path: Path, *, column: str) -> dict[int, str]:
	out: dict[int, str] = {}
	try:
		with path.open("r", encoding="utf-8", newline="") as fh:
			reader = csv.DictReader(fh, delimiter="\t")
			for row in reader:
				raw_id = row.get("cluster_id", None)
				if raw_id is None:
					continue
				token = str(raw_id).strip()
				if not token:
					continue
				try:
					cluster_id = int(token)
				except ValueError:
					continue
				value = row.get(column, None)
				if value is None:
					continue
				label = str(value).strip()
				if not label:
					continue
				out[cluster_id] = label
	except OSError:
		return {}
	return out


def read_per_cluster_spike_counts(spikesort_outputs_dir: Path) -> dict[int, int]:
	"""Return `cluster_id -> spike_count` from `spike_clusters.npy`.

	Returns an empty dict if numpy isn't importable, the file is missing,
	or the array can't be loaded. Callers must tolerate `None` counts.
	"""
	try:
		import numpy as np  # local import to keep this module light
	except ImportError:
		return {}
	for candidate in _candidate_sorter_output_dirs(spikesort_outputs_dir):
		path = candidate / "spike_clusters.npy"
		if not path.is_file():
			continue
		try:
			arr = np.load(path)
		except (OSError, ValueError):
			continue
		if arr.size == 0:
			return {}
		# Use bincount + nonzero for an efficient per-cluster count over int arrays.
		try:
			int_arr = arr.astype(int, copy=False)
		except (TypeError, ValueError):
			return {}
		if int_arr.min() < 0:
			# Negative cluster_ids would break bincount. Fall back to Counter.
			from collections import Counter

			return {int(k): int(v) for k, v in Counter(int_arr.tolist()).items()}
		counts = np.bincount(int_arr)
		nonzero = counts.nonzero()[0]
		return {int(idx): int(counts[idx]) for idx in nonzero}
	return {}


def lookup_label(labels: dict[int, str], unit_id: Any) -> str | None:
	"""Look up a label by unit_id (integer key)."""
	try:
		key = int(unit_id)
	except (TypeError, ValueError):
		return None
	return labels.get(key, None)


def lookup_spike_count(counts: dict[int, int], unit_id: Any) -> int | None:
	try:
		key = int(unit_id)
	except (TypeError, ValueError):
		return None
	return counts.get(key, None)
