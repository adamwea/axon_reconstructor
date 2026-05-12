from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from axon_recon.pipeline.stages.spikesort.legacy_runner import (
	LEGACY_SPIKESORTING_OUTPUTS_DIRNAME,
	SPIKESORTING_OUTPUTS_DIRNAME,
)


DEFAULT_UNIT_LABEL_FILTER: tuple[str, ...] = ("good", "non_soma_good")


def normalize_unit_token(value: Any) -> str:
	return str(value).strip()


def normalize_label_token(value: Any) -> str:
	return str(value).strip().lower()


def load_unit_labels_from_spikesorting(well_out_dir: Path) -> dict[str, str] | None:
	# Prefer cluster_KSLabel.tsv / cluster_group.tsv in sorter_output: post-SLAy
	# this file is the only complete view (bombcell pass1 wrote its labels for
	# pre-merge units, then SLAy appended inherited labels for the new merged
	# unit IDs). bombcell_labels.json is pre-merge-only and is kept as a fallback
	# for runs where bombcell ran with apply_to_sorter_output=false (dry_run).
	for labels_tsv in _candidate_cluster_label_tsv_paths(well_out_dir):
		labels = _read_cluster_label_tsv(labels_tsv)
		if labels:
			return labels

	for labels_json in _candidate_bombcell_label_json_paths(well_out_dir):
		labels = _read_bombcell_labels_json(labels_json)
		if labels:
			return labels

	for labels_xlsx in _candidate_qm_xlsx_paths(well_out_dir):
		labels = _read_qm_label_xlsx(labels_xlsx)
		if labels:
			return labels

	return None


def filter_unit_ids_by_labels(
	unit_ids: list[Any],
	labels_by_unit: dict[str, str],
	allowed_labels: tuple[str, ...] | list[str],
) -> list[Any]:
	allowed = {normalize_label_token(label) for label in allowed_labels if normalize_label_token(label)}
	if not allowed:
		return list(unit_ids)
	return [
		unit_id
		for unit_id in unit_ids
		if normalize_label_token(labels_by_unit.get(normalize_unit_token(unit_id), "")) in allowed
	]


def count_labels(labels_by_unit: dict[str, str]) -> dict[str, int]:
	counts: dict[str, int] = {}
	for label in labels_by_unit.values():
		label_token = normalize_label_token(label)
		if not label_token:
			continue
		counts[label_token] = int(counts.get(label_token, 0)) + 1
	return dict(sorted(counts.items(), key=lambda item: item[0]))


def _candidate_spikesort_roots(well_out_dir: Path) -> list[Path]:
	roots: list[Path] = []
	for dirname in (SPIKESORTING_OUTPUTS_DIRNAME, LEGACY_SPIKESORTING_OUTPUTS_DIRNAME):
		root = well_out_dir / dirname
		if root not in roots:
			roots.append(root)
	return roots


def _candidate_bombcell_label_json_paths(well_out_dir: Path) -> list[Path]:
	paths: list[Path] = []
	for root in _candidate_spikesort_roots(well_out_dir):
		paths.append(root / "bombcell_label_outputs" / "bombcell_labels.json")
		paths.append(root / "merge_units" / "bombcell_label_outputs" / "bombcell_labels.json")
	return paths


def _candidate_cluster_label_tsv_paths(well_out_dir: Path) -> list[Path]:
	paths: list[Path] = []
	for root in _candidate_spikesort_roots(well_out_dir):
		for sorter_root in (
			root / "sorter_output",
			root / "sorter_output" / "sorter_output",
			root / "merge_units" / "sorter_output",
			root / "merge_units" / "sorter_output" / "sorter_output",
		):
			paths.append(sorter_root / "cluster_KSLabel.tsv")
			paths.append(sorter_root / "cluster_group.tsv")
	return paths


def _candidate_qm_xlsx_paths(well_out_dir: Path) -> list[Path]:
	return [root / "qm_unfiltered.xlsx" for root in _candidate_spikesort_roots(well_out_dir)]


def _read_bombcell_labels_json(path: Path) -> dict[str, str] | None:
	if not path.exists():
		return None
	try:
		payload = json.loads(path.read_text())
	except Exception:
		return None
	labels_raw = payload.get("labels_by_unit", {}) if isinstance(payload, dict) else {}
	if not isinstance(labels_raw, dict):
		return None
	labels = {
		normalize_unit_token(unit_id): str(label).strip()
		for unit_id, label in labels_raw.items()
		if normalize_unit_token(unit_id) and str(label).strip()
	}
	return labels or None


def _read_cluster_label_tsv(path: Path) -> dict[str, str] | None:
	if not path.exists():
		return None
	try:
		with path.open("r", newline="") as handle:
			reader = csv.DictReader(handle, delimiter="\t")
			if reader.fieldnames is None:
				return None
			unit_column = _first_existing_field(reader.fieldnames, ("cluster_id", "id", "unit_id"))
			label_column = _first_existing_field(reader.fieldnames, ("KSLabel", "group", "label"))
			if unit_column is None or label_column is None:
				return None
			labels = {
				normalize_unit_token(row.get(unit_column, "")): str(row.get(label_column, "")).strip()
				for row in reader
				if normalize_unit_token(row.get(unit_column, "")) and str(row.get(label_column, "")).strip()
			}
	except Exception:
		return None
	return labels or None


def _read_qm_label_xlsx(path: Path) -> dict[str, str] | None:
	if not path.exists():
		return None
	try:
		import pandas as pd  # type: ignore[import-not-found]

		df = pd.read_excel(path, index_col=0)
	except Exception:
		return None
	label_column = _first_existing_field(
		[str(column) for column in getattr(df, "columns", [])],
		("label", "KSLabel", "group", "cluster_group", "bombcell_label", "bc_label"),
	)
	if label_column is None:
		return None
	labels: dict[str, str] = {}
	try:
		for unit_id, row in df.iterrows():
			unit_token = normalize_unit_token(unit_id)
			label = str(row.get(label_column, "")).strip()
			if unit_token and label:
				labels[unit_token] = label
	except Exception:
		return None
	return labels or None


def _first_existing_field(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
	by_lower = {str(field).strip().lower(): str(field) for field in fieldnames}
	for candidate in candidates:
		match = by_lower.get(str(candidate).strip().lower())
		if match is not None:
			return match
	return None