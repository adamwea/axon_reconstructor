from __future__ import annotations

import json
from pathlib import Path

from ..spikesort_metrics import (
	CANONICAL_BOMBCELL_LABELS,
	SPIKESORT_AGG_COLUMNS,
	collect_spikesort_unit_counts,
)


def _write_json(path: Path, payload: dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload), encoding="utf-8")


def test_aggregates_include_every_declared_column() -> None:
	"""The aggregates dict always has one key per SPIKESORT_AGG_COLUMNS entry,
	so callers can plug into a fixed parquet schema without surprises."""
	aggregates = collect_spikesort_unit_counts(well_out_dir=Path("/tmp/does-not-exist"))
	assert set(aggregates.keys()) == set(SPIKESORT_AGG_COLUMNS)
	# Nothing on disk → every value is None.
	assert all(value is None for value in aggregates.values())


def test_collect_reads_pre_and_post_merge_metadata(tmp_path: Path) -> None:
	merge_dir = tmp_path / "spikesort_outputs" / "merge_SLAy"
	_write_json(
		merge_dir / "pre_merge_metadata_summary.json",
		{"sorter": {"unit_count": 438}},
	)
	_write_json(
		merge_dir / "post_merge_metadata_summary.json",
		{"sorter": {"unit_count": 410}},
	)
	_write_json(
		merge_dir / "merge_stage_summary.json",
		{"n_merge_groups": 12, "n_candidate_pairs": 47},
	)

	aggs = collect_spikesort_unit_counts(well_out_dir=tmp_path)

	assert aggs["unit_count_pre_merge"] == 438
	assert aggs["unit_count_ks_raw"] == 438  # pre-merge sorter IS the raw KS output
	assert aggs["unit_count_post_merge"] == 410
	assert aggs["unit_count_merge_groups"] == 12
	assert aggs["unit_count_merge_candidate_pairs"] == 47


def test_collect_reads_full_bombcell_label_histogram(tmp_path: Path) -> None:
	bombcell_dir = tmp_path / "spikesort_outputs" / "bombcell_label_outputs"
	_write_json(
		bombcell_dir / "bombcell_label_summary.json",
		{
			"counts_by_label": {
				"good": 1,
				"non_soma_good": 0,
				"mua": 134,
				"non_soma_mua": 0,
				"noise": 302,
				"non_soma_noise": 0,
			}
		},
	)
	aggs = collect_spikesort_unit_counts(well_out_dir=tmp_path)

	# Every canonical label gets a dedicated column, even when zero.
	for label in CANONICAL_BOMBCELL_LABELS:
		assert aggs[f"bombcell_count_{label}"] is not None, f"missing column for {label}"
	assert aggs["bombcell_count_good"] == 1
	assert aggs["bombcell_count_mua"] == 134
	assert aggs["bombcell_count_noise"] == 302
	assert aggs["bombcell_count_non_soma_good"] == 0
	assert aggs["bombcell_count_other"] == 0
	assert aggs["bombcell_count_total"] == 437


def test_collect_rolls_unknown_bombcell_labels_into_other(tmp_path: Path) -> None:
	bombcell_dir = tmp_path / "spikesort_outputs" / "bombcell_label_outputs"
	_write_json(
		bombcell_dir / "bombcell_label_summary.json",
		{"counts_by_label": {"good": 5, "experimental_label": 3, "weirdo": 2}},
	)
	aggs = collect_spikesort_unit_counts(well_out_dir=tmp_path)

	assert aggs["bombcell_count_good"] == 5
	assert aggs["bombcell_count_mua"] == 0  # canonical, zero-filled
	assert aggs["bombcell_count_other"] == 5  # experimental_label + weirdo
	assert aggs["bombcell_count_total"] == 10


def test_collect_handles_malformed_or_missing_json(tmp_path: Path) -> None:
	# Pre-merge metadata is malformed JSON; post-merge is missing entirely.
	merge_dir = tmp_path / "spikesort_outputs" / "merge_SLAy"
	merge_dir.mkdir(parents=True, exist_ok=True)
	(merge_dir / "pre_merge_metadata_summary.json").write_text("{not json")

	aggs = collect_spikesort_unit_counts(well_out_dir=tmp_path)

	# Failed reads should leave everything as None rather than raising.
	assert aggs["unit_count_pre_merge"] is None
	assert aggs["unit_count_post_merge"] is None
	assert aggs["unit_count_merge_groups"] is None


def test_collect_respects_custom_output_rel_root(tmp_path: Path) -> None:
	# Some configs override `spikesort_outputs` to something else; the
	# helper must follow.
	merge_dir = tmp_path / "alt_spikesort" / "merge_SLAy"
	_write_json(
		merge_dir / "pre_merge_metadata_summary.json",
		{"sorter": {"unit_count": 99}},
	)
	aggs = collect_spikesort_unit_counts(
		well_out_dir=tmp_path,
		output_rel_root="alt_spikesort",
	)
	assert aggs["unit_count_pre_merge"] == 99
	assert aggs["unit_count_ks_raw"] == 99
