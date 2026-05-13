"""Read well-level spikesort artifacts and surface their unit counts.

Pure function helpers that the analysis stage uses to populate
spikesort-derived columns in `well_summary.parquet`. No heavy IO —
just reads the small JSON summaries the spikesort stage already
writes per well.

Sources read (all optional; missing files produce ``None`` columns):

  - ``<well>/spikesort_outputs/merge_SLAy/pre_merge_metadata_summary.json``
    → `sorter.unit_count` becomes both `unit_count_ks_raw` and
    `unit_count_pre_merge` (they match: pre-merge sorter IS the
    Kilosort output).
  - ``<well>/spikesort_outputs/merge_SLAy/post_merge_metadata_summary.json``
    → `sorter.unit_count` becomes `unit_count_post_merge`.
  - ``<well>/spikesort_outputs/merge_SLAy/merge_stage_summary.json``
    → `n_merge_groups` becomes `unit_count_merge_groups`;
      `n_candidate_pairs` becomes `unit_count_merge_candidate_pairs`.
  - ``<well>/spikesort_outputs/bombcell_label_outputs/bombcell_label_summary.json``
    → `counts_by_label` is split into one column per canonical
    bombcell label (good, non_soma_good, mua, non_soma_mua, noise,
    non_soma_noise), plus a `bombcell_count_total` sanity sum.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# Canonical bombcell labels. Any label outside this set is rolled into
# `bombcell_count_other` to keep the parquet schema stable.
CANONICAL_BOMBCELL_LABELS: tuple[str, ...] = (
	"good",
	"non_soma_good",
	"mua",
	"non_soma_mua",
	"noise",
	"non_soma_noise",
)


SPIKESORT_AGG_COLUMNS: tuple[str, ...] = (
	"unit_count_ks_raw",
	"unit_count_pre_merge",
	"unit_count_post_merge",
	"unit_count_merge_groups",
	"unit_count_merge_candidate_pairs",
	*(f"bombcell_count_{label}" for label in CANONICAL_BOMBCELL_LABELS),
	"bombcell_count_other",
	"bombcell_count_total",
)


def _empty_aggregates() -> dict[str, Any]:
	return {column: None for column in SPIKESORT_AGG_COLUMNS}


def _safe_read_json(path: Path) -> dict[str, Any] | None:
	if not path.is_file():
		return None
	try:
		with path.open("r", encoding="utf-8") as fh:
			payload = json.load(fh)
	except (json.JSONDecodeError, OSError):
		return None
	return payload if isinstance(payload, dict) else None


def collect_spikesort_unit_counts(
	*,
	well_out_dir: Path,
	output_rel_root: str = "spikesort_outputs",
) -> dict[str, Any]:
	"""Return a flat dict of well-level spikesort counts for well_summary.

	Every key in the returned dict is also present in
	`SPIKESORT_AGG_COLUMNS`. Missing source files leave the value at
	``None`` rather than raising, so this helper can be called
	unconditionally without worrying about whether spikesort completed.
	"""
	aggregates = _empty_aggregates()

	base = Path(well_out_dir) / str(output_rel_root or "spikesort_outputs").strip().lstrip("/")
	merge_dir = base / "merge_SLAy"
	bombcell_dir = base / "bombcell_label_outputs"

	pre_payload = _safe_read_json(merge_dir / "pre_merge_metadata_summary.json")
	if pre_payload is not None:
		sorter = pre_payload.get("sorter") if isinstance(pre_payload.get("sorter"), dict) else None
		if sorter is not None:
			pre_count = sorter.get("unit_count")
			try:
				if pre_count is not None:
					aggregates["unit_count_pre_merge"] = int(pre_count)
					# Pre-merge sorter IS the Kilosort output, so use the same
					# value for "raw KS clusters" — they're definitionally equal.
					aggregates["unit_count_ks_raw"] = int(pre_count)
			except (TypeError, ValueError):
				pass

	post_payload = _safe_read_json(merge_dir / "post_merge_metadata_summary.json")
	if post_payload is not None:
		sorter = post_payload.get("sorter") if isinstance(post_payload.get("sorter"), dict) else None
		if sorter is not None:
			post_count = sorter.get("unit_count")
			try:
				if post_count is not None:
					aggregates["unit_count_post_merge"] = int(post_count)
			except (TypeError, ValueError):
				pass

	merge_payload = _safe_read_json(merge_dir / "merge_stage_summary.json")
	if merge_payload is not None:
		try:
			ng = merge_payload.get("n_merge_groups")
			if ng is not None:
				aggregates["unit_count_merge_groups"] = int(ng)
		except (TypeError, ValueError):
			pass
		try:
			ncp = merge_payload.get("n_candidate_pairs")
			if ncp is not None:
				aggregates["unit_count_merge_candidate_pairs"] = int(ncp)
		except (TypeError, ValueError):
			pass

	bombcell_payload = _safe_read_json(bombcell_dir / "bombcell_label_summary.json")
	if bombcell_payload is not None:
		counts = bombcell_payload.get("counts_by_label")
		if isinstance(counts, dict):
			canonical_set = set(CANONICAL_BOMBCELL_LABELS)
			total = 0
			other_total = 0
			for label, count in counts.items():
				try:
					ival = int(count)
				except (TypeError, ValueError):
					continue
				total += ival
				key = str(label)
				if key in canonical_set:
					aggregates[f"bombcell_count_{key}"] = ival
				else:
					other_total += ival
			# Zero-fill canonical labels that weren't in counts_by_label so the
			# downstream parquet schema stays dense.
			for label in CANONICAL_BOMBCELL_LABELS:
				column = f"bombcell_count_{label}"
				if aggregates[column] is None:
					aggregates[column] = 0
			aggregates["bombcell_count_other"] = int(other_total)
			aggregates["bombcell_count_total"] = int(total)

	return aggregates
