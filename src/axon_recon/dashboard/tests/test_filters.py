from __future__ import annotations

import pandas as pd
import pytest

from ..filters import (
	apply_filter_spec,
	filter_bombcell_allowlist,
	filter_min_numeric,
	filter_multiselect,
	filter_numeric_range,
	filter_recon_status_ok,
	mask_all,
)


def _df():
	return pd.DataFrame(
		{
			"recon_status": ["ok", "ok", "error", "ok"],
			"bombcell_label": ["good", "non_soma_good", "mua", None],
			"num_spikes": [100, 50, 10, 1000],
			"num_branches": [3, 1, 0, 5],
			"recon_quality_score": [None, 0.7, 0.3, None],
			"project": ["A", "A", "B", "B"],
			"DIV": [6, 18, 36, 12],
			"genotype": ["WT", "KO", "WT", "KO"],
		}
	)


def test_mask_all_aligns_with_index() -> None:
	df = _df()
	mask = mask_all(df)
	assert mask.all()
	assert list(mask.index) == list(df.index)


def test_filter_recon_status_ok_default_drops_error_rows() -> None:
	df = _df()
	mask = filter_recon_status_ok(df, on=True)
	assert list(mask) == [True, True, False, True]


def test_filter_recon_status_ok_pass_through_when_off() -> None:
	df = _df()
	mask = filter_recon_status_ok(df, on=False)
	assert mask.all()


def test_filter_bombcell_allowlist_with_none_passes_unlabeled() -> None:
	df = _df()
	mask = filter_bombcell_allowlist(df, ["good", None])
	assert list(mask) == [True, False, False, True]


def test_filter_bombcell_allowlist_empty_drops_everything() -> None:
	df = _df()
	mask = filter_bombcell_allowlist(df, [])
	assert not mask.any()


def test_filter_bombcell_allowlist_none_is_pass_through() -> None:
	df = _df()
	mask = filter_bombcell_allowlist(df, None)
	assert mask.all()


def test_filter_min_numeric_keeps_above_threshold_and_nan_rows() -> None:
	df = _df()
	mask = filter_min_numeric(df, "recon_quality_score", 0.5)
	# Row 0 (None) and Row 1 (0.7) pass; Row 2 (0.3) drops; Row 3 (None) passes.
	assert list(mask) == [True, True, False, True]


def test_filter_min_numeric_threshold_none_passes_all() -> None:
	df = _df()
	mask = filter_min_numeric(df, "num_spikes", None)
	assert mask.all()


def test_filter_min_numeric_missing_column_passes_all() -> None:
	df = _df()
	mask = filter_min_numeric(df, "doesnt_exist", 5)
	assert mask.all()


def test_filter_multiselect_keeps_matching_values() -> None:
	df = _df()
	mask = filter_multiselect(df, "project", ["A"])
	assert list(mask) == [True, True, False, False]


def test_filter_multiselect_empty_or_none_passes_all() -> None:
	df = _df()
	assert filter_multiselect(df, "project", None).all()
	assert filter_multiselect(df, "project", []).all()


def test_filter_numeric_range_inclusive_bounds() -> None:
	df = _df()
	mask = filter_numeric_range(df, "DIV", lo=10, hi=20)
	assert list(mask) == [False, True, False, True]


def test_apply_filter_spec_combines_filters() -> None:
	df = _df()
	filtered = apply_filter_spec(
		df,
		{
			"require_recon_ok": True,
			"bombcell_allowlist": ["good", "non_soma_good"],
			"min_num_spikes": 30,
			"div_lo": 5,
			"div_hi": 30,
		},
	)
	# Expected: index 0 (ok/good/100 spikes/DIV=6) PASS; index 1 (ok/non_soma_good/50/18) PASS;
	# index 2 dropped by recon_status; index 3 dropped by bombcell allowlist (None not allowed).
	assert len(filtered) == 2
	assert list(filtered["bombcell_label"]) == ["good", "non_soma_good"]


def test_apply_filter_spec_default_inputs_keep_all_ok_rows() -> None:
	df = _df()
	filtered = apply_filter_spec(df, {})
	# Default: require_recon_ok=True drops 1 error row; nothing else applied.
	assert len(filtered) == 3
	assert all(s == "ok" for s in filtered["recon_status"])


# ---------- slice 6: filter + plot spec JSON round-trip ----------


def test_filter_spec_json_round_trip_preserves_filter_and_plot_contents() -> None:
	import json as _json

	from ..filters import SPEC_SCHEMA_VERSION, filter_spec_from_json, filter_spec_to_json

	filter_spec = {
		"require_recon_ok": True,
		"bombcell_allowlist": ["good", "non_soma_good", None],
		"min_num_spikes": 50,
		"min_num_branches": 2,
		"min_recon_quality_score": 0.5,
		"project": ["Media_Density_T5_02182026_AR"],
		"chip_id": ["M08073"],
		"div_lo": 6,
		"div_hi": 36,
	}
	plot_spec = {
		"active_tab": "box",
		"box": {"value_col": "branch_count", "group_col": "genotype", "test": "mann_whitney", "correction": "bh"},
		"scatter": {"x": "branch_count", "y": "total_branch_length_um", "color": "genotype", "facet_col": "DIV"},
	}
	text = filter_spec_to_json(filter_spec, plot_spec=plot_spec)
	assert SPEC_SCHEMA_VERSION in text
	# Sorted, indented JSON: parseable round-trip.
	parsed_raw = _json.loads(text)
	assert parsed_raw["schema_version"] == SPEC_SCHEMA_VERSION
	round_tripped = filter_spec_from_json(text)
	assert round_tripped["filters"] == filter_spec
	assert round_tripped["plot"] == plot_spec


def test_filter_spec_from_json_handles_bytes_payload() -> None:
	from ..filters import filter_spec_from_json, filter_spec_to_json

	text = filter_spec_to_json({"require_recon_ok": False}, plot_spec={"active_tab": "histogram"})
	parsed = filter_spec_from_json(text.encode("utf-8"))
	assert parsed["filters"] == {"require_recon_ok": False}
	assert parsed["plot"] == {"active_tab": "histogram"}


def test_filter_spec_from_json_rejects_non_object_payload() -> None:
	from ..filters import filter_spec_from_json

	with pytest.raises(ValueError):
		filter_spec_from_json("[1, 2, 3]")


def test_filter_spec_to_json_serializes_non_str_types_via_default() -> None:
	from pathlib import Path

	from ..filters import filter_spec_to_json

	# A path object must survive serialization (the `default=str` fallback
	# in the writer turns it into a string instead of raising TypeError).
	text = filter_spec_to_json({"output_path": Path("/tmp/something")})
	assert "/tmp/something" in text
