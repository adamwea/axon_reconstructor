from __future__ import annotations

import math

from ..metrics import (
	WELL_SUMMARY_METRIC_COLUMNS,
	branch_count,
	compute_unit_metrics,
	compute_well_summary,
	passthrough_grid_sort_metric,
	recon_density,
	template_density,
	total_branch_length_um,
)


def _isnan(value: float) -> bool:
	return value != value


# ---------- branch_count ----------


def test_branch_count_counts_branches() -> None:
	payload = {"unit_id": 1, "branches": [{"branch_index": 0}, {"branch_index": 1}, {"branch_index": 2}]}
	assert branch_count(payload) == 3.0


def test_branch_count_empty_list() -> None:
	assert branch_count({"branches": []}) == 0.0


def test_branch_count_missing_payload_returns_nan() -> None:
	assert _isnan(branch_count(None))


def test_branch_count_payload_without_branches_key_returns_nan() -> None:
	assert _isnan(branch_count({"unit_id": 1}))


# ---------- total_branch_length_um ----------


def test_total_branch_length_uses_distances_when_present() -> None:
	payload = {
		"branches": [
			{"distances": [1.0, 2.0, 3.0], "polyline_xy": [[0, 0], [0, 100], [0, 300]]},
			{"distances": [4.0], "polyline_xy": [[0, 0], [0, 4]]},
		]
	}
	assert total_branch_length_um(payload) == 10.0


def test_total_branch_length_falls_back_to_polyline_when_distances_missing() -> None:
	payload = {
		"branches": [
			# distances missing; polyline path length = 3+4 = 7 (3-4-5 triangle)
			{"polyline_xy": [[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]]},
		]
	}
	assert math.isclose(total_branch_length_um(payload), 7.0, rel_tol=1e-6)


def test_total_branch_length_one_node_branch_contributes_zero() -> None:
	payload = {"branches": [{"polyline_xy": [[0.0, 0.0]]}]}
	assert total_branch_length_um(payload) == 0.0


def test_total_branch_length_empty_branches_returns_nan() -> None:
	assert _isnan(total_branch_length_um({"branches": []}))


def test_total_branch_length_missing_payload_returns_nan() -> None:
	assert _isnan(total_branch_length_um(None))


# ---------- template_density ----------


def test_template_density_reads_grid_sort_metrics() -> None:
	payload = {"grid_sort_metrics": {"template_density": 0.42}}
	assert math.isclose(template_density(payload), 0.42, rel_tol=1e-9)


def test_template_density_missing_value_returns_nan() -> None:
	assert _isnan(template_density({"grid_sort_metrics": {}}))


def test_template_density_missing_payload_returns_nan() -> None:
	assert _isnan(template_density(None))


def test_template_density_non_dict_grid_sort_metrics_returns_nan() -> None:
	assert _isnan(template_density({"grid_sort_metrics": "not-a-dict"}))


# ---------- recon_density ----------


def test_recon_density_computes_electrodes_per_bbox_area() -> None:
	# bbox: x in [0, 10], y in [0, 5] -> area = 50; 100 electrodes / 50 = 2.0
	merged = {"electrode_ids": [str(i) for i in range(100)]}
	branches = {
		"branches": [
			{"polyline_xy": [[0.0, 0.0], [10.0, 0.0]]},
			{"polyline_xy": [[5.0, 5.0], [5.0, 0.0]]},
		]
	}
	assert math.isclose(recon_density(merged, branches), 2.0, rel_tol=1e-9)


def test_recon_density_zero_bbox_area_returns_nan() -> None:
	# All nodes collinear at y=0 -> height=0 -> area=0
	merged = {"electrode_ids": ["a", "b", "c"]}
	branches = {"branches": [{"polyline_xy": [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]]}]}
	assert _isnan(recon_density(merged, branches))


def test_recon_density_empty_branches_returns_nan() -> None:
	merged = {"electrode_ids": ["a"]}
	assert _isnan(recon_density(merged, {"branches": []}))


def test_recon_density_missing_electrodes_returns_nan() -> None:
	branches = {"branches": [{"polyline_xy": [[0.0, 0.0], [1.0, 1.0]]}]}
	assert _isnan(recon_density({"electrode_ids": []}, branches))
	assert _isnan(recon_density(None, branches))


# ---------- passthrough_grid_sort_metric ----------


def test_passthrough_reads_named_key() -> None:
	payload = {"grid_sort_metrics": {"max_amplitude": -123.4, "max_ptp": 250.0}}
	assert math.isclose(passthrough_grid_sort_metric(payload, key="max_amplitude"), -123.4)
	assert math.isclose(passthrough_grid_sort_metric(payload, key="max_ptp"), 250.0)


def test_passthrough_missing_key_returns_nan() -> None:
	assert _isnan(passthrough_grid_sort_metric({"grid_sort_metrics": {}}, key="max_amplitude"))


def test_passthrough_missing_payload_returns_nan() -> None:
	assert _isnan(passthrough_grid_sort_metric(None, key="max_amplitude"))


# ---------- compute_unit_metrics (aggregator) ----------


def test_compute_unit_metrics_aggregates_all_fields() -> None:
	unit_summary = {
		"grid_sort_metrics": {
			"template_density": 0.5,
			"max_amplitude": -200.0,
			"max_ptp": 400.0,
			"max_delay": 1.25,
		}
	}
	branches = {"branches": [{"distances": [3.0, 4.0]}]}
	merged = {"electrode_ids": ["a", "b", "c"]}
	templates = {"unit_location": {"x_um": 100.5, "y_um": 200.5}}

	metrics = compute_unit_metrics(
		unit_summary_payload=unit_summary,
		branches_payload=branches,
		merged_payload=merged,
		templates_payload=templates,
	)

	# 3+4 = 7 path length sum
	assert metrics["branch_count"] == 1.0
	assert math.isclose(metrics["total_branch_length_um"], 7.0)
	assert math.isclose(metrics["template_density"], 0.5)
	# recon_density needs a bbox; with no polyline_xy provided the result is NaN.
	assert _isnan(metrics["recon_density"])
	assert math.isclose(metrics["max_amplitude_uv"], -200.0)
	assert math.isclose(metrics["max_ptp_uv"], 400.0)
	assert math.isclose(metrics["max_delay_ms"], 1.25)
	assert math.isclose(metrics["unit_location_x_um"], 100.5)
	assert math.isclose(metrics["unit_location_y_um"], 200.5)


def test_compute_unit_metrics_all_none_payloads_returns_nan() -> None:
	metrics = compute_unit_metrics(
		unit_summary_payload=None,
		branches_payload=None,
		merged_payload=None,
		templates_payload=None,
	)
	for key, value in metrics.items():
		assert _isnan(value), f"expected NaN for {key}, got {value}"


# ---------- compute_well_summary ----------


def _make_units_df():
	"""Return a small synthetic units DataFrame covering the well_summary cases."""
	import pandas as pd

	return pd.DataFrame(
		[
			{
				"recon_status": "ok",
				"bombcell_label": "good",
				"branch_count": 1.0,
				"total_branch_length_um": 100.0,
				"template_density": 0.4,
				"recon_density": 0.2,
			},
			{
				"recon_status": "ok",
				"bombcell_label": "non_soma_good",
				"branch_count": 3.0,
				"total_branch_length_um": 300.0,
				"template_density": 0.6,
				"recon_density": 0.4,
			},
			{
				"recon_status": "ok",
				"bombcell_label": "mua",
				"branch_count": 5.0,
				"total_branch_length_um": 500.0,
				"template_density": 0.8,
				"recon_density": 0.6,
			},
			# error row — must not contribute to the metric aggregates.
			{
				"recon_status": "error",
				"bombcell_label": None,
				"branch_count": float("nan"),
				"total_branch_length_um": float("nan"),
				"template_density": float("nan"),
				"recon_density": float("nan"),
			},
		]
	)


def test_compute_well_summary_counts_and_aggregates() -> None:
	df = _make_units_df()
	identity = {
		"project": "P",
		"well_id": "well000",
		"DIV": 36,
		"genotype": "WT",
	}
	summary = compute_well_summary(df, identity)

	# Identity passes through verbatim.
	assert summary["project"] == "P"
	assert summary["well_id"] == "well000"
	assert summary["DIV"] == 36
	assert summary["genotype"] == "WT"

	# Counts cover the whole table; the recon_ok count excludes the error row.
	assert summary["unit_count_total"] == 4
	assert summary["unit_count_recon_ok"] == 3
	assert summary["unit_count_bombcell_good"] == 1
	assert summary["unit_count_bombcell_non_soma_good"] == 1

	# Means/medians cover only the 3 ok rows.
	assert math.isclose(summary["mean_branch_count"], 3.0, rel_tol=1e-9)
	assert math.isclose(summary["median_branch_count"], 3.0, rel_tol=1e-9)
	assert math.isclose(summary["mean_total_branch_length_um"], 300.0, rel_tol=1e-9)
	assert math.isclose(summary["median_total_branch_length_um"], 300.0, rel_tol=1e-9)
	assert math.isclose(summary["mean_template_density"], 0.6, rel_tol=1e-9)
	assert math.isclose(summary["median_template_density"], 0.6, rel_tol=1e-9)
	assert math.isclose(summary["mean_recon_density"], 0.4, rel_tol=1e-9)
	assert math.isclose(summary["median_recon_density"], 0.4, rel_tol=1e-9)


def test_compute_well_summary_empty_df_returns_zero_counts_and_nan_metrics() -> None:
	import pandas as pd

	empty = pd.DataFrame()
	summary = compute_well_summary(empty, {"well_id": "well000"})
	assert summary["well_id"] == "well000"
	assert summary["unit_count_total"] == 0
	assert summary["unit_count_recon_ok"] == 0
	assert summary["unit_count_bombcell_good"] == 0
	assert summary["unit_count_bombcell_non_soma_good"] == 0
	for metric in WELL_SUMMARY_METRIC_COLUMNS:
		assert _isnan(summary[f"mean_{metric}"])
		assert _isnan(summary[f"median_{metric}"])


def test_compute_well_summary_no_ok_rows_returns_nan_metrics() -> None:
	import pandas as pd

	df = pd.DataFrame(
		[{"recon_status": "error", "bombcell_label": None, "branch_count": float("nan")}]
	)
	summary = compute_well_summary(df, {})
	assert summary["unit_count_total"] == 1
	assert summary["unit_count_recon_ok"] == 0
	for metric in WELL_SUMMARY_METRIC_COLUMNS:
		assert _isnan(summary[f"mean_{metric}"])
		assert _isnan(summary[f"median_{metric}"])


def test_compute_well_summary_handles_none_units_df() -> None:
	summary = compute_well_summary(None, {"well_id": "well000"})
	assert summary["unit_count_total"] == 0
	assert summary["unit_count_recon_ok"] == 0
	for metric in WELL_SUMMARY_METRIC_COLUMNS:
		assert _isnan(summary[f"mean_{metric}"])
