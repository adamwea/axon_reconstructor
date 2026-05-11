from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ..significance import (
	DEFAULT_THRESHOLDS,
	apply_correction,
	asterisks_for_p,
	compute_pairwise_pvalues,
	kruskal_wallis_omnibus,
	significance_brackets,
)


def _two_group_df(*, n_per_group: int = 20, mean_a: float = 0.0, mean_b: float = 0.0, sd: float = 1.0, seed: int = 0) -> pd.DataFrame:
	rng = np.random.default_rng(seed)
	a = rng.normal(loc=mean_a, scale=sd, size=n_per_group)
	b = rng.normal(loc=mean_b, scale=sd, size=n_per_group)
	return pd.DataFrame(
		{
			"group": ["A"] * n_per_group + ["B"] * n_per_group,
			"value": list(a) + list(b),
		}
	)


def _three_group_df(*, n: int = 15, means: tuple[float, float, float] = (0.0, 0.0, 5.0), sd: float = 1.0, seed: int = 0) -> pd.DataFrame:
	rng = np.random.default_rng(seed)
	data: list[tuple[str, float]] = []
	for label, mu in zip("ABC", means):
		for value in rng.normal(loc=mu, scale=sd, size=n):
			data.append((label, value))
	return pd.DataFrame(data, columns=["group", "value"])


# ---------- compute_pairwise_pvalues ----------


def test_mann_whitney_returns_low_pvalue_for_separated_groups() -> None:
	df = _two_group_df(mean_a=0.0, mean_b=5.0, sd=0.5)
	result = compute_pairwise_pvalues(df, group_col="group", value_col="value", test="mann_whitney")
	assert ("A", "B") in result or ("B", "A") in result
	p = next(iter(result.values()))
	assert p < 0.001


def test_mann_whitney_returns_large_pvalue_for_overlapping_groups() -> None:
	df = _two_group_df(mean_a=0.0, mean_b=0.0, sd=1.0, seed=42)
	result = compute_pairwise_pvalues(df, group_col="group", value_col="value", test="mann_whitney")
	p = next(iter(result.values()))
	assert p > 0.05


def test_welch_t_returns_low_pvalue_for_separated_groups() -> None:
	df = _two_group_df(mean_a=0.0, mean_b=10.0, sd=0.5)
	result = compute_pairwise_pvalues(df, group_col="group", value_col="value", test="welch_t")
	p = next(iter(result.values()))
	assert p < 1e-10


def test_tukey_hsd_returns_pairs_for_three_groups() -> None:
	df = _three_group_df()
	result = compute_pairwise_pvalues(df, group_col="group", value_col="value", test="tukey_hsd")
	# Tukey HSD yields 3 pairs for 3 groups.
	assert len(result) == 3


def test_compute_pairwise_unsupported_test_raises() -> None:
	df = _two_group_df()
	with pytest.raises(ValueError):
		compute_pairwise_pvalues(df, group_col="group", value_col="value", test="not_a_real_test")


def test_compute_pairwise_missing_columns_returns_empty() -> None:
	df = pd.DataFrame({"x": [1, 2, 3]})
	result = compute_pairwise_pvalues(df, group_col="missing", value_col="x", test="mann_whitney")
	assert result == {}


def test_compute_pairwise_handles_single_group() -> None:
	df = pd.DataFrame({"group": ["A"] * 10, "value": list(range(10))})
	result = compute_pairwise_pvalues(df, group_col="group", value_col="value", test="mann_whitney")
	assert result == {}


# ---------- kruskal_wallis_omnibus ----------


def test_kruskal_wallis_separated_groups_returns_low_p() -> None:
	df = _three_group_df(means=(0.0, 0.0, 5.0))
	p = kruskal_wallis_omnibus(df, group_col="group", value_col="value")
	assert p is not None and p < 0.001


def test_kruskal_wallis_overlapping_groups_returns_high_p() -> None:
	df = _three_group_df(means=(0.0, 0.0, 0.0), seed=42)
	p = kruskal_wallis_omnibus(df, group_col="group", value_col="value")
	assert p is not None and p > 0.1


def test_kruskal_wallis_single_group_returns_none() -> None:
	df = pd.DataFrame({"group": ["A"] * 10, "value": list(range(10))})
	assert kruskal_wallis_omnibus(df, group_col="group", value_col="value") is None


# ---------- apply_correction ----------


def test_apply_correction_none_returns_raw() -> None:
	pvalues = {("A", "B"): 0.04, ("A", "C"): 0.001}
	out = apply_correction(pvalues, method="none")
	assert out == pvalues


def test_apply_correction_bonferroni_inflates_pvalues_and_caps_at_one() -> None:
	pvalues = {("A", "B"): 0.04, ("A", "C"): 0.5, ("B", "C"): 0.6}
	out = apply_correction(pvalues, method="bonferroni")
	# Bonferroni multiplies by the number of tests and caps at 1.
	for pair, p in pvalues.items():
		assert out[pair] >= p
	assert all(out[pair] <= 1.0 for pair in pvalues)
	assert out[("A", "B")] == pytest.approx(min(0.04 * 3, 1.0))


def test_apply_correction_holm_preserves_or_inflates_p() -> None:
	pvalues = {("A", "B"): 0.01, ("A", "C"): 0.04, ("B", "C"): 0.03}
	out = apply_correction(pvalues, method="holm")
	for pair, p in pvalues.items():
		assert out[pair] >= p
		assert out[pair] <= 1.0 + 1e-9


def test_apply_correction_bh_preserves_ordering() -> None:
	pvalues = {("A", "B"): 0.001, ("A", "C"): 0.01, ("B", "C"): 0.05}
	out = apply_correction(pvalues, method="bh")
	# Benjamini-Hochberg preserves the rank order of raw p-values.
	order_raw = sorted(pvalues.keys(), key=lambda k: pvalues[k])
	order_adj = sorted(out.keys(), key=lambda k: out[k])
	assert order_raw == order_adj


def test_apply_correction_unsupported_raises() -> None:
	with pytest.raises(ValueError):
		apply_correction({("A", "B"): 0.5}, method="bogus")


def test_apply_correction_empty_returns_empty() -> None:
	assert apply_correction({}, method="bonferroni") == {}


# ---------- asterisks_for_p ----------


def test_asterisks_for_p_renders_stars_above_thresholds() -> None:
	assert asterisks_for_p(0.0009) == "***"
	assert asterisks_for_p(0.009) == "**"
	assert asterisks_for_p(0.04) == "*"
	assert asterisks_for_p(0.5) == "n.s."
	assert asterisks_for_p(float("nan")) == ""


def test_asterisks_for_p_respects_custom_thresholds() -> None:
	# With thresholds (0.05,), only one star tier exists.
	assert asterisks_for_p(0.04, thresholds=(0.05,)) == "*"
	assert asterisks_for_p(0.001, thresholds=(0.05,)) == "*"


# ---------- significance_brackets (integration with plotly) ----------


def test_significance_brackets_adds_shape_and_annotation_for_significant_pair() -> None:
	import plotly.express as px

	df = _two_group_df(mean_a=0.0, mean_b=5.0, sd=0.3)
	raw = compute_pairwise_pvalues(df, group_col="group", value_col="value", test="mann_whitney")
	corrected = apply_correction(raw, method="none")
	fig = px.box(df, x="group", y="value")
	out = significance_brackets(fig, corrected)
	assert len(out.layout.shapes) == 1
	assert len(out.layout.annotations) == 1
	assert out.layout.annotations[0].text == "***"


def test_significance_brackets_hides_non_significant_pairs_by_default() -> None:
	import plotly.express as px

	df = _two_group_df(mean_a=0.0, mean_b=0.0, sd=1.0, seed=42)
	raw = compute_pairwise_pvalues(df, group_col="group", value_col="value", test="mann_whitney")
	corrected = apply_correction(raw, method="none")
	fig = px.box(df, x="group", y="value")
	out = significance_brackets(fig, corrected)
	assert len(out.layout.shapes) == 0
	assert len(out.layout.annotations) == 0


def test_significance_brackets_empty_pvalues_passthrough() -> None:
	import plotly.express as px

	fig = px.box(pd.DataFrame({"x": [], "y": []}), x="x", y="y")
	out = significance_brackets(fig, {})
	assert out is fig  # no shapes added


def test_significance_brackets_position_annotation_at_categorical_midpoint() -> None:
	"""Asterisks should land between the two boxes, not docked to one end."""
	import plotly.express as px

	df = _two_group_df(mean_a=0.0, mean_b=5.0, sd=0.3)
	raw = compute_pairwise_pvalues(df, group_col="group", value_col="value", test="mann_whitney")
	corrected = apply_correction(raw, method="none")
	fig = px.box(df, x="group", y="value")
	significance_brackets(fig, corrected)

	# Plotly assigns categorical positions 0 and 1; bracket should span them
	# and the asterisk annotation should land at 0.5 (the midpoint).
	shape = fig.layout.shapes[0]
	annotation = fig.layout.annotations[0]
	assert {float(shape.x0), float(shape.x1)} == {0.0, 1.0}
	assert float(annotation.x) == 0.5
	assert annotation.text == "***"


def test_significance_brackets_multiple_pairs_stack_at_midpoints() -> None:
	import plotly.express as px

	df = _three_group_df(means=(0.0, 3.0, 6.0), sd=0.3)
	raw = compute_pairwise_pvalues(df, group_col="group", value_col="value", test="mann_whitney")
	corrected = apply_correction(raw, method="none")
	fig = px.box(df, x="group", y="value")
	significance_brackets(fig, corrected)

	# 3 group pairs (A,B), (A,C), (B,C) — all strongly significant.
	annotation_xs = [float(a.x) for a in fig.layout.annotations]
	# Midpoints between integer category positions: 0.5, 1.0, 1.5.
	for ann_x in annotation_xs:
		# Every label must sit on a half-integer between adjacent boxes.
		assert ann_x in {0.5, 1.0, 1.5}, f"unexpected midpoint x={ann_x}"
