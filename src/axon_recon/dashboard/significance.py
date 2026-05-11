"""Statistical helpers for box-plot significance markers.

Pure functions, no Dash imports. Only `pandas`, `numpy`, `scipy`, and
`statsmodels` are touched at runtime. The Dash app composes:

  1. `compute_pairwise_pvalues` (or `kruskal_wallis_omnibus`)
  2. `apply_correction` (none / bonferroni / holm / bh)
  3. `significance_brackets(fig, corrected)` to draw the bracket + asterisks.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


SUPPORTED_PAIRWISE_TESTS: tuple[str, ...] = ("mann_whitney", "welch_t", "tukey_hsd")
SUPPORTED_OMNIBUS_TESTS: tuple[str, ...] = ("kruskal_wallis",)
SUPPORTED_CORRECTIONS: tuple[str, ...] = ("none", "bonferroni", "holm", "bh")

DEFAULT_THRESHOLDS: tuple[float, ...] = (0.05, 0.01, 0.001)


def _group_arrays(
	df: pd.DataFrame, *, group_col: str, value_col: str
) -> dict[Any, np.ndarray]:
	"""Return `{group_label: 1-D float array of non-NaN values}`.

	Groups that end up empty after dropping NaNs are excluded.
	"""
	if group_col not in df.columns or value_col not in df.columns:
		return {}
	out: dict[Any, np.ndarray] = {}
	values = pd.to_numeric(df[value_col], errors="coerce")
	for label, indices in df.groupby(group_col, dropna=True).indices.items():
		series = values.iloc[indices].dropna()
		if series.empty:
			continue
		out[label] = np.asarray(series.to_numpy(), dtype=float)
	return out


def _ordered_group_pairs(labels: Iterable[Any]) -> list[tuple[Any, Any]]:
	items = list(labels)
	return [(items[i], items[j]) for i in range(len(items)) for j in range(i + 1, len(items))]


def compute_pairwise_pvalues(
	df: pd.DataFrame,
	*,
	group_col: str,
	value_col: str,
	test: str,
) -> dict[tuple[Any, Any], float]:
	"""Pairwise p-values across every group pair in `df[group_col]`.

	Returns `{(group_a, group_b): p_value}`. Pairs that can't be computed
	(too few samples, all-NaN, etc.) are silently dropped from the result.
	"""
	test_key = str(test).strip().lower()
	if test_key not in SUPPORTED_PAIRWISE_TESTS:
		raise ValueError(
			f"Unsupported pairwise test: {test!r}; supported: {SUPPORTED_PAIRWISE_TESTS}"
		)
	groups = _group_arrays(df, group_col=group_col, value_col=value_col)
	labels = list(groups.keys())
	if len(labels) < 2:
		return {}

	if test_key == "tukey_hsd":
		return _tukey_hsd_pairwise(groups)

	from scipy import stats

	out: dict[tuple[Any, Any], float] = {}
	for a, b in _ordered_group_pairs(labels):
		arr_a = groups[a]
		arr_b = groups[b]
		if arr_a.size < 2 or arr_b.size < 2:
			continue
		try:
			if test_key == "mann_whitney":
				result = stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")
			else:  # welch_t
				result = stats.ttest_ind(arr_a, arr_b, equal_var=False, nan_policy="omit")
			p_value = float(getattr(result, "pvalue", float("nan")))
		except (ValueError, TypeError):
			continue
		if not np.isfinite(p_value):
			continue
		out[(a, b)] = p_value
	return out


def _tukey_hsd_pairwise(groups: dict[Any, np.ndarray]) -> dict[tuple[Any, Any], float]:
	from statsmodels.stats.multicomp import pairwise_tukeyhsd

	labels = list(groups.keys())
	flat_values: list[float] = []
	flat_labels: list[Any] = []
	for label in labels:
		arr = groups[label]
		flat_values.extend(arr.tolist())
		flat_labels.extend([label] * arr.size)
	if len(set(flat_labels)) < 2 or len(flat_values) < 3:
		return {}
	try:
		result = pairwise_tukeyhsd(np.asarray(flat_values), np.asarray(flat_labels, dtype=object))
	except (ValueError, TypeError):
		return {}
	out: dict[tuple[Any, Any], float] = {}
	for row in getattr(result, "_results_table", None).data[1:] if hasattr(result, "_results_table") else []:
		# Columns: group1, group2, meandiff, p-adj, lower, upper, reject
		if len(row) < 4:
			continue
		a, b, _meandiff, p_adj = row[0], row[1], row[2], row[3]
		try:
			out[(a, b)] = float(p_adj)
		except (TypeError, ValueError):
			continue
	return out


def kruskal_wallis_omnibus(
	df: pd.DataFrame,
	*,
	group_col: str,
	value_col: str,
) -> float | None:
	"""Single omnibus p-value across all groups; None if undefined."""
	groups = _group_arrays(df, group_col=group_col, value_col=value_col)
	arrays = [arr for arr in groups.values() if arr.size >= 2]
	if len(arrays) < 2:
		return None
	from scipy import stats

	try:
		result = stats.kruskal(*arrays)
	except (ValueError, TypeError):
		return None
	p_value = float(getattr(result, "pvalue", float("nan")))
	return p_value if np.isfinite(p_value) else None


def apply_correction(
	pvalues: dict[tuple[Any, Any], float],
	*,
	method: str,
) -> dict[tuple[Any, Any], float]:
	"""Apply multiple-testing correction. Returns a new `{pair: adj_p}` dict.

	`method`: `none`, `bonferroni`, `holm`, `bh` (Benjamini-Hochberg FDR).
	`pvalues` order is preserved (Python 3.7+ dict-insertion order).
	"""
	method_key = str(method).strip().lower()
	if method_key not in SUPPORTED_CORRECTIONS:
		raise ValueError(
			f"Unsupported correction: {method!r}; supported: {SUPPORTED_CORRECTIONS}"
		)
	if not pvalues:
		return {}
	if method_key == "none":
		return dict(pvalues)

	from statsmodels.stats.multitest import multipletests

	pairs = list(pvalues.keys())
	raw = [float(pvalues[pair]) for pair in pairs]
	method_map = {"bonferroni": "bonferroni", "holm": "holm", "bh": "fdr_bh"}
	_, adjusted, _, _ = multipletests(raw, method=method_map[method_key])
	return {pair: float(adj) for pair, adj in zip(pairs, adjusted)}


def asterisks_for_p(p_value: float, thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS) -> str:
	"""Plot-side significance markers (`*`, `**`, `***`, `n.s.`).

	`thresholds` are alpha cutoffs in *increasing* significance — the
	default `(0.05, 0.01, 0.001)` yields `*` / `**` / `***`.
	"""
	if not np.isfinite(p_value):
		return ""
	sorted_thresholds = sorted(thresholds)
	stars = ""
	for alpha in sorted_thresholds:
		if p_value <= alpha:
			stars += "*"
	return stars if stars else "n.s."


def significance_brackets(
	fig: Any,
	corrected_pvalues: dict[tuple[Any, Any], float],
	*,
	thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
	hide_ns: bool = True,
	y_top: float | None = None,
	step: float | None = None,
) -> Any:
	"""Add bracket lines + asterisks above the existing figure.

	`y_top` and `step` default to figure-derived values when None. Pairs with
	non-significant p-values are skipped when `hide_ns` is True (default) so
	the plot doesn't get cluttered with `n.s.` labels.
	"""
	if fig is None or not corrected_pvalues:
		return fig

	# Determine an existing y range from the figure traces.
	max_y = _figure_y_max(fig)
	if max_y is None:
		max_y = 1.0
	if y_top is None:
		y_top = float(max_y) + max(abs(float(max_y)) * 0.10, 0.5)
	if step is None:
		step = max(abs(float(y_top)) * 0.08, 0.5)

	x_positions = _figure_x_positions(fig)

	for index, (pair, p_value) in enumerate(corrected_pvalues.items()):
		stars = asterisks_for_p(float(p_value), thresholds=thresholds)
		if not stars or (hide_ns and stars == "n.s."):
			continue
		try:
			x_a = x_positions[pair[0]]
			x_b = x_positions[pair[1]]
		except KeyError:
			continue
		bracket_y = float(y_top) + index * float(step)
		# Bracket shape: ⎻ ⎻ ⎻ with little drops at the ends.
		fig.add_shape(
			type="line",
			xref="x",
			yref="y",
			x0=x_a,
			x1=x_b,
			y0=bracket_y,
			y1=bracket_y,
			line={"width": 1, "color": "#333"},
		)
		# Label placed mid-bracket.
		fig.add_annotation(
			x=(x_a + x_b) / 2 if isinstance(x_a, (int, float)) and isinstance(x_b, (int, float)) else x_a,
			y=bracket_y,
			text=stars,
			showarrow=False,
			yshift=8,
			font={"size": 12, "color": "#333"},
		)
	return fig


def _figure_y_max(fig: Any) -> float | None:
	values: list[float] = []
	for trace in getattr(fig, "data", []) or []:
		y = getattr(trace, "y", None)
		if y is None:
			continue
		try:
			arr = np.asarray(y, dtype=float)
		except (TypeError, ValueError):
			continue
		if arr.size == 0:
			continue
		try:
			values.append(float(np.nanmax(arr)))
		except (TypeError, ValueError):
			continue
	if not values:
		return None
	return max(values)


def _figure_x_positions(fig: Any) -> dict[Any, Any]:
	"""Map category labels to their plotted x-axis positions.

	For categorical axes (Plotly box plots over string groups), the labels
	themselves are the x-positions, so the dict is identity. For numeric
	axes, callers can substitute their own positions.
	"""
	categories: list[Any] = []
	for trace in getattr(fig, "data", []) or []:
		x = getattr(trace, "x", None)
		if x is None:
			continue
		try:
			for value in x:
				if value not in categories:
					categories.append(value)
		except TypeError:
			continue
	return {value: value for value in categories}
