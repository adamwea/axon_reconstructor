from __future__ import annotations

import csv
import json
import math
import os
import shutil
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from ...execution.results import TargetStageResult
from .models.results import AnalysisResult


META_COLUMNS = {"h5_path", "stream_id", "well_id", "unit_id", "status", "metric"}


@dataclass(frozen=True)
class _WellContext:
    well_key: str
    well_id: str
    result: AnalysisResult
    target: TargetStageResult


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _resolve_output_path(*, base_dir: Path, relpath: str) -> Path:
    raw = Path(str(relpath)).expanduser()
    if raw.is_absolute():
        return raw
    return base_dir / raw


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _resolve_cross_well_root(*, wells: list[_WellContext], output_rel_root: str) -> Path:
    analysis_dirs = [ctx.result.analysis_out_dir for ctx in wells]
    if not analysis_dirs:
        raise ValueError("Cannot resolve cross-well output path without successful analysis directories")

    well_dirs = [path.parent for path in analysis_dirs]
    if len(well_dirs) == 1:
        common_root = well_dirs[0].parent
    else:
        common_root = Path(os.path.commonpath([str(path) for path in well_dirs]))

    return common_root / str(output_rel_root) / "cross_well"


def _ordered_well_ids(*, well_ids_in_config_order: list[str], ordering: str) -> list[str]:
    if str(ordering).strip().lower() == "alphabetical":
        return sorted(well_ids_in_config_order)

    deduped: list[str] = []
    seen: set[str] = set()
    for well_id in well_ids_in_config_order:
        if well_id in seen:
            continue
        seen.add(well_id)
        deduped.append(well_id)
    return deduped


def _extract_per_unit_values(
    *,
    context: _WellContext,
    source_metric: str,
    rows_cache: dict[str, list[dict[str, str]]],
    warnings: list[str],
) -> list[float]:
    source = str(source_metric or "").strip()
    if not source:
        return []

    if "." in source:
        metric_name, column = source.split(".", 1)
    else:
        metric_name = source
        column = "value"

    output_key = f"per_unit.{metric_name}"
    csv_path_raw = context.result.outputs.get(output_key)
    if not csv_path_raw:
        warnings.append(
            f"Cross-well source '{output_key}' is missing for well '{context.well_key}'; metric '{source_metric}' omitted."
        )
        return []

    csv_path = Path(str(csv_path_raw)).expanduser()
    cache_key = str(csv_path)
    rows = rows_cache.get(cache_key)
    if rows is None:
        rows = _load_csv_rows(csv_path)
        rows_cache[cache_key] = rows

    values: list[float] = []
    for row in rows:
        row_n = _as_float(row.get("n"))
        if column != "n" and row_n is not None and int(row_n) <= 0:
            continue
        parsed = _as_float(row.get(column))
        if parsed is not None:
            values.append(parsed)
    return values


def _extract_per_well_scalar(
    *,
    context: _WellContext,
    source_metric: str,
    source_column: str | None,
    rows_cache: dict[str, list[dict[str, str]]],
    warnings: list[str],
) -> float | None:
    metric_name = str(source_metric or "").strip()
    if not metric_name:
        return None

    output_key = f"per_well.{metric_name}"
    csv_path_raw = context.result.outputs.get(output_key)
    if not csv_path_raw:
        warnings.append(
            f"Cross-well source '{output_key}' is missing for well '{context.well_key}'; bar metric omitted."
        )
        return None

    csv_path = Path(str(csv_path_raw)).expanduser()
    cache_key = str(csv_path)
    rows = rows_cache.get(cache_key)
    if rows is None:
        rows = _load_csv_rows(csv_path)
        rows_cache[cache_key] = rows

    if not rows:
        return None
    row = rows[0]

    row_n = _as_float(row.get("n"))
    if source_column is not None and source_column != "n" and row_n is not None and int(row_n) <= 0:
        return None

    if source_column:
        return _as_float(row.get(source_column))

    if "value" in row:
        return _as_float(row.get("value"))

    if metric_name in row:
        return _as_float(row.get(metric_name))

    for key, value in row.items():
        if key in META_COLUMNS:
            continue
        parsed = _as_float(value)
        if parsed is not None:
            return parsed
    return None


def _try_import_matplotlib() -> Any | None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(float(z) / math.sqrt(2.0))


def _rankdata_average(arr: np.ndarray) -> np.ndarray:
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=float)
    i = 0
    while i < arr.size:
        j = i
        while j + 1 < arr.size and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def _mann_whitney_u(x: list[float], y: list[float]) -> tuple[float, float]:
    x_arr = np.asarray([float(v) for v in x if _as_float(v) is not None], dtype=float)
    y_arr = np.asarray([float(v) for v in y if _as_float(v) is not None], dtype=float)
    n1 = int(x_arr.size)
    n2 = int(y_arr.size)
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))

    pooled = np.concatenate([x_arr, y_arr])
    ranks = _rankdata_average(pooled)
    r1 = float(np.sum(ranks[:n1]))
    u1 = r1 - (n1 * (n1 + 1) / 2.0)
    u2 = float(n1 * n2) - u1
    u_stat = float(min(u1, u2))

    mu = float(n1 * n2) / 2.0
    tie_correction = 1.0
    _, counts = np.unique(pooled, return_counts=True)
    if counts.size > 0:
        t_sum = float(np.sum(counts.astype(float) ** 3 - counts.astype(float)))
        denom = float(pooled.size ** 3 - pooled.size)
        if denom > 0:
            tie_correction = max(0.0, 1.0 - (t_sum / denom))

    sigma = math.sqrt(float(n1 * n2 * (n1 + n2 + 1) / 12.0) * tie_correction)
    if sigma <= 0:
        return (u_stat, 1.0)

    continuity = 0.5 if u_stat < mu else -0.5
    z = (u_stat - mu + continuity) / sigma
    p_value = 2.0 * _norm_sf(abs(float(z)))
    return (u_stat, float(max(0.0, min(1.0, p_value))))


def _cohens_d(x: list[float], y: list[float]) -> float | None:
    x_arr = np.asarray([float(v) for v in x if _as_float(v) is not None], dtype=float)
    y_arr = np.asarray([float(v) for v in y if _as_float(v) is not None], dtype=float)
    n1 = int(x_arr.size)
    n2 = int(y_arr.size)
    if n1 < 2 or n2 < 2:
        return None

    mean_diff = float(np.mean(x_arr) - np.mean(y_arr))
    s1 = float(np.var(x_arr, ddof=1))
    s2 = float(np.var(y_arr, ddof=1))
    pooled_var = ((n1 - 1) * s1 + (n2 - 1) * s2) / float(n1 + n2 - 2)
    if pooled_var <= 0:
        return None
    return float(mean_diff / math.sqrt(pooled_var))


def _cliffs_delta(x: list[float], y: list[float]) -> float | None:
    x_arr = np.asarray([float(v) for v in x if _as_float(v) is not None], dtype=float)
    y_arr = np.asarray([float(v) for v in y if _as_float(v) is not None], dtype=float)
    n1 = int(x_arr.size)
    n2 = int(y_arr.size)
    if n1 == 0 or n2 == 0:
        return None
    diffs = x_arr[:, None] - y_arr[None, :]
    greater = float(np.sum(diffs > 0))
    less = float(np.sum(diffs < 0))
    return float((greater - less) / float(n1 * n2))


def _adjust_pvalues(p_values: list[float], method: str) -> list[float]:
    clean_method = str(method or "none").strip().lower()
    m = len(p_values)
    if m == 0 or clean_method in {"none", ""}:
        return [float(p) for p in p_values]

    p = np.asarray([float(max(0.0, min(1.0, v))) for v in p_values], dtype=float)
    if clean_method == "bonferroni":
        return [float(min(1.0, v * m)) for v in p.tolist()]

    if clean_method == "holm":
        order = np.argsort(p)
        adjusted = np.zeros(m, dtype=float)
        running = 0.0
        for rank, idx in enumerate(order.tolist()):
            factor = float(m - rank)
            value = min(1.0, float(p[idx]) * factor)
            running = max(running, value)
            adjusted[idx] = running
        return [float(v) for v in adjusted.tolist()]

    if clean_method == "fdr_bh":
        order = np.argsort(p)
        adjusted_sorted = np.zeros(m, dtype=float)
        running = 1.0
        for rev_rank, idx in enumerate(order[::-1].tolist(), start=1):
            i = m - rev_rank + 1
            value = float(p[idx]) * float(m) / float(i)
            running = min(running, value)
            adjusted_sorted[idx] = min(1.0, running)
        return [float(v) for v in adjusted_sorted.tolist()]

    return [float(v) for v in p.tolist()]


def _stars_for_pvalue(*, p_value: float, thresholds: dict[str, Any]) -> str:
    p = float(p_value)
    t1 = _as_float(thresholds.get("one_star_max_p"))
    t2 = _as_float(thresholds.get("two_star_max_p"))
    t3 = _as_float(thresholds.get("three_star_max_p"))
    t4 = _as_float(thresholds.get("four_star_max_p"))
    t1 = 0.05 if t1 is None else t1
    t2 = 0.01 if t2 is None else t2
    t3 = 0.001 if t3 is None else t3
    t4 = 0.0001 if t4 is None else t4

    if p <= t4:
        return "****"
    if p <= t3:
        return "***"
    if p <= t2:
        return "**"
    if p <= t1:
        return "*"
    return "ns"


def _format_annotation_text(*, row: dict[str, Any], annotation_cfg: dict[str, Any], alpha: float) -> str:
    mode = str(annotation_cfg.get("mode", "stars_and_pvalue") or "stars_and_pvalue").strip().lower()
    hide_ns = _as_bool(annotation_cfg.get("hide_ns", False), False)
    include_effect_size = _as_bool(annotation_cfg.get("include_effect_size", False), False)
    thresholds = annotation_cfg.get("star_thresholds", {}) if isinstance(annotation_cfg.get("star_thresholds", {}), dict) else {}

    p_adj = _as_float(row.get("p_value_adj"))
    p_raw = _as_float(row.get("p_value"))
    p_value = p_adj if p_adj is not None else p_raw
    if p_value is None:
        return ""

    stars = _stars_for_pvalue(p_value=float(p_value), thresholds=thresholds)
    if hide_ns and stars == "ns":
        return ""

    p_part = f"p={float(p_value):.3g}"
    if mode == "stars":
        text = stars
    elif mode == "pvalue":
        text = p_part
    else:
        text = f"{stars} ({p_part})"

    if include_effect_size:
        effect_size = _as_float(row.get("effect_size"))
        effect_name = str(row.get("effect_size_name", "effect") or "effect")
        if effect_size is not None:
            text = f"{text} {effect_name}={effect_size:.3g}"

    if _as_bool(row.get("significant", False), False) and p_value <= float(alpha):
        return text
    if hide_ns:
        return ""
    return text


def _annotate_pairwise(
    *,
    ax: Any,
    x_left: float,
    x_right: float,
    text: str,
    y_offset_fraction: float,
    bracket_linewidth: float,
    text_fontsize: float,
) -> None:
    if not text:
        return

    y_min, y_max = ax.get_ylim()
    span = max(1e-9, float(y_max - y_min))
    y = float(y_max + span * max(0.01, y_offset_fraction))
    h = float(span * 0.02)

    ax.plot([x_left, x_left, x_right, x_right], [y, y + h, y + h, y], lw=bracket_linewidth, c="black")
    ax.text((x_left + x_right) / 2.0, y + h, text, ha="center", va="bottom", fontsize=text_fontsize)
    ax.set_ylim(y_min, y + h + span * 0.05)


def _well_labels(
    *,
    well_ids: list[str],
    counts: dict[str, int],
    show_counts: bool,
    count_format: str,
) -> list[str]:
    if not show_counts:
        return [str(w) for w in well_ids]

    labels: list[str] = []
    for well in well_ids:
        n = int(counts.get(well, 0))
        try:
            suffix = str(count_format).format(n=n, well=well)
        except Exception:
            suffix = f"n={n}"
        labels.append(f"{well}\n{suffix}")
    return labels


def _compute_pairwise_rows(
    *,
    metric_id: str,
    plot_kind: str,
    series_by_well: dict[str, list[float]],
    stats_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    min_samples = int(max(1, int(stats_cfg.get("min_samples_per_well", 3) or 3)))
    alpha = float(_as_float(stats_cfg.get("alpha")) or 0.05)
    effect_cfg = stats_cfg.get("effect_size", {}) if isinstance(stats_cfg.get("effect_size", {}), dict) else {}
    effect_enabled = _as_bool(effect_cfg.get("compute", False), False)

    eligible = {
        well_id: [float(v) for v in values if _as_float(v) is not None]
        for well_id, values in series_by_well.items()
        if len([v for v in values if _as_float(v) is not None]) >= min_samples
    }

    rows: list[dict[str, Any]] = []
    for well_a, well_b in combinations(sorted(eligible.keys()), 2):
        values_a = eligible[well_a]
        values_b = eligible[well_b]
        u_stat, p_value = _mann_whitney_u(values_a, values_b)
        row: dict[str, Any] = {
            "metric_id": metric_id,
            "plot_kind": plot_kind,
            "test_name": "mannwhitneyu",
            "well_a": well_a,
            "well_b": well_b,
            "n_a": int(len(values_a)),
            "n_b": int(len(values_b)),
            "u_statistic": u_stat,
            "p_value": p_value,
        }

        if effect_enabled:
            nonparam_name = str(effect_cfg.get("nonparametric_metric", "cliffs_delta") or "cliffs_delta").strip().lower()
            if nonparam_name == "cohens_d":
                row["effect_size_name"] = "cohens_d"
                row["effect_size"] = _cohens_d(values_a, values_b)
            else:
                row["effect_size_name"] = "cliffs_delta"
                row["effect_size"] = _cliffs_delta(values_a, values_b)

        rows.append(row)

    correction_cfg = (
        stats_cfg.get("multiple_testing_correction", {})
        if isinstance(stats_cfg.get("multiple_testing_correction", {}), dict)
        else {}
    )
    correction_enabled = _as_bool(correction_cfg.get("enable", False), False)
    correction_method = str(correction_cfg.get("method", "none") or "none") if correction_enabled else "none"
    adjusted = _adjust_pvalues(
        [float(_as_float(row.get("p_value")) or 1.0) for row in rows],
        method=correction_method,
    )

    for row, p_adj in zip(rows, adjusted, strict=False):
        row["p_value_adj"] = float(p_adj)
        row["alpha"] = float(alpha)
        row["significant"] = bool(float(p_adj) <= float(alpha))
        row["p_adjust_method"] = str(correction_method)

    return rows


def _plot_box_and_whisker(
    *,
    plt: Any,
    out_path: Path,
    metric_name: str,
    values_by_well: dict[str, list[float]],
    base_well_by_key: dict[str, str],
    well_order: list[str],
    plot_defaults: dict[str, Any],
    metric_cfg: dict[str, Any],
    stats_row_for_annotation: dict[str, Any] | None,
    stats_cfg: dict[str, Any],
) -> None:
    use_wells = [well for well in well_order if values_by_well.get(well)]
    if not use_wells:
        return

    fig_width = float(_as_float(plot_defaults.get("fig_width_inches")) or 7.2)
    fig_height = float(_as_float(plot_defaults.get("fig_height_inches")) or 4.8)
    dpi = int(_as_float(plot_defaults.get("dpi")) or 300)
    style = str(plot_defaults.get("style", "default") or "default")
    grid_alpha = float(_as_float(plot_defaults.get("grid_alpha")) or 0.2)
    rotate_labels = float(_as_float(plot_defaults.get("rotate_xtick_labels_deg")) or 0.0)
    tight_layout = _as_bool(plot_defaults.get("tight_layout", True), True)

    title_fontsize = float(_as_float(plot_defaults.get("title_fontsize")) or 11.0)
    axis_label_fontsize = float(_as_float(plot_defaults.get("axis_label_fontsize")) or 10.0)
    tick_label_fontsize = float(_as_float(plot_defaults.get("tick_label_fontsize")) or 9.0)

    plt.style.use(style)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    box_defaults = metric_cfg.get("_box_defaults", {})
    show_points = _as_bool(metric_cfg.get("show_points", box_defaults.get("show_points", True)), True)
    point_alpha = float(_as_float(metric_cfg.get("point_alpha", box_defaults.get("point_alpha", 0.45))) or 0.45)
    point_size = float(_as_float(metric_cfg.get("point_size", box_defaults.get("point_size", 18))) or 18.0)
    jitter = float(_as_float(metric_cfg.get("jitter", box_defaults.get("jitter", 0.08))) or 0.08)
    whiskers = metric_cfg.get("whisker_percentiles", box_defaults.get("whisker_percentiles", [5, 95]))
    whis = [5, 95]
    if isinstance(whiskers, (list, tuple)) and len(whiskers) == 2:
        a = _as_float(whiskers[0])
        b = _as_float(whiskers[1])
        if a is not None and b is not None:
            whis = [float(a), float(b)]
    show_notch = _as_bool(metric_cfg.get("show_notch", box_defaults.get("show_notch", False)), False)
    use_log_y = _as_bool(metric_cfg.get("use_log_y", box_defaults.get("use_log_y", False)), False)

    show_counts = _as_bool(plot_defaults.get("show_unit_counts_on_x", True), True)
    count_fmt = str(plot_defaults.get("unit_count_label_format", "n={n}"))
    counts = {well: int(len(values_by_well.get(well, []))) for well in use_wells}
    labels = _well_labels(well_ids=use_wells, counts=counts, show_counts=show_counts, count_format=count_fmt)

    data = [values_by_well[well] for well in use_wells]
    box = ax.boxplot(
        data,
        labels=labels,
        whis=whis,
        notch=show_notch,
        patch_artist=True,
    )

    well_colors = plot_defaults.get("well_colors", {}) if isinstance(plot_defaults.get("well_colors", {}), dict) else {}
    for patch, well in zip(box.get("boxes", []), use_wells, strict=False):
        base_well = str(base_well_by_key.get(well, ""))
        color = str(well_colors.get(well, well_colors.get(base_well, "#4c72b0")))
        patch.set_facecolor(color)
        patch.set_alpha(0.45)

    if show_points:
        rng = np.random.default_rng(1337)
        for idx, well in enumerate(use_wells, start=1):
            values = np.asarray(values_by_well[well], dtype=float)
            if values.size == 0:
                continue
            xs = rng.normal(loc=idx, scale=max(0.0, jitter), size=values.size)
            ax.scatter(xs, values, s=point_size, alpha=point_alpha, color="black", linewidths=0)

    ax.set_title(str(metric_cfg.get("title", metric_name)), fontsize=title_fontsize)
    ax.set_ylabel(str(metric_cfg.get("y_label", metric_name)), fontsize=axis_label_fontsize)
    ax.tick_params(axis="x", labelsize=tick_label_fontsize, rotation=rotate_labels)
    ax.tick_params(axis="y", labelsize=tick_label_fontsize)
    ax.grid(True, axis="y", alpha=grid_alpha)
    if use_log_y:
        ax.set_yscale("log")

    annotation_cfg = stats_cfg.get("annotation", {}) if isinstance(stats_cfg.get("annotation", {}), dict) else {}
    if stats_row_for_annotation is not None and _as_bool(annotation_cfg.get("mark_on_plot", True), True):
        well_a = str(stats_row_for_annotation.get("well_a", ""))
        well_b = str(stats_row_for_annotation.get("well_b", ""))
        if well_a in use_wells and well_b in use_wells:
            x_left = float(use_wells.index(well_a) + 1)
            x_right = float(use_wells.index(well_b) + 1)
            text = _format_annotation_text(
                row=stats_row_for_annotation,
                annotation_cfg=annotation_cfg,
                alpha=float(_as_float(stats_cfg.get("alpha")) or 0.05),
            )
            _annotate_pairwise(
                ax=ax,
                x_left=x_left,
                x_right=x_right,
                text=text,
                y_offset_fraction=float(_as_float(annotation_cfg.get("y_offset_fraction")) or 0.04),
                bracket_linewidth=float(_as_float(annotation_cfg.get("bracket_linewidth")) or 1.0),
                text_fontsize=float(_as_float(annotation_cfg.get("text_fontsize")) or 9.0),
            )

    if tight_layout:
        fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_bar(
    *,
    plt: Any,
    out_path: Path,
    metric_name: str,
    values_by_well: dict[str, float],
    errors_by_well: dict[str, float | None],
    sample_counts: dict[str, int],
    base_well_by_key: dict[str, str],
    well_order: list[str],
    plot_defaults: dict[str, Any],
    metric_cfg: dict[str, Any],
    stats_row_for_annotation: dict[str, Any] | None,
    stats_cfg: dict[str, Any],
) -> None:
    use_wells = [well for well in well_order if _as_float(values_by_well.get(well)) is not None]
    if not use_wells:
        return

    fig_width = float(_as_float(plot_defaults.get("fig_width_inches")) or 7.2)
    fig_height = float(_as_float(plot_defaults.get("fig_height_inches")) or 4.8)
    dpi = int(_as_float(plot_defaults.get("dpi")) or 300)
    style = str(plot_defaults.get("style", "default") or "default")
    grid_alpha = float(_as_float(plot_defaults.get("grid_alpha")) or 0.2)
    rotate_labels = float(_as_float(plot_defaults.get("rotate_xtick_labels_deg")) or 0.0)
    tight_layout = _as_bool(plot_defaults.get("tight_layout", True), True)

    title_fontsize = float(_as_float(plot_defaults.get("title_fontsize")) or 11.0)
    axis_label_fontsize = float(_as_float(plot_defaults.get("axis_label_fontsize")) or 10.0)
    tick_label_fontsize = float(_as_float(plot_defaults.get("tick_label_fontsize")) or 9.0)

    plt.style.use(style)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    bar_defaults = metric_cfg.get("_bar_defaults", {})
    bar_width = float(_as_float(metric_cfg.get("bar_width", bar_defaults.get("bar_width", 0.65))) or 0.65)
    capsize = float(_as_float(metric_cfg.get("capsize", bar_defaults.get("capsize", 4))) or 4.0)
    annotate_values = _as_bool(metric_cfg.get("annotate_values", bar_defaults.get("annotate_values", True)), True)
    value_decimals = int(_as_float(metric_cfg.get("value_decimals", bar_defaults.get("value_decimals", 2))) or 2)
    use_log_y = _as_bool(metric_cfg.get("use_log_y", bar_defaults.get("use_log_y", False)), False)

    show_counts = _as_bool(plot_defaults.get("show_unit_counts_on_x", True), True)
    count_fmt = str(plot_defaults.get("unit_count_label_format", "n={n}"))
    labels = _well_labels(well_ids=use_wells, counts=sample_counts, show_counts=show_counts, count_format=count_fmt)

    heights = [float(values_by_well[well]) for well in use_wells]
    y_errors = [errors_by_well.get(well) for well in use_wells]
    use_yerr = any(_as_float(err) is not None for err in y_errors)
    yerr_values = (
        [float(_as_float(err)) if _as_float(err) is not None else float("nan") for err in y_errors]
        if use_yerr
        else None
    )

    x = np.arange(len(use_wells), dtype=float)
    well_colors = plot_defaults.get("well_colors", {}) if isinstance(plot_defaults.get("well_colors", {}), dict) else {}
    colors = [
        str(well_colors.get(well, well_colors.get(str(base_well_by_key.get(well, "")), "#4c72b0")))
        for well in use_wells
    ]

    bars = ax.bar(x, heights, yerr=yerr_values, width=bar_width, capsize=capsize, color=colors)
    ax.set_xticks(x, labels)
    ax.set_title(str(metric_cfg.get("title", metric_name)), fontsize=title_fontsize)
    ax.set_ylabel(str(metric_cfg.get("y_label", metric_name)), fontsize=axis_label_fontsize)
    ax.tick_params(axis="x", labelsize=tick_label_fontsize, rotation=rotate_labels)
    ax.tick_params(axis="y", labelsize=tick_label_fontsize)
    ax.grid(True, axis="y", alpha=grid_alpha)
    if use_log_y:
        ax.set_yscale("log")

    if annotate_values:
        for rect, value in zip(bars, heights, strict=False):
            height = float(rect.get_height())
            ax.text(
                float(rect.get_x() + rect.get_width() / 2.0),
                height,
                f"{value:.{value_decimals}f}",
                ha="center",
                va="bottom",
                fontsize=tick_label_fontsize,
            )

    annotation_cfg = stats_cfg.get("annotation", {}) if isinstance(stats_cfg.get("annotation", {}), dict) else {}
    if stats_row_for_annotation is not None and _as_bool(annotation_cfg.get("mark_on_plot", True), True):
        well_a = str(stats_row_for_annotation.get("well_a", ""))
        well_b = str(stats_row_for_annotation.get("well_b", ""))
        if well_a in use_wells and well_b in use_wells:
            x_left = float(use_wells.index(well_a))
            x_right = float(use_wells.index(well_b))
            text = _format_annotation_text(
                row=stats_row_for_annotation,
                annotation_cfg=annotation_cfg,
                alpha=float(_as_float(stats_cfg.get("alpha")) or 0.05),
            )
            _annotate_pairwise(
                ax=ax,
                x_left=x_left,
                x_right=x_right,
                text=text,
                y_offset_fraction=float(_as_float(annotation_cfg.get("y_offset_fraction")) or 0.04),
                bracket_linewidth=float(_as_float(annotation_cfg.get("bracket_linewidth")) or 1.0),
                text_fontsize=float(_as_float(annotation_cfg.get("text_fontsize")) or 9.0),
            )

    if tight_layout:
        fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def generate_cross_well_artifacts(
    *,
    target_results: list[TargetStageResult],
    metrics_cfg: dict[str, Any],
    output_rel_root: str,
    force_restart: bool = False,
    force_replot: bool = False,
) -> tuple[dict[str, str], list[str]]:
    outputs: dict[str, str] = {}
    warnings: list[str] = []

    cross_cfg = metrics_cfg.get("cross_well", {}) if isinstance(metrics_cfg.get("cross_well", {}), dict) else {}
    if not cross_cfg:
        return outputs, warnings

    successful_wells: list[_WellContext] = []
    staged_items: list[TargetStageResult] = []
    for item in target_results:
        if item.status != "ok" or not isinstance(item.result, AnalysisResult):
            continue
        staged_items.append(item)

    well_id_counts: dict[str, int] = {}
    for item in staged_items:
        base_well_id = str(item.target.stream_id)
        well_id_counts[base_well_id] = int(well_id_counts.get(base_well_id, 0)) + 1

    for item in staged_items:
        base_well_id = str(item.target.stream_id)
        if int(well_id_counts.get(base_well_id, 0)) > 1:
            well_key = f"{item.target.dataset_index}:{base_well_id}"
        else:
            well_key = base_well_id

        successful_wells.append(
            _WellContext(
                well_key=well_key,
                well_id=base_well_id,
                result=item.result,
                target=item,
            )
        )

    if not successful_wells:
        warnings.append("Cross-well artifacts skipped: no successful well-level analysis results were available.")
        return outputs, warnings

    if len(successful_wells) < 2:
        warnings.append("Cross-well artifacts are generated with fewer than two wells; statistical tests may be unavailable.")

    cross_root = _resolve_cross_well_root(wells=successful_wells, output_rel_root=output_rel_root)
    full_restart = bool(force_restart) and (not bool(force_replot))
    if full_restart and cross_root.exists():
        shutil.rmtree(cross_root)
    cross_root.mkdir(parents=True, exist_ok=True)

    ordering = str(cross_cfg.get("ordering", "config") or "config")
    well_order = _ordered_well_ids(
        well_ids_in_config_order=[ctx.well_key for ctx in successful_wells],
        ordering=ordering,
    )
    context_by_well = {ctx.well_key: ctx for ctx in successful_wells}
    base_well_by_key = {ctx.well_key: ctx.well_id for ctx in successful_wells}

    plot_defaults = cross_cfg.get("plot_defaults", {}) if isinstance(cross_cfg.get("plot_defaults", {}), dict) else {}
    stats_cfg = (
        cross_cfg.get("statistical_testing", {})
        if isinstance(cross_cfg.get("statistical_testing", {}), dict)
        else {}
    )
    stats_enabled = _as_bool(stats_cfg.get("enable", False), False)
    apply_to = str(stats_cfg.get("apply_to", "both") or "both").strip().lower()

    rows_cache: dict[str, list[dict[str, str]]] = {}
    pairwise_rows: list[dict[str, Any]] = []

    plt = _try_import_matplotlib()
    if plt is None:
        warnings.append("matplotlib is not available; cross-well plots were skipped.")

    box_cfg = cross_cfg.get("box_and_whisker_plots", {}) if isinstance(cross_cfg.get("box_and_whisker_plots", {}), dict) else {}
    box_defaults = box_cfg.get("defaults", {}) if isinstance(box_cfg.get("defaults", {}), dict) else {}
    box_dir = cross_root / str(box_cfg.get("reldir", "box_and_whisker_plots/"))

    for metric_name, metric_cfg_raw in box_cfg.items():
        if metric_name in {"reldir", "defaults"}:
            continue
        metric_cfg = metric_cfg_raw if isinstance(metric_cfg_raw, dict) else {}
        source_level = str(metric_cfg.get("source_level", "per_unit") or "per_unit").strip().lower()
        source_metric = str(metric_cfg.get("source_metric", "") or "").strip()
        if source_level != "per_unit":
            warnings.append(
                f"Cross-well box metric '{metric_name}' uses unsupported source_level '{source_level}'; expected per_unit."
            )
            continue
        if not source_metric:
            warnings.append(f"Cross-well box metric '{metric_name}' is missing source_metric.")
            continue

        values_by_well: dict[str, list[float]] = {}
        for well_id in well_order:
            ctx = context_by_well.get(well_id)
            if ctx is None:
                continue
            values_by_well[well_id] = _extract_per_unit_values(
                context=ctx,
                source_metric=source_metric,
                rows_cache=rows_cache,
                warnings=warnings,
            )

        if not any(len(values) > 0 for values in values_by_well.values()):
            warnings.append(
                f"Cross-well box metric '{metric_name}' has no numeric values across selected wells; plot emission skipped."
            )
            continue

        metric_pairwise_rows: list[dict[str, Any]] = []
        if stats_enabled and apply_to in {"box_and_whisker_plots", "both"}:
            metric_pairwise_rows = _compute_pairwise_rows(
                metric_id=f"box_and_whisker_plots.{metric_name}",
                plot_kind="box_and_whisker_plots",
                series_by_well=values_by_well,
                stats_cfg=stats_cfg,
            )
            pairwise_rows.extend(metric_pairwise_rows)

        if plt is None:
            continue

        metric_cfg_local = dict(metric_cfg)
        metric_cfg_local["_box_defaults"] = box_defaults

        write_png = _as_bool(metric_cfg.get("write_png", box_defaults.get("write_png", True)), True)
        write_pdf = _as_bool(metric_cfg.get("write_pdf", box_defaults.get("write_pdf", False)), False)

        annotation_row = metric_pairwise_rows[0] if len(metric_pairwise_rows) == 1 else None

        if write_png:
            png_relpath = str(metric_cfg.get("png_relpath", f"{metric_name}.png"))
            png_path = _resolve_output_path(base_dir=box_dir, relpath=png_relpath)
            _plot_box_and_whisker(
                plt=plt,
                out_path=png_path,
                metric_name=metric_name,
                values_by_well=values_by_well,
                base_well_by_key=base_well_by_key,
                well_order=well_order,
                plot_defaults=plot_defaults,
                metric_cfg=metric_cfg_local,
                stats_row_for_annotation=annotation_row,
                stats_cfg=stats_cfg,
            )
            if png_path.exists():
                outputs[f"cross_well.box_and_whisker_plots.{metric_name}.png"] = str(png_path)
            else:
                warnings.append(
                    f"Cross-well box metric '{metric_name}' PNG was requested but no file was produced."
                )

        if write_pdf:
            pdf_relpath = str(metric_cfg.get("pdf_relpath", f"{metric_name}.pdf"))
            pdf_path = _resolve_output_path(base_dir=box_dir, relpath=pdf_relpath)
            _plot_box_and_whisker(
                plt=plt,
                out_path=pdf_path,
                metric_name=metric_name,
                values_by_well=values_by_well,
                base_well_by_key=base_well_by_key,
                well_order=well_order,
                plot_defaults=plot_defaults,
                metric_cfg=metric_cfg_local,
                stats_row_for_annotation=annotation_row,
                stats_cfg=stats_cfg,
            )
            if pdf_path.exists():
                outputs[f"cross_well.box_and_whisker_plots.{metric_name}.pdf"] = str(pdf_path)
            else:
                warnings.append(
                    f"Cross-well box metric '{metric_name}' PDF was requested but no file was produced."
                )

    bar_cfg = cross_cfg.get("bar_plots", {}) if isinstance(cross_cfg.get("bar_plots", {}), dict) else {}
    bar_defaults = bar_cfg.get("defaults", {}) if isinstance(bar_cfg.get("defaults", {}), dict) else {}
    bar_dir = cross_root / str(bar_cfg.get("reldir", "bar_plots/"))

    for metric_name, metric_cfg_raw in bar_cfg.items():
        if metric_name in {"reldir", "defaults"}:
            continue
        metric_cfg = metric_cfg_raw if isinstance(metric_cfg_raw, dict) else {}
        source_level = str(metric_cfg.get("source_level", "per_well") or "per_well").strip().lower()
        source_metric = str(metric_cfg.get("source_metric", "") or "").strip()
        source_column = metric_cfg.get("source_column")
        source_column = str(source_column).strip() if source_column is not None else None

        values_by_well: dict[str, float] = {}
        errors_by_well: dict[str, float | None] = {}
        sample_counts: dict[str, int] = {}

        for well_id in well_order:
            ctx = context_by_well.get(well_id)
            if ctx is None:
                continue

            if source_level == "per_well":
                value = _extract_per_well_scalar(
                    context=ctx,
                    source_metric=source_metric,
                    source_column=source_column,
                    rows_cache=rows_cache,
                    warnings=warnings,
                )
                if value is not None:
                    values_by_well[well_id] = float(value)
                    errors_by_well[well_id] = None
                sample_counts[well_id] = 1
                continue

            if source_level == "per_well_stats":
                value = _extract_per_well_scalar(
                    context=ctx,
                    source_metric=source_metric,
                    source_column=source_column,
                    rows_cache=rows_cache,
                    warnings=warnings,
                )
                if value is not None:
                    values_by_well[well_id] = float(value)

                error_bar_mode = str(metric_cfg.get("error_bar", bar_defaults.get("error_bar", "sem")) or "sem").strip().lower()
                if error_bar_mode == "none":
                    errors_by_well[well_id] = None
                elif error_bar_mode == "sd":
                    errors_by_well[well_id] = _extract_per_well_scalar(
                        context=ctx,
                        source_metric=source_metric,
                        source_column="std",
                        rows_cache=rows_cache,
                        warnings=warnings,
                    )
                elif error_bar_mode == "ci95":
                    sem_value = _extract_per_well_scalar(
                        context=ctx,
                        source_metric=source_metric,
                        source_column="sem",
                        rows_cache=rows_cache,
                        warnings=warnings,
                    )
                    errors_by_well[well_id] = float(1.96 * sem_value) if sem_value is not None else None
                else:
                    errors_by_well[well_id] = _extract_per_well_scalar(
                        context=ctx,
                        source_metric=source_metric,
                        source_column="sem",
                        rows_cache=rows_cache,
                        warnings=warnings,
                    )

                n_value = _extract_per_well_scalar(
                    context=ctx,
                    source_metric=source_metric,
                    source_column="n",
                    rows_cache=rows_cache,
                    warnings=warnings,
                )
                sample_counts[well_id] = int(n_value) if n_value is not None else 0
                continue

            warnings.append(
                f"Cross-well bar metric '{metric_name}' uses unsupported source_level '{source_level}'; metric omitted."
            )

        if not any(_as_float(value) is not None for value in values_by_well.values()):
            warnings.append(
                f"Cross-well bar metric '{metric_name}' has no numeric values across selected wells; plot emission skipped."
            )
            continue

        metric_pairwise_rows: list[dict[str, Any]] = []
        metric_stats_cfg = metric_cfg.get("stats", {}) if isinstance(metric_cfg.get("stats", {}), dict) else {}
        metric_stats_enabled = _as_bool(metric_stats_cfg.get("enable", True), True)
        if stats_enabled and metric_stats_enabled and apply_to in {"bar_plots", "both"}:
            stats_source_level = str(metric_cfg.get("stats_source_level", "per_unit") or "per_unit").strip().lower()
            stats_source_metric = str(metric_cfg.get("stats_source_metric", "") or "").strip()

            if stats_source_level == "per_unit" and stats_source_metric:
                series_by_well: dict[str, list[float]] = {}
                for well_id in well_order:
                    ctx = context_by_well.get(well_id)
                    if ctx is None:
                        continue
                    series_by_well[well_id] = _extract_per_unit_values(
                        context=ctx,
                        source_metric=stats_source_metric,
                        rows_cache=rows_cache,
                        warnings=warnings,
                    )

                metric_pairwise_rows = _compute_pairwise_rows(
                    metric_id=f"bar_plots.{metric_name}",
                    plot_kind="bar_plots",
                    series_by_well=series_by_well,
                    stats_cfg=stats_cfg,
                )
                pairwise_rows.extend(metric_pairwise_rows)
            elif metric_stats_enabled:
                warnings.append(
                    f"Cross-well bar metric '{metric_name}' cannot run statistical testing without stats_source_metric from per_unit."
                )

        if plt is None:
            continue

        metric_cfg_local = dict(metric_cfg)
        metric_cfg_local["_bar_defaults"] = bar_defaults
        write_png = _as_bool(metric_cfg.get("write_png", bar_defaults.get("write_png", True)), True)
        write_pdf = _as_bool(metric_cfg.get("write_pdf", bar_defaults.get("write_pdf", False)), False)

        annotation_row = metric_pairwise_rows[0] if len(metric_pairwise_rows) == 1 else None

        if write_png:
            png_relpath = str(metric_cfg.get("png_relpath", f"{metric_name}.png"))
            png_path = _resolve_output_path(base_dir=bar_dir, relpath=png_relpath)
            _plot_bar(
                plt=plt,
                out_path=png_path,
                metric_name=metric_name,
                values_by_well=values_by_well,
                errors_by_well=errors_by_well,
                sample_counts=sample_counts,
                base_well_by_key=base_well_by_key,
                well_order=well_order,
                plot_defaults=plot_defaults,
                metric_cfg=metric_cfg_local,
                stats_row_for_annotation=annotation_row,
                stats_cfg=stats_cfg,
            )
            if png_path.exists():
                outputs[f"cross_well.bar_plots.{metric_name}.png"] = str(png_path)
            else:
                warnings.append(
                    f"Cross-well bar metric '{metric_name}' PNG was requested but no file was produced."
                )

        if write_pdf:
            pdf_relpath = str(metric_cfg.get("pdf_relpath", f"{metric_name}.pdf"))
            pdf_path = _resolve_output_path(base_dir=bar_dir, relpath=pdf_relpath)
            _plot_bar(
                plt=plt,
                out_path=pdf_path,
                metric_name=metric_name,
                values_by_well=values_by_well,
                errors_by_well=errors_by_well,
                sample_counts=sample_counts,
                base_well_by_key=base_well_by_key,
                well_order=well_order,
                plot_defaults=plot_defaults,
                metric_cfg=metric_cfg_local,
                stats_row_for_annotation=annotation_row,
                stats_cfg=stats_cfg,
            )
            if pdf_path.exists():
                outputs[f"cross_well.bar_plots.{metric_name}.pdf"] = str(pdf_path)
            else:
                warnings.append(
                    f"Cross-well bar metric '{metric_name}' PDF was requested but no file was produced."
                )

    pairwise_csv = cross_root / "pairwise_tests_mannwhitneyu.csv"
    _write_rows_csv(pairwise_csv, pairwise_rows)
    outputs["cross_well.pairwise_tests_csv"] = str(pairwise_csv)

    summary_json = cross_root / "cross_well_summary.json"
    summary_payload = {
        "cross_well_out_dir": str(cross_root),
        "well_order": list(well_order),
        "successful_wells": [
            {
                "well_key": ctx.well_key,
                "well_id": ctx.well_id,
                "dataset_id": str(ctx.target.target.dataset_id),
            }
            for ctx in successful_wells
        ],
        "statistical_testing_enabled": bool(stats_enabled),
        "pairwise_tests_count": int(len(pairwise_rows)),
        "outputs": outputs,
        "warnings": warnings,
    }
    _write_json(summary_json, summary_payload)
    outputs["cross_well.summary_json"] = str(summary_json)

    return outputs, warnings
