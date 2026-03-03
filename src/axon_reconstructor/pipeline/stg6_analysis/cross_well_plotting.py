from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .cross_well_stats import _iqr_outlier_mask


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _build_condition_colors(conditions: list[str]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    unique: list[str] = []
    seen: set[str] = set()
    for c in conditions:
        cc = str(c)
        if cc in seen:
            continue
        seen.add(cc)
        unique.append(cc)

    cmap = plt.get_cmap("tab10")
    return {cond: cmap(i % 10) for i, cond in enumerate(unique)}


def _compute_positions_by_div(
    *,
    group_div_labels: list[str],
    intra_step: float = 1.0,
    inter_gap: float = 1.6,
) -> tuple[np.ndarray, list[tuple[str, float]], list[float]]:
    if not group_div_labels:
        return np.asarray([], dtype=float), [], []

    positions: list[float] = []
    centers: list[tuple[str, float]] = []
    separators: list[float] = []

    idx = 0
    x = 1.0
    n = len(group_div_labels)
    while idx < n:
        div = str(group_div_labels[idx])
        start_x = x
        count = 0
        while idx < n and str(group_div_labels[idx]) == div:
            positions.append(x)
            x += intra_step
            idx += 1
            count += 1

        end_x = start_x + intra_step * (count - 1)
        centers.append((div, (start_x + end_x) / 2.0))
        if idx < n:
            separators.append(end_x + intra_step / 2.0 + inter_gap / 2.0)
            x += inter_gap

    return np.asarray(positions, dtype=float), centers, separators


def _text_height_in_data_units(*, ax: Any, fig: Any, sample_text: str, fontsize: float = 7.0) -> float:
    """Estimate text height in y-data units for current axes transform."""

    tmp = ax.text(0.0, 0.0, sample_text, fontsize=fontsize, alpha=0.0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = tmp.get_window_extent(renderer=renderer)
    tmp.remove()

    inv = ax.transData.inverted()
    y0 = float(inv.transform((0.0, 0.0))[1])
    y1 = float(inv.transform((0.0, float(bbox.height)))[1])
    dy = abs(y1 - y0)
    if not np.isfinite(dy) or dy <= 0:
        ylo, yhi = ax.get_ylim()
        dy = max(1e-9, abs(float(yhi) - float(ylo))) * 0.02
    return float(dy)


def _dynamic_annotation_spacing(*, ax: Any, fig: Any) -> dict[str, float]:
    """Compute annotation spacing from rendered text geometry (no fixed absolute pads)."""

    h_n = _text_height_in_data_units(ax=ax, fig=fig, sample_text="n=999", fontsize=7.0)
    h_star = _text_height_in_data_units(ax=ax, fig=fig, sample_text="***", fontsize=9.0)
    y0, y1 = ax.get_ylim()
    y_range = max(1e-9, abs(float(y1) - float(y0)))

    marker_height = max(0.8 * h_star, 0.012 * y_range)
    star_text_offset = max(0.15 * h_star, 0.006 * y_range)
    star_text_height = h_star

    # Ensure the next marker starts above the previous star text by a y-range fraction.
    stack_clearance = max(0.55 * h_n, 0.028 * y_range)
    step = max(
        1.1 * (h_n + h_star),
        marker_height + star_text_offset + star_text_height + stack_clearance,
    )

    # Gap from n-label baseline to first significance marker baseline.
    n_to_sig_gap = max(0.8 * h_n, 0.038 * y_range)

    return {
        "label_offset": max(0.8 * h_n, 0.012 * y_range),
        "n_to_sig_gap": n_to_sig_gap,
        "marker_height": marker_height,
        "star_text_offset": star_text_offset,
        "step": step,
        "top_padding": max(h_n, h_star, 0.015 * y_range),
        "star_text_height": star_text_height,
    }


def plot_boxplot_with_stars(
    *,
    out_path: Path,
    title: str,
    ylabel: str,
    groups: list[np.ndarray],
    group_labels: list[str],
    pairwise_tests: list[dict[str, Any]],
    outlier_iqr_k: float = 1.5,
    exclude_outliers_in_tests: bool = False,
    group_div_labels: list[str] | None = None,
    display_group_labels: list[str] | None = None,
    group_conditions: list[str] | None = None,
    highlight_outliers: bool = True,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=150)
    boxplot_obj = None
    x_positions, div_centers, div_separators = _compute_positions_by_div(
        group_div_labels=(group_div_labels or []),
        intra_step=1.0,
        inter_gap=1.6,
    )
    if x_positions.size != len(groups):
        x_positions = np.arange(1, len(groups) + 1, dtype=float)
        div_centers = []
        div_separators = []
    try:
        boxplot_obj = ax.boxplot(
            groups,
            tick_labels=group_labels,
            showfliers=False,
            patch_artist=True,
            positions=x_positions,
            widths=0.92,
        )
    except TypeError:
        # Older Matplotlib
        boxplot_obj = ax.boxplot(
            groups,
            labels=group_labels,
            showfliers=False,
            patch_artist=True,
            positions=x_positions,
            widths=0.92,
        )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.3)
    shown_labels = display_group_labels if (display_group_labels and len(display_group_labels) == len(group_labels)) else group_labels
    if div_centers:
        ax.set_xticks([c for _, c in div_centers])
        ax.set_xticklabels([d for d, _ in div_centers], rotation=0, ha="center")
    else:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(shown_labels, rotation=35, ha="right")

    density_colors = _build_condition_colors(group_conditions or shown_labels)
    if boxplot_obj is not None and group_conditions is not None and len(group_conditions) == len(groups):
        for patch, cond in zip(boxplot_obj.get("boxes", []), group_conditions, strict=False):
            color = density_colors.get(str(cond), "#4C78A8")
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
            patch.set_edgecolor("#333333")
            patch.set_linewidth(0.8)

    # Scatter overlay (jittered) with outlier highlighting.
    rng = np.random.default_rng(0)
    any_outliers = False
    for i, g in enumerate(groups, start=1):
        if g.size == 0:
            continue
        jitter = rng.uniform(-0.09, 0.09, size=g.size)
        x = float(x_positions[i - 1]) + jitter
        if highlight_outliers:
            out_mask = _iqr_outlier_mask(g, k=float(outlier_iqr_k))
            any_outliers = any_outliers or bool(out_mask.any())
            if (~out_mask).any():
                ax.scatter(x[~out_mask], g[~out_mask], s=5, alpha=0.5, linewidths=0, c="black")
            if out_mask.any():
                ax.scatter(x[out_mask], g[out_mask], s=9, alpha=0.8, linewidths=0.5, edgecolors="black", c="red")
        else:
            ax.scatter(x, g, s=5, alpha=0.5, linewidths=0, c="black")

    y_max = max([float(np.max(g)) for g in groups if g.size > 0] + [0.0])
    y_min = min([float(np.min(g)) for g in groups if g.size > 0] + [0.0])
    y_span = max(1e-9, y_max - y_min)

    spacing = _dynamic_annotation_spacing(ax=ax, fig=fig)

    # Add per-group n labels dynamically above each group's highest plotted point.
    y_top_by_group: list[float] = []
    for g in groups:
        if g.size == 0:
            y_top_by_group.append(np.nan)
            continue
        try:
            y_top_by_group.append(float(np.nanmax(g)))
        except Exception:
            y_top_by_group.append(np.nan)

    label_offset = float(spacing["label_offset"])
    n_label_y_by_group: list[float] = []
    max_annotation_y = y_max
    for x_pos, y_top, g in zip(x_positions, y_top_by_group, groups, strict=False):
        if not np.isfinite(y_top):
            n_label_y_by_group.append(np.nan)
            continue
        n_label_y = float(y_top + label_offset)
        max_annotation_y = max(max_annotation_y, n_label_y)
        n_label_y_by_group.append(n_label_y)
        nlab = f"n={int(g.size)}"
        ax.text(
            float(x_pos),
            n_label_y,
            nlab,
            ha="center",
            va="bottom",
            fontsize=7,
            color="#333333",
            bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none", "pad": 1.0},
        )

    # Draw significance markers per DIV block so different DIVs can share the same y-levels.
    marker_height = float(spacing["marker_height"])
    step = float(spacing["step"])
    n_to_sig_gap = float(spacing["n_to_sig_gap"])
    star_text_offset = float(spacing["star_text_offset"])
    star_text_height = float(spacing["star_text_height"])

    group_idx_by_label = {lab: idx for idx, lab in enumerate(group_labels)}
    if group_div_labels is not None and len(group_div_labels) == len(group_labels):
        group_block_labels = [str(d) for d in group_div_labels]
    else:
        group_block_labels = ["all"] * len(group_labels)

    block_base_y: dict[str, float] = {}
    for idx, block in enumerate(group_block_labels):
        y_n = n_label_y_by_group[idx] if idx < len(n_label_y_by_group) else np.nan
        y_top = y_top_by_group[idx] if idx < len(y_top_by_group) else np.nan
        y_ref = y_n if np.isfinite(y_n) else (float(y_top) if np.isfinite(y_top) else y_max)
        current = block_base_y.get(block)
        if current is None or y_ref > current:
            block_base_y[block] = float(y_ref)

    block_drawn: dict[str, int] = {}
    for t in pairwise_tests:
        stars = t.get("stars") or ""
        if stars == "":
            continue
        ga = str(t.get("group_a", ""))
        gb = str(t.get("group_b", ""))
        ia = group_idx_by_label.get(ga)
        ib = group_idx_by_label.get(gb)
        if ia is None or ib is None:
            continue

        block_a = group_block_labels[ia]
        block_b = group_block_labels[ib]
        if block_a != block_b:
            continue
        block = block_a

        i = float(x_positions[ia])
        j = float(x_positions[ib])
        if j < i:
            i, j = j, i

        drawn_in_block = int(block_drawn.get(block, 0))
        base_y = float(block_base_y.get(block, y_max)) + n_to_sig_gap
        y = base_y + drawn_in_block * step
        ax.plot([i, i, j, j], [y, y + marker_height, y + marker_height, y], lw=1.0, c="black")
        star_y = y + marker_height + star_text_offset
        ax.text((i + j) / 2.0, star_y, stars, ha="center", va="bottom")
        max_annotation_y = max(max_annotation_y, star_y + star_text_height)
        block_drawn[block] = drawn_in_block + 1

    # Ensure headroom so top labels/markers do not touch the plot ceiling.
    y_low, y_high = ax.get_ylim()
    required_top = float(max_annotation_y + float(spacing["top_padding"]))
    if required_top > y_high:
        ax.set_ylim(y_low, required_top)

    # Legend/key: outliers + star thresholds.
    legend_handles: list[Any] = []
    legend_labels: list[str] = []

    for cond, color in density_colors.items():
        legend_handles.append(Line2D([0], [0], color=color, lw=6))
        legend_labels.append(str(cond))

    if any_outliers:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                markerfacecolor="red",
                markeredgecolor="black",
                markersize=6,
                linestyle="",
            )
        )
        legend_labels.append(f"outliers (IQR k={float(outlier_iqr_k):g})")

    sig_key = "* p<0.05, ** p<0.005, *** p<0.001 (Mann–Whitney U)"
    tests_key = "tests exclude outliers" if exclude_outliers_in_tests else "tests include outliers"
    legend_handles.append(Line2D([], [], linestyle="", color="none"))
    legend_labels.append(f"{sig_key}; {tests_key}")

    if legend_handles:
        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.23),
            frameon=False,
            fontsize=7,
            title="Density",
            ncol=min(4, max(1, len(density_colors))),
        )

    for x_sep in div_separators:
        ax.axvline(x_sep, color="#999999", linewidth=0.8, alpha=0.6)

    if x_positions.size > 0:
        ax.set_xlim(float(np.min(x_positions)) - 0.6, float(np.max(x_positions)) + 0.6)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.17, top=0.80)
    _ensure_dir(out_path.parent)
    fig.savefig(out_path)
    plt.close(fig)


def plot_mean_sem_bar_with_stars(
    *,
    out_path: Path,
    title: str,
    ylabel: str,
    groups: list[np.ndarray],
    group_labels: list[str],
    group_div_labels: list[str],
    pairwise_tests: list[dict[str, Any]],
    outlier_iqr_k: float = 1.5,
    exclude_outliers_in_tests: bool = False,
    display_group_labels: list[str] | None = None,
    group_conditions: list[str] | None = None,
    highlight_outliers: bool = True,
    show_n_labels: bool = True,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    means = np.asarray([float(np.mean(g)) if g.size > 0 else np.nan for g in groups], dtype=float)
    stds = np.asarray(
        [
            float(np.std(g, ddof=1))
            if g.size > 1
            else (0.0 if g.size == 1 else np.nan)
            for g in groups
        ],
        dtype=float,
    )

    x, div_centers, div_separators = _compute_positions_by_div(
        group_div_labels=group_div_labels,
        intra_step=1.0,
        inter_gap=1.6,
    )
    if x.size != len(group_labels):
        x = np.arange(1, len(group_labels) + 1, dtype=float)
        div_centers = []
        div_separators = []
    valid = np.isfinite(means)

    shown_labels = display_group_labels if (display_group_labels and len(display_group_labels) == len(group_labels)) else group_labels
    density_colors = _build_condition_colors(group_conditions or shown_labels)
    bar_colors = []
    if group_conditions is not None and len(group_conditions) == len(group_labels):
        for cond in group_conditions:
            bar_colors.append(density_colors.get(str(cond), "#4C78A8"))
    else:
        for lab in shown_labels:
            bar_colors.append(density_colors.get(str(lab), "#4C78A8"))

    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=150)
    if valid.any():
        ax.bar(
            x[valid],
            means[valid],
            yerr=stds[valid],
            width=0.95,
            capsize=3,
            alpha=0.55,
            color=np.asarray(bar_colors, dtype=object)[valid],
            edgecolor="#333333",
            linewidth=0.8,
        )

    rng = np.random.default_rng(0)
    any_outliers = False
    y_top_by_group: list[float] = []
    for idx, g in enumerate(groups):
        arr = np.asarray(g, dtype=float)
        if arr.size == 0:
            y_top_by_group.append(np.nan)
            continue
        jitter = rng.uniform(-0.09, 0.09, size=arr.size)
        xs = float(x[idx]) + jitter
        if highlight_outliers:
            out_mask = _iqr_outlier_mask(arr, k=float(outlier_iqr_k))
            any_outliers = any_outliers or bool(out_mask.any())
            if (~out_mask).any():
                ax.scatter(xs[~out_mask], arr[~out_mask], s=5, alpha=0.5, linewidths=0, c="black")
            if out_mask.any():
                ax.scatter(xs[out_mask], arr[out_mask], s=9, alpha=0.8, linewidths=0.5, edgecolors="black", c="red")
        else:
            ax.scatter(xs, arr, s=5, alpha=0.5, linewidths=0, c="black")
        y_top_by_group.append(float(np.nanmax(arr)))

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.3)
    if div_centers:
        ax.set_xticks([c for _, c in div_centers])
        ax.set_xticklabels([d for d, _ in div_centers], rotation=0, ha="center")
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(shown_labels, rotation=35, ha="right")

    legend_handles = [Line2D([0], [0], color=color, lw=6) for _, color in density_colors.items()]
    legend_labels = [str(cond) for cond in density_colors.keys()]
    sig_key = "* p<0.05, ** p<0.005, *** p<0.001 (Mann–Whitney U)"
    tests_key = "tests exclude outliers" if exclude_outliers_in_tests else "tests include outliers"
    if any_outliers:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                markerfacecolor="red",
                markeredgecolor="black",
                markersize=6,
                linestyle="",
            )
        )
        legend_labels.append(f"outliers (IQR k={float(outlier_iqr_k):g})")
    legend_handles.append(Line2D([], [], linestyle="", color="none"))
    legend_labels.append(f"{sig_key}; {tests_key}")
    if legend_handles:
        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.23),
            frameon=False,
            fontsize=7,
            title="Density",
            ncol=min(4, max(1, len(density_colors))),
        )

    y_vals = means[np.isfinite(means)]
    if y_vals.size > 0:
        y_max = float(np.max(y_vals))
        y_min = float(np.min(y_vals))
    else:
        y_max = 1.0
        y_min = 0.0

    for m, s in zip(means, stds, strict=False):
        if np.isfinite(m) and np.isfinite(s):
            y_max = max(y_max, float(m + s))

    y_span = max(1e-9, y_max - y_min)

    spacing = _dynamic_annotation_spacing(ax=ax, fig=fig)

    # Per-group n labels above top observed values (same spacing as boxplot path).
    label_offset = float(spacing["label_offset"])
    n_label_y_by_group: list[float] = []
    max_annotation_y = y_max
    for x_pos, y_top, g in zip(x, y_top_by_group, groups, strict=False):
        if not np.isfinite(y_top):
            n_label_y_by_group.append(np.nan)
            continue
        n_label_y = float(y_top + label_offset)
        n_label_y_by_group.append(n_label_y)
        if show_n_labels:
            max_annotation_y = max(max_annotation_y, n_label_y)
            ax.text(
                float(x_pos),
                n_label_y,
                f"n={int(np.asarray(g).size)}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#333333",
                bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none", "pad": 1.0},
            )

    marker_height = float(spacing["marker_height"])
    n_to_sig_gap = float(spacing["n_to_sig_gap"])
    step = float(spacing["step"])
    star_text_offset = float(spacing["star_text_offset"])
    star_text_height = float(spacing["star_text_height"])

    group_idx_by_label = {lab: idx for idx, lab in enumerate(group_labels)}
    if group_div_labels is not None and len(group_div_labels) == len(group_labels):
        group_block_labels = [str(d) for d in group_div_labels]
    else:
        group_block_labels = ["all"] * len(group_labels)

    block_base_y: dict[str, float] = {}
    for idx, block in enumerate(group_block_labels):
        y_n = n_label_y_by_group[idx] if idx < len(n_label_y_by_group) else np.nan
        y_top = y_top_by_group[idx] if idx < len(y_top_by_group) else np.nan
        y_ref = y_n if np.isfinite(y_n) else (float(y_top) if np.isfinite(y_top) else y_max)
        current = block_base_y.get(block)
        if current is None or y_ref > current:
            block_base_y[block] = float(y_ref)

    block_drawn: dict[str, int] = {}
    for t in pairwise_tests:
        stars = t.get("stars") or ""
        if not stars:
            continue
        ga = str(t.get("group_a", ""))
        gb = str(t.get("group_b", ""))
        ia = group_idx_by_label.get(ga)
        ib = group_idx_by_label.get(gb)
        if ia is None or ib is None:
            continue
        block_a = group_block_labels[ia]
        block_b = group_block_labels[ib]
        if block_a != block_b:
            continue
        block = block_a

        i = float(x[ia])
        j = float(x[ib])
        if j < i:
            i, j = j, i

        drawn_in_block = int(block_drawn.get(block, 0))
        base_y = float(block_base_y.get(block, y_max)) + n_to_sig_gap
        y = base_y + drawn_in_block * step
        ax.plot([i, i, j, j], [y, y + marker_height, y + marker_height, y], lw=1.0, c="black")
        star_y = y + marker_height + star_text_offset
        ax.text((i + j) / 2.0, star_y, stars, ha="center", va="bottom")
        max_annotation_y = max(max_annotation_y, star_y + star_text_height)
        block_drawn[block] = drawn_in_block + 1

    y_low, y_high = ax.get_ylim()
    required_top = float(max_annotation_y + float(spacing["top_padding"]))
    if required_top > y_high:
        ax.set_ylim(y_low, required_top)

    for x_sep in div_separators:
        ax.axvline(x_sep, color="#999999", linewidth=0.8, alpha=0.6)

    if x.size > 0:
        ax.set_xlim(float(np.min(x)) - 0.6, float(np.max(x)) + 0.6)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.17, top=0.80)
    _ensure_dir(out_path.parent)
    fig.savefig(out_path)
    plt.close(fig)


def _well_positions_by_div(
    rows: list[dict[str, Any]],
    *,
    intra_step: float = 1.0,
    inter_gap: float = 1.6,
) -> tuple[np.ndarray, list[str], list[tuple[str, float]], list[float]]:
    positions: list[float] = []
    div_labels: list[str] = []
    centers: list[tuple[str, float]] = []
    separators: list[float] = []
    if not rows:
        return np.asarray([], dtype=float), div_labels, centers, separators

    x = 1.0
    idx = 0
    n = len(rows)
    while idx < n:
        div = rows[idx].get("div")
        div_label = f"DIV{int(div)}" if div is not None else "DIV?"
        start_x = x
        count = 0
        while idx < n and rows[idx].get("div") == div:
            positions.append(x)
            div_labels.append(div_label)
            x += intra_step
            idx += 1
            count += 1
        end_x = start_x + intra_step * (count - 1)
        centers.append((div_label, (start_x + end_x) / 2.0))
        if idx < n:
            separators.append(end_x + intra_step / 2.0 + inter_gap / 2.0)
            x += inter_gap

    return np.asarray(positions, dtype=float), div_labels, centers, separators


def plot_units_detected_vs_reconstructed_per_well(
    *,
    out_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if not rows:
        return
    x, _, div_centers, div_separators = _well_positions_by_div(rows)
    vals_detected = np.asarray([float(r.get("n_units_detected") or 0) for r in rows], dtype=float)
    vals_reconstructed = np.asarray([float(r.get("n_units_reconstructed") or 0) for r in rows], dtype=float)
    conds = [str(r.get("condition", "")) for r in rows]
    density_colors = _build_condition_colors(conds)
    colors = [density_colors.get(c, "#4C78A8") for c in conds]

    fig, ax = plt.subplots(figsize=(10.8, 5.0), dpi=150)
    w = 0.38
    ax.bar(
        x - w / 2.0,
        vals_detected,
        width=w,
        alpha=0.55,
        color=colors,
        edgecolor="#333333",
        linewidth=0.8,
        label="detected",
    )
    ax.bar(
        x + w / 2.0,
        vals_reconstructed,
        width=w,
        alpha=0.55,
        color=colors,
        edgecolor="#333333",
        linewidth=0.8,
        hatch="//",
        label="reconstructed",
    )

    # % reconstructed labels above each bar pair.
    pair_top = np.maximum(vals_detected, vals_reconstructed)
    y_max_pair = float(np.max(pair_top)) if pair_top.size else 0.0
    y_min_pair = float(np.min(np.minimum(vals_detected, vals_reconstructed))) if pair_top.size else 0.0
    y_span_pair = max(1e-9, y_max_pair - y_min_pair)
    pct_offset = max(0.02 * y_span_pair, 0.35)
    for xi, vd, vr, top in zip(x, vals_detected, vals_reconstructed, pair_top, strict=False):
        pct = (100.0 * vr / vd) if vd > 0 else np.nan
        txt = f"{pct:.1f}%" if np.isfinite(pct) else "n/a"
        ax.text(float(xi), float(top + pct_offset), txt, ha="center", va="bottom", fontsize=7, color="#333333")

    ax.set_title("n_counts: units detected vs reconstructed / well")
    ax.set_ylabel("count")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.3)
    if div_centers:
        ax.set_xticks([c for _, c in div_centers])
        ax.set_xticklabels([d for d, _ in div_centers], rotation=0, ha="center")
    if x.size > 0:
        ax.set_xlim(float(np.min(x)) - 0.8, float(np.max(x)) + 0.8)
    for x_sep in div_separators:
        ax.axvline(x_sep, color="#999999", linewidth=0.8, alpha=0.6)

    legend_handles = [Patch(facecolor=color, edgecolor="#333333", alpha=0.55) for _, color in density_colors.items()]
    legend_labels = [str(cond) for cond in density_colors.keys()]
    if legend_handles:
        density_leg = ax.legend(
            legend_handles,
            legend_labels,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.22),
            frameon=False,
            fontsize=7,
            title="Density",
            ncol=min(4, max(1, len(density_colors))),
        )
        ax.add_artist(density_leg)

    style_handles = [
        Patch(facecolor="#B0B0B0", edgecolor="#333333", alpha=0.55, label="detected"),
        Patch(facecolor="#B0B0B0", edgecolor="#333333", alpha=0.55, hatch="//", label="reconstructed"),
    ]
    ax.legend(
        handles=style_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.22),
        frameon=False,
        fontsize=8,
        title="Bar Type",
    )

    y0, y1 = ax.get_ylim()
    needed_top = float(np.max(pair_top + pct_offset)) + max(0.05 * y_span_pair, 0.5)
    if needed_top > y1:
        ax.set_ylim(y0, needed_top)

    fig.tight_layout()
    fig.subplots_adjust(top=0.78)
    _ensure_dir(out_path.parent)
    fig.savefig(out_path)
    plt.close(fig)


def plot_percent_reconstructed_per_well(
    *,
    out_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if not rows:
        return

    x, _, div_centers, div_separators = _well_positions_by_div(rows)
    vals_detected = np.asarray([float(r.get("n_units_detected") or 0) for r in rows], dtype=float)
    vals_reconstructed = np.asarray([float(r.get("n_units_reconstructed") or 0) for r in rows], dtype=float)
    pct = np.where(vals_detected > 0, 100.0 * vals_reconstructed / vals_detected, np.nan)
    pct_for_bars = np.where(np.isfinite(pct), pct, 0.0)

    conds = [str(r.get("condition", "")) for r in rows]
    density_colors = _build_condition_colors(conds)
    colors = [density_colors.get(c, "#4C78A8") for c in conds]

    fig, ax = plt.subplots(figsize=(10.8, 5.0), dpi=150)
    ax.bar(
        x,
        pct_for_bars,
        width=0.92,
        alpha=0.55,
        color=colors,
        edgecolor="#333333",
        linewidth=0.8,
    )

    y_max = float(np.nanmax(pct_for_bars)) if pct_for_bars.size else 0.0
    y_span = max(1e-9, y_max)
    label_offset = max(0.02 * y_span, 1.0)
    for xi, p in zip(x, pct, strict=False):
        label = f"{p:.1f}%" if np.isfinite(p) else "n/a"
        ytxt = (float(p) if np.isfinite(p) else 0.0) + label_offset
        ax.text(float(xi), ytxt, label, ha="center", va="bottom", fontsize=7, color="#333333")

    ax.set_title("n_counts: % reconstructed / well")
    ax.set_ylabel("%")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.3)
    if div_centers:
        ax.set_xticks([c for _, c in div_centers])
        ax.set_xticklabels([d for d, _ in div_centers], rotation=0, ha="center")
    if x.size > 0:
        ax.set_xlim(float(np.min(x)) - 0.6, float(np.max(x)) + 0.6)
    for x_sep in div_separators:
        ax.axvline(x_sep, color="#999999", linewidth=0.8, alpha=0.6)

    legend_handles = [Patch(facecolor=color, edgecolor="#333333", alpha=0.55) for _, color in density_colors.items()]
    legend_labels = [str(cond) for cond in density_colors.keys()]
    if legend_handles:
        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.20),
            frameon=False,
            fontsize=7,
            title="Density",
            ncol=min(4, max(1, len(density_colors))),
        )

    y0, y1 = ax.get_ylim()
    needed_top = float(np.nanmax(pct_for_bars + label_offset)) + max(0.05 * y_span, 2.0)
    if needed_top > y1:
        ax.set_ylim(y0, needed_top)

    fig.tight_layout()
    fig.subplots_adjust(top=0.80)
    _ensure_dir(out_path.parent)
    fig.savefig(out_path)
    plt.close(fig)
