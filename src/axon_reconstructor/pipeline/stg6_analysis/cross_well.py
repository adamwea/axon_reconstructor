#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from axon_reconstructor.env_utils import load_env_file_into_os
from ..stg2_spikesorting.runner import LEGACY_SPIKESORTING_OUTPUTS_DIRNAME, SPIKESORTING_OUTPUTS_DIRNAME
from . import cross_well_decks as cw_decks
from . import cross_well_plotting as cw_plots
from . import cross_well_stats as cw_stats


@dataclass(frozen=True)
class WellSpec:
    well_id: str
    condition: str
    plating_density_nbp: int
    genotype: str


@dataclass(frozen=True)
class DatasetSpec:
    raw_data_h5_path: Path
    dataset_output_root: Path
    div: int | None
    wells: list[WellSpec]


@dataclass(frozen=True)
class CrossWellConfig:
    analysis_name: str
    out_dir: Path
    electrode_pitch_um: float
    runtime_env_file: Path | None
    datasets: list[DatasetSpec]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_dataset_output_root_from_raw_h5(raw_data_h5_path: Path) -> Path:
    """Infer the per-dataset output root directory from the raw-data H5 path.

    Expected pattern (example):
      .../raw_data/<DS>/<DS>/<date>/<mouse>/<modality>/<run_id>/data.raw.h5

    Output pattern:
      .../outputs/<DS>/<date>/<mouse>/<modality>/<run_id>

    We drop the duplicated <DS> folder if present.
    """

    p = raw_data_h5_path.expanduser().resolve()
    parts = list(p.parts)
    if "raw_data" not in parts:
        raise ValueError(f"Cannot infer output root: path does not contain 'raw_data': {p}")
    i = parts.index("raw_data")
    if i + 1 >= len(parts):
        raise ValueError(f"Cannot infer output root: missing dataset name after raw_data: {p}")
    ds_name = parts[i + 1]

    # Remaining path after dataset name
    rest = parts[i + 2 :]
    if rest and rest[0] == ds_name:
        rest = rest[1:]

    # Remove filename if it looks like an H5
    if rest and rest[-1].lower().endswith((".h5", ".hdf5")):
        rest = rest[:-1]

    out_parts = parts[:i] + ["outputs", ds_name] + rest
    return Path(*out_parts)


def make_dataset_key(*, raw_data_h5_path: Path, dataset_output_root: Path) -> str:
    """Create a stable, human-readable dataset key for row-level joins.

    Example: Media_Density_T3_07012025_AR__250728__M07137__000225
    """

    p = raw_data_h5_path.expanduser().resolve()
    parts = list(p.parts)
    ds_name = "dataset"
    date = "date"
    mouse = "mouse"
    run_id = "run"

    if "raw_data" in parts:
        i = parts.index("raw_data")
        if i + 1 < len(parts):
            ds_name = parts[i + 1]

        # After ds_name there may be a duplicated ds_name folder.
        j = i + 2
        if j < len(parts) and parts[j] == ds_name:
            j += 1

        if j < len(parts):
            date = parts[j]
        if j + 1 < len(parts):
            mouse = parts[j + 1]

    try:
        run_id = str(dataset_output_root.name)
    except Exception:
        pass

    return f"{ds_name}__{date}__{mouse}__{run_id}"


def _read_yaml(path: Path) -> Any:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_cross_well_config(path: Path) -> CrossWellConfig:
    path = Path(path)
    if path.suffix.lower() in {".yml", ".yaml"}:
        raw = _read_yaml(path)
    else:
        # Back-compat: allow the older JSON spec to load as a single-dataset config.
        raw = _read_json(path)

    electrode_pitch_um = float(raw.get("electrode_pitch_um", 17.5))
    analysis_name = raw.get("analysis_name") or raw.get("name") or path.stem

    out_dir_raw = raw.get("out_dir")
    if out_dir_raw is None:
        raise ValueError("Missing required config key: out_dir")
    out_dir = Path(out_dir_raw).expanduser().resolve()

    runtime_env_file_raw = raw.get("runtime_env_file")
    runtime_env_file = Path(runtime_env_file_raw).expanduser().resolve() if runtime_env_file_raw else None

    datasets_raw = raw.get("datasets")
    if datasets_raw is None:
        # Older JSON shape
        datasets_raw = [
            {
                "dataset_output_root": raw.get("dataset_output_root"),
                "wells": raw.get("wells", []),
                "raw_data_h5_path": raw.get("raw_data_h5_path"),
            }
        ]

    datasets: list[DatasetSpec] = []
    for d in datasets_raw:
        raw_h5 = d.get("raw_data_h5_path")
        dataset_output_root_raw = d.get("dataset_output_root")

        if raw_h5 is None and dataset_output_root_raw is None:
            raise ValueError("Each dataset must specify raw_data_h5_path (preferred) or dataset_output_root")

        raw_data_h5_path = Path(raw_h5).expanduser().resolve() if raw_h5 else Path(".")
        dataset_output_root = (
            infer_dataset_output_root_from_raw_h5(raw_data_h5_path)
            if raw_h5
            else Path(dataset_output_root_raw).expanduser().resolve()
        )
        div_raw = d.get("DIV")
        div: int | None = None
        if div_raw is not None:
            try:
                div = int(div_raw)
            except Exception:
                div = None

        wells_raw = d.get("wells") or []
        wells: list[WellSpec] = []
        for w in wells_raw:
            wells.append(
                WellSpec(
                    well_id=w["well_id"],
                    condition=w["condition"],
                    plating_density_nbp=int(w["plating_density_nbp"]),
                    genotype=w.get("genotype", ""),
                )
            )

        datasets.append(
            DatasetSpec(
                raw_data_h5_path=raw_data_h5_path,
                dataset_output_root=dataset_output_root,
                div=div,
                wells=wells,
            )
        )

    return CrossWellConfig(
        analysis_name=analysis_name,
        out_dir=out_dir,
        electrode_pitch_um=electrode_pitch_um,
        runtime_env_file=runtime_env_file,
        datasets=datasets,
    )


def iter_unit_dirs(by_unit_dir: Path) -> Iterable[Path]:
    if not by_unit_dir.exists():
        return []
    return sorted([p for p in by_unit_dir.glob("unit_*") if p.is_dir()])


def parse_unit_id(unit_dir: Path) -> int:
    name = unit_dir.name
    if not name.startswith("unit_"):
        raise ValueError(f"Unexpected unit dir: {unit_dir}")
    return int(name.split("unit_", 1)[1])


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def branch_length_um(branch: dict[str, Any]) -> float | None:
    d = branch.get("distances")
    if not isinstance(d, list) or len(d) == 0:
        return None
    values = [
        float(v)
        for v in d
        if isinstance(v, (int, float)) and (not math.isnan(float(v))) and (not math.isinf(float(v)))
    ]
    if not values:
        return None
    return float(max(values))


def load_detected_unit_ids(well_dir: Path) -> list[int] | None:
    info_path = (
        well_dir
        / SPIKESORTING_OUTPUTS_DIRNAME
        / "analyzer_output"
        / "sorting"
        / "numpysorting_info.json"
    )
    if not info_path.exists():
        legacy_info_path = (
            well_dir
            / LEGACY_SPIKESORTING_OUTPUTS_DIRNAME
            / "analyzer_output"
            / "sorting"
            / "numpysorting_info.json"
        )
        if legacy_info_path.exists():
            info_path = legacy_info_path
    if not info_path.exists():
        return None
    info = _read_json(info_path)
    unit_ids = info.get("unit_ids")
    if not isinstance(unit_ids, list):
        return None
    out: list[int] = []
    for u in unit_ids:
        try:
            out.append(int(u))
        except Exception:
            continue
    return out


def extract_metrics_for_well(
    *,
    dataset_root: Path,
    well: WellSpec,
    electrode_pitch_um: float,
    dataset_key: str,
    raw_data_h5_path: Path,
) -> dict[str, Any]:
    well_dir = dataset_root / well.well_id
    by_unit_dir = well_dir / "stg5_reconstruction_outputs" / "by_unit"
    unit_dirs = list(iter_unit_dirs(by_unit_dir))

    detected_unit_ids = load_detected_unit_ids(well_dir)
    n_detected_units = len(detected_unit_ids) if detected_unit_ids is not None else None

    units_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []

    for unit_dir in unit_dirs:
        unit_id = parse_unit_id(unit_dir)

        branches_raw_path = unit_dir / "branches_raw.json"
        branches_clean_path = unit_dir / "branches.json"
        heuristics_path = unit_dir / "heuristics.json"

        branches_raw = _read_json(branches_raw_path)["branches"] if branches_raw_path.exists() else []
        branches_clean = _read_json(branches_clean_path)["branches"] if branches_clean_path.exists() else []

        reconstructed = bool(branches_clean_path.exists()) and isinstance(branches_clean, list) and len(branches_clean) > 0

        n_branches_raw = len(branches_raw) if isinstance(branches_raw, list) else 0
        n_branches_clean = len(branches_clean) if isinstance(branches_clean, list) else 0

        branch_lengths_raw: list[float] = []
        branch_velocities_raw: list[float] = []
        branch_lengths_clean: list[float] = []
        branch_velocities_clean: list[float] = []

        def add_branch_rows(*, branches: list[dict[str, Any]], source: str) -> None:
            nonlocal branch_rows, branch_lengths_raw, branch_velocities_raw, branch_lengths_clean, branch_velocities_clean

            for bi, br in enumerate(branches):
                if not isinstance(br, dict):
                    continue
                length_um = branch_length_um(br)
                vel = _safe_float(br.get("velocity"))

                if source == "raw":
                    if length_um is not None:
                        branch_lengths_raw.append(float(length_um))
                    if vel is not None:
                        branch_velocities_raw.append(float(vel))
                elif source == "clean":
                    if length_um is not None:
                        branch_lengths_clean.append(float(length_um))
                    if vel is not None:
                        branch_velocities_clean.append(float(vel))
                else:
                    raise ValueError(f"Unknown branch source: {source}")

                branch_rows.append(
                    {
                        "dataset_key": dataset_key,
                        "dataset_output_root": str(dataset_root),
                        "raw_data_h5_path": str(raw_data_h5_path),
                        "well_id": well.well_id,
                        "condition": well.condition,
                        "plating_density_nbp": well.plating_density_nbp,
                        "genotype": well.genotype,
                        "unit_id": unit_id,
                        "branch_source": source,
                        "branch_index": br.get("branch_index", bi),
                        "branch_length_um": length_um,
                        "branch_velocity": vel,
                        "branch_r2": _safe_float(br.get("r2")),
                    }
                )

        if isinstance(branches_raw, list):
            add_branch_rows(branches=branches_raw, source="raw")
        if isinstance(branches_clean, list):
            add_branch_rows(branches=branches_clean, source="clean")

        n_selected_channels = None
        axon_area_um2 = None
        if heuristics_path.exists():
            heur = _read_json(heuristics_path)
            heur_h = heur.get("heuristics", {}) if isinstance(heur, dict) else {}
            if isinstance(heur_h, dict):
                n_selected_channels = heur_h.get("n_selected_channels")
                if n_selected_channels is None:
                    sc = heur_h.get("selected_channels")
                    if isinstance(sc, list):
                        n_selected_channels = len(sc)
                try:
                    if n_selected_channels is not None:
                        n_selected_channels = int(n_selected_channels)
                        axon_area_um2 = float(n_selected_channels) * float(electrode_pitch_um) ** 2
                except Exception:
                    n_selected_channels = None
                    axon_area_um2 = None

        units_rows.append(
            {
                "dataset_key": dataset_key,
                "dataset_output_root": str(dataset_root),
                "raw_data_h5_path": str(raw_data_h5_path),
                "well_id": well.well_id,
                "condition": well.condition,
                "plating_density_nbp": well.plating_density_nbp,
                "genotype": well.genotype,
                "unit_id": unit_id,
                "reconstructed": reconstructed,
                "n_branches_raw": n_branches_raw,
                "n_branches_clean": n_branches_clean,
                # Raw (unit-level)
                "total_branch_length_um": float(np.sum(branch_lengths_raw)) if branch_lengths_raw else None,
                "mean_branch_length_um": float(np.mean(branch_lengths_raw)) if branch_lengths_raw else None,
                "median_branch_length_um": float(np.median(branch_lengths_raw)) if branch_lengths_raw else None,
                "mean_branch_velocity": float(np.mean(branch_velocities_raw)) if branch_velocities_raw else None,
                "median_branch_velocity": float(np.median(branch_velocities_raw)) if branch_velocities_raw else None,
                # Clean (unit-level)
                "total_branch_length_um_clean": float(np.sum(branch_lengths_clean)) if branch_lengths_clean else None,
                "mean_branch_length_um_clean": float(np.mean(branch_lengths_clean)) if branch_lengths_clean else None,
                "median_branch_length_um_clean": float(np.median(branch_lengths_clean)) if branch_lengths_clean else None,
                "mean_branch_velocity_clean": float(np.mean(branch_velocities_clean)) if branch_velocities_clean else None,
                "median_branch_velocity_clean": float(np.median(branch_velocities_clean)) if branch_velocities_clean else None,
                "n_selected_channels": n_selected_channels,
                "axon_area_um2": axon_area_um2,
            }
        )

    n_reconstructed_units = sum(1 for r in units_rows if r.get("reconstructed"))
    return {
        "dataset_key": dataset_key,
        "dataset_output_root": str(dataset_root),
        "raw_data_h5_path": str(raw_data_h5_path),
        "well_id": well.well_id,
        "condition": well.condition,
        "plating_density_nbp": well.plating_density_nbp,
        "genotype": well.genotype,
        "n_detected_units": n_detected_units,
        "n_units_with_recon_dir": len(unit_dirs),
        "n_reconstructed_units": n_reconstructed_units,
        "units": units_rows,
        "branches": branch_rows,
    }


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _extract_recording_date_ymd_from_raw_h5(raw_data_h5_path: Path) -> str | None:
    """Best-effort parse of recording date (YYYY-MM-DD) from raw_data_h5_path.

    Expected segment in path is YYMMDD (e.g. 250708).
    """

    parts = list(raw_data_h5_path.expanduser().resolve().parts)
    if "raw_data" not in parts:
        return None
    i = parts.index("raw_data")

    if i + 1 >= len(parts):
        return None
    ds_name = parts[i + 1]

    j = i + 2
    if j < len(parts) and parts[j] == ds_name:
        j += 1
    if j >= len(parts):
        return None

    raw_date = str(parts[j]).strip()
    if len(raw_date) == 6 and raw_date.isdigit():
        yy = int(raw_date[0:2])
        year = 2000 + yy
        month = int(raw_date[2:4])
        day = int(raw_date[4:6])
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except Exception:
            return None

    if len(raw_date) == 8 and raw_date.isdigit():
        try:
            return datetime.strptime(raw_date, "%Y%m%d").strftime("%Y-%m-%d")
        except Exception:
            return None

    return None


def _annotate_rows_with_div(*, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Annotate rows with `recording_date`, `div_index`, and `div_label`.

    DIV is computed relative to the earliest recording date present in rows.
    """

    out: list[dict[str, Any]] = []

    parsed_dates: list[datetime] = []
    for r in rows:
        explicit_div = r.get("div")
        if explicit_div is not None:
            continue
        d_raw = r.get("recording_date")
        if d_raw is None:
            continue
        try:
            parsed_dates.append(datetime.strptime(str(d_raw), "%Y-%m-%d"))
        except Exception:
            continue

    min_date = min(parsed_dates) if parsed_dates else None

    for r in rows:
        rr = dict(r)
        explicit_div = rr.get("div")
        if explicit_div is not None:
            try:
                div_value = int(explicit_div)
                rr["div"] = div_value
                rr["div_index"] = div_value
                rr["div_label"] = f"DIV{div_value}"
                out.append(rr)
                continue
            except Exception:
                pass

        d_raw = rr.get("recording_date")
        if d_raw is None or min_date is None:
            rr["div"] = None
            rr["div_index"] = None
            rr["div_label"] = None
            out.append(rr)
            continue

        try:
            d = datetime.strptime(str(d_raw), "%Y-%m-%d")
            div_index = int((d - min_date).days)
            rr["div"] = div_index
            rr["div_index"] = div_index
            rr["div_label"] = f"DIV{div_index}"
        except Exception:
            rr["div"] = None
            rr["div_index"] = None
            rr["div_label"] = None
        out.append(rr)

    return out


def _compute_div_density_group_order(*, rows: list[dict[str, Any]]) -> list[tuple[int, int, str]]:
    seen: set[tuple[int, int, str]] = set()
    order: list[tuple[int, int, str]] = []
    for r in rows:
        div = r.get("div")
        density = r.get("plating_density_nbp")
        cond = r.get("condition")
        if div is None or density is None or cond is None:
            continue
        try:
            key = (int(div), int(density), str(cond))
        except Exception:
            continue
        if key not in seen:
            seen.add(key)
            order.append(key)
    order.sort(key=lambda x: (x[0], x[1], x[2]))
    return order


def _values_by_div_density(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    group_order: list[tuple[int, int, str]],
    where: dict[str, Any] | None = None,
    min_value: float | None = None,
) -> tuple[list[np.ndarray], list[str], list[str], list[str], list[str]]:
    groups: list[np.ndarray] = []
    group_labels: list[str] = []
    density_tick_labels: list[str] = []
    div_labels: list[str] = []
    conditions_for_groups: list[str] = []

    for div, density, cond in group_order:
        vals: list[float] = []
        for r in rows:
            if where is not None:
                ok = True
                for k, v in where.items():
                    if r.get(k) != v:
                        ok = False
                        break
                if not ok:
                    continue

            try:
                if int(r.get("div")) != int(div):
                    continue
                if int(r.get("plating_density_nbp")) != int(density):
                    continue
            except Exception:
                continue
            if str(r.get("condition")) != str(cond):
                continue

            v = r.get(metric)
            if v is None:
                continue
            try:
                vv = float(v)
            except Exception:
                continue
            if min_value is not None and vv < float(min_value):
                continue
            vals.append(vv)

        groups.append(np.asarray(vals, dtype=float))
        div_label = f"DIV{int(div)}"
        group_labels.append(f"{div_label} | {cond}")
        density_tick_labels.append(str(cond))
        div_labels.append(div_label)
        conditions_for_groups.append(str(cond))

    return groups, group_labels, div_labels, density_tick_labels, conditions_for_groups


def _rows_with_positive_metric(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        v = r.get(metric)
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if fv > 0:
            out.append(r)
    return out


def plot_grouped_bar_by_div_and_well(
    *,
    out_path: Path,
    title: str,
    ylabel: str,
    rows: list[dict[str, Any]],
    metric: str,
) -> None:
    import matplotlib.pyplot as plt

    numeric_rows: list[dict[str, Any]] = []
    for r in rows:
        div_idx = r.get("div_index")
        div_label = r.get("div_label")
        well_id = r.get("well_id")
        v = r.get(metric)
        if div_idx is None or div_label is None or well_id is None or v is None:
            continue
        try:
            vv = float(v)
        except Exception:
            continue
        numeric_rows.append(
            {
                "div_index": int(div_idx),
                "div_label": str(div_label),
                "well_id": str(well_id),
                "value": vv,
            }
        )

    if not numeric_rows:
        return

    div_order = sorted({int(r["div_index"]) for r in numeric_rows})
    div_labels = [f"DIV{d}" for d in div_order]
    wells_order = sorted({str(r["well_id"]) for r in numeric_rows})

    value_map: dict[tuple[int, str], float] = {}
    for r in numeric_rows:
        key = (int(r["div_index"]), str(r["well_id"]))
        value_map[key] = float(r["value"])

    x = np.arange(len(div_order), dtype=float)
    n_wells = max(1, len(wells_order))
    group_width = 0.84
    bar_w = group_width / n_wells

    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=150)
    for wi, well_id in enumerate(wells_order):
        offset = -group_width / 2 + wi * bar_w + bar_w / 2
        xs = x + offset
        ys = []
        for d in div_order:
            y = value_map.get((d, well_id))
            ys.append(np.nan if y is None else float(y))
        ys_arr = np.asarray(ys, dtype=float)
        mask = np.isfinite(ys_arr)
        if mask.any():
            ax.bar(xs[mask], ys_arr[mask], width=bar_w * 0.95, label=well_id, alpha=0.9)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Days in vitro (DIV)")
    ax.set_xticks(x)
    ax.set_xticklabels(div_labels)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Well", ncol=min(6, len(wells_order)), fontsize=8)

    fig.tight_layout()
    _ensure_dir(out_path.parent)
    fig.savefig(out_path)
    plt.close(fig)


def _p_to_stars(p: float) -> str | None:
    if p < 0.001:
        return "***"
    if p < 0.005:
        return "**"
    if p < 0.05:
        return "*"
    return None


def _pairwise_mannwhitneyu(groups: list[np.ndarray], group_labels: list[str]) -> list[dict[str, Any]]:
    from scipy.stats import mannwhitneyu

    out: list[dict[str, Any]] = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a = groups[i]
            b = groups[j]
            if a.size == 0 or b.size == 0:
                continue
            res = mannwhitneyu(a, b, alternative="two-sided")
            p = float(res.pvalue)
            out.append(
                {
                    "group_a": group_labels[i],
                    "group_b": group_labels[j],
                    "n_a": int(a.size),
                    "n_b": int(b.size),
                    "u": float(res.statistic),
                    "p": p,
                    "stars": _p_to_stars(p) or "",
                }
            )
    return out


def _pairwise_mannwhitneyu_within_blocks(
    groups: list[np.ndarray],
    group_labels: list[str],
    *,
    block_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    from scipy.stats import mannwhitneyu

    out: list[dict[str, Any]] = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if block_labels is not None:
                if i >= len(block_labels) or j >= len(block_labels):
                    continue
                if str(block_labels[i]) != str(block_labels[j]):
                    continue

            a = groups[i]
            b = groups[j]
            if a.size == 0 or b.size == 0:
                continue
            res = mannwhitneyu(a, b, alternative="two-sided")
            p = float(res.pvalue)
            out.append(
                {
                    "group_a": group_labels[i],
                    "group_b": group_labels[j],
                    "n_a": int(a.size),
                    "n_b": int(b.size),
                    "u": float(res.statistic),
                    "p": p,
                    "stars": _p_to_stars(p) or "",
                }
            )
    return out


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


def _iqr_outlier_mask(values: np.ndarray, *, k: float = 1.5) -> np.ndarray:
    """Return a boolean mask marking IQR outliers.

    Outliers are values < Q1 - k*IQR or > Q3 + k*IQR.
    For small samples (<4) or constant arrays, returns all-False.
    """

    v = np.asarray(values, dtype=float)
    if v.size < 4:
        return np.zeros(v.shape, dtype=bool)
    q1 = float(np.percentile(v, 25))
    q3 = float(np.percentile(v, 75))
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        return np.zeros(v.shape, dtype=bool)
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return (v < lo) | (v > hi)


def _prepare_groups_for_tests(
    groups: list[np.ndarray],
    labels: list[str],
    *,
    exclude_outliers: bool,
    outlier_k: float,
) -> tuple[list[np.ndarray], dict[str, dict[str, int]]]:
    """Optionally exclude IQR outliers per group; return groups used + stats per label."""

    used: list[np.ndarray] = []
    stats: dict[str, dict[str, int]] = {}
    for g, lab in zip(groups, labels, strict=True):
        g = np.asarray(g, dtype=float)
        mask = _iqr_outlier_mask(g, k=outlier_k) if exclude_outliers else np.zeros(g.shape, dtype=bool)
        g_used = g[~mask]
        used.append(g_used)
        stats[lab] = {
            "n_total": int(g.size),
            "n_outliers": int(mask.sum()),
            "n_used": int(g_used.size),
        }
    return used, stats


def _exclude_group_outliers(
    groups: list[np.ndarray],
    *,
    outlier_k: float,
) -> tuple[list[np.ndarray], dict[int, dict[str, int]]]:
    """Return outlier-filtered groups and per-group exclusion stats."""

    filtered: list[np.ndarray] = []
    stats: dict[int, dict[str, int]] = {}
    for idx, g in enumerate(groups):
        arr = np.asarray(g, dtype=float)
        mask = _iqr_outlier_mask(arr, k=outlier_k)
        kept = arr[~mask]
        filtered.append(kept)
        stats[idx] = {
            "n_total": int(arr.size),
            "n_outliers": int(mask.sum()),
            "n_kept": int(kept.size),
        }
    return filtered, stats


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


def _build_well_count_rows(
    *,
    wells_summary: list[dict[str, Any]],
    all_units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one row per (dataset_key, well_id) for detected/reconstructed counts."""

    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for r in wells_summary:
        dk = str(r.get("dataset_key", ""))
        wid = str(r.get("well_id", ""))
        if not dk or not wid:
            continue
        key = (dk, wid)
        rows_by_key[key] = {
            "dataset_key": dk,
            "well_id": wid,
            "condition": r.get("condition"),
            "plating_density_nbp": r.get("plating_density_nbp"),
            "div": r.get("div"),
            "n_units_detected": r.get("n_detected_units"),
            "n_units_reconstructed": int(r.get("n_reconstructed_units") or 0),
        }

    out = list(rows_by_key.values())
    out.sort(
        key=lambda r: (
            int(r.get("div")) if r.get("div") is not None else 10**9,
            str(r.get("well_id", "")),
        )
    )
    return out


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


def _values_by_condition(units_rows: list[dict[str, Any]], metric: str, conditions_order: list[str]) -> tuple[list[np.ndarray], list[str]]:
    groups: list[np.ndarray] = []
    labels: list[str] = []
    for cond in conditions_order:
        vals: list[float] = []
        for r in units_rows:
            if r.get("condition") != cond:
                continue
            v = r.get(metric)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except Exception:
                continue
        groups.append(np.asarray(vals, dtype=float))
        labels.append(cond)
    return groups, labels


def _values_by_condition_from_rows(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    conditions_order: list[str],
    where: dict[str, Any] | None = None,
) -> tuple[list[np.ndarray], list[str]]:
    groups: list[np.ndarray] = []
    labels: list[str] = []
    for cond in conditions_order:
        vals: list[float] = []
        for r in rows:
            if r.get("condition") != cond:
                continue
            if where is not None:
                ok = True
                for k, v in where.items():
                    if r.get(k) != v:
                        ok = False
                        break
                if not ok:
                    continue
            v = r.get(metric)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except Exception:
                continue
        groups.append(np.asarray(vals, dtype=float))
        labels.append(cond)
    return groups, labels


def _try_get_spikeinterface_recording_info(*, h5_path: Path, stream_id: str) -> tuple[dict[str, str], str | None]:
    """Best-effort SpikeInterface metadata extraction (kept intentionally small)."""

    try:
        import spikeinterface.extractors as se

        rec = se.read_maxwell(str(h5_path), stream_id=stream_id)
        info: dict[str, str] = {}
        try:
            info["sampling_frequency_hz"] = str(float(rec.get_sampling_frequency()))
        except Exception:
            pass
        try:
            info["num_channels"] = str(len(rec.get_channel_ids()))
        except Exception:
            pass
        try:
            nseg = int(rec.get_num_segments())
            info["num_segments"] = str(nseg)
        except Exception:
            nseg = 1
        try:
            fs = float(rec.get_sampling_frequency())
            seg_frames = []
            for si in range(nseg):
                seg_frames.append(int(rec.get_num_frames(segment_index=si)))
            if seg_frames and fs > 0:
                total_frames = sum(seg_frames)
                info["duration_s_total"] = f"{total_frames / fs:.2f}"
        except Exception:
            pass
        return info, None
    except Exception as e:
        return {}, f"SpikeInterface read_maxwell failed: {e}"


def _parse_env_assignments(path: Path) -> dict[str, str]:
    """Parse simple KEY=VALUE lines from a dotenv-like file."""

    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        out[key] = value
    return out


def _feature_key_from_plot_name(name: str) -> str | None:
    lower = name.lower()
    if (
        "n_branches" in lower
        or "n_units_detected" in lower
        or "n_units_reconstructed" in lower
        or "pct_reconstructed" in lower
    ):
        return "n_counts"
    if "area" in lower:
        return "area"
    if "velocity" in lower:
        return "velocity"
    if "length" in lower:
        return "length"
    return None


def _plot_source_from_name(name: str) -> str:
    lower = name.lower()
    if "branch_source-raw" in lower:
        return "raw"
    if "branch_source-clean" in lower:
        return "clean"
    if "n_counts__n_branches_raw" in lower:
        return "raw"
    if "n_counts__n_branches_clean" in lower:
        return "clean"
    if "_clean" in lower:
        return "clean"
    if "_raw" in lower:
        return "raw"
    if lower.startswith("unit__") and any(
        token in lower
        for token in [
            "mean_branch_velocity",
            "mean_branch_length_um",
            "total_branch_length_um",
            "n_branches_raw",
        ]
    ):
        return "raw"
    return "na"


def _plot_exclusion_level(name: str) -> int:
    lower = name.lower()
    level = 0
    if "positive_only" in lower:
        level += 1
    if "outliers_excluded" in lower:
        level += 1
    return level


def _plot_kind_rank(name: str) -> int:
    lower = name.lower()
    if lower.startswith("branch__"):
        return 0  # individual data-point level plots
    if "mean_" in lower:
        return 1
    if "total_" in lower:
        return 2
    return 3


def _sorted_feature_plots(*, plots_dir: Path, feature: str) -> list[Path]:
    all_pngs = sorted([p for p in Path(plots_dir).glob("*.png") if p.is_file()])
    feature_pngs = [p for p in all_pngs if _feature_key_from_plot_name(p.name) == feature]
    return sorted(
        feature_pngs,
        key=lambda p: (
            0
            if _plot_source_from_name(p.name) == "raw"
            else (1 if _plot_source_from_name(p.name) == "clean" else 2),
            _plot_kind_rank(p.name),
            _plot_exclusion_level(p.name),
            p.name,
        ),
    )


def _feature_av_param_keys(feature: str) -> list[str]:
    feature = str(feature)
    common_graph = [
        "AXON_RECON_AV_MAX_DISTANCE_FOR_EDGE",
        "AXON_RECON_AV_MAX_DISTANCE_TO_INIT",
        "AXON_RECON_AV_N_NEIGHBORS",
        "AXON_RECON_AV_DISTANCE_EXP",
    ]
    if feature == "length":
        return [
            "AXON_RECON_AV_MIN_PATH_LENGTH",
            "AXON_RECON_AV_MIN_PATH_POINTS",
            "AXON_RECON_AV_MIN_POINTS_AFTER_BRANCHING",
            "AXON_RECON_AV_SPLIT_PATHS",
            "AXON_RECON_AV_MAX_PEAK_LATENCY_FOR_SPLITTING",
            *common_graph,
        ]
    if feature == "velocity":
        return [
            "AXON_RECON_AV_R2_THRESHOLD",
            "AXON_RECON_AV_R2_THRESHOLD_FOR_OUTLIERS",
            "AXON_RECON_AV_THEILSEN_MAXITER",
            "AXON_RECON_AV_MAD_THRESHOLD",
            "AXON_RECON_AV_MIN_OUTLIER_TRACKING_ERROR",
            "AXON_RECON_AV_UPSAMPLE",
            "AXON_RECON_AV_MAX_PEAK_LATENCY_FOR_SPLITTING",
            *common_graph,
        ]
    if feature == "area":
        return [
            "AXON_RECON_AV_DETECT_THRESHOLD",
            "AXON_RECON_AV_DETECTION_TYPE",
            "AXON_RECON_AV_KURT_THRESHOLD",
            "AXON_RECON_AV_REMOVE_ISOLATED",
            "AXON_RECON_AV_MIN_SELECTED_POINTS",
            "AXON_RECON_AV_NEIGHBOR_RADIUS",
            *common_graph,
        ]
    if feature == "n_counts":
        return [
            "AXON_RECON_AV_MIN_POINTS_AFTER_BRANCHING",
            "AXON_RECON_AV_SPLIT_PATHS",
            "AXON_RECON_AV_MAX_PEAK_LATENCY_FOR_SPLITTING",
            "AXON_RECON_AV_MIN_PATH_POINTS",
            "AXON_RECON_AV_MIN_PATH_LENGTH",
            "AXON_RECON_AV_N_NEIGHBORS",
            "AXON_RECON_AV_DISTANCE_EXP",
        ]
    return []


def _av_default_values() -> dict[str, str]:
    return {
        "AXON_RECON_AV_UPSAMPLE": "1",
        "AXON_RECON_AV_INIT_DELAY": "0",
        "AXON_RECON_AV_DETECT_THRESHOLD": "0.01",
        "AXON_RECON_AV_DETECTION_TYPE": "relative",
        "AXON_RECON_AV_KURT_THRESHOLD": "0.3",
        "AXON_RECON_AV_PEAK_STD_THRESHOLD": "null",
        "AXON_RECON_AV_PEAK_STD_DISTANCE": "30",
        "AXON_RECON_AV_REMOVE_ISOLATED": "true",
        "AXON_RECON_AV_MIN_SELECTED_POINTS": "30",
        "AXON_RECON_AV_MIN_PATH_LENGTH": "100",
        "AXON_RECON_AV_MIN_PATH_POINTS": "5",
        "AXON_RECON_AV_MIN_POINTS_AFTER_BRANCHING": "3",
        "AXON_RECON_AV_MAX_DISTANCE_FOR_EDGE": "300",
        "AXON_RECON_AV_MAX_DISTANCE_TO_INIT": "200",
        "AXON_RECON_AV_N_NEIGHBORS": "3",
        "AXON_RECON_AV_R2_THRESHOLD": "0.9",
        "AXON_RECON_AV_MAD_THRESHOLD": "8",
        "AXON_RECON_AV_INIT_AMP_PEAK_RATIO": "0.2",
        "AXON_RECON_AV_EDGE_DIST_AMP_RATIO": "0.3",
        "AXON_RECON_AV_DISTANCE_EXP": "2",
        "AXON_RECON_AV_MAX_PEAK_LATENCY_FOR_SPLITTING": "0.5",
        "AXON_RECON_AV_R2_THRESHOLD_FOR_OUTLIERS": "0.98",
        "AXON_RECON_AV_MIN_OUTLIER_TRACKING_ERROR": "50",
        "AXON_RECON_AV_THEILSEN_MAXITER": "2000",
        "AXON_RECON_AV_NEIGHBOR_RADIUS": "100",
        "AXON_RECON_AV_SPLIT_PATHS": "true",
    }


def _format_av_value_with_default(*, key: str, values: dict[str, str]) -> str:
    if key in values:
        return str(values[key])
    defaults = _av_default_values()
    if key in defaults:
        return f"{defaults[key]} (default)"
    return "<unset>"


def _add_recording_info_slide(
    *,
    prs: Any,
    cfg: CrossWellConfig,
    config_path: Path,
    title: str,
) -> None:
    from pptx.util import Inches, Pt

    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    box = slide.shapes.add_textbox(Inches(0.6), Inches(0.4), Inches(12.2), Inches(6.8))
    tf = box.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = title
    p0.font.size = Pt(28)
    p0.font.bold = True

    def add_line(text: str, *, size: int = 14, bold: bool = False) -> None:
        para = tf.add_paragraph()
        para.text = text
        para.font.size = Pt(size)
        para.font.bold = bool(bold)

    add_line(f"Config: {config_path}")
    add_line(f"Analysis name: {cfg.analysis_name}")
    add_line(f"Out dir: {cfg.out_dir}")
    add_line(f"Electrode pitch (um): {cfg.electrode_pitch_um}")
    if cfg.runtime_env_file is not None:
        add_line(f"Runtime env file: {cfg.runtime_env_file}")
    add_line(f"Generated: {datetime.now().isoformat(timespec='seconds')}")

    for di, d in enumerate(cfg.datasets, start=1):
        add_line(f"Dataset {di}:", bold=True)
        add_line(f"  raw_data_h5_path: {d.raw_data_h5_path}")
        add_line(f"  inferred_output_root: {d.dataset_output_root}")
        if d.wells:
            add_line("  wells (density order from config):")
            for w in d.wells:
                add_line(
                    f"    - {w.well_id}: {w.condition} (density={w.plating_density_nbp}, genotype={w.genotype})",
                    size=12,
                )
            stream_id = d.wells[0].well_id
            si_info, si_err = _try_get_spikeinterface_recording_info(h5_path=d.raw_data_h5_path, stream_id=stream_id)
            if si_info:
                add_line(f"  recording info (SpikeInterface, stream={stream_id}):")
                for k in ["sampling_frequency_hz", "num_channels", "num_segments", "duration_s_total"]:
                    if k in si_info:
                        add_line(f"    - {k}: {si_info[k]}", size=12)
            elif si_err is not None:
                add_line(f"  recording info unavailable ({stream_id}): {si_err}", size=12)


def _add_feature_av_params_slide(
    *,
    prs: Any,
    feature: str,
    env_path: Path | None,
) -> None:
    from pptx.util import Inches, Pt

    feature_title = feature.replace("_", " ")
    values = _parse_env_assignments(env_path) if env_path is not None else {}
    keys = _feature_av_param_keys(feature)

    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    box = slide.shapes.add_textbox(Inches(0.6), Inches(0.4), Inches(12.2), Inches(6.8))
    tf = box.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = f"Axon velocity params for {feature_title}"
    p0.font.size = Pt(28)
    p0.font.bold = True

    def add_line(text: str, *, size: int = 14, bold: bool = False) -> None:
        para = tf.add_paragraph()
        para.text = text
        para.font.size = Pt(size)
        para.font.bold = bool(bold)

    add_line(f"Source env: {env_path if env_path is not None else '<none>'}")
    add_line("Relevant AXON_RECON_AV_* keys:", bold=True)
    if not keys:
        add_line("  - none configured for this feature", size=12)
    else:
        for key in keys:
            val = _format_av_value_with_default(key=key, values=values)
            add_line(f"  - {key} = {val}", size=12)


def _convert_pptx_to_pdf(*, deck_path: Path, out_dir: Path) -> Path | None:
    import shutil
    import subprocess

    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        return None
    cmd = [
        soffice,
        "--headless",
        "--nologo",
        "--nodefault",
        "--nolockcheck",
        "--norestore",
        "--convert-to",
        "pdf",
        "--outdir",
        str(out_dir),
        str(deck_path),
    ]
    try:
        subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception:
        return None
    pdf_path = out_dir / f"{deck_path.stem}.pdf"
    return pdf_path if pdf_path.exists() else None


def write_feature_slide_decks(*, out_dir: Path, cfg: CrossWellConfig, plots_dir: Path, config_path: Path) -> list[Path]:
    """Write one deck per feature (length/velocity/area/n_branches), plus PDF conversions."""

    import logging

    logger = logging.getLogger("projects.cross_well_feature_decks")

    try:
        from pptx import Presentation  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]
    except Exception as e:
        logger.warning("Skipping feature deck export (missing deps): %s", e)
        return []

    features = ["length", "velocity", "area", "n_counts"]
    written: list[Path] = []

    for feature in features:
        feature_pngs = _sorted_feature_plots(plots_dir=plots_dir, feature=feature)
        if not feature_pngs:
            continue

        prs = Presentation()
        prs.slide_width = 9144000 * 4 // 3
        prs.slide_height = 6858000
        slide_w = int(prs.slide_width)
        slide_h = int(prs.slide_height)

        blank_layout = prs.slide_layouts[6]
        for png_path in feature_pngs:
            s = prs.slides.add_slide(blank_layout)
            with Image.open(png_path) as im:
                iw, ih = im.size
            img_ratio = iw / float(ih)
            slide_ratio = slide_w / float(slide_h)

            if img_ratio >= slide_ratio:
                pic_w = slide_w
                pic_h = int(slide_w / img_ratio)
                left = 0
                top = int((slide_h - pic_h) / 2)
            else:
                pic_h = slide_h
                pic_w = int(slide_h * img_ratio)
                top = 0
                left = int((slide_w - pic_w) / 2)

            s.shapes.add_picture(str(png_path), left, top, width=pic_w, height=pic_h)
            try:
                s.notes_slide.notes_text_frame.text = png_path.name
            except Exception:
                pass

        _add_recording_info_slide(
            prs=prs,
            cfg=cfg,
            config_path=config_path,
            title=f"{feature.replace('_', ' ').title()} summary",
        )
        _add_feature_av_params_slide(
            prs=prs,
            feature=feature,
            env_path=cfg.runtime_env_file,
        )

        deck_path = Path(out_dir) / f"cross_well_{feature}_summary.pptx"
        prs.save(str(deck_path))
        written.append(deck_path)

        pdf_path = _convert_pptx_to_pdf(deck_path=deck_path, out_dir=Path(out_dir))
        if pdf_path is not None:
            written.append(pdf_path)

    return written


def write_feature_pdf_decks(*, out_dir: Path, cfg: CrossWellConfig, plots_dir: Path, config_path: Path) -> list[Path]:
    """Write one PDF deck per feature without relying on LibreOffice."""

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    features = ["length", "velocity", "area", "n_counts"]
    written: list[Path] = []

    for feature in features:
        feature_pngs = _sorted_feature_plots(plots_dir=plots_dir, feature=feature)
        if not feature_pngs:
            continue

        pdf_path = Path(out_dir) / f"cross_well_{feature}_summary.pdf"
        values = _parse_env_assignments(cfg.runtime_env_file) if cfg.runtime_env_file is not None else {}
        keys = _feature_av_param_keys(feature)

        with PdfPages(pdf_path) as pdf:
            # Plot pages first
            for png in feature_pngs:
                img = plt.imread(str(png))
                fig = plt.figure(figsize=(13.333, 7.5), dpi=150)
                fig.patch.set_facecolor("white")
                ax = fig.add_subplot(111)
                ax.imshow(img)
                ax.set_title(png.name, fontsize=10)
                ax.axis("off")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            # Recording/runtime info page at end
            fig = plt.figure(figsize=(13.333, 7.5), dpi=150)
            fig.patch.set_facecolor("white")
            ax = fig.add_subplot(111)
            ax.axis("off")

            lines: list[str] = [
                f"{feature.replace('_', ' ').title()} summary",
                "",
                f"Config: {config_path}",
                f"Analysis name: {cfg.analysis_name}",
                f"Out dir: {cfg.out_dir}",
                f"Electrode pitch (um): {cfg.electrode_pitch_um}",
                f"Runtime env file: {cfg.runtime_env_file if cfg.runtime_env_file is not None else '<none>'}",
                f"Generated: {datetime.now().isoformat(timespec='seconds')}",
                "",
            ]
            for di, d in enumerate(cfg.datasets, start=1):
                lines.append(f"Dataset {di}:")
                lines.append(f"  raw_data_h5_path: {d.raw_data_h5_path}")
                lines.append(f"  inferred_output_root: {d.dataset_output_root}")
                if d.wells:
                    lines.append("  wells:")
                    for w in d.wells:
                        lines.append(
                            f"    - {w.well_id}: {w.condition} (density={w.plating_density_nbp}, genotype={w.genotype})"
                        )
            ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Feature-relevant AV params page at end
            fig = plt.figure(figsize=(13.333, 7.5), dpi=150)
            fig.patch.set_facecolor("white")
            ax = fig.add_subplot(111)
            ax.axis("off")
            param_lines = [
                f"Axon velocity params for {feature.replace('_', ' ')}",
                "",
                f"Source env: {cfg.runtime_env_file if cfg.runtime_env_file is not None else '<none>'}",
                "",
                "Relevant AXON_RECON_AV_* keys:",
            ]
            if keys:
                for key in keys:
                    param_lines.append(f"  - {key} = {_format_av_value_with_default(key=key, values=values)}")
            else:
                param_lines.append("  - none configured for this feature")
            ax.text(0.02, 0.98, "\n".join(param_lines), va="top", ha="left", fontsize=11, family="monospace")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        written.append(pdf_path)

    return written


def write_cross_well_slide_deck(*, out_dir: Path, cfg: CrossWellConfig, plots_dir: Path, config_path: Path) -> Path | None:
    """Write a PPTX deck with a summary slide + one slide per plot PNG."""

    import logging
    import shutil
    import subprocess
    from datetime import datetime

    logger = logging.getLogger("projects.cross_well_deck")

    try:
        from pptx import Presentation  # type: ignore[import-not-found]
        from pptx.util import Inches, Pt
        from PIL import Image  # type: ignore[import-not-found]
    except Exception as e:
        logger.warning("Skipping slide deck export (missing deps): %s", e)
        return None

    deck_path = Path(out_dir) / "cross_well_summary.pptx"
    pdf_out_dir = Path(out_dir)

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    slide_w = int(prs.slide_width)
    slide_h = int(prs.slide_height)

    # --- Summary slide ---
    _add_recording_info_slide(
        prs=prs,
        cfg=cfg,
        config_path=config_path,
        title="Cross-well summary",
    )

    # --- Plot slides ---
    blank_layout = prs.slide_layouts[6]
    plot_paths = sorted([p for p in Path(plots_dir).glob("*.png") if p.is_file()])
    for png_path in plot_paths:
        s = prs.slides.add_slide(blank_layout)
        with Image.open(png_path) as im:
            iw, ih = im.size
        img_ratio = iw / float(ih)
        slide_ratio = slide_w / float(slide_h)

        if img_ratio >= slide_ratio:
            pic_w = slide_w
            pic_h = int(slide_w / img_ratio)
            left = 0
            top = int((slide_h - pic_h) / 2)
        else:
            pic_h = slide_h
            pic_w = int(slide_h * img_ratio)
            top = 0
            left = int((slide_w - pic_w) / 2)

        s.shapes.add_picture(str(png_path), left, top, width=pic_w, height=pic_h)
        try:
            s.notes_slide.notes_text_frame.text = png_path.name
        except Exception:
            pass

    prs.save(str(deck_path))

    # Best-effort PDF export (LibreOffice)
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice:
        cmd = [
            soffice,
            "--headless",
            "--nologo",
            "--nodefault",
            "--nolockcheck",
            "--norestore",
            "--convert-to",
            "pdf",
            "--outdir",
            str(pdf_out_dir),
            str(deck_path),
        ]
        try:
            subprocess.run(cmd, check=False, capture_output=True, text=True)
        except Exception:
            pass

    return deck_path


def main(argv: list[str] | None = None, *, default_config_path: Path | None = None) -> None:
    ap = argparse.ArgumentParser(description="Cross-well summary stats and plots")
    ap.add_argument(
        "--config",
        type=Path,
        default=(Path(default_config_path) if default_config_path is not None else (Path.cwd() / "cross_well_config.yml")),
        help="Path to YAML config (cross_well_config.yml)",
    )
    args = ap.parse_args(argv)

    cfg = load_cross_well_config(Path(args.config))
    if cfg.runtime_env_file is not None:
        load_env_file_into_os(env_file=cfg.runtime_env_file, override_existing=False)

    plot_sources = str(os.environ.get("AXON_RECON_CROSS_WELL_PLOT_SOURCES", "both") or "both").strip().lower()
    if plot_sources not in {"raw", "clean", "both"}:
        raise ValueError(
            "AXON_RECON_CROSS_WELL_PLOT_SOURCES must be one of: raw, clean, both. "
            f"Got: {plot_sources!r}"
        )

    exclude_outliers_in_tests = str(os.environ.get("AXON_RECON_CROSS_WELL_EXCLUDE_OUTLIERS_IN_TESTS", "1")).strip().lower()
    exclude_outliers = exclude_outliers_in_tests in {"1", "true", "yes", "y", "on"}
    try:
        outlier_k = float(os.environ.get("AXON_RECON_CROSS_WELL_OUTLIER_IQR_K", "1.5"))
    except Exception:
        outlier_k = 1.5

    out_dir = cfg.out_dir
    _ensure_dir(out_dir)

    all_units: list[dict[str, Any]] = []
    all_branches: list[dict[str, Any]] = []
    wells_summary: list[dict[str, Any]] = []

    for d in cfg.datasets:
        recording_date = _extract_recording_date_ymd_from_raw_h5(d.raw_data_h5_path)
        dataset_key = make_dataset_key(
            raw_data_h5_path=d.raw_data_h5_path,
            dataset_output_root=d.dataset_output_root,
        )
        for w in d.wells:
            metrics = extract_metrics_for_well(
                dataset_root=d.dataset_output_root,
                well=w,
                electrode_pitch_um=cfg.electrode_pitch_um,
                dataset_key=dataset_key,
                raw_data_h5_path=d.raw_data_h5_path,
            )
            wells_summary.append(
                {
                    "dataset_key": dataset_key,
                    "dataset_output_root": str(d.dataset_output_root),
                    "raw_data_h5_path": str(d.raw_data_h5_path),
                    "recording_date": recording_date,
                    "div": d.div,
                    "well_id": metrics["well_id"],
                    "condition": metrics["condition"],
                    "plating_density_nbp": metrics["plating_density_nbp"],
                    "genotype": metrics["genotype"],
                    "n_detected_units": metrics["n_detected_units"],
                    "n_units_with_recon_dir": metrics["n_units_with_recon_dir"],
                    "n_reconstructed_units": metrics["n_reconstructed_units"],
                }
            )
            metrics_units = [dict(u, recording_date=recording_date, div=d.div) for u in metrics["units"]]
            metrics_branches = [dict(b, recording_date=recording_date, div=d.div) for b in metrics["branches"]]
            all_units.extend(metrics_units)
            all_branches.extend(metrics_branches)

    _write_json(
        out_dir / "config_resolved.json",
        {
            "config_path": str(Path(args.config).resolve()),
            "analysis_name": cfg.analysis_name,
            "out_dir": str(cfg.out_dir),
            "electrode_pitch_um": cfg.electrode_pitch_um,
            "runtime_env_file": str(cfg.runtime_env_file) if cfg.runtime_env_file else None,
            "datasets": [
                {
                    "raw_data_h5_path": str(d.raw_data_h5_path),
                    "dataset_output_root": str(d.dataset_output_root),
                    "DIV": d.div,
                    "wells": [w.__dict__ for w in d.wells],
                }
                for d in cfg.datasets
            ],
        },
    )
    wells_summary = cw_stats._annotate_rows_with_div(rows=wells_summary)
    all_units = cw_stats._annotate_rows_with_div(rows=all_units)
    all_branches = cw_stats._annotate_rows_with_div(rows=all_branches)

    _write_csv(out_dir / "well_summary.csv", wells_summary)
    _write_csv(out_dir / "unit_metrics.csv", all_units)
    _write_csv(out_dir / "branch_metrics.csv", all_branches)

    # Plot ordering: by DIV then density then condition.
    group_order = cw_stats._compute_div_density_group_order(rows=wells_summary)
    plots_dir = out_dir / "plots"
    _ensure_dir(plots_dir)

    # Avoid stale plots from previous versions being accidentally included in the deck.
    for old in plots_dir.glob("*.png"):
        try:
            old.unlink()
        except Exception:
            pass

    tests_rows: list[dict[str, Any]] = []

    # --- Unit-level plots (raw + clean variants where applicable) ---
    unit_metrics_to_plot: list[tuple[str, str, str]] = []
    if plot_sources in {"raw", "both"}:
        unit_metrics_to_plot.extend(
            [
                ("total_branch_length_um", "Total branch length / unit (raw)", "µm"),
                ("mean_branch_velocity", "Mean branch velocity / unit (raw)", "(branch velocity units)"),
                ("mean_branch_length_um", "Mean branch length / unit (raw)", "µm"),
            ]
        )
    if plot_sources in {"clean", "both"}:
        unit_metrics_to_plot.extend(
            [
                ("total_branch_length_um_clean", "Total branch length / unit (clean)", "µm"),
                ("mean_branch_velocity_clean", "Mean branch velocity / unit (clean)", "(branch velocity units)"),
                ("mean_branch_length_um_clean", "Mean branch length / unit (clean)", "µm"),
            ]
        )

    # Always include axon area (not raw/clean-specific).
    unit_metrics_to_plot.append(("axon_area_um2", "Axon area / unit", "µm²"))

    for metric, title, ylabel in unit_metrics_to_plot:
        unit_rows_for_metric = all_units
        if (
            metric.startswith("n_branches_")
            or ("branch_length" in metric)
            or ("branch_velocity" in metric)
        ):
            unit_rows_for_metric = cw_stats._rows_with_positive_metric(all_units, metric)

        groups, labels, div_labels, density_tick_labels, conditions_for_groups = cw_stats._values_by_div_density(
            unit_rows_for_metric,
            metric=metric,
            group_order=group_order,
        )

        groups_used, group_stats = cw_stats._prepare_groups_for_tests(
            groups,
            labels,
            exclude_outliers=exclude_outliers,
            outlier_k=outlier_k,
        )
        pairwise = cw_stats._pairwise_mannwhitneyu_within_blocks(
            groups_used,
            labels,
            block_labels=div_labels,
        )
        for r in pairwise:
            r["metric"] = metric
            r["level"] = "unit"
            r["exclude_outliers"] = bool(exclude_outliers)
            r["outlier_iqr_k"] = float(outlier_k)
            r["n_outliers_a"] = group_stats.get(r["group_a"], {}).get("n_outliers")
            r["n_outliers_b"] = group_stats.get(r["group_b"], {}).get("n_outliers")
            r["n_used_a"] = group_stats.get(r["group_a"], {}).get("n_used")
            r["n_used_b"] = group_stats.get(r["group_b"], {}).get("n_used")
        tests_rows.extend(pairwise)
        cw_plots.plot_boxplot_with_stars(
            out_path=plots_dir / f"unit__{metric}.png",
            title=title,
            ylabel=ylabel,
            groups=groups,
            group_labels=labels,
            pairwise_tests=pairwise,
            outlier_iqr_k=outlier_k,
            exclude_outliers_in_tests=exclude_outliers,
            group_div_labels=div_labels,
            display_group_labels=density_tick_labels,
            group_conditions=conditions_for_groups,
        )

        groups_no_outliers, _ = cw_stats._exclude_group_outliers(groups, outlier_k=outlier_k)
        pairwise_no_outliers = cw_stats._pairwise_mannwhitneyu_within_blocks(
            groups_no_outliers,
            labels,
            block_labels=div_labels,
        )
        for r in pairwise_no_outliers:
            r["metric"] = metric
            r["level"] = "unit"
            r["exclude_outliers"] = True
            r["outlier_iqr_k"] = float(outlier_k)
            r["plot_variant"] = "outliers_excluded"
        tests_rows.extend(pairwise_no_outliers)
        cw_plots.plot_boxplot_with_stars(
            out_path=plots_dir / f"unit__{metric}__outliers_excluded.png",
            title=f"{title} (outliers excluded)",
            ylabel=ylabel,
            groups=groups_no_outliers,
            group_labels=labels,
            pairwise_tests=pairwise_no_outliers,
            outlier_iqr_k=outlier_k,
            exclude_outliers_in_tests=True,
            group_div_labels=div_labels,
            display_group_labels=density_tick_labels,
            group_conditions=conditions_for_groups,
            highlight_outliers=False,
        )

    # --- Branch-level plots (treat each branch as a data point) ---
    branch_metrics_to_plot: list[tuple[str, str, str, dict[str, Any]]] = []
    if plot_sources in {"raw", "both"}:
        branch_metrics_to_plot.extend(
            [
                ("branch_velocity", "Branch velocity (raw)", "(branch velocity units)", {"branch_source": "raw"}),
                ("branch_length_um", "Branch length (raw)", "µm", {"branch_source": "raw"}),
            ]
        )
    if plot_sources in {"clean", "both"}:
        branch_metrics_to_plot.extend(
            [
                ("branch_velocity", "Branch velocity (clean)", "(branch velocity units)", {"branch_source": "clean"}),
                ("branch_length_um", "Branch length (clean)", "µm", {"branch_source": "clean"}),
            ]
        )

    for metric, title, ylabel, where in branch_metrics_to_plot:
        groups, labels, div_labels, density_tick_labels, conditions_for_groups = cw_stats._values_by_div_density(
            all_branches,
            metric=metric,
            group_order=group_order,
            where=where,
        )

        groups_used, group_stats = cw_stats._prepare_groups_for_tests(
            groups,
            labels,
            exclude_outliers=exclude_outliers,
            outlier_k=outlier_k,
        )
        pairwise = cw_stats._pairwise_mannwhitneyu_within_blocks(
            groups_used,
            labels,
            block_labels=div_labels,
        )
        for r in pairwise:
            r["metric"] = metric
            r["level"] = "branch"
            r["where"] = json.dumps(where, sort_keys=True)
            r["exclude_outliers"] = bool(exclude_outliers)
            r["outlier_iqr_k"] = float(outlier_k)
            r["n_outliers_a"] = group_stats.get(r["group_a"], {}).get("n_outliers")
            r["n_outliers_b"] = group_stats.get(r["group_b"], {}).get("n_outliers")
            r["n_used_a"] = group_stats.get(r["group_a"], {}).get("n_used")
            r["n_used_b"] = group_stats.get(r["group_b"], {}).get("n_used")
        tests_rows.extend(pairwise)
        suffix = "_" + "_".join(f"{k}-{v}" for k, v in where.items())
        cw_plots.plot_boxplot_with_stars(
            out_path=plots_dir / f"branch__{metric}{suffix}.png",
            title=title,
            ylabel=ylabel,
            groups=groups,
            group_labels=labels,
            pairwise_tests=pairwise,
            outlier_iqr_k=outlier_k,
            exclude_outliers_in_tests=exclude_outliers,
            group_div_labels=div_labels,
            display_group_labels=density_tick_labels,
            group_conditions=conditions_for_groups,
        )

        groups_no_outliers, _ = cw_stats._exclude_group_outliers(groups, outlier_k=outlier_k)
        pairwise_no_outliers = cw_stats._pairwise_mannwhitneyu_within_blocks(
            groups_no_outliers,
            labels,
            block_labels=div_labels,
        )
        for r in pairwise_no_outliers:
            r["metric"] = metric
            r["level"] = "branch"
            r["where"] = json.dumps(where, sort_keys=True)
            r["exclude_outliers"] = True
            r["outlier_iqr_k"] = float(outlier_k)
            r["plot_variant"] = "outliers_excluded"
        tests_rows.extend(pairwise_no_outliers)
        cw_plots.plot_boxplot_with_stars(
            out_path=plots_dir / f"branch__{metric}{suffix}__outliers_excluded.png",
            title=f"{title} (outliers excluded)",
            ylabel=ylabel,
            groups=groups_no_outliers,
            group_labels=labels,
            pairwise_tests=pairwise_no_outliers,
            outlier_iqr_k=outlier_k,
            exclude_outliers_in_tests=True,
            group_div_labels=div_labels,
            display_group_labels=density_tick_labels,
            group_conditions=conditions_for_groups,
            highlight_outliers=False,
        )

        # Optional positive-only velocity view (for exploratory handling of negative values).
        if metric == "branch_velocity":
            groups_pos, labels_pos, div_labels_pos, density_labels_pos, conds_pos = cw_stats._values_by_div_density(
                all_branches,
                metric=metric,
                group_order=group_order,
                where=where,
                min_value=0.0,
            )

            groups_pos_used, group_stats_pos = cw_stats._prepare_groups_for_tests(
                groups_pos,
                labels_pos,
                exclude_outliers=exclude_outliers,
                outlier_k=outlier_k,
            )
            pairwise_pos = cw_stats._pairwise_mannwhitneyu_within_blocks(
                groups_pos_used,
                labels_pos,
                block_labels=div_labels_pos,
            )
            for r in pairwise_pos:
                r["metric"] = f"{metric}_positive_only"
                r["level"] = "branch"
                r["where"] = json.dumps(where, sort_keys=True)
                r["exclude_outliers"] = bool(exclude_outliers)
                r["outlier_iqr_k"] = float(outlier_k)
                r["n_outliers_a"] = group_stats_pos.get(r["group_a"], {}).get("n_outliers")
                r["n_outliers_b"] = group_stats_pos.get(r["group_b"], {}).get("n_outliers")
                r["n_used_a"] = group_stats_pos.get(r["group_a"], {}).get("n_used")
                r["n_used_b"] = group_stats_pos.get(r["group_b"], {}).get("n_used")
            tests_rows.extend(pairwise_pos)

            cw_plots.plot_boxplot_with_stars(
                out_path=plots_dir / f"branch__{metric}{suffix}__positive_only.png",
                title=f"{title} (non-negative only)",
                ylabel=ylabel,
                groups=groups_pos,
                group_labels=labels_pos,
                pairwise_tests=pairwise_pos,
                outlier_iqr_k=outlier_k,
                exclude_outliers_in_tests=exclude_outliers,
                group_div_labels=div_labels_pos,
                display_group_labels=density_labels_pos,
                group_conditions=conds_pos,
            )

            groups_pos_no_outliers, _ = cw_stats._exclude_group_outliers(groups_pos, outlier_k=outlier_k)
            pairwise_pos_no_outliers = cw_stats._pairwise_mannwhitneyu_within_blocks(
                groups_pos_no_outliers,
                labels_pos,
                block_labels=div_labels_pos,
            )
            for r in pairwise_pos_no_outliers:
                r["metric"] = f"{metric}_positive_only"
                r["level"] = "branch"
                r["where"] = json.dumps(where, sort_keys=True)
                r["exclude_outliers"] = True
                r["outlier_iqr_k"] = float(outlier_k)
                r["plot_variant"] = "outliers_excluded"
            tests_rows.extend(pairwise_pos_no_outliers)

            cw_plots.plot_boxplot_with_stars(
                out_path=plots_dir / f"branch__{metric}{suffix}__positive_only__outliers_excluded.png",
                title=f"{title} (non-negative only, outliers excluded)",
                ylabel=ylabel,
                groups=groups_pos_no_outliers,
                group_labels=labels_pos,
                pairwise_tests=pairwise_pos_no_outliers,
                outlier_iqr_k=outlier_k,
                exclude_outliers_in_tests=True,
                group_div_labels=div_labels_pos,
                display_group_labels=density_labels_pos,
                group_conditions=conds_pos,
                highlight_outliers=False,
            )

    # --- N-counts plots (bar + SEM): n_branches, n_units_detected, n_units_reconstructed ---
    n_count_specs: list[tuple[str, str, str, list[dict[str, Any]], str]] = []
    if plot_sources in {"raw", "both"}:
        n_count_specs.append(
            ("n_branches_raw", "n_counts: branches / unit (raw)", "count", all_units, "n_counts__n_branches_raw")
        )
    if plot_sources in {"clean", "both"}:
        n_count_specs.append(
            (
                "n_branches_clean",
                "n_counts: branches / unit (clean)",
                "count",
                all_units,
                "n_counts__n_branches_clean",
            )
        )

    for metric, title, ylabel, source_rows, stem in n_count_specs:
        rows_for_metric = cw_stats._rows_with_positive_metric(source_rows, metric) if metric.startswith("n_branches") else source_rows
        groups, labels, div_labels, density_tick_labels, conditions_for_groups = cw_stats._values_by_div_density(
            rows_for_metric,
            metric=metric,
            group_order=group_order,
        )

        groups_used, group_stats = cw_stats._prepare_groups_for_tests(
            groups,
            labels,
            exclude_outliers=exclude_outliers,
            outlier_k=outlier_k,
        )
        pairwise = cw_stats._pairwise_mannwhitneyu_within_blocks(
            groups_used,
            labels,
            block_labels=div_labels,
        )
        for r in pairwise:
            r["metric"] = metric
            r["level"] = "n_counts"
            r["exclude_outliers"] = bool(exclude_outliers)
            r["outlier_iqr_k"] = float(outlier_k)
            r["n_outliers_a"] = group_stats.get(r["group_a"], {}).get("n_outliers")
            r["n_outliers_b"] = group_stats.get(r["group_b"], {}).get("n_outliers")
            r["n_used_a"] = group_stats.get(r["group_a"], {}).get("n_used")
            r["n_used_b"] = group_stats.get(r["group_b"], {}).get("n_used")
        tests_rows.extend(pairwise)

        cw_plots.plot_mean_sem_bar_with_stars(
            out_path=plots_dir / f"{stem}.png",
            title=title,
            ylabel=ylabel,
            groups=groups,
            group_labels=labels,
            group_div_labels=div_labels,
            pairwise_tests=pairwise,
            outlier_iqr_k=outlier_k,
            exclude_outliers_in_tests=exclude_outliers,
            display_group_labels=density_tick_labels,
            group_conditions=conditions_for_groups,
            highlight_outliers=True,
        )

        groups_no_outliers, _ = cw_stats._exclude_group_outliers(groups, outlier_k=outlier_k)
        pairwise_no_outliers = cw_stats._pairwise_mannwhitneyu_within_blocks(
            groups_no_outliers,
            labels,
            block_labels=div_labels,
        )
        for r in pairwise_no_outliers:
            r["metric"] = metric
            r["level"] = "n_counts"
            r["exclude_outliers"] = True
            r["outlier_iqr_k"] = float(outlier_k)
            r["plot_variant"] = "outliers_excluded"
        tests_rows.extend(pairwise_no_outliers)

        cw_plots.plot_mean_sem_bar_with_stars(
            out_path=plots_dir / f"{stem}__outliers_excluded.png",
            title=f"{title} (outliers excluded)",
            ylabel=ylabel,
            groups=groups_no_outliers,
            group_labels=labels,
            group_div_labels=div_labels,
            pairwise_tests=pairwise_no_outliers,
            outlier_iqr_k=outlier_k,
            exclude_outliers_in_tests=True,
            display_group_labels=density_tick_labels,
            group_conditions=conditions_for_groups,
            highlight_outliers=False,
        )

    # Dedicated per-well counts (no outlier logic):
    # - detected units: single bar per well
    # - reconstructed units: paired raw/clean bars per well
    well_count_rows = cw_stats._build_well_count_rows(
        wells_summary=wells_summary,
        all_units=all_units,
    )
    cw_plots.plot_units_detected_vs_reconstructed_per_well(
        out_path=plots_dir / "n_counts__n_units_detected_vs_reconstructed_per_well.png",
        rows=well_count_rows,
    )
    cw_plots.plot_percent_reconstructed_per_well(
        out_path=plots_dir / "n_counts__pct_reconstructed_per_well.png",
        rows=well_count_rows,
    )

    _write_csv(out_dir / "pairwise_tests_mannwhitneyu.csv", tests_rows)

    deck = cw_decks.write_cross_well_slide_deck(
        out_dir=out_dir,
        cfg=cfg,
        plots_dir=plots_dir,
        config_path=Path(args.config).resolve(),
    )
    if deck is not None:
        print(f"Wrote deck: {deck}")

    feature_decks = cw_decks.write_feature_slide_decks(
        out_dir=out_dir,
        cfg=cfg,
        plots_dir=plots_dir,
        config_path=Path(args.config).resolve(),
    )
    for p in feature_decks:
        print(f"Wrote feature deck artifact: {p}")

    feature_pdfs = cw_decks.write_feature_pdf_decks(
        out_dir=out_dir,
        cfg=cfg,
        plots_dir=plots_dir,
        config_path=Path(args.config).resolve(),
    )
    for p in feature_pdfs:
        print(f"Wrote feature deck artifact: {p}")

    print(f"Wrote: {out_dir}")


if __name__ == "__main__":
    main()
