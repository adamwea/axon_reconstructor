from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir

from .models.inputs import AnalysisInputs
from .models.results import AnalysisResult


DEFAULT_STATS_COLUMNS = (
    "n",
    "total",
    "mean",
    "median",
    "min",
    "max",
    "std",
    "sem",
    "iqr",
    "p05",
    "p25",
    "p75",
    "p95",
)

TEMPLATE_ARRAY_CANDIDATES = (
    ("merged_template_npy", "merged_template_channel_locations_npy"),
    ("full_template_npy", "full_template_channel_locations_npy"),
    ("scan_template_npy", "scan_template_channel_locations_npy"),
    ("square_template_npy", "square_template_channel_locations_npy"),
)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _try_read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = _read_json(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


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


def _resolve_output_path(*, base_dir: Path, relpath: str) -> Path:
    raw = Path(str(relpath)).expanduser()
    if raw.is_absolute():
        return raw
    return base_dir / raw


def _resolve_summary_path(*, candidates: list[Path]) -> Path:
    if not candidates:
        raise ValueError("At least one summary path candidate is required")

    existing: list[tuple[float, Path]] = []
    for path in candidates:
        try:
            if path.exists():
                existing.append((float(path.stat().st_mtime), path))
        except Exception:
            continue

    if not existing:
        return candidates[0]

    existing.sort(key=lambda item: item[0], reverse=True)
    return existing[0][1]


def _as_unit_id_key(value: Any) -> str:
    try:
        return str(int(value))
    except Exception:
        return str(value)


def _parse_unit_id_for_output(value: str) -> Any:
    try:
        return int(value)
    except Exception:
        return value


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _branch_length_um(branch: dict[str, Any]) -> float | None:
    distances = branch.get("distances")
    if not isinstance(distances, list) or not distances:
        return None
    finite_values = [_as_float(x) for x in distances]
    cleaned = [v for v in finite_values if v is not None]
    if not cleaned:
        return None
    return float(max(cleaned))


def _branch_nodes(branch: dict[str, Any]) -> int | None:
    channels = branch.get("channels")
    if not isinstance(channels, list):
        return None
    return int(len(channels))


def _branch_tortuosity(*, branch: dict[str, Any], path_length_um: float | None) -> float | None:
    if path_length_um is None or path_length_um <= 0:
        return None
    polyline = branch.get("polyline_xy")
    if not isinstance(polyline, list) or len(polyline) < 2:
        return None
    try:
        coords = np.asarray(polyline, dtype=float)
    except Exception:
        return None
    if coords.ndim != 2 or coords.shape[0] < 2 or coords.shape[1] < 2:
        return None

    dx = float(coords[-1, 0] - coords[0, 0])
    dy = float(coords[-1, 1] - coords[0, 1])
    euclidean_um = float(math.hypot(dx, dy))
    if euclidean_um <= 0:
        return None
    return float(path_length_um / euclidean_um)


def _extract_discovered_unit_ids(
    *,
    templates_summary: dict[str, Any] | None,
    reconstruction_summary: dict[str, Any] | None,
) -> list[Any]:
    recon_units = reconstruction_summary.get("units", []) if isinstance(reconstruction_summary, dict) else []
    if isinstance(recon_units, list) and recon_units:
        out: list[Any] = []
        for row in recon_units:
            if isinstance(row, dict) and ("unit_id" in row):
                out.append(row.get("unit_id"))
        if out:
            return out

    templates_units = templates_summary.get("units", []) if isinstance(templates_summary, dict) else []
    if isinstance(templates_units, list) and templates_units:
        out = []
        for row in templates_units:
            if isinstance(row, dict) and ("unit_id" in row):
                out.append(row.get("unit_id"))
        return out

    return []


def _extract_branch_rows(*, unit_id: Any, payload: dict[str, Any] | None, branch_source: str) -> list[dict[str, Any]]:
    branches = payload.get("branches") if isinstance(payload, dict) else None
    if not isinstance(branches, list):
        return []

    out: list[dict[str, Any]] = []
    for idx, raw_branch in enumerate(branches):
        if not isinstance(raw_branch, dict):
            continue

        branch_index_raw = raw_branch.get("branch_index", idx)
        try:
            branch_index = int(branch_index_raw)
        except Exception:
            branch_index = int(idx)

        length_um = _branch_length_um(raw_branch)
        velocity = _as_float(raw_branch.get("velocity"))
        nodes = _branch_nodes(raw_branch)
        nodes_per_length = (
            (float(nodes) / float(length_um))
            if (nodes is not None and length_um is not None and float(length_um) > 0)
            else None
        )
        tortuosity = _branch_tortuosity(branch=raw_branch, path_length_um=length_um)

        out.append(
            {
                "unit_id": unit_id,
                "branch_index": branch_index,
                "branch_source": branch_source,
                "velocity": velocity,
                "length": length_um,
                "nodes": (None if nodes is None else int(nodes)),
                "nodes_per_length": nodes_per_length,
                "tortuosity": tortuosity,
            }
        )

    return out


def _normalize_template_channels_first(template: np.ndarray, n_channels: int) -> np.ndarray | None:
    if template.ndim != 2:
        return None
    if int(template.shape[0]) == int(n_channels):
        return template
    if int(template.shape[1]) == int(n_channels):
        return template.T
    return None


def _load_template_arrays_from_outputs(outputs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    for template_key, locs_key in TEMPLATE_ARRAY_CANDIDATES:
        template_path_raw = outputs.get(template_key)
        locs_path_raw = outputs.get(locs_key)
        if not template_path_raw or not locs_path_raw:
            continue

        template_path = Path(str(template_path_raw)).expanduser()
        locs_path = Path(str(locs_path_raw)).expanduser()
        if not template_path.exists() or not locs_path.exists():
            continue

        try:
            template = np.asarray(np.load(template_path), dtype=float)
            locs = np.asarray(np.load(locs_path), dtype=float)
        except Exception:
            continue

        if locs.ndim != 2 or locs.shape[0] <= 0 or locs.shape[1] < 2:
            continue

        channels_first = _normalize_template_channels_first(template, n_channels=int(locs.shape[0]))
        if channels_first is None:
            continue

        return channels_first, locs

    return None


def _has_template_array_key_pair(outputs: dict[str, Any]) -> bool:
    for template_key, locs_key in TEMPLATE_ARRAY_CANDIDATES:
        if outputs.get(template_key) and outputs.get(locs_key):
            return True
    return False


def _compute_template_metrics(
    *,
    outputs: dict[str, Any],
    probe_pitch_um: float | None,
) -> dict[str, Any]:
    loaded = _load_template_arrays_from_outputs(outputs)
    if loaded is None:
        return {}

    channels_by_time, locs = loaded
    if channels_by_time.size == 0 or locs.size == 0:
        return {}

    try:
        ptp = np.ptp(channels_by_time, axis=1)
    except Exception:
        return {}

    finite_idx = np.where(np.isfinite(ptp))[0]
    if finite_idx.size == 0:
        return {}

    active_idx = np.where(np.isfinite(ptp) & (ptp > 0))[0]
    selected_idx = active_idx if active_idx.size > 0 else finite_idx

    selected_locs = locs[selected_idx, :2]
    extent_x_um = float(np.max(selected_locs[:, 0]) - np.min(selected_locs[:, 0])) if selected_locs.size > 0 else None
    extent_y_um = float(np.max(selected_locs[:, 1]) - np.min(selected_locs[:, 1])) if selected_locs.size > 0 else None

    template_area = None
    if extent_x_um is not None and extent_y_um is not None and extent_x_um > 0 and extent_y_um > 0:
        template_area = float(extent_x_um * extent_y_um)
    elif probe_pitch_um is not None and probe_pitch_um > 0:
        template_area = float(len(selected_idx) * (probe_pitch_um ** 2))

    template_channel_density = None
    if template_area is not None and template_area > 0:
        template_channel_density = float(len(selected_idx) / template_area)

    strongest_local_idx = int(selected_idx[int(np.nanargmax(ptp[selected_idx]))])
    unit_x_um = float(locs[strongest_local_idx, 0])
    unit_y_um = float(locs[strongest_local_idx, 1])

    return {
        "template_area": template_area,
        "template_channel_density": template_channel_density,
        "template_extent_x_um": extent_x_um,
        "template_extent_y_um": extent_y_um,
        "unit_location_x_um": unit_x_um,
        "unit_location_y_um": unit_y_um,
    }


def _compute_sholl_metrics(*, branch_lengths_um: list[float], probe_pitch_um: float | None) -> dict[str, Any]:
    cleaned = [float(x) for x in branch_lengths_um if _as_float(x) is not None and float(x) > 0]
    if not cleaned:
        return {
            "sholl_analysis": None,
            "sholl_peak_intersections": None,
            "sholl_critical_radius_um": None,
        }

    max_length = float(max(cleaned))
    if max_length <= 0:
        return {
            "sholl_analysis": None,
            "sholl_peak_intersections": None,
            "sholl_critical_radius_um": None,
        }

    step_um = max(10.0, float((probe_pitch_um or 17.5) * 2.0))
    radii = np.arange(step_um, max_length + step_um, step_um, dtype=float)
    if radii.size == 0:
        radii = np.asarray([max_length], dtype=float)

    counts = [int(sum(1 for length in cleaned if float(length) >= float(radius))) for radius in radii]
    peak_intersections = int(max(counts)) if counts else 0
    critical_radius_um = None
    if counts and peak_intersections > 0:
        peak_index = counts.index(peak_intersections)
        critical_radius_um = float(radii[peak_index])

    profile = [
        {
            "radius_um": float(radius),
            "intersections": int(intersections),
        }
        for radius, intersections in zip(radii.tolist(), counts, strict=False)
    ]

    return {
        "sholl_analysis": json.dumps(profile),
        "sholl_peak_intersections": peak_intersections,
        "sholl_critical_radius_um": critical_radius_um,
    }


def _compute_stats(values: list[float], columns: list[str] | tuple[str, ...]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if _as_float(v) is not None], dtype=float)
    n = int(arr.size)

    out: dict[str, Any] = {}
    for column in columns:
        key = str(column)
        if key == "n":
            out[key] = n
        elif key == "total":
            out[key] = float(np.sum(arr)) if n > 0 else None
        elif n == 0:
            out[key] = None
        elif key == "mean":
            out[key] = float(np.mean(arr))
        elif key == "median":
            out[key] = float(np.median(arr))
        elif key == "min":
            out[key] = float(np.min(arr))
        elif key == "max":
            out[key] = float(np.max(arr))
        elif key == "std":
            out[key] = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        elif key == "sem":
            out[key] = float(np.std(arr, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        elif key == "iqr":
            out[key] = float(np.percentile(arr, 75) - np.percentile(arr, 25))
        elif key == "p05":
            out[key] = float(np.percentile(arr, 5))
        elif key == "p25":
            out[key] = float(np.percentile(arr, 25))
        elif key == "p75":
            out[key] = float(np.percentile(arr, 75))
        elif key == "p95":
            out[key] = float(np.percentile(arr, 95))
        else:
            out[key] = None

    return out


def _resolve_per_unit_source_values(
    *,
    source_metric: str,
    unit_keys: list[str],
    per_unit_scalars: dict[str, dict[str, Any]],
    per_unit_stats_tables: dict[str, dict[str, dict[str, Any]]],
) -> list[float]:
    source = str(source_metric or "").strip()
    if not source:
        return []

    values: list[float] = []
    if "." in source:
        metric_name, column = source.split(".", 1)
        by_unit = per_unit_stats_tables.get(metric_name, {})
        for unit_key in unit_keys:
            row = by_unit.get(unit_key)
            if not isinstance(row, dict):
                continue
            # If a stats row has n<=0, treat all non-n columns as missing data.
            row_n = _as_float(row.get("n"))
            if column != "n" and row_n is not None and int(row_n) <= 0:
                continue
            parsed = _as_float(row.get(column))
            if parsed is not None:
                values.append(parsed)
        return values

    by_unit_scalar = per_unit_scalars.get(source, {})
    for unit_key in unit_keys:
        parsed = _as_float(by_unit_scalar.get(unit_key))
        if parsed is not None:
            values.append(parsed)
    return values


def _unit_metadata_row(*, inputs: AnalysisInputs, unit_key: str, status: str) -> dict[str, Any]:
    return {
        "h5_path": str(inputs.h5_path),
        "stream_id": str(inputs.stream_id),
        "well_id": str(inputs.stream_id),
        "unit_id": _parse_unit_id_for_output(unit_key),
        "status": str(status),
    }


def _well_metadata_row(*, inputs: AnalysisInputs) -> dict[str, Any]:
    return {
        "h5_path": str(inputs.h5_path),
        "stream_id": str(inputs.stream_id),
        "well_id": str(inputs.stream_id),
    }


def run_analysis_stage_core(inputs: AnalysisInputs) -> AnalysisResult:
    well_out_dir = compute_mea_analysis_output_dir(
        output_root=inputs.mea_output_root,
        data_file=inputs.h5_path,
        well=inputs.stream_id,
    )
    analysis_out_dir = well_out_dir / str(inputs.output_rel_root)
    full_restart = bool(inputs.force_restart) and (not bool(inputs.force_replot))
    if full_restart and analysis_out_dir.exists():
        shutil.rmtree(analysis_out_dir)
    analysis_out_dir.mkdir(parents=True, exist_ok=True)

    templates_summary_json = _resolve_summary_path(
        candidates=[
            well_out_dir / "template_outputs" / "templates_summary.json",
            well_out_dir / "templates_outputs" / "templates_summary.json",
            well_out_dir / "stg4_templates_outputs" / "templates_summary.json",
        ]
    )
    reconstruction_summary_json = _resolve_summary_path(
        candidates=[
            well_out_dir / "recon_outputs" / "reconstruction_summary.json",
            well_out_dir / "reconstruction_outputs" / "reconstruction_summary.json",
        ]
    )
    templates_summary = _try_read_json(templates_summary_json)
    reconstruction_summary = _try_read_json(reconstruction_summary_json)

    templates_units_raw = templates_summary.get("units", []) if isinstance(templates_summary, dict) else []
    templates_units = templates_units_raw if isinstance(templates_units_raw, list) else []
    runtime_warnings: list[str] = []
    if templates_summary is None:
        runtime_warnings.append("templates_summary.json is missing or unreadable; template-derived metrics may be empty.")
    elif not templates_units:
        runtime_warnings.append("templates_summary.json has no units entries; template-derived metrics may be empty.")
    templates_by_id = {
        _as_unit_id_key(row.get("unit_id")): row
        for row in templates_units
        if isinstance(row, dict) and ("unit_id" in row)
    }

    recon_units_raw = reconstruction_summary.get("units", []) if isinstance(reconstruction_summary, dict) else []
    recon_units = recon_units_raw if isinstance(recon_units_raw, list) else []
    if reconstruction_summary is None:
        runtime_warnings.append("reconstruction_summary.json is missing or unreadable; branch-derived metrics may be empty.")
    elif not recon_units:
        runtime_warnings.append("reconstruction_summary.json has no units entries; branch-derived metrics may be empty.")
    recon_by_id = {
        _as_unit_id_key(row.get("unit_id")): row
        for row in recon_units
        if isinstance(row, dict) and ("unit_id" in row)
    }

    discovered_unit_ids = _extract_discovered_unit_ids(
        templates_summary=templates_summary,
        reconstruction_summary=reconstruction_summary,
    )

    if inputs.unit_ids is not None:
        selected_keys = [_as_unit_id_key(unit_id) for unit_id in inputs.unit_ids]
    else:
        selected_keys = [_as_unit_id_key(unit_id) for unit_id in discovered_unit_ids]

    if not selected_keys:
        selected_keys = sorted(set(recon_by_id.keys()) | set(templates_by_id.keys()))

    deduped_keys: list[str] = []
    seen_keys: set[str] = set()
    for key in selected_keys:
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped_keys.append(key)

    if inputs.unit_limit is not None:
        deduped_keys = deduped_keys[: int(inputs.unit_limit)]

    per_unit_scalars: dict[str, dict[str, Any]] = {
        "n_branches": {},
        "sholl_analysis": {},
        "sholl_peak_intersections": {},
        "sholl_critical_radius_um": {},
        "template_area": {},
        "template_channel_density": {},
        "template_extent_x_um": {},
        "template_extent_y_um": {},
        "branch_points_per_100um": {},
    }
    per_unit_status: dict[str, str] = {}
    unit_locations: list[dict[str, Any]] = []
    units_missing_branch_sources = 0
    units_missing_template_sources = 0
    units_template_source_load_failures = 0

    branch_rows: list[dict[str, Any]] = []
    branch_values_by_metric: dict[str, dict[str, list[float]]] = {
        "velocity": {},
        "length": {},
        "nodes": {},
        "nodes_per_length": {},
        "tortuosity": {},
    }

    for unit_key in deduped_keys:
        recon_row = recon_by_id.get(unit_key, {}) if isinstance(recon_by_id.get(unit_key, {}), dict) else {}
        recon_outputs = recon_row.get("outputs", {}) if isinstance(recon_row.get("outputs", {}), dict) else {}
        template_row = templates_by_id.get(unit_key, {}) if isinstance(templates_by_id.get(unit_key, {}), dict) else {}
        template_outputs = template_row.get("outputs", {}) if isinstance(template_row.get("outputs", {}), dict) else {}

        status = str(recon_row.get("status", "missing"))
        status_norm = status.strip().lower()
        per_unit_status[unit_key] = status

        has_branch_source_paths = bool(recon_outputs.get("branches_json")) or bool(recon_outputs.get("branches_raw_json"))
        if status_norm == "ok" and not has_branch_source_paths:
            units_missing_branch_sources += 1

        branches_clean = _try_read_json(
            Path(str(recon_outputs.get("branches_json", ""))).expanduser()
            if recon_outputs.get("branches_json")
            else None
        )
        branches_raw = _try_read_json(
            Path(str(recon_outputs.get("branches_raw_json", ""))).expanduser()
            if recon_outputs.get("branches_raw_json")
            else None
        )

        unit_id_value = _parse_unit_id_for_output(unit_key)
        branch_rows_clean = _extract_branch_rows(unit_id=unit_id_value, payload=branches_clean, branch_source="clean")
        branch_rows_raw = _extract_branch_rows(unit_id=unit_id_value, payload=branches_raw, branch_source="raw")
        unit_branch_rows = branch_rows_clean if branch_rows_clean else branch_rows_raw

        for branch_row in unit_branch_rows:
            branch_row.update(
                {
                    "h5_path": str(inputs.h5_path),
                    "stream_id": str(inputs.stream_id),
                    "well_id": str(inputs.stream_id),
                    "unit_key": unit_key,
                }
            )
            branch_rows.append(branch_row)

            for metric_name in branch_values_by_metric.keys():
                parsed = _as_float(branch_row.get(metric_name))
                if parsed is None:
                    continue
                by_unit = branch_values_by_metric[metric_name].setdefault(unit_key, [])
                by_unit.append(parsed)

        branch_lengths = branch_values_by_metric.get("length", {}).get(unit_key, [])
        n_branches = int(len(unit_branch_rows))
        total_branch_length = float(sum(branch_lengths)) if branch_lengths else 0.0
        branch_points_per_100um = (
            float((float(n_branches) / total_branch_length) * 100.0)
            if total_branch_length > 0
            else None
        )

        sholl = _compute_sholl_metrics(branch_lengths_um=branch_lengths, probe_pitch_um=inputs.probe_pitch_um)
        template_has_source_keys = _has_template_array_key_pair(template_outputs)
        template_metrics = _compute_template_metrics(outputs=template_outputs, probe_pitch_um=inputs.probe_pitch_um)
        if status_norm == "ok" and not template_metrics:
            if template_has_source_keys:
                units_template_source_load_failures += 1
            else:
                units_missing_template_sources += 1

        per_unit_scalars["n_branches"][unit_key] = n_branches
        per_unit_scalars["branch_points_per_100um"][unit_key] = branch_points_per_100um
        per_unit_scalars["sholl_analysis"][unit_key] = sholl.get("sholl_analysis")
        per_unit_scalars["sholl_peak_intersections"][unit_key] = sholl.get("sholl_peak_intersections")
        per_unit_scalars["sholl_critical_radius_um"][unit_key] = sholl.get("sholl_critical_radius_um")
        per_unit_scalars["template_area"][unit_key] = template_metrics.get("template_area")
        per_unit_scalars["template_channel_density"][unit_key] = template_metrics.get("template_channel_density")
        per_unit_scalars["template_extent_x_um"][unit_key] = template_metrics.get("template_extent_x_um")
        per_unit_scalars["template_extent_y_um"][unit_key] = template_metrics.get("template_extent_y_um")

        unit_locations.append(
            {
                **_unit_metadata_row(inputs=inputs, unit_key=unit_key, status=status),
                "x_um": template_metrics.get("unit_location_x_um"),
                "y_um": template_metrics.get("unit_location_y_um"),
            }
        )

    metrics_cfg = inputs.metrics if isinstance(inputs.metrics, dict) else {}
    outputs: dict[str, str] = {}

    if units_missing_branch_sources > 0:
        runtime_warnings.append(
            "Branch-source outputs are missing for "
            f"{units_missing_branch_sources} reconstructed units "
            "(expected outputs keys: branches_json and/or branches_raw_json)."
        )
    if units_missing_template_sources > 0:
        runtime_warnings.append(
            "Template-array outputs are missing for "
            f"{units_missing_template_sources} reconstructed units "
            "(expected templates outputs keys include merged_template_npy and merged_template_channel_locations_npy, "
            "or full/scan/square equivalents)."
        )
    if units_template_source_load_failures > 0:
        runtime_warnings.append(
            "Template-array outputs were referenced but could not be loaded or validated for "
            f"{units_template_source_load_failures} reconstructed units."
        )

    per_branch_cfg = metrics_cfg.get("per_branch", {}) if isinstance(metrics_cfg.get("per_branch", {}), dict) else {}
    per_branch_dir = analysis_out_dir / str(per_branch_cfg.get("reldir", "branch_metrics/"))
    for metric_name, metric_cfg_raw in per_branch_cfg.items():
        if metric_name == "reldir":
            continue
        metric_cfg = metric_cfg_raw if isinstance(metric_cfg_raw, dict) else {}
        if not bool(metric_cfg.get("write_csv", False)):
            continue

        csv_relpath = str(metric_cfg.get("csv_relpath", f"{metric_name}.csv"))
        csv_path = _resolve_output_path(base_dir=per_branch_dir, relpath=csv_relpath)
        rows_for_metric: list[dict[str, Any]] = []
        for row in branch_rows:
            rows_for_metric.append(
                {
                    "h5_path": row.get("h5_path"),
                    "stream_id": row.get("stream_id"),
                    "well_id": row.get("well_id"),
                    "unit_id": row.get("unit_id"),
                    "branch_index": row.get("branch_index"),
                    "branch_source": row.get("branch_source"),
                    "value": row.get(metric_name),
                    metric_name: row.get(metric_name),
                }
            )
        _write_rows_csv(csv_path, rows_for_metric)
        outputs[f"per_branch.{metric_name}"] = str(csv_path)

    per_unit_cfg = metrics_cfg.get("per_unit", {}) if isinstance(metrics_cfg.get("per_unit", {}), dict) else {}
    per_unit_dir = analysis_out_dir / str(per_unit_cfg.get("reldir", "unit_metrics/"))
    per_unit_stats_tables: dict[str, dict[str, dict[str, Any]]] = {}

    for metric_name, metric_cfg_raw in per_unit_cfg.items():
        if metric_name == "reldir":
            continue
        metric_cfg = metric_cfg_raw if isinstance(metric_cfg_raw, dict) else {}
        write_csv = bool(metric_cfg.get("write_csv", False))
        if not write_csv:
            continue

        csv_relpath = str(metric_cfg.get("csv_relpath", f"{metric_name}.csv"))
        csv_path = _resolve_output_path(base_dir=per_unit_dir, relpath=csv_relpath)
        source_level = metric_cfg.get("source_level")

        if source_level is not None:
            source_level_norm = str(source_level).strip().lower()
            if source_level_norm != "per_branch":
                runtime_warnings.append(
                    f"Per-unit metric '{metric_name}' has unsupported source_level '{source_level_norm}'; expected per_branch."
                )
                _write_rows_csv(csv_path, [])
                outputs[f"per_unit.{metric_name}"] = str(csv_path)
                continue

            source_metric = str(metric_cfg.get("source_metric", "")).strip()
            columns = metric_cfg.get("columns", list(DEFAULT_STATS_COLUMNS))
            columns_list = [str(x) for x in columns] if isinstance(columns, (list, tuple)) else list(DEFAULT_STATS_COLUMNS)
            by_unit_values = branch_values_by_metric.get(source_metric, {})

            rows_for_metric: list[dict[str, Any]] = []
            by_unit_rows: dict[str, dict[str, Any]] = {}
            for unit_key in deduped_keys:
                stats = _compute_stats(by_unit_values.get(unit_key, []), columns_list)
                row = {
                    **_unit_metadata_row(inputs=inputs, unit_key=unit_key, status=per_unit_status.get(unit_key, "missing")),
                    **stats,
                }
                rows_for_metric.append(row)
                by_unit_rows[unit_key] = row

            per_unit_stats_tables[metric_name] = by_unit_rows
            _write_rows_csv(csv_path, rows_for_metric)
            outputs[f"per_unit.{metric_name}"] = str(csv_path)
            continue

        scalar_values = per_unit_scalars.get(metric_name, {})
        rows_for_metric = [
            {
                **_unit_metadata_row(inputs=inputs, unit_key=unit_key, status=per_unit_status.get(unit_key, "missing")),
                "value": scalar_values.get(unit_key),
            }
            for unit_key in deduped_keys
        ]
        _write_rows_csv(csv_path, rows_for_metric)
        outputs[f"per_unit.{metric_name}"] = str(csv_path)

    n_units_total = int(len(deduped_keys))
    n_units_reconstructed = int(
        sum(1 for unit_key in deduped_keys if str(per_unit_status.get(unit_key, "")).strip().lower() == "ok")
    )
    n_units_with_branches = int(sum(1 for unit_key in deduped_keys if int(per_unit_scalars["n_branches"].get(unit_key, 0) or 0) > 0))
    frac_units_with_branches = (
        float(n_units_with_branches / n_units_total) if n_units_total > 0 else None
    )

    per_well_cfg = metrics_cfg.get("per_well", {}) if isinstance(metrics_cfg.get("per_well", {}), dict) else {}
    per_well_dir = analysis_out_dir / str(per_well_cfg.get("reldir", "well_metrics/"))

    counter_values: dict[str, Any] = {
        "n_units_total": n_units_total,
        "n_units_reconstructed": n_units_reconstructed,
        "n_units_with_branches": n_units_with_branches,
        "frac_units_with_branches": frac_units_with_branches,
    }

    for metric_name, metric_cfg_raw in per_well_cfg.items():
        if metric_name == "reldir":
            continue
        metric_cfg = metric_cfg_raw if isinstance(metric_cfg_raw, dict) else {}
        if not bool(metric_cfg.get("write_csv", False)):
            continue

        csv_relpath = str(metric_cfg.get("csv_relpath", f"{metric_name}.csv"))
        csv_path = _resolve_output_path(base_dir=per_well_dir, relpath=csv_relpath)

        if metric_name == "unit_locations":
            _write_rows_csv(csv_path, unit_locations)
            outputs[f"per_well.{metric_name}"] = str(csv_path)
            continue

        if metric_name == "unit_loc_density":
            finite_points = [
                (float(row["x_um"]), float(row["y_um"]))
                for row in unit_locations
                if _as_float(row.get("x_um")) is not None and _as_float(row.get("y_um")) is not None
            ]
            density_row = _well_metadata_row(inputs=inputs)
            density_row["n_units_with_location"] = int(len(finite_points))
            if finite_points:
                pts = np.asarray(finite_points, dtype=float)
                x_span = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
                y_span = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
                bbox_area_um2 = float(x_span * y_span) if x_span > 0 and y_span > 0 else None
                density_row["bbox_area_um2"] = bbox_area_um2
                density_row["units_per_mm2"] = (
                    float(len(finite_points) / (bbox_area_um2 / 1_000_000.0))
                    if bbox_area_um2 is not None and bbox_area_um2 > 0
                    else None
                )

                if pts.shape[0] > 1:
                    diffs = pts[:, None, :] - pts[None, :, :]
                    distances = np.sqrt(np.sum(diffs ** 2, axis=2))
                    np.fill_diagonal(distances, np.inf)
                    nn = np.min(distances, axis=1)
                    density_row["mean_nearest_neighbor_distance_um"] = float(np.mean(nn))
                else:
                    density_row["mean_nearest_neighbor_distance_um"] = None
            else:
                density_row["bbox_area_um2"] = None
                density_row["units_per_mm2"] = None
                density_row["mean_nearest_neighbor_distance_um"] = None

            _write_rows_csv(csv_path, [density_row])
            outputs[f"per_well.{metric_name}"] = str(csv_path)
            continue

        source_level = metric_cfg.get("source_level")
        if source_level is not None:
            source_level_norm = str(source_level).strip().lower()
            if source_level_norm != "per_unit":
                runtime_warnings.append(
                    f"Per-well metric '{metric_name}' has unsupported source_level '{source_level_norm}'; expected per_unit."
                )
                _write_rows_csv(csv_path, [])
                outputs[f"per_well.{metric_name}"] = str(csv_path)
                continue

            source_metric = str(metric_cfg.get("source_metric", "")).strip()
            columns = metric_cfg.get("columns", list(DEFAULT_STATS_COLUMNS))
            columns_list = [str(x) for x in columns] if isinstance(columns, (list, tuple)) else list(DEFAULT_STATS_COLUMNS)
            values = _resolve_per_unit_source_values(
                source_metric=source_metric,
                unit_keys=deduped_keys,
                per_unit_scalars=per_unit_scalars,
                per_unit_stats_tables=per_unit_stats_tables,
            )
            stats = _compute_stats(values, columns_list)
            row = {
                **_well_metadata_row(inputs=inputs),
                **stats,
            }
            _write_rows_csv(csv_path, [row])
            outputs[f"per_well.{metric_name}"] = str(csv_path)
            continue

        if metric_name in counter_values:
            row = {
                **_well_metadata_row(inputs=inputs),
                "metric": metric_name,
                "value": counter_values[metric_name],
                "n_units_total": n_units_total,
                "n_units_reconstructed": n_units_reconstructed,
                "n_units_with_branches": n_units_with_branches,
            }
            _write_rows_csv(csv_path, [row])
            outputs[f"per_well.{metric_name}"] = str(csv_path)
            continue

        runtime_warnings.append(f"Per-well metric '{metric_name}' is configured but not recognized by current runner.")
        _write_rows_csv(csv_path, [])
        outputs[f"per_well.{metric_name}"] = str(csv_path)

    warnings = list(inputs.deferred_warnings) + runtime_warnings

    summary_json = analysis_out_dir / "analysis_summary.json"
    summary_payload = {
        "h5_path": str(inputs.h5_path),
        "stream_id": str(inputs.stream_id),
        "well_out_dir": str(well_out_dir),
        "analysis_out_dir": str(analysis_out_dir),
        "inputs": {
            "output_rel_root": str(inputs.output_rel_root),
            "unit_ids": [str(u) for u in (inputs.unit_ids or [])],
            "unit_limit": (None if inputs.unit_limit is None else int(inputs.unit_limit)),
            "force_restart": bool(inputs.force_restart),
            "force_replot": bool(inputs.force_replot),
            "n_jobs": int(max(1, int(inputs.n_jobs))),
            "probe_pitch_um": (None if inputs.probe_pitch_um is None else float(inputs.probe_pitch_um)),
        },
        "inputs_available": {
            "templates_summary_json": str(templates_summary_json) if templates_summary_json.exists() else None,
            "reconstruction_summary_json": str(reconstruction_summary_json) if reconstruction_summary_json.exists() else None,
        },
        "unit_counts": {
            "n_units_total": n_units_total,
            "n_units_reconstructed": n_units_reconstructed,
            "n_units_with_branches": n_units_with_branches,
            "frac_units_with_branches": frac_units_with_branches,
        },
        "rows_generated": {
            "branch_rows": int(len(branch_rows)),
            "unit_rows": int(len(deduped_keys)),
            "unit_location_rows": int(len(unit_locations)),
        },
        "outputs": outputs,
        "deferred_warnings": warnings,
    }
    _write_json(summary_json, summary_payload)

    return AnalysisResult(
        well_out_dir=well_out_dir,
        analysis_out_dir=analysis_out_dir,
        summary_json=summary_json,
        outputs=outputs,
        deferred_warnings=warnings,
    )
