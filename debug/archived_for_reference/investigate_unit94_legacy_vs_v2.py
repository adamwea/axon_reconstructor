from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from axon_recon.pipeline.config import load_pipeline_runtime_bundle
from axon_recon.pipeline.stages.reconstruct.config import parse_reconstruction_stage_config
from axon_recon.pipeline.stages.reconstruct.integrations.axon_velocity import compute_graph_tracking
from axon_recon.pipeline.stages.reconstruct.integrations.axon_velocity import import_axon_velocity


DEFAULT_LEGACY_FULL_TEMPLATE = Path(
    "/home/adamm/dev/symlinks/local_RBS_data/outputs/Media_Density_T5_02182026_AR/260312/"
    "M08073/AxonTracking/000146/well001/stg4_templates_outputs/templates/full/unit_94/full_template.npy"
)
DEFAULT_LEGACY_FULL_LOCS = Path(
    "/home/adamm/dev/symlinks/local_RBS_data/outputs/Media_Density_T5_02182026_AR/260312/"
    "M08073/AxonTracking/000146/well001/stg4_templates_outputs/templates/full/unit_94/full_channel_locations_xy.npy"
)
DEFAULT_LEGACY_FULL_META = Path(
    "/home/adamm/dev/symlinks/local_RBS_data/outputs/Media_Density_T5_02182026_AR/260312/"
    "M08073/AxonTracking/000146/well001/stg4_templates_outputs/templates/full/unit_94/full_template_meta.json"
)

DEFAULT_V2_MERGED_TEMPLATE = Path(
    "/home/adamm/dev/symlinks/local_RBS_data/outputs/Media_Density_T5_02182026_AR/260312/"
    "M08073/AxonTracking/000146/well001/template_outputs/units/0094/merged_template.npy"
)
DEFAULT_V2_MERGED_LOCS = Path(
    "/home/adamm/dev/symlinks/local_RBS_data/outputs/Media_Density_T5_02182026_AR/260312/"
    "M08073/AxonTracking/000146/well001/template_outputs/templates/merged/unit_94/merged_contributing_channel_locations.npy"
)

DEFAULT_RUNTIME_CONFIG = REPO_ROOT / "tools" / "debug" / "debug.runtime.yml"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "tools" / "debug" / "outputs" / "unit94_run146_legacy_vs_current_report.json"
DEFAULT_OUTPUT_TEMPLATE = REPO_ROOT / "tools" / "debug" / "outputs" / "unit94_run146_generated_full_from_current_merged.npy"
DEFAULT_OUTPUT_LOCS = REPO_ROOT / "tools" / "debug" / "outputs" / "unit94_run146_generated_full_locations_xy.npy"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sampling_hz_from_meta(path: Path) -> float:
    if not path.exists():
        return 10_000.0
    try:
        data = _read_json(path)
    except Exception:
        return 10_000.0
    for key in ("sampling_frequency_hz", "effective_sampling_rate_hz", "native_sampling_frequency_hz"):
        if data.get(key) is not None:
            return float(data[key])
    return 10_000.0


def _to_ch_by_t(template_any: np.ndarray, n_channels: int) -> np.ndarray:
    arr = np.asarray(template_any, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Template must be 2D, got shape {arr.shape}")
    if int(arr.shape[0]) == int(n_channels):
        return arr
    if int(arr.shape[1]) == int(n_channels):
        return np.asarray(arr.T, dtype=float)
    raise ValueError(f"Cannot orient template {arr.shape} to n_channels={n_channels}")


def _loc_key(xy: np.ndarray) -> tuple[float, float]:
    return (float(np.round(float(xy[0]), 6)), float(np.round(float(xy[1]), 6)))


def _loc_index(locs_xy: np.ndarray) -> dict[tuple[float, float], int]:
    idx: dict[tuple[float, float], int] = {}
    for i in range(int(locs_xy.shape[0])):
        key = _loc_key(locs_xy[i, :2])
        if key not in idx:
            idx[key] = int(i)
    return idx


def _project_to_target_grid(
    merged_ch_by_t: np.ndarray,
    merged_locs_xy: np.ndarray,
    target_locs_xy: np.ndarray,
) -> np.ndarray:
    out = np.zeros((int(target_locs_xy.shape[0]), int(merged_ch_by_t.shape[1])), dtype=float)
    target_idx = _loc_index(target_locs_xy)
    for src_i in range(min(int(merged_ch_by_t.shape[0]), int(merged_locs_xy.shape[0]))):
        key = _loc_key(merged_locs_xy[src_i, :2])
        dst_i = target_idx.get(key)
        if dst_i is None:
            continue
        out[int(dst_i), :] = np.asarray(merged_ch_by_t[src_i, :], dtype=float)
    return out


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def _mean_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def _align_common_time_samples(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.ndim != 2 or bb.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got {aa.shape} and {bb.shape}")
    n = int(min(int(aa.shape[1]), int(bb.shape[1])))
    if n <= 0:
        raise ValueError(f"No common time samples between shapes {aa.shape} and {bb.shape}")
    return np.asarray(aa[:, :n], dtype=float), np.asarray(bb[:, :n], dtype=float), n


def _run_av(
    *,
    template_ch_by_t: np.ndarray,
    locs_xy: np.ndarray,
    sampling_hz: float,
    runtime_config: Path,
) -> dict[str, Any]:
    bundle = load_pipeline_runtime_bundle(config_path=str(runtime_config))
    stage_cfg = parse_reconstruction_stage_config(
        runtime_config=bundle.runtime_config,
        unit_id_override=94,
        force_restart_override=True,
        force_replot_override=False,
    )
    av = import_axon_velocity(repo_root=None)
    try:
        gtr = compute_graph_tracking(
            av=av,
            template_ch_by_t=np.asarray(template_ch_by_t, dtype=float),
            locs_xy=np.asarray(locs_xy, dtype=float),
            sampling_frequency_hz=float(sampling_hz),
            params=dict(stage_cfg.axon_velocity_params),
        )
        branches = getattr(gtr, "branches", None)
        n_branches = int(len(branches)) if isinstance(branches, (list, tuple)) else 0
        return {
            "status": "ok",
            "n_branches": n_branches,
            "error": None,
            "params": dict(stage_cfg.axon_velocity_params),
        }
    except Exception as exc:
        return {
            "status": "error",
            "n_branches": 0,
            "error": str(exc),
            "params": dict(stage_cfg.axon_velocity_params),
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Investigate unit-94 legacy full vs v2 merged-derived full behavior with AV checks"
    )
    parser.add_argument("--legacy-full-template", type=Path, default=DEFAULT_LEGACY_FULL_TEMPLATE)
    parser.add_argument("--legacy-full-locs", type=Path, default=DEFAULT_LEGACY_FULL_LOCS)
    parser.add_argument("--legacy-full-meta", type=Path, default=DEFAULT_LEGACY_FULL_META)
    parser.add_argument("--v2-merged-template", type=Path, default=DEFAULT_V2_MERGED_TEMPLATE)
    parser.add_argument("--v2-merged-locs", type=Path, default=DEFAULT_V2_MERGED_LOCS)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--write-generated", action="store_true", default=True)
    parser.add_argument("--generated-template", type=Path, default=DEFAULT_OUTPUT_TEMPLATE)
    parser.add_argument("--generated-locs", type=Path, default=DEFAULT_OUTPUT_LOCS)
    args = parser.parse_args()

    for required in (
        args.legacy_full_template,
        args.legacy_full_locs,
        args.v2_merged_template,
        args.v2_merged_locs,
        args.runtime_config,
    ):
        if not Path(required).exists():
            raise FileNotFoundError(f"Missing required input: {required}")

    legacy_locs_xy = np.asarray(np.load(args.legacy_full_locs)[:, :2], dtype=float)
    legacy_template_ch_by_t = _to_ch_by_t(np.load(args.legacy_full_template), n_channels=int(legacy_locs_xy.shape[0]))

    v2_merged_locs_xy = np.asarray(np.load(args.v2_merged_locs)[:, :2], dtype=float)
    v2_merged_ch_by_t = _to_ch_by_t(np.load(args.v2_merged_template), n_channels=int(v2_merged_locs_xy.shape[0]))

    generated_full_ch_by_t = _project_to_target_grid(
        merged_ch_by_t=v2_merged_ch_by_t,
        merged_locs_xy=v2_merged_locs_xy,
        target_locs_xy=legacy_locs_xy,
    )

    if bool(args.write_generated):
        args.generated_template.parent.mkdir(parents=True, exist_ok=True)
        args.generated_locs.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.generated_template, generated_full_ch_by_t)
        np.save(args.generated_locs, legacy_locs_xy)

    legacy_idx = _loc_index(legacy_locs_xy)
    v2_idx = _loc_index(v2_merged_locs_xy)
    common_keys = sorted(set(legacy_idx.keys()).intersection(set(v2_idx.keys())))

    merged_vs_generated_max_abs = None
    merged_vs_generated_mean_abs = None
    if common_keys:
        merged_rows = np.asarray([v2_merged_ch_by_t[v2_idx[k], :] for k in common_keys], dtype=float)
        generated_rows = np.asarray([generated_full_ch_by_t[legacy_idx[k], :] for k in common_keys], dtype=float)
        merged_vs_generated_max_abs = _max_abs(merged_rows, generated_rows)
        merged_vs_generated_mean_abs = _mean_abs(merged_rows, generated_rows)

    legacy_aligned, generated_aligned, n_common_samples = _align_common_time_samples(
        legacy_template_ch_by_t,
        generated_full_ch_by_t,
    )
    legacy_vs_generated_max_abs = _max_abs(legacy_aligned, generated_aligned)
    legacy_vs_generated_mean_abs = _mean_abs(legacy_aligned, generated_aligned)

    legacy_fs_hz = _sampling_hz_from_meta(args.legacy_full_meta)
    legacy_av = _run_av(
        template_ch_by_t=legacy_template_ch_by_t,
        locs_xy=legacy_locs_xy,
        sampling_hz=float(legacy_fs_hz),
        runtime_config=args.runtime_config,
    )
    generated_av = _run_av(
        template_ch_by_t=generated_full_ch_by_t,
        locs_xy=legacy_locs_xy,
        sampling_hz=float(legacy_fs_hz),
        runtime_config=args.runtime_config,
    )

    generated_nonzero_mask = np.any(np.abs(generated_full_ch_by_t) > 1e-12, axis=1)
    legacy_nonzero_mask = np.any(np.abs(legacy_template_ch_by_t) > 1e-12, axis=1)

    payload: dict[str, Any] = {
        "paths": {
            "legacy_full_template": str(args.legacy_full_template),
            "legacy_full_locs": str(args.legacy_full_locs),
            "legacy_full_meta": str(args.legacy_full_meta),
            "v2_merged_template": str(args.v2_merged_template),
            "v2_merged_locs": str(args.v2_merged_locs),
            "runtime_config": str(args.runtime_config),
            "generated_template": str(args.generated_template) if bool(args.write_generated) else None,
            "generated_locs": str(args.generated_locs) if bool(args.write_generated) else None,
        },
        "shapes": {
            "legacy_full_ch_by_t": list(legacy_template_ch_by_t.shape),
            "legacy_full_locs": list(legacy_locs_xy.shape),
            "v2_merged_ch_by_t": list(v2_merged_ch_by_t.shape),
            "v2_merged_locs": list(v2_merged_locs_xy.shape),
            "v2_generated_full_ch_by_t": list(generated_full_ch_by_t.shape),
            "v2_generated_full_locs": list(legacy_locs_xy.shape),
        },
        "parity": {
            "legacy_vs_generated_max_abs_diff": legacy_vs_generated_max_abs,
            "legacy_vs_generated_mean_abs_diff": legacy_vs_generated_mean_abs,
            "legacy_samples": int(legacy_template_ch_by_t.shape[1]),
            "generated_samples": int(generated_full_ch_by_t.shape[1]),
            "common_samples_used_for_legacy_vs_generated": int(n_common_samples),
            "merged_vs_generated_common_locations": int(len(common_keys)),
            "merged_vs_generated_max_abs_diff": merged_vs_generated_max_abs,
            "merged_vs_generated_mean_abs_diff": merged_vs_generated_mean_abs,
            "generated_nonzero_channels": int(np.sum(generated_nonzero_mask)),
            "legacy_nonzero_channels": int(np.sum(legacy_nonzero_mask)),
            "nonzero_overlap": int(np.sum(generated_nonzero_mask & legacy_nonzero_mask)),
            "generated_channels": int(generated_full_ch_by_t.shape[0]),
            "generated_is_full_grid_26400": bool(int(generated_full_ch_by_t.shape[0]) == 26400),
        },
        "av": {
            "sampling_frequency_hz": float(legacy_fs_hz),
            "legacy_full": legacy_av,
            "v2_generated_full": generated_av,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("legacy_full shape:", tuple(legacy_template_ch_by_t.shape))
    print("v2_merged shape:", tuple(v2_merged_ch_by_t.shape))
    print("v2_generated_full shape:", tuple(generated_full_ch_by_t.shape))
    print("legacy AV:", legacy_av["status"], "n_branches=", legacy_av["n_branches"])
    print("v2_generated AV:", generated_av["status"], "n_branches=", generated_av["n_branches"])
    print("report:", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())