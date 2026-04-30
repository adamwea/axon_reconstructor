from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from axon_recon.pipeline.config import load_pipeline_runtime_bundle
from axon_recon.pipeline.stages.reconstruct.config import parse_reconstruction_stage_config
from axon_recon.pipeline.stages.reconstruct.integrations.axon_velocity import compute_graph_tracking
from axon_recon.pipeline.stages.reconstruct.integrations.axon_velocity import import_axon_velocity


DEFAULT_MERGED_TEMPLATE = Path(
    "/home/adamm/dev/symlinks/local_RBS_data/outputs/Media_Density_T5_02182026_AR/260312/"
    "M08073/AxonTracking/000146/well001/template_outputs/units/0094/merged_template.npy"
)
DEFAULT_MERGED_LOCS = Path(
    "/home/adamm/dev/symlinks/local_RBS_data/outputs/Media_Density_T5_02182026_AR/260312/"
    "M08073/AxonTracking/000146/well001/template_outputs/templates/merged/unit_94/merged_contributing_channel_locations.npy"
)
DEFAULT_TARGET_FULL_LOCS = Path(
    "/home/adamm/dev/symlinks/local_RBS_data/outputs/Media_Density_T5_02182026_AR/260312/"
    "M08073/AxonTracking/000146/well001/stg4_templates_outputs/templates/full/unit_94/full_channel_locations_xy.npy"
)
DEFAULT_TARGET_META = Path(
    "/home/adamm/dev/symlinks/local_RBS_data/outputs/Media_Density_T5_02182026_AR/260312/"
    "M08073/AxonTracking/000146/well001/stg4_templates_outputs/templates/full/unit_94/full_template_meta.json"
)
DEFAULT_RUNTIME_CONFIG = REPO_ROOT / "tools" / "debug" / "debug.runtime.yml"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "tools" / "debug" / "outputs" / "unit94_reconstruct_current_full_from_merged.json"
DEFAULT_OUTPUT_PNG = REPO_ROOT / "tools" / "debug" / "outputs" / "unit94_reconstruct_current_full_from_merged_native.png"


def _as_dict(obj: Any) -> dict[str, Any]:
    return obj if isinstance(obj, dict) else {}


def _sampling_hz_from_meta(meta_path: Path) -> float:
    if not meta_path.exists():
        return 10_000.0
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return 10_000.0
    for key in ("sampling_frequency_hz", "effective_sampling_rate_hz", "native_sampling_frequency_hz"):
        if data.get(key) is not None:
            return float(data.get(key))
    return 10_000.0


def _to_ch_by_t(template_any: np.ndarray, n_channels: int) -> np.ndarray:
    arr = np.asarray(template_any, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Template must be 2D, got {arr.shape}")
    if int(arr.shape[0]) == int(n_channels):
        return arr
    if int(arr.shape[1]) == int(n_channels):
        return np.asarray(arr.T, dtype=float)
    raise ValueError(f"Cannot orient template {arr.shape} to n_channels={n_channels}")


def _loc_key(xy: np.ndarray) -> tuple[float, float]:
    return (float(np.round(float(xy[0]), 6)), float(np.round(float(xy[1]), 6)))


def _loc_index(locs_xy: np.ndarray) -> dict[tuple[float, float], int]:
    out: dict[tuple[float, float], int] = {}
    for i in range(int(locs_xy.shape[0])):
        k = _loc_key(locs_xy[i, :2])
        if k not in out:
            out[k] = int(i)
    return out


def _project_to_full_grid(merged_ch_by_t: np.ndarray, merged_locs_xy: np.ndarray, full_locs_xy: np.ndarray) -> np.ndarray:
    out = np.zeros((int(full_locs_xy.shape[0]), int(merged_ch_by_t.shape[1])), dtype=float)
    full_index = _loc_index(full_locs_xy)
    for i in range(min(int(merged_ch_by_t.shape[0]), int(merged_locs_xy.shape[0]))):
        dst = full_index.get(_loc_key(merged_locs_xy[i, :2]))
        if dst is None:
            continue
        out[int(dst), :] = merged_ch_by_t[i, :]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct current full_from_merged template for one unit and run AV graph tracking"
    )
    parser.add_argument("--unit-id", type=int, default=94)
    parser.add_argument("--merged-template", type=Path, default=DEFAULT_MERGED_TEMPLATE)
    parser.add_argument("--merged-locs", type=Path, default=DEFAULT_MERGED_LOCS)
    parser.add_argument("--target-full-locs", type=Path, default=DEFAULT_TARGET_FULL_LOCS)
    parser.add_argument("--target-meta", type=Path, default=DEFAULT_TARGET_META)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    parser.add_argument(
        "--plot-mode",
        type=str,
        default="clean",
        choices=("clean", "raw"),
        help="Use axon_velocity native clean or raw branch plot",
    )
    args = parser.parse_args()

    if not args.merged_template.exists():
        raise FileNotFoundError(f"Missing merged template: {args.merged_template}")
    if not args.merged_locs.exists():
        raise FileNotFoundError(f"Missing merged locations: {args.merged_locs}")
    if not args.target_full_locs.exists():
        raise FileNotFoundError(f"Missing target full locations: {args.target_full_locs}")
    if not args.runtime_config.exists():
        raise FileNotFoundError(f"Missing runtime config: {args.runtime_config}")

    bundle = load_pipeline_runtime_bundle(config_path=str(args.runtime_config))
    stage_cfg = parse_reconstruction_stage_config(
        runtime_config=bundle.runtime_config,
        unit_id_override=int(args.unit_id),
        force_restart_override=True,
        force_replot_override=False,
    )

    merged_locs_xy = np.asarray(np.load(args.merged_locs)[:, :2], dtype=float)
    merged_ch_by_t = _to_ch_by_t(np.load(args.merged_template), n_channels=int(merged_locs_xy.shape[0]))
    gtr_locs_xy = np.asarray(np.load(args.target_full_locs)[:, :2], dtype=float)
    gtr_template_ch_by_t = _project_to_full_grid(merged_ch_by_t, merged_locs_xy, gtr_locs_xy)
    fs_hz = _sampling_hz_from_meta(args.target_meta)
    selected_source = "current_full_from_merged"

    av = import_axon_velocity(repo_root=None)

    payload: dict[str, Any] = {
        "unit_id": int(args.unit_id),
        "merged_template": str(args.merged_template),
        "merged_locs": str(args.merged_locs),
        "target_full_locs": str(args.target_full_locs),
        "target_meta": str(args.target_meta),
        "runtime_config": str(args.runtime_config),
        "selected_source": str(selected_source),
        "merged_shape_ch_by_t": list(np.asarray(merged_ch_by_t).shape),
        "merged_locs_shape": list(np.asarray(merged_locs_xy).shape),
        "template_shape_ch_by_t": list(np.asarray(gtr_template_ch_by_t).shape),
        "locs_shape": list(np.asarray(gtr_locs_xy).shape),
        "sampling_frequency_hz": float(fs_hz),
        "av_params": dict(stage_cfg.axon_velocity_params),
        "output_png": str(args.output_png),
        "plot_mode": str(args.plot_mode),
    }

    try:
        gtr = compute_graph_tracking(
            av=av,
            template_ch_by_t=np.asarray(gtr_template_ch_by_t, dtype=float),
            locs_xy=np.asarray(gtr_locs_xy, dtype=float),
            sampling_frequency_hz=float(fs_hz),
            params=dict(stage_cfg.axon_velocity_params),
        )
        branches = getattr(gtr, "branches", None)
        payload["status"] = "ok"
        payload["n_branches"] = int(len(branches)) if isinstance(branches, (list, tuple)) else 0
        payload["first_branch_keys"] = (
            sorted(list(_as_dict(branches[0]).keys())) if isinstance(branches, list) and branches else []
        )

        plot_fn_name = "plot_clean_branches" if str(args.plot_mode) == "clean" else "plot_raw_branches"
        plot_fn = getattr(gtr, plot_fn_name, None)
        if callable(plot_fn):
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            _ = plot_fn(plot_full_template=True, ax=ax)
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            args.output_png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.output_png, dpi=200, bbox_inches="tight")
            plt.close(fig)
            payload["plot_status"] = "ok"
            payload["plot_function"] = plot_fn_name
        else:
            payload["plot_status"] = "error"
            payload["plot_function"] = plot_fn_name
            payload["plot_error"] = f"Graph tracking object missing callable {plot_fn_name}"

        print("status=ok")
        print(f"selected_source={selected_source}")
        print(f"template_shape_ch_by_t={tuple(np.asarray(gtr_template_ch_by_t).shape)}")
        print(f"locs_shape={tuple(np.asarray(gtr_locs_xy).shape)}")
        print(f"n_branches={payload['n_branches']}")
        print(f"plot_status={payload.get('plot_status')}")
        print(f"plot_png={args.output_png}")
    except Exception as exc:
        payload["status"] = "error"
        payload["n_branches"] = 0
        payload["error"] = str(exc)
        print("status=error")
        print(f"selected_source={selected_source}")
        print(f"template_shape_ch_by_t={tuple(np.asarray(gtr_template_ch_by_t).shape)}")
        print(f"locs_shape={tuple(np.asarray(gtr_locs_xy).shape)}")
        print(f"error={exc}")
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 1

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote={args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())