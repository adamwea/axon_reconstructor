from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

LOCAL_SPIKEINTERFACE_SRC = Path("/home/adamm/dev/pkgs/spikeinterface/src")
LOCAL_SLAY_ROOT = Path("/home/adamm/dev/pkgs/SLAy")
for candidate in (LOCAL_SPIKEINTERFACE_SRC, LOCAL_SLAY_ROOT / "src", LOCAL_SLAY_ROOT):
    if candidate.exists():
        candidate_text = str(candidate)
        if candidate_text not in sys.path:
            sys.path.insert(0, candidate_text)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import spikeinterface.full as si


DEFAULT_WELL_OUTPUT_ROOT = Path(
    "/home/adamm/scratch/axon_recon_scratch/outputs/Media_Density_T3_07012025_AR/250728/M07137/AxonTracking/000225/well001"
)

DEFAULT_SORTER_SOURCE = Path(
    DEFAULT_WELL_OUTPUT_ROOT / "spikesort_outputs/sorter_output"
)
DEFAULT_MODEL_PATH = Path(
    DEFAULT_WELL_OUTPUT_ROOT / "spikesort_outputs/merge_output/cache/slay_model/ae.pt"
)
DEFAULT_OUT_ROOT = Path(
    "/home/adamm/dev/pkgs/axon_reconstructor/tools/debug/outputs/original_slay_upstream"
)
DEFAULT_ALLOWED_LABELS = ("good", "mua")


def _install_numpy_cupy_fallback_module() -> None:
    shim = ModuleType("cupy")
    shim.array = np.array  # type: ignore[attr-defined]
    shim.asarray = np.asarray  # type: ignore[attr-defined]
    shim.asnumpy = np.asarray  # type: ignore[attr-defined]
    shim.mean = np.mean  # type: ignore[attr-defined]
    shim.zeros = np.zeros  # type: ignore[attr-defined]
    shim.float32 = np.float32  # type: ignore[attr-defined]
    shim.ndarray = np.ndarray  # type: ignore[attr-defined]
    sys.modules["cupy"] = shim


def _import_slay_run_function() -> Callable[[dict[str, Any]], None]:
    import importlib

    def _patch_parse_kilosort_params(module: Any) -> None:
        original = getattr(module, "parse_kilosort_params", None)
        if not callable(original):
            return

        def _patched(args: dict[str, Any]) -> dict[str, Any]:
            import os

            ks_folder = str(args.get("KS_folder", "")).strip()
            if not ks_folder:
                return original(args)

            ksparam_path = os.path.join(ks_folder, "params.py")
            ksparams: dict[str, Any] = {}
            with open(ksparam_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    ksparams[str(key).strip()] = eval(str(value).strip())

            dat_path = ksparams.pop("dat_path", None)
            if isinstance(dat_path, (list, tuple)):
                dat_path = (dat_path[0] if len(dat_path) > 0 else None)
            if dat_path is not None:
                dat_path_s = str(dat_path)
                if os.path.isabs(dat_path_s):
                    ksparams["data_filepath"] = dat_path_s
                else:
                    ksparams["data_filepath"] = os.path.join(ks_folder, dat_path_s)
            if "n_channels_dat" in ksparams:
                ksparams["n_chan"] = ksparams.pop("n_channels_dat")
            args.update(ksparams)
            return args

        setattr(module, "parse_kilosort_params", _patched)

    try:
        module = importlib.import_module("slay.run")
        _patch_parse_kilosort_params(module)
        run_slay = getattr(module, "run_slay", None)
        if callable(run_slay):
            return run_slay
        raise RuntimeError("slay.run.run_slay_not_callable")
    except ModuleNotFoundError as exc:
        if str(getattr(exc, "name", "")) != "cupy":
            raise
        _install_numpy_cupy_fallback_module()
        module = importlib.import_module("slay.run")
        _patch_parse_kilosort_params(module)
        run_slay = getattr(module, "run_slay", None)
        if callable(run_slay):
            return run_slay
        raise RuntimeError("slay.run.run_slay_not_callable_after_numpy_fallback")


def _resolve_sorting_data_dir(sorter_dir: Path) -> Path:
    sorter_dir = Path(sorter_dir).resolve()
    nested = (sorter_dir / "sorter_output").resolve()
    if nested.exists() and (nested / "spike_clusters.npy").exists() and (nested / "spike_times.npy").exists():
        return nested
    return sorter_dir


def _normalize_slay_kilosort_dir(sorter_dir: Path) -> Path:
    sorter_dir = Path(sorter_dir).resolve()
    if (sorter_dir / "params.py").exists():
        return sorter_dir
    nested = (sorter_dir / "sorter_output").resolve()
    if (nested / "params.py").exists():
        return nested
    raise FileNotFoundError(f"missing_params_py_in_sorter_dir:{sorter_dir}")


def _copy_sorter_to_cache(*, source_dir: Path, cache_dir: Path) -> Path:
    source_dir = Path(source_dir).resolve()
    cache_dir = Path(cache_dir).resolve()
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, cache_dir)
    return cache_dir


def _find_recording_json(sorter_dir: Path) -> Path:
    sorter_dir = Path(sorter_dir).resolve()
    candidates = [
        sorter_dir / "spikeinterface_recording.json",
        sorter_dir.parent / "spikeinterface_recording.json",
        sorter_dir.parent.parent / "spikeinterface_recording.json",
    ]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing_recording_json_near:{sorter_dir}")


def _load_recording(sorter_dir: Path) -> Any:
    recording_json = _find_recording_json(sorter_dir)
    load_fn = getattr(si, "load", None)
    if callable(load_fn):
        return load_fn(str(recording_json))
    return si.load_extractor(recording_json)


def _read_labels_from_sorter(sorter_dir: Path) -> dict[str, str]:
    import csv

    sorting_data_dir = _resolve_sorting_data_dir(sorter_dir)
    for file_name in ("cluster_group.tsv", "cluster_KSLabel.tsv"):
        path = (sorting_data_dir / file_name).resolve()
        if not path.exists():
            continue
        labels: dict[str, str] = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                cluster_id = str((row.get("cluster_id") or row.get("cluster_id\ufeff") or "")).strip()
                label = str((row.get("KSLabel") or row.get("group") or "")).strip().lower()
                if cluster_id:
                    labels[cluster_id] = label
        if labels:
            return labels
    return {}


def _load_sorting(sorter_dir: Path, recording: Any, allowed_labels: set[str] | None) -> tuple[Any, dict[str, str]]:
    sorting_data_dir = _resolve_sorting_data_dir(sorter_dir)
    spike_times = np.asarray(np.load(sorting_data_dir / "spike_times.npy"), dtype=np.int64).reshape(-1)
    spike_labels = np.asarray(np.load(sorting_data_dir / "spike_clusters.npy"), dtype=np.int64).reshape(-1)
    if spike_times.shape[0] != spike_labels.shape[0]:
        raise RuntimeError(f"spike_times_labels_length_mismatch:{spike_times.shape[0]}:{spike_labels.shape[0]}")

    unit_ids = np.unique(spike_labels)
    sampling_frequency = float(getattr(recording, "sampling_frequency", None) or recording.get_sampling_frequency())
    from_samples_and_labels = getattr(si.NumpySorting, "from_samples_and_labels", None)
    if callable(from_samples_and_labels):
        sorting = from_samples_and_labels(
            samples_list=[spike_times],
            labels_list=[spike_labels],
            sampling_frequency=sampling_frequency,
            unit_ids=unit_ids,
        )
    else:
        sorting = si.NumpySorting.from_times_labels(
            times_list=[spike_times],
            labels_list=[spike_labels],
            sampling_frequency=sampling_frequency,
            unit_ids=unit_ids,
        )

    labels_by_unit = _read_labels_from_sorter(sorter_dir)
    if labels_by_unit and allowed_labels:
        selected_unit_ids: list[Any] = []
        for uid in list(sorting.unit_ids):
            if labels_by_unit.get(str(uid), "") in allowed_labels:
                selected_unit_ids.append(uid)
        if selected_unit_ids:
            sorting = sorting.select_units(unit_ids=selected_unit_ids)
            labels_by_unit = {str(uid): labels_by_unit.get(str(uid), "") for uid in selected_unit_ids}

    return sorting, labels_by_unit


def _compute_extension(analyzer: Any, extension_name: str, **kwargs: Any) -> None:
    compute_extension = getattr(analyzer, "compute", None)
    if not callable(compute_extension):
        raise RuntimeError("analyzer_compute_unavailable")

    try:
        compute_extension(extension_name, **kwargs)
        return
    except Exception:
        pass

    compute_extension([extension_name], **kwargs)


def _compute_required_extensions(analyzer: Any, extension_names: list[str]) -> None:
    has_extension = getattr(analyzer, "has_extension", None)

    pending: list[str] = []
    for extension_name in extension_names:
        already = False
        if callable(has_extension):
            try:
                already = bool(has_extension(extension_name))
            except Exception:
                already = False
        if not already:
            pending.append(str(extension_name))

    if "random_spikes" in pending:
        _compute_extension(analyzer, "random_spikes", method="all")
        pending = [name for name in pending if name != "random_spikes"]

    for extension_name in pending:
        _compute_extension(analyzer, extension_name)


def _build_analyzer(*, sorting: Any, recording: Any, analyzer_dir: Path) -> Any:
    analyzer_dir = Path(analyzer_dir).resolve()
    if analyzer_dir.exists():
        shutil.rmtree(analyzer_dir, ignore_errors=True)
    analyzer = si.create_sorting_analyzer(
        sorting=sorting,
        recording=recording,
        format="binary_folder",
        folder=analyzer_dir,
        sparse=False,
        overwrite=True,
    )
    return analyzer


def _ensure_templates_extension(analyzer: Any) -> Any:
    if not analyzer.has_extension("templates"):
        _compute_required_extensions(analyzer, ["random_spikes", "waveforms", "templates"])
    return analyzer.get_extension("templates")


def _unit_id_map(analyzer: Any) -> dict[str, Any]:
    return {str(uid): uid for uid in list(analyzer.sorting.unit_ids)}


def _resolve_sparse_channel_indices(analyzer: Any, canonical_uid: Any) -> np.ndarray | None:
    sparsity = getattr(analyzer, "sparsity", None)
    if sparsity is None:
        return None

    mapping = getattr(sparsity, "unit_id_to_channel_indices", None)
    if callable(mapping):
        try:
            idx = mapping(canonical_uid)
            if idx is not None:
                return np.asarray(idx, dtype=int)
        except Exception:
            pass
    elif isinstance(mapping, dict):
        for key in (canonical_uid, str(canonical_uid)):
            if key in mapping:
                try:
                    return np.asarray(mapping[key], dtype=int)
                except Exception:
                    pass

    for name in ("get_channel_indices", "get_channel_indices_for_unit"):
        fn = getattr(sparsity, name, None)
        if callable(fn):
            try:
                idx = fn(canonical_uid)
                if idx is not None:
                    return np.asarray(idx, dtype=int)
            except Exception:
                pass

    return None


def _load_template_and_locs(
    analyzer: Any,
    templates_ext: Any,
    uid_text: str,
    uid_lookup: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    canonical_uid = uid_lookup.get(str(uid_text), None)
    if canonical_uid is None:
        return None, None, "unit_not_found"

    try:
        template = np.asarray(templates_ext.get_unit_template(unit_id=canonical_uid), dtype=float)
    except Exception as exc:
        return None, None, f"template_load_failed:{type(exc).__name__}:{exc}"

    if template.ndim != 2:
        return None, None, "template_not_2d"

    try:
        locs = np.asarray(analyzer.recording.get_channel_locations(), dtype=float)
    except Exception as exc:
        return None, None, f"channel_locations_failed:{type(exc).__name__}:{exc}"

    if locs.ndim != 2 or locs.shape[1] < 2:
        return None, None, "locations_not_2d"

    if template.shape[0] != locs.shape[0] and template.shape[1] == locs.shape[0]:
        template = template.T

    if template.shape[0] != locs.shape[0]:
        idx = _resolve_sparse_channel_indices(analyzer, canonical_uid)
        if idx is not None and idx.ndim == 1 and template.shape[0] == idx.shape[0]:
            try:
                locs = locs[idx, :]
            except Exception:
                pass

    if template.shape[0] != locs.shape[0]:
        return None, None, "template_channel_count_mismatch"

    return template, locs[:, :2], None


def _chip_extent_from_analyzers(analyzers: list[Any]) -> tuple[float, float, float, float] | None:
    x_vals: list[float] = []
    y_vals: list[float] = []
    for analyzer in analyzers:
        recording = getattr(analyzer, "recording", None)
        if recording is None:
            continue
        get_channel_locations = getattr(recording, "get_channel_locations", None)
        if not callable(get_channel_locations):
            continue
        try:
            locs = np.asarray(get_channel_locations(), dtype=float)
        except Exception:
            continue
        if locs.ndim != 2 or locs.shape[1] < 2:
            continue
        x = np.asarray(locs[:, 0], dtype=float)
        y = np.asarray(locs[:, 1], dtype=float)
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        if x.size == 0 or y.size == 0:
            continue
        x_vals.extend([float(np.min(x)), float(np.max(x))])
        y_vals.extend([float(np.min(y)), float(np.max(y))])

    if not x_vals or not y_vals:
        return None

    x_min = float(min(x_vals))
    x_max = float(max(x_vals))
    y_min = float(min(y_vals))
    y_max = float(max(y_vals))
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    return x_min, x_max, y_min, y_max


def _plot_amp_map_log(
    ax: Any,
    template: np.ndarray,
    locs: np.ndarray,
    title: str,
    chip_extent: tuple[float, float, float, float] | None,
    marker_size: float = 2.0,
    eps: float = 1e-3,
) -> None:
    amp = np.ptp(np.asarray(template, dtype=float), axis=1)
    amp = np.asarray(amp, dtype=float)
    valid_mask = np.isfinite(amp) & (amp > 0.0)
    valid = amp[valid_mask]
    if valid.size == 0:
        raise ValueError("nonpositive_amplitude")

    locs_valid = np.asarray(locs, dtype=float)[valid_mask, :]
    amp_valid = np.asarray(valid, dtype=float)
    vmin = max(float(np.min(valid)), float(eps))
    vmax = max(float(np.max(valid)), vmin * (1.0 + 1e-6))

    ax.scatter(
        locs_valid[:, 0],
        locs_valid[:, 1],
        c=np.clip(amp_valid, vmin, None),
        s=float(marker_size),
        cmap="viridis",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax.set_title(str(title))
    ax.set_xlabel("x_um")
    ax.set_ylabel("y_um")
    if chip_extent is not None:
        x_min, x_max, y_min, y_max = chip_extent
        ax.set_xlim(float(x_min), float(x_max))
        ax.set_ylim(float(y_min), float(y_max))
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)


def _safe_token(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return "unknown"
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in text).strip("_") or "unknown"


def _write_json(path: Path, payload: Any) -> None:
    def _jsonable(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [_jsonable(item) for item in value]
        if isinstance(value, dict):
            return {str(key): _jsonable(val) for key, val in value.items()}
        try:
            return value.item()
        except Exception:
            return str(value)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def _read_merge_groups_from_slay(ks_dir: Path) -> dict[str, list[str]]:
    new2old_path = (Path(ks_dir).resolve() / "automerge" / "new2old.json").resolve()
    if not new2old_path.exists():
        return {}
    payload = json.loads(new2old_path.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    if isinstance(payload, dict):
        for raw_new_id, raw_members in payload.items():
            members = [str(member) for member in list(raw_members or [])]
            out[str(raw_new_id)] = members
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy the original sorter to debug cache, run the local SLAy repo, and plot pre/post templates."
    )
    parser.add_argument("--sorter-source", type=Path, default=DEFAULT_SORTER_SOURCE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--model-path", type=Path, default=(DEFAULT_MODEL_PATH if DEFAULT_MODEL_PATH.exists() else None))
    parser.add_argument("--min-spikes", type=int, default=50)
    parser.add_argument("--max-spikes", type=int, default=-1)
    parser.add_argument("--good-labels", nargs="+", default=list(DEFAULT_ALLOWED_LABELS))
    parser.add_argument("--window-size", type=float, default=0.1)
    parser.add_argument("--xcorr-bin-width", type=float, default=0.0005)
    parser.add_argument("--overlap-tol", type=float, default=5.0 / 30000.0)
    parser.add_argument("--min-xcorr-rate", type=float, default=800.0)
    parser.add_argument("--xcorr-coeff", type=float, default=0.25)
    parser.add_argument("--ref-pen-bin-width", type=float, default=1.0)
    parser.add_argument("--max-viol", type=float, default=0.15)
    parser.add_argument("--ref-pen-coeff", type=float, default=1.0)
    parser.add_argument("--sim-thresh", type=float, default=0.4)
    parser.add_argument("--ae-pre", type=int, default=10)
    parser.add_argument("--ae-post", type=int, default=30)
    parser.add_argument("--ae-chan", type=int, default=8)
    parser.add_argument("--ae-noise", action="store_true", default=True)
    parser.add_argument("--no-ae-noise", action="store_false", dest="ae_noise")
    parser.add_argument("--ae-shft", action="store_true", default=False)
    parser.add_argument("--ae-epochs", type=int, default=50)
    parser.add_argument("--final-thresh", type=float, default=0.5)
    parser.add_argument("--max-dist", type=int, default=10)
    parser.add_argument("--auto-accept-merges", action="store_true", default=True)
    parser.add_argument("--plot-merges", action="store_true", default=True)
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    cache_root = (out_root / "cache").resolve()
    sorter_cache_dir = (cache_root / "original_sorter_output").resolve()
    pre_analyzer_dir = (cache_root / "pre_analyzer_output").resolve()
    post_analyzer_dir = (cache_root / "post_analyzer_output").resolve()
    plots_dir = (out_root / "plots").resolve()
    run_output_json = (out_root / "run-output.json").resolve()
    summary_json = (out_root / "original_slay_summary.json").resolve()

    copied_sorter_dir = _copy_sorter_to_cache(source_dir=args.sorter_source, cache_dir=sorter_cache_dir)
    ks_dir = _normalize_slay_kilosort_dir(copied_sorter_dir)
    print(f"copied_sorter_cache={copied_sorter_dir}")
    print(f"ks_dir={ks_dir}")

    allowed_labels = {str(label).strip().lower() for label in list(args.good_labels or [])}
    pre_recording = _load_recording(copied_sorter_dir)
    pre_sorting, pre_labels_by_unit = _load_sorting(copied_sorter_dir, pre_recording, allowed_labels)
    print(f"pre_selected_unit_count={len(pre_sorting.unit_ids)}")

    pre_analyzer = _build_analyzer(sorting=pre_sorting, recording=pre_recording, analyzer_dir=pre_analyzer_dir)
    _compute_required_extensions(pre_analyzer, ["random_spikes", "waveforms", "templates"])
    pre_templates = _ensure_templates_extension(pre_analyzer)
    pre_uid_lookup = _unit_id_map(pre_analyzer)

    run_slay = _import_slay_run_function()
    run_args: dict[str, Any] = {
        "KS_folder": str(ks_dir),
        "output_json": str(run_output_json),
        "min_spikes": int(args.min_spikes),
        "max_spikes": int(args.max_spikes),
        "good_lbls": sorted(list(allowed_labels)),
        "window_size": float(args.window_size),
        "xcorr_bin_width": float(args.xcorr_bin_width),
        "overlap_tol": float(args.overlap_tol),
        "min_xcorr_rate": float(args.min_xcorr_rate),
        "xcorr_coeff": float(args.xcorr_coeff),
        "ref_pen_bin_width": float(args.ref_pen_bin_width),
        "max_viol": float(args.max_viol),
        "ref_pen_coeff": float(args.ref_pen_coeff),
        "sim_thresh": float(args.sim_thresh),
        "ae_pre": int(args.ae_pre),
        "ae_post": int(args.ae_post),
        "ae_chan": int(args.ae_chan),
        "ae_noise": bool(args.ae_noise),
        "ae_shft": bool(args.ae_shft),
        "ae_epochs": int(args.ae_epochs),
        "final_thresh": float(args.final_thresh),
        "max_dist": int(args.max_dist),
        "auto_accept_merges": bool(args.auto_accept_merges),
        "plot_merges": bool(args.plot_merges),
    }
    if args.model_path is not None:
        run_args["model_path"] = str(Path(args.model_path).resolve())

    run_slay(run_args)
    merge_groups = _read_merge_groups_from_slay(ks_dir)
    print(f"slay_merge_group_count={len(merge_groups)}")

    if not bool(args.auto_accept_merges):
        raise RuntimeError("post_templates_require_auto_accept_merges_true")

    post_recording = _load_recording(copied_sorter_dir)
    post_sorting, post_labels_by_unit = _load_sorting(copied_sorter_dir, post_recording, None)
    print(f"post_unit_count={len(post_sorting.unit_ids)}")
    post_analyzer = _build_analyzer(sorting=post_sorting, recording=post_recording, analyzer_dir=post_analyzer_dir)
    _compute_required_extensions(post_analyzer, ["random_spikes", "waveforms", "templates"])
    post_templates = _ensure_templates_extension(post_analyzer)
    post_uid_lookup = _unit_id_map(post_analyzer)
    chip_extent = _chip_extent_from_analyzers([pre_analyzer, post_analyzer])

    plots_dir.mkdir(parents=True, exist_ok=True)
    merge_rows: list[dict[str, Any]] = []
    sorted_merge_items = sorted(merge_groups.items(), key=lambda item: int(item[0]))
    for group_index, (post_uid, pre_uids) in enumerate(sorted_merge_items, start=1):
        pre_unit_ids = [str(uid) for uid in list(pre_uids or [])]
        ncols = int(len(pre_unit_ids) + 1)
        fig, axes = plt.subplots(1, ncols, figsize=(4.0 * ncols, 4.0), constrained_layout=True)
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes])

        row: dict[str, Any] = {
            "group_index": int(group_index),
            "pre_unit_ids": list(pre_unit_ids),
            "post_unit_id": str(post_uid),
            "plot_png": None,
        }

        for ax_index, pre_uid in enumerate(pre_unit_ids):
            ax = axes[ax_index]
            tmpl, locs, err = _load_template_and_locs(pre_analyzer, pre_templates, pre_uid, pre_uid_lookup)
            if tmpl is None or locs is None:
                ax.text(0.5, 0.5, f"pre {pre_uid}\n{err}", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
                continue
            try:
                _plot_amp_map_log(ax, tmpl, locs, f"Pre {pre_uid}", chip_extent=chip_extent)
            except Exception as exc:
                ax.text(0.5, 0.5, f"pre {pre_uid}\n{type(exc).__name__}:{exc}", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")

        post_ax = axes[-1]
        post_tmpl, post_locs, post_err = _load_template_and_locs(post_analyzer, post_templates, post_uid, post_uid_lookup)
        if post_tmpl is None or post_locs is None:
            post_ax.text(0.5, 0.5, f"post {post_uid}\n{post_err}", ha="center", va="center", transform=post_ax.transAxes)
            post_ax.axis("off")
        else:
            try:
                _plot_amp_map_log(post_ax, post_tmpl, post_locs, f"Post {post_uid}", chip_extent=chip_extent)
            except Exception as exc:
                post_ax.text(0.5, 0.5, f"post {post_uid}\n{type(exc).__name__}:{exc}", ha="center", va="center", transform=post_ax.transAxes)
                post_ax.axis("off")

        out_path = (plots_dir / f"slay_group_{int(group_index):03d}__post_{_safe_token(post_uid)}.png").resolve()
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        row["plot_png"] = str(out_path)
        merge_rows.append(row)
        print(f"wrote {out_path}")

    label_counts: dict[str, int] = {}
    for label in pre_labels_by_unit.values():
        label_counts[str(label)] = int(label_counts.get(str(label), 0) + 1)

    summary_payload = {
        "spikeinterface_src": str(LOCAL_SPIKEINTERFACE_SRC),
        "slay_root": str(LOCAL_SLAY_ROOT),
        "sorter_source": str(Path(args.sorter_source).resolve()),
        "sorter_cache_dir": str(copied_sorter_dir),
        "ks_dir": str(ks_dir),
        "sorting_data_dir": str(_resolve_sorting_data_dir(copied_sorter_dir)),
        "analyzer_sparse": False,
        "template_random_spikes_method": "all",
        "allowed_labels": sorted(list(allowed_labels)),
        "pre_selected_unit_count": int(len(pre_sorting.unit_ids)),
        "pre_selected_label_counts": label_counts,
        "post_unit_count": int(len(post_sorting.unit_ids)),
        "pre_analyzer_dir": str(pre_analyzer_dir),
        "post_analyzer_dir": str(post_analyzer_dir),
        "run_output_json": str(run_output_json),
        "model_path": (str(Path(args.model_path).resolve()) if args.model_path is not None else None),
        "run_args": run_args,
        "merge_group_count": int(len(merge_rows)),
        "merge_rows": merge_rows,
        "plots_dir": str(plots_dir),
        "automerge_dir": str((ks_dir / "automerge").resolve()),
        "new2old_json": str((ks_dir / "automerge" / "new2old.json").resolve()),
        "metrics_tsv": str((ks_dir / "automerge" / "metrics.tsv").resolve()),
        "post_labels_present": int(len(post_labels_by_unit)),
    }
    _write_json(summary_json, summary_payload)
    print(f"summary_json={summary_json}")


if __name__ == "__main__":
    main()