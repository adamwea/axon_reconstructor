from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

LOCAL_SPIKEINTERFACE_SRC = Path("/home/adamm/dev/pkgs/spikeinterface/src")
if LOCAL_SPIKEINTERFACE_SRC.exists():
    local_src_text = str(LOCAL_SPIKEINTERFACE_SRC)
    if local_src_text not in sys.path:
        sys.path.insert(0, local_src_text)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import spikeinterface.full as si
from spikeinterface.curation import compute_merge_unit_groups


DEFAULT_SORTER_SOURCE = Path(
    "/home/adamm/dev/scratch_outputs/Media_Density_T3_07012025_AR/250728/M07137/AxonTracking/000225/well001/spikesort_outputs/sorter_output"
)
DEFAULT_OUT_ROOT = Path(
    "/home/adamm/dev/pkgs/axon_reconstructor/tools/debug/outputs/spikeinterface_slay_upstream"
)
DEFAULT_ALLOWED_LABELS = ("good", "mua")


def _resolve_sorting_data_dir(sorter_dir: Path) -> Path:
    sorter_dir = Path(sorter_dir).resolve()
    nested = (sorter_dir / "sorter_output").resolve()
    if nested.exists() and (nested / "spike_clusters.npy").exists() and (nested / "spike_times.npy").exists():
        return nested
    return sorter_dir


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


def _load_sorting(sorter_dir: Path, recording: Any, allowed_labels: set[str]) -> tuple[Any, dict[str, str]]:
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
    if labels_by_unit:
        selected_unit_ids: list[Any] = []
        for uid in list(sorting.unit_ids):
            label = labels_by_unit.get(str(uid), "")
            if label in allowed_labels:
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
        _compute_required_extensions(analyzer, ["random_spikes", "templates"])
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
        pass
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def _make_new_unit_ids(existing_unit_ids: list[Any], n_groups: int) -> list[int]:
    numeric_ids: list[int] = []
    for uid in existing_unit_ids:
        try:
            numeric_ids.append(int(uid))
        except Exception:
            continue
    start = (max(numeric_ids) + 1) if numeric_ids else 0
    return [int(start + idx) for idx in range(int(n_groups))]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy the original sorter to debug cache, run local SpikeInterface SLAy merges, and plot pre/post templates."
    )
    parser.add_argument("--sorter-source", type=Path, default=DEFAULT_SORTER_SOURCE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    cache_root = (out_root / "cache").resolve()
    sorter_cache_dir = (cache_root / "original_sorter_output").resolve()
    pre_analyzer_dir = (cache_root / "pre_analyzer_output").resolve()
    post_analyzer_dir = (cache_root / "post_analyzer_output").resolve()
    plots_dir = (out_root / "plots").resolve()
    summary_json = (out_root / "spikeinterface_slay_summary.json").resolve()

    print(f"spikeinterface_src={LOCAL_SPIKEINTERFACE_SRC}")
    copied_sorter_dir = _copy_sorter_to_cache(source_dir=args.sorter_source, cache_dir=sorter_cache_dir)
    print(f"copied_sorter_cache={copied_sorter_dir}")

    allowed_labels = set(str(label).strip().lower() for label in DEFAULT_ALLOWED_LABELS)
    recording = _load_recording(copied_sorter_dir)
    sorting, labels_by_unit = _load_sorting(copied_sorter_dir, recording, allowed_labels)
    print(f"selected_unit_count={len(sorting.unit_ids)}")

    pre_analyzer = _build_analyzer(sorting=sorting, recording=recording, analyzer_dir=pre_analyzer_dir)
    _compute_required_extensions(
        pre_analyzer,
        ["random_spikes", "templates", "template_similarity", "correlograms"],
    )
    pre_templates = _ensure_templates_extension(pre_analyzer)
    pre_uid_lookup = _unit_id_map(pre_analyzer)

    merge_unit_groups = compute_merge_unit_groups(
        sorting_analyzer=pre_analyzer,
        preset="slay",
        resolve_graph=True,
        compute_needed_extensions=False,
    )
    merge_unit_groups = [list(group) for group in list(merge_unit_groups or []) if len(list(group)) >= 2]
    print(f"merge_group_count={len(merge_unit_groups)}")

    new_unit_ids = _make_new_unit_ids(list(pre_analyzer.sorting.unit_ids), len(merge_unit_groups))
    if post_analyzer_dir.exists():
        shutil.rmtree(post_analyzer_dir, ignore_errors=True)

    if merge_unit_groups:
        merged_result = pre_analyzer.merge_units(
            merge_unit_groups=merge_unit_groups,
            new_unit_ids=new_unit_ids,
            return_new_unit_ids=True,
            format="binary_folder",
            folder=post_analyzer_dir,
            raise_error_if_overlap_fails=False,
        )
        post_analyzer, returned_new_unit_ids = merged_result
    else:
        returned_new_unit_ids = []
        post_analyzer = _build_analyzer(sorting=sorting, recording=recording, analyzer_dir=post_analyzer_dir)

    _compute_required_extensions(post_analyzer, ["random_spikes", "templates"])
    post_templates = _ensure_templates_extension(post_analyzer)
    post_uid_lookup = _unit_id_map(post_analyzer)
    chip_extent = _chip_extent_from_analyzers([pre_analyzer, post_analyzer])

    plots_dir.mkdir(parents=True, exist_ok=True)
    merge_rows: list[dict[str, Any]] = []
    for idx, group in enumerate(merge_unit_groups, start=1):
        pre_unit_ids = [str(uid) for uid in list(group)]
        post_unit_id = str(returned_new_unit_ids[idx - 1])
        ncols = int(len(pre_unit_ids) + 1)
        fig, axes = plt.subplots(1, ncols, figsize=(4.0 * ncols, 4.0), constrained_layout=True)
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes])

        row: dict[str, Any] = {
            "group_index": int(idx),
            "pre_unit_ids": list(pre_unit_ids),
            "post_unit_id": str(post_unit_id),
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
        post_tmpl, post_locs, post_err = _load_template_and_locs(post_analyzer, post_templates, post_unit_id, post_uid_lookup)
        if post_tmpl is None or post_locs is None:
            post_ax.text(0.5, 0.5, f"post {post_unit_id}\n{post_err}", ha="center", va="center", transform=post_ax.transAxes)
            post_ax.axis("off")
        else:
            try:
                _plot_amp_map_log(post_ax, post_tmpl, post_locs, f"Post {post_unit_id}", chip_extent=chip_extent)
            except Exception as exc:
                post_ax.text(0.5, 0.5, f"post {post_unit_id}\n{type(exc).__name__}:{exc}", ha="center", va="center", transform=post_ax.transAxes)
                post_ax.axis("off")

        out_path = (plots_dir / f"slay_group_{int(idx):03d}__post_{_safe_token(post_unit_id)}.png").resolve()
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        row["plot_png"] = str(out_path)
        merge_rows.append(row)
        print(f"wrote {out_path}")

    summary_payload = {
        "spikeinterface_src": str(LOCAL_SPIKEINTERFACE_SRC),
        "sorter_source": str(Path(args.sorter_source).resolve()),
        "sorter_cache_dir": str(copied_sorter_dir),
        "sorting_data_dir": str(_resolve_sorting_data_dir(copied_sorter_dir)),
        "analyzer_sparse": False,
        "template_random_spikes_method": "all",
        "allowed_labels": sorted(list(allowed_labels)),
        "selected_unit_count": int(len(sorting.unit_ids)),
        "selected_label_counts": {
            label: int(sum(1 for value in labels_by_unit.values() if value == label))
            for label in sorted(list(allowed_labels))
        },
        "pre_analyzer_dir": str(pre_analyzer_dir),
        "post_analyzer_dir": str(post_analyzer_dir),
        "merge_group_count": int(len(merge_unit_groups)),
        "merge_rows": merge_rows,
        "new_unit_ids": [str(uid) for uid in list(returned_new_unit_ids)],
        "plots_dir": str(plots_dir),
    }
    _write_json(summary_json, summary_payload)
    print(f"summary_json={summary_json}")


if __name__ == "__main__":
    main()