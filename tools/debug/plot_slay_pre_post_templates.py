from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import spikeinterface.full as si


DEFAULT_PRE_ANALYZER = Path(
    "/home/adamm/dev/scratch_outputs/Media_Density_T3_07012025_AR/250728/M07137/AxonTracking/000225/well001/spikesort_outputs/merge_output/cache/merge_workspace/pre_merge_analyzer_output"
)
DEFAULT_POST_ANALYZER = Path(
    "/home/adamm/dev/scratch_outputs/Media_Density_T3_07012025_AR/250728/M07137/AxonTracking/000225/well001/spikesort_outputs/merge_output/cache/merge_workspace/analyzer_output"
)
DEFAULT_PRE_SORTER = Path(
    "/home/adamm/dev/scratch_outputs/Media_Density_T3_07012025_AR/250728/M07137/AxonTracking/000225/well001/spikesort_outputs/sorter_output"
)
DEFAULT_POST_SORTER = Path(
    "/home/adamm/dev/scratch_outputs/Media_Density_T3_07012025_AR/250728/M07137/AxonTracking/000225/well001/spikesort_outputs/merge_output/cache/merge_workspace/sorter_output/sorter_output"
)
DEFAULT_MAPPING_TSV = Path(
    "/home/adamm/dev/scratch_outputs/Media_Density_T3_07012025_AR/250728/M07137/AxonTracking/000225/well001/spikesort_outputs/merge_output/SLAy_outputs/recommended_merge_candidates.tsv"
)
DEFAULT_OUT_DIR = Path("/home/adamm/dev/pkgs/axon_reconstructor/tools/debug/outputs/slay_template_maps")


def _parse_unit_list(raw: str) -> list[str]:
    token = str(raw or "").strip()
    if not token:
        return []
    token = token.replace(",", "|")
    out: list[str] = []
    for part in token.split("|"):
        uid = part.strip()
        if uid and uid not in out:
            out.append(uid)
    return out


def _add_unique(items: list[str], value: str) -> None:
    token = str(value or "").strip()
    if token and token not in items:
        items.append(token)


def _read_slay_groups(tsv_path: Path) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            group_id = str((row.get("group_id") or "")).strip()
            if not group_id:
                continue

            grp = groups.get(group_id)
            if grp is None:
                grp = {"group_id": group_id, "pre_unit_ids": [], "post_unit_id": group_id}
                groups[group_id] = grp

            members = _parse_unit_list(str(row.get("group_members") or ""))
            for uid in members:
                _add_unique(grp["pre_unit_ids"], uid)

            for col in ("cluster_a", "cluster_b"):
                uid = str((row.get(col) or "")).strip()
                _add_unique(grp["pre_unit_ids"], uid)

            for col in ("post_unit_id", "final_post_unit_id", "merged_unit_id"):
                post_uid = str((row.get(col) or "")).strip()
                if post_uid:
                    grp["post_unit_id"] = post_uid
                    break

    def _sort_key(item: dict[str, Any]) -> tuple[int, Any]:
        gid = str(item.get("group_id", ""))
        try:
            return (0, int(gid))
        except Exception:
            return (1, gid)

    return sorted(groups.values(), key=_sort_key)


def _ensure_templates_extension(analyzer: Any) -> Any:
    if not analyzer.has_extension("templates"):
        _compute_template_extensions(analyzer)
    return analyzer.get_extension("templates")


def _resolve_sorting_data_dir(sorter_dir: Path) -> Path:
    sorter_dir = Path(sorter_dir).resolve()
    nested = (sorter_dir / "sorter_output").resolve()
    if nested.exists() and (nested / "spike_clusters.npy").exists() and (nested / "spike_times.npy").exists():
        return nested
    return sorter_dir


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


def _load_recording_from_sorter_dir(sorter_dir: Path) -> Any:
    recording_json = _find_recording_json(sorter_dir)

    load_fn = getattr(si, "load", None)
    if callable(load_fn):
        return load_fn(str(recording_json))
    return si.load_extractor(recording_json)


def _load_sorting_from_sorter_dir(sorter_dir: Path, recording: Any) -> Any:
    sorter_dir = Path(sorter_dir).resolve()
    sorting_data_dir = _resolve_sorting_data_dir(sorter_dir)
    spike_times_path = sorting_data_dir / "spike_times.npy"
    spike_clusters_path = sorting_data_dir / "spike_clusters.npy"

    if spike_times_path.exists() and spike_clusters_path.exists() and hasattr(si, "NumpySorting"):
        spike_times = np.asarray(np.load(spike_times_path), dtype=np.int64).reshape(-1)
        spike_labels = np.asarray(np.load(spike_clusters_path), dtype=np.int64).reshape(-1)
        if spike_times.shape[0] != spike_labels.shape[0]:
            raise RuntimeError(
                f"spike_times_labels_length_mismatch:{spike_times.shape[0]}:{spike_labels.shape[0]}"
            )
        unit_ids = np.unique(spike_labels)
        sampling_frequency = float(getattr(recording, "sampling_frequency", None) or recording.get_sampling_frequency())
        from_samples_and_labels = getattr(si.NumpySorting, "from_samples_and_labels", None)
        if callable(from_samples_and_labels):
            return from_samples_and_labels(
                samples_list=[spike_times],
                labels_list=[spike_labels],
                sampling_frequency=sampling_frequency,
                unit_ids=unit_ids,
            )

        return si.NumpySorting.from_times_labels(
            times_list=[spike_times],
            labels_list=[spike_labels],
            sampling_frequency=sampling_frequency,
            unit_ids=unit_ids,
        )

    return si.read_sorter_folder(sorter_dir, raise_error=True)


def _compute_template_extensions(analyzer: Any) -> None:
    has_extension = getattr(analyzer, "has_extension", None)
    compute_extension = getattr(analyzer, "compute", None)
    if not callable(compute_extension):
        raise RuntimeError("analyzer_compute_unavailable")

    pending: list[str] = []
    for extension_name in ("random_spikes", "waveforms", "templates"):
        if callable(has_extension):
            try:
                if bool(has_extension(extension_name)):
                    continue
            except Exception:
                pass
        pending.append(str(extension_name))

    if "random_spikes" in pending:
        try:
            compute_extension("random_spikes", method="all")
        except Exception:
            compute_extension(["random_spikes"], method="all")
        pending = [name for name in pending if name != "random_spikes"]

    for extension_name in pending:
        try:
            compute_extension(extension_name)
            continue
        except Exception:
            pass
        compute_extension([extension_name])


def _rebuild_analyzer_from_sorter(*, sorter_dir: Path, analyzer_dir: Path, label: str) -> Any:
    sorter_dir = Path(sorter_dir).resolve()
    analyzer_dir = Path(analyzer_dir).resolve()
    if analyzer_dir.exists():
        shutil.rmtree(analyzer_dir, ignore_errors=True)

    recording = _load_recording_from_sorter_dir(sorter_dir)
    sorting = _load_sorting_from_sorter_dir(sorter_dir, recording)
    print(f"{label}_sorter_source={_resolve_sorting_data_dir(sorter_dir)}")
    print(f"{label}_sorting_unit_count={len(sorting.unit_ids)}")
    analyzer = si.create_sorting_analyzer(
        sorting=sorting,
        recording=recording,
        format="binary_folder",
        folder=analyzer_dir,
        sparse=False,
        overwrite=True,
    )
    _compute_template_extensions(analyzer)
    print(f"rebuilt_{label}_analyzer={analyzer_dir}")
    return analyzer


def _unit_id_map(analyzer: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for uid in list(analyzer.sorting.unit_ids):
        out[str(uid)] = uid
    return out


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
    eps: float = 1e-3,
    marker_size: float = 2.0,
    chip_extent: tuple[float, float, float, float] | None = None,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot pre/post log-amplitude template maps from SLAy merge candidates.")
    parser.add_argument("--pre-sorter", type=Path, default=DEFAULT_PRE_SORTER)
    parser.add_argument("--post-sorter", type=Path, default=DEFAULT_POST_SORTER)
    parser.add_argument("--pre-analyzer", type=Path, default=DEFAULT_PRE_ANALYZER)
    parser.add_argument("--post-analyzer", type=Path, default=DEFAULT_POST_ANALYZER)
    parser.add_argument("--mapping-tsv", type=Path, default=DEFAULT_MAPPING_TSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    pre_analyzer = _rebuild_analyzer_from_sorter(
        sorter_dir=args.pre_sorter,
        analyzer_dir=args.pre_analyzer,
        label="pre",
    )
    post_analyzer = _rebuild_analyzer_from_sorter(
        sorter_dir=args.post_sorter,
        analyzer_dir=args.post_analyzer,
        label="post",
    )
    pre_templates = _ensure_templates_extension(pre_analyzer)
    post_templates = _ensure_templates_extension(post_analyzer)
    pre_uid_lookup = _unit_id_map(pre_analyzer)
    post_uid_lookup = _unit_id_map(post_analyzer)
    chip_extent = _chip_extent_from_analyzers([pre_analyzer, post_analyzer])

    groups = _read_slay_groups(args.mapping_tsv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"groups={len(groups)}")
    for group in groups:
        group_id = str(group.get("group_id"))
        pre_ids = list(group.get("pre_unit_ids") or [])
        post_id = str(group.get("post_unit_id") or "")

        ncols = max(1, len(pre_ids) + 1)
        fig, axes = plt.subplots(1, ncols, figsize=(4.0 * ncols, 4.0), constrained_layout=True)
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes])

        for idx, pre_uid in enumerate(pre_ids):
            ax = axes[idx]
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

        post_ax = axes[len(pre_ids)] if len(pre_ids) < len(axes) else axes[-1]
        post_tmpl, post_locs, post_err = _load_template_and_locs(post_analyzer, post_templates, post_id, post_uid_lookup)
        if post_tmpl is None or post_locs is None:
            post_ax.text(0.5, 0.5, f"post {post_id or 'n/a'}\n{post_err}", ha="center", va="center", transform=post_ax.transAxes)
            post_ax.axis("off")
        else:
            try:
                _plot_amp_map_log(post_ax, post_tmpl, post_locs, f"Post {post_id}", chip_extent=chip_extent)
            except Exception as exc:
                post_ax.text(0.5, 0.5, f"post {post_id}\n{type(exc).__name__}:{exc}", ha="center", va="center", transform=post_ax.transAxes)
                post_ax.axis("off")

        out_path = args.out_dir / f"group_{_safe_token(group_id)}.png"
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
