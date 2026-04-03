"""Radivojevic_2023 axonal reconstruction strategy (standalone, not wired).

Implements the method described in:
  Radivojevic & Rostedt Punga, eLife 2023;12:e86512

Scope implemented from the publication text:
- Adaptive thresholding in 3 steps on dV/dt electrical images:
  step1: 9*noise_std (global)
  step2: 2*noise_std, spatially confined to 50 um, temporally to [t-1, t, t+1]
  step3: 1*noise_std, spatially confined to 100 um, temporally to [t-1, t, t+1]
- 3-step trajectory tracking:
  step1: direct links between consecutive frames, <=100 um
  step2: skeletonization-assisted links between consecutive frames, <=200 um
  step3: indirect links every-other frame, <=400 um, with middle-frame prediction
- Velocity-consistency rejection: discard candidate links whose velocities deviate by >50%
  from previously estimated velocities.

Notes:
- This module is intentionally not wired into the active reconstruction runner.
- Inputs can be provided as already-framed electrical images, or generated from full-template
    unit artifacts (preferred) / merged contributing templates (compatibility fallback).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from ...stg1_preprocessing.constants import LEGACY_PREPROCESS_OUTPUTS_DIRNAME, PREPROCESS_OUTPUTS_DIRNAME
from ...stg2_spikesorting.runner import LEGACY_SPIKESORTING_OUTPUTS_DIRNAME, SPIKESORTING_OUTPUTS_DIRNAME


@dataclass(frozen=True)
class Radivojevic2023Params:
    interframe_us: float = 50.0
    n_timeframes_expected: int = 400

    threshold_std_step1: float = 9.0
    threshold_std_step2: float = 2.0
    threshold_std_step3: float = 1.0

    confined_radius_step2_um: float = 50.0
    confined_radius_step3_um: float = 100.0
    temporal_neighbor_half_window_frames: int = 1

    direct_link_max_distance_um: float = 100.0
    skeleton_link_max_distance_um: float = 200.0
    indirect_link_max_distance_um: float = 400.0
    max_velocity_deviation_fraction: float = 0.5

    # Local peak selector radius in the channel cloud.
    local_peak_radius_um: float = 25.0

    # Skeleton-like routing helper.
    support_level_std: float = 1.0
    support_graph_neighbor_radius_um: float = 25.0
    support_path_slack_factor: float = 2.5
    skeleton_grid_step_um: float = 10.0

    # If True, enforce exact paper framing assumptions.
    strict_paper_mode: bool = False

    # Optional BOTM-inspired noise estimation from spike-free (quiet) windows.
    use_quiet_period_noise_estimation: bool = False
    quiet_noise_n_windows: int = 2000
    quiet_noise_seed: int = 0
    quiet_noise_exclude_all_units: bool = False


@dataclass(frozen=True)
class Radivojevic2023Input:
    dvdt_frames_uv_per_us: np.ndarray
    channel_locations_um: np.ndarray
    noise_std_uv_per_us: float
    sampling_frequency_hz: float
    source: str


@dataclass(frozen=True)
class DetectedPeak:
    peak_id: int
    frame_idx: int
    channel_idx: int
    x_um: float
    y_um: float
    dvdt_uv_per_us: float
    detection_step: int


@dataclass(frozen=True)
class PeakLink:
    src_peak_id: int
    dst_peak_id: int
    tracking_step: int
    link_kind: str
    dt_frames: int
    distance_um: float
    velocity_m_per_s: float
    predicted_midpoint_xy_um: tuple[float, float] | None


@dataclass(frozen=True)
class Radivojevic2023Result:
    params: Radivojevic2023Params
    n_frames: int
    n_channels: int
    peaks: list[DetectedPeak]
    links: list[PeakLink]
    trajectories: list[list[int]]
    velocity_m_per_s_median: float | None

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "params": self.params.__dict__,
            "n_frames": self.n_frames,
            "n_channels": self.n_channels,
            "peaks": [p.__dict__ for p in self.peaks],
            "links": [
                {
                    **l.__dict__,
                    "predicted_midpoint_xy_um": list(l.predicted_midpoint_xy_um)
                    if l.predicted_midpoint_xy_um is not None
                    else None,
                }
                for l in self.links
            ],
            "trajectories": self.trajectories,
            "velocity_m_per_s_median": self.velocity_m_per_s_median,
        }


def _robust_noise_std(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        raise ValueError("Cannot estimate noise: empty values")
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    sigma = mad / 0.6744897501960817 if mad > 0 else float(np.std(v)) # TODO: Why this constant? (From scipy.stats.median_abs_deviation with scale='normal')
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(v))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Cannot estimate positive finite noise std")
    return sigma


def _pairwise_dist_um(locs_xy: np.ndarray) -> np.ndarray:
    d = locs_xy[:, None, :] - locs_xy[None, :, :]
    return np.sqrt(np.sum(d * d, axis=2))


def _time_derivative_uv_per_us(template_samples_by_channels: np.ndarray, fs_hz: float) -> np.ndarray:
    if fs_hz <= 0:
        raise ValueError(f"Invalid sampling frequency: {fs_hz}")
    dt_us = 1e6 / float(fs_hz)
    return np.diff(np.asarray(template_samples_by_channels, dtype=float), axis=0) / dt_us


def _any_spike_in_interval(*, spikes_sorted: np.ndarray, start: int, end: int) -> bool:
    s = int(start)
    e = int(end)
    if e <= s:
        return True
    sp = np.asarray(spikes_sorted, dtype=np.int64)
    if sp.size == 0:
        return False
    i = int(np.searchsorted(sp, s, side="left"))
    return bool(i < int(sp.size) and int(sp[i]) < e)


def _load_sorting(*, sorter_output_dir: Path, sorter: str) -> Any:
    import spikeinterface.full as si  # type: ignore[import-not-found]

    if hasattr(si, "read_sorter_folder"):
        try:
            return si.read_sorter_folder(sorter_output_dir, sorter_name=str(sorter))
        except TypeError:
            return si.read_sorter_folder(sorter_output_dir, str(sorter))

    try:
        return si.load_extractor(sorter_output_dir)
    except Exception as e:
        raise RuntimeError(
            f"Could not load sorting from sorter_output_dir={sorter_output_dir} (sorter={sorter}): {e}"
        ) from e


def _load_preprocessed_concat_recording(*, well_out_dir: Path) -> Any:
    import spikeinterface.full as si  # type: ignore[import-not-found]

    rec_dir = Path(well_out_dir) / PREPROCESS_OUTPUTS_DIRNAME / "preprocessed_recording"
    if not rec_dir.exists():
        legacy_rec_dir = Path(well_out_dir) / LEGACY_PREPROCESS_OUTPUTS_DIRNAME / "preprocessed_recording"
        if legacy_rec_dir.exists():
            rec_dir = legacy_rec_dir
    if not rec_dir.exists():
        raise FileNotFoundError(f"Missing preprocessed recording dir: {rec_dir}")

    try:
        return si.load(rec_dir)
    except Exception:
        return si.load_extractor(rec_dir)


def _estimate_quiet_noise_std_uv_per_us(
    *,
    well_out_dir: Path,
    unit_id: Any,
    template_n_samples: int,
    channel_ids: list[int] | None,
    n_noise_windows: int,
    seed: int,
    sorter: str,
    exclude_all_units: bool,
) -> float:
    if int(n_noise_windows) <= 0:
        raise ValueError(f"quiet noise n_noise_windows must be > 0, got {n_noise_windows}")

    sorter_output_dir = Path(well_out_dir) / SPIKESORTING_OUTPUTS_DIRNAME / "sorter_output"
    if not sorter_output_dir.exists():
        legacy_sorter_output_dir = Path(well_out_dir) / LEGACY_SPIKESORTING_OUTPUTS_DIRNAME / "sorter_output"
        if legacy_sorter_output_dir.exists():
            sorter_output_dir = legacy_sorter_output_dir
        else:
            raise FileNotFoundError(f"Missing sorter output dir: {sorter_output_dir}")

    sorting = _load_sorting(sorter_output_dir=sorter_output_dir, sorter=str(sorter))
    rec = _load_preprocessed_concat_recording(well_out_dir=Path(well_out_dir))

    fs_hz = float(rec.get_sampling_frequency())
    if fs_hz <= 0:
        raise RuntimeError("Could not determine sampling frequency from preprocessed recording")

    if int(template_n_samples) < 3:
        raise ValueError(f"template_n_samples must be >= 3, got {template_n_samples}")

    pre_samples = int(template_n_samples // 2)
    post_samples = int(template_n_samples - pre_samples)

    if exclude_all_units:
        all_spikes: list[np.ndarray] = []
        for uid2 in sorting.get_unit_ids():
            st = sorting.get_unit_spike_train(unit_id=uid2, segment_index=0)
            st = np.asarray(st, dtype=np.int64)
            if st.size:
                all_spikes.append(st)
        spikes_guard = np.unique(np.concatenate(all_spikes, axis=0)).astype(np.int64, copy=False) if all_spikes else np.asarray([], dtype=np.int64)
    else:
        st_unit = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=0)
        spikes_guard = np.asarray(st_unit, dtype=np.int64)
        spikes_guard.sort()

    rec_channel_ids = list(rec.get_channel_ids())
    rec_channel_set = set(rec_channel_ids)
    selected_channels: list[Any]
    if channel_ids:
        selected_channels = [ch for ch in channel_ids if ch in rec_channel_set]
        if not selected_channels:
            raise RuntimeError("No merged-template channels available in preprocessed recording")
    else:
        selected_channels = rec_channel_ids

    n_ch = int(len(selected_channels))
    sum_x = np.zeros((n_ch,), dtype=float)
    sum_x2 = np.zeros((n_ch,), dtype=float)
    n_x = np.zeros((n_ch,), dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    n_total = int(rec.get_num_samples())
    center_min = int(pre_samples)
    center_max = int(n_total - post_samples)
    if center_max <= center_min:
        raise RuntimeError(
            f"Recording too short for quiet-noise windows: n_total={n_total}, window={template_n_samples}"
        )

    max_tries = int(max(10_000, 20 * int(n_noise_windows)))
    n_accepted = 0
    tries = 0
    dt_us = 1e6 / float(fs_hz)

    while n_accepted < int(n_noise_windows) and tries < max_tries:
        tries += 1
        center = int(rng.integers(center_min, center_max))
        if _any_spike_in_interval(
            spikes_sorted=spikes_guard,
            start=center - int(pre_samples),
            end=center + int(post_samples),
        ):
            continue

        x = rec.get_traces(
            start_frame=int(center - pre_samples),
            end_frame=int(center + post_samples),
            channel_ids=selected_channels,
        )
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or int(x.shape[0]) != int(template_n_samples) or int(x.shape[1]) != n_ch:
            continue

        dvdt = np.diff(x, axis=0) / dt_us
        if dvdt.size == 0:
            continue

        sum_x += np.sum(dvdt, axis=0)
        sum_x2 += np.sum(dvdt * dvdt, axis=0)
        n_x += int(dvdt.shape[0])
        n_accepted += 1

    if n_accepted < int(n_noise_windows):
        raise RuntimeError(
            "Failed to sample required spike-free quiet windows: "
            f"needed={int(n_noise_windows)} got={n_accepted} tries={tries}"
        )

    eps = 1e-12
    sigma_per_ch: list[float] = []
    for ci in range(n_ch):
        if int(n_x[ci]) <= 1:
            continue
        mean = float(sum_x[ci]) / float(n_x[ci])
        ex2 = float(sum_x2[ci]) / float(n_x[ci])
        var = float(max(ex2 - mean * mean, 0.0))
        sigma = float(np.sqrt(var if var > eps else eps))
        if np.isfinite(sigma) and sigma > 0:
            sigma_per_ch.append(sigma)

    if not sigma_per_ch:
        raise RuntimeError("Could not compute any valid channelwise quiet-noise sigma values")

    sigma_global = float(np.median(np.asarray(sigma_per_ch, dtype=float)))
    if not np.isfinite(sigma_global) or sigma_global <= 0:
        raise RuntimeError(f"Invalid global quiet-noise sigma: {sigma_global}")
    return sigma_global


def _frame_from_derivative(
    dvdt_samples_by_channels: np.ndarray,
    *,
    fs_hz: float,
    interframe_us: float,
) -> np.ndarray:
    frame_samples = int(round((interframe_us * 1e-6) * float(fs_hz)))
    if frame_samples <= 0:
        raise ValueError("interframe_us/frame_samples must be > 0")
    n_total, n_channels = dvdt_samples_by_channels.shape
    n_frames = n_total // frame_samples
    if n_frames <= 0:
        raise ValueError(
            f"Not enough derivative samples ({n_total}) for frame size {frame_samples}."
        )

    trimmed = dvdt_samples_by_channels[: n_frames * frame_samples, :]
    blocks = trimmed.reshape(n_frames, frame_samples, n_channels)
    return np.min(blocks, axis=1)


class Radivojevic2023Reconstructor:
    """Standalone implementation of the Radivojevic_2023 strategy."""

    def __init__(self, params: Radivojevic2023Params | None = None) -> None:
        self.params = params or Radivojevic2023Params()

    @staticmethod
    def load_full_template_inputs(*, full_unit_dir: Path) -> dict[str, Any]:
        unit_dir = Path(full_unit_dir)
        template_npy = unit_dir / "full_template.npy"
        locs_npy = unit_dir / "full_channel_locations_xy.npy"
        meta_json = unit_dir / "full_template_meta.json"
        channel_ids_npy = unit_dir / "full_channel_ids.npy"
        contrib_idx_npy = unit_dir / "contributing_full_channel_indices.npy"

        if not template_npy.exists() or not locs_npy.exists() or not meta_json.exists():
            raise FileNotFoundError(
                "Expected full-template files: "
                f"{template_npy}, {locs_npy}, {meta_json}"
            )

        template = np.asarray(np.load(template_npy), dtype=float)
        locs = np.asarray(np.load(locs_npy), dtype=float)
        meta = json.loads(meta_json.read_text(encoding="utf-8"))

        if template.ndim != 2:
            raise ValueError(f"Expected full_template.npy as 2D [samples, channels], got {template.shape}")
        if locs.ndim != 2 or locs.shape[1] < 2:
            raise ValueError(f"Expected full_channel_locations_xy.npy as [channels,2], got {locs.shape}")
        if int(template.shape[1]) != int(locs.shape[0]):
            raise ValueError("Mismatch between full template channel count and location count")

        if channel_ids_npy.exists():
            channel_ids_arr = np.load(channel_ids_npy, allow_pickle=True)
            channel_ids = [int(v) for v in np.asarray(channel_ids_arr).tolist()]
        else:
            channel_ids = list(range(int(template.shape[1])))

        # Keep only informative channels so downstream pairwise-distance computations stay tractable.
        keep_mask = np.any(template != 0.0, axis=0)
        if contrib_idx_npy.exists():
            contrib_idx = np.asarray(np.load(contrib_idx_npy), dtype=np.int64)
            if contrib_idx.size:
                keep_mask = np.zeros((template.shape[1],), dtype=bool)
                keep_mask[np.clip(contrib_idx, 0, template.shape[1] - 1)] = True

        keep_idx = np.where(keep_mask)[0]
        if keep_idx.size == 0:
            raise ValueError("Full template has no non-zero/contributing channels after filtering")

        template_kept = template[:, keep_idx]
        locs_kept = locs[keep_idx, :2]
        channel_ids_kept = [int(channel_ids[int(i)]) for i in keep_idx.tolist()]

        return {
            "template_samples_by_channels": template_kept,
            "channel_locations_um": locs_kept,
            "channel_ids": channel_ids_kept,
            "meta": meta,
            "source_base": "full_template",
        }

    @staticmethod
    def load_merged_contributing_inputs(*, merged_unit_dir: Path) -> dict[str, Any]:
        unit_dir = Path(merged_unit_dir)
        template_npy = unit_dir / "merged_contributing_template.npy"
        locs_npy = unit_dir / "merged_contributing_channel_locations.npy"
        meta_json = unit_dir / "merged_contributing_template_meta.json"

        if not template_npy.exists() or not locs_npy.exists() or not meta_json.exists():
            raise FileNotFoundError(
                "Expected merged contributing files: "
                f"{template_npy}, {locs_npy}, {meta_json}"
            )

        template = np.load(template_npy)
        locs = np.load(locs_npy)
        meta = json.loads(meta_json.read_text(encoding="utf-8"))

        return {
            "template_samples_by_channels": np.asarray(template, dtype=float),
            "channel_locations_um": np.asarray(locs, dtype=float)[:, :2],
            "channel_ids": [int(ch) for ch in meta.get("channel_ids", [])] if isinstance(meta.get("channel_ids"), list) else None,
            "meta": meta,
            "source_base": "merged_contributing_template",
        }

    def _load_preferred_template_inputs(self, *, unit_dir: Path) -> dict[str, Any]:
        p = Path(unit_dir)
        if (p / "full_template.npy").exists() and (p / "full_template_meta.json").exists():
            return self.load_full_template_inputs(full_unit_dir=p)
        return self.load_merged_contributing_inputs(merged_unit_dir=p)

    def make_input_from_full_template(
        self,
        *,
        full_unit_dir: Path,
        noise_std_uv_per_us: float | None = None,
        well_out_dir: Path | None = None,
        unit_id: Any | None = None,
        sorter: str = "kilosort4",
    ) -> Radivojevic2023Input:
        loaded = self.load_full_template_inputs(full_unit_dir=full_unit_dir)
        return self._make_input_from_loaded_template(
            loaded=loaded,
            noise_std_uv_per_us=noise_std_uv_per_us,
            well_out_dir=well_out_dir,
            unit_id=unit_id,
            sorter=sorter,
        )

    def make_input_from_merged_contributing(
        self,
        *,
        merged_unit_dir: Path,
        noise_std_uv_per_us: float | None = None,
        well_out_dir: Path | None = None,
        unit_id: Any | None = None,
        sorter: str = "kilosort4",
    ) -> Radivojevic2023Input:
        loaded = self._load_preferred_template_inputs(unit_dir=Path(merged_unit_dir))
        return self._make_input_from_loaded_template(
            loaded=loaded,
            noise_std_uv_per_us=noise_std_uv_per_us,
            well_out_dir=well_out_dir,
            unit_id=unit_id,
            sorter=sorter,
        )

    def _make_input_from_loaded_template(
        self,
        *,
        loaded: dict[str, Any],
        noise_std_uv_per_us: float | None,
        well_out_dir: Path | None,
        unit_id: Any | None,
        sorter: str,
    ) -> Radivojevic2023Input:
        template = loaded["template_samples_by_channels"]
        locs = loaded["channel_locations_um"]
        source_base = str(loaded.get("source_base") or "template")
        meta = loaded["meta"]

        fs_hz = float(meta.get("sampling_frequency_hz") or 0.0)
        if fs_hz <= 0:
            raise ValueError("template meta missing valid sampling_frequency_hz")

        dvdt = _time_derivative_uv_per_us(template, fs_hz)
        frames = _frame_from_derivative(
            dvdt,
            fs_hz=fs_hz,
            interframe_us=self.params.interframe_us,
        )

        resolved_unit_id = unit_id if unit_id is not None else meta.get("unit_id")
        loaded_channel_ids = loaded.get("channel_ids")
        channel_ids = [int(ch) for ch in loaded_channel_ids] if isinstance(loaded_channel_ids, list) else None

        if noise_std_uv_per_us is not None:
            sigma = float(noise_std_uv_per_us)
            source = f"{source_base}_user_noise"
        elif self.params.use_quiet_period_noise_estimation:
            if well_out_dir is None:
                raise ValueError(
                    "use_quiet_period_noise_estimation=True requires well_out_dir"
                )
            if resolved_unit_id is None:
                raise ValueError(
                    "use_quiet_period_noise_estimation=True requires unit_id or unit_id in template meta"
                )
            sigma = _estimate_quiet_noise_std_uv_per_us(
                well_out_dir=Path(well_out_dir),
                unit_id=resolved_unit_id,
                template_n_samples=int(template.shape[0]),
                channel_ids=channel_ids,
                n_noise_windows=int(self.params.quiet_noise_n_windows),
                seed=int(self.params.quiet_noise_seed),
                sorter=str(sorter),
                exclude_all_units=bool(self.params.quiet_noise_exclude_all_units),
            )
            source = f"{source_base}_quiet_noise"
        else:
            sigma = _robust_noise_std(dvdt)
            source = source_base

        return Radivojevic2023Input(
            dvdt_frames_uv_per_us=frames,
            channel_locations_um=locs,
            noise_std_uv_per_us=sigma,
            sampling_frequency_hz=fs_hz,
            source=source,
        )

    def run(self, inputs: Radivojevic2023Input) -> Radivojevic2023Result:
        frames = np.asarray(inputs.dvdt_frames_uv_per_us, dtype=float)
        locs = np.asarray(inputs.channel_locations_um, dtype=float)
        noise_std = float(inputs.noise_std_uv_per_us)

        if frames.ndim != 2:
            raise ValueError(f"Expected dvdt frames as 2D array [n_frames, n_channels], got {frames.shape}")
        if locs.ndim != 2 or locs.shape[1] < 2:
            raise ValueError(f"Expected channel locations as [n_channels,2], got {locs.shape}")
        if frames.shape[1] != locs.shape[0]:
            raise ValueError("Mismatch: n_channels in frames and locations differ")
        if noise_std <= 0 or not np.isfinite(noise_std):
            raise ValueError(f"Invalid noise std: {noise_std}")

        n_frames, n_channels = frames.shape
        if self.params.strict_paper_mode:
            if int(n_frames) != int(self.params.n_timeframes_expected):
                raise ValueError(
                    "strict_paper_mode=True requires exactly "
                    f"{self.params.n_timeframes_expected} frames at {self.params.interframe_us} us; got {n_frames}"
                )

        dists = _pairwise_dist_um(locs[:, :2])

        peaks1 = self._detect_global_step(frames, locs, dists, threshold_std=self.params.threshold_std_step1, noise_std=noise_std)
        peaks2 = self._detect_confined_step(
            frames,
            locs,
            dists,
            threshold_std=self.params.threshold_std_step2,
            noise_std=noise_std,
            radius_um=self.params.confined_radius_step2_um,
            seed_peaks=peaks1,
            detection_step=2,
        )
        peaks12 = self._merge_peaks(peaks1 + peaks2)
        peaks3 = self._detect_confined_step(
            frames,
            locs,
            dists,
            threshold_std=self.params.threshold_std_step3,
            noise_std=noise_std,
            radius_um=self.params.confined_radius_step3_um,
            seed_peaks=peaks12,
            detection_step=3,
        )
        all_peaks = self._merge_peaks(peaks12 + peaks3)

        links = self._track_three_steps(frames=frames, locs=locs, peaks=all_peaks, noise_std=noise_std)
        trajectories = self._build_trajectories(peaks=all_peaks, links=links)

        vel_vals = [l.velocity_m_per_s for l in links if np.isfinite(l.velocity_m_per_s) and l.velocity_m_per_s > 0]
        vel_med = float(np.median(vel_vals)) if vel_vals else None

        return Radivojevic2023Result(
            params=self.params,
            n_frames=n_frames,
            n_channels=n_channels,
            peaks=all_peaks,
            links=links,
            trajectories=trajectories,
            velocity_m_per_s_median=vel_med,
        )

    def _detect_global_step(
        self,
        frames: np.ndarray,
        locs: np.ndarray,
        dists: np.ndarray,
        *,
        threshold_std: float,
        noise_std: float,
    ) -> list[DetectedPeak]:
        peaks: list[DetectedPeak] = []
        peak_id = 0
        thr = -float(threshold_std) * float(noise_std)
        for fi in range(frames.shape[0]):
            vals = frames[fi]
            candidates = np.where(vals <= thr)[0]
            for ci in candidates.tolist():
                if self._is_local_minimum(channel_idx=ci, values=vals, dists=dists):
                    peaks.append(
                        DetectedPeak(
                            peak_id=peak_id,
                            frame_idx=fi,
                            channel_idx=int(ci),
                            x_um=float(locs[ci, 0]),
                            y_um=float(locs[ci, 1]),
                            dvdt_uv_per_us=float(vals[ci]),
                            detection_step=1,
                        )
                    )
                    peak_id += 1
        return peaks

    def _detect_confined_step(
        self,
        frames: np.ndarray,
        locs: np.ndarray,
        dists: np.ndarray,
        *,
        threshold_std: float,
        noise_std: float,
        radius_um: float,
        seed_peaks: list[DetectedPeak],
        detection_step: int,
    ) -> list[DetectedPeak]:
        thr = -float(threshold_std) * float(noise_std)
        peak_id_start = 0 if not seed_peaks else (max(p.peak_id for p in seed_peaks) + 1)
        out: list[DetectedPeak] = []
        seen: set[tuple[int, int]] = set()
        next_peak_id = peak_id_start

        seed_by_frame: dict[int, list[DetectedPeak]] = {}
        for p in seed_peaks:
            seed_by_frame.setdefault(p.frame_idx, []).append(p)

        t_half = int(self.params.temporal_neighbor_half_window_frames)
        for fi in range(frames.shape[0]):
            vals = frames[fi]
            for fj in range(max(0, fi - t_half), min(frames.shape[0] - 1, fi + t_half) + 1):
                for sp in seed_by_frame.get(fj, []):
                    c0 = sp.channel_idx
                    nearby = np.where(dists[c0] <= float(radius_um))[0]
                    if nearby.size == 0:
                        continue
                    passing = nearby[vals[nearby] <= thr]
                    for ci in passing.tolist():
                        key = (fi, int(ci))
                        if key in seen:
                            continue
                        if self._is_local_minimum(channel_idx=int(ci), values=vals, dists=dists):
                            seen.add(key)
                            out.append(
                                DetectedPeak(
                                    peak_id=next_peak_id,
                                    frame_idx=fi,
                                    channel_idx=int(ci),
                                    x_um=float(locs[ci, 0]),
                                    y_um=float(locs[ci, 1]),
                                    dvdt_uv_per_us=float(vals[ci]),
                                    detection_step=int(detection_step),
                                )
                            )
                            next_peak_id += 1
        return out

    def _is_local_minimum(self, *, channel_idx: int, values: np.ndarray, dists: np.ndarray) -> bool:
        neigh = np.where(dists[channel_idx] <= float(self.params.local_peak_radius_um))[0]
        if neigh.size == 0:
            return True
        return bool(values[channel_idx] <= np.min(values[neigh]))

    def _merge_peaks(self, peaks: list[DetectedPeak]) -> list[DetectedPeak]:
        best: dict[tuple[int, int], DetectedPeak] = {}
        for p in peaks:
            key = (p.frame_idx, p.channel_idx)
            cur = best.get(key)
            if cur is None:
                best[key] = p
                continue
            if p.detection_step > cur.detection_step:
                best[key] = p
            elif p.detection_step == cur.detection_step and p.dvdt_uv_per_us < cur.dvdt_uv_per_us:
                best[key] = p

        merged = sorted(best.values(), key=lambda x: (x.frame_idx, x.channel_idx))
        reindexed: list[DetectedPeak] = []
        for i, p in enumerate(merged):
            reindexed.append(
                DetectedPeak(
                    peak_id=i,
                    frame_idx=p.frame_idx,
                    channel_idx=p.channel_idx,
                    x_um=p.x_um,
                    y_um=p.y_um,
                    dvdt_uv_per_us=p.dvdt_uv_per_us,
                    detection_step=p.detection_step,
                )
            )
        return reindexed

    def _track_three_steps(
        self,
        *,
        frames: np.ndarray,
        locs: np.ndarray,
        peaks: list[DetectedPeak],
        noise_std: float,
    ) -> list[PeakLink]:
        by_frame: dict[int, list[DetectedPeak]] = {}
        for p in peaks:
            by_frame.setdefault(p.frame_idx, []).append(p)

        links: list[PeakLink] = []
        used_src: set[int] = set()
        used_dst: set[int] = set()

        dt_s = float(self.params.interframe_us) * 1e-6

        def _velocity_ok(v: float, existing: list[PeakLink]) -> bool:
            if not np.isfinite(v) or v <= 0:
                return False
            if not existing:
                return True
            ref = float(np.median([x.velocity_m_per_s for x in existing if x.velocity_m_per_s > 0]))
            if ref <= 0 or not np.isfinite(ref):
                return True
            return abs(v - ref) / ref <= float(self.params.max_velocity_deviation_fraction)

        # (I) Direct interconnection (consecutive, <=100 um)
        for fi in range(frames.shape[0] - 1):
            srcs = [p for p in by_frame.get(fi, []) if p.peak_id not in used_src]
            dsts = [p for p in by_frame.get(fi + 1, []) if p.peak_id not in used_dst]
            for sp in srcs:
                best: tuple[float, DetectedPeak] | None = None
                for dp in dsts:
                    d = float(np.hypot(dp.x_um - sp.x_um, dp.y_um - sp.y_um))
                    if d > float(self.params.direct_link_max_distance_um):
                        continue
                    if best is None or d < best[0]:
                        best = (d, dp)
                if best is None:
                    continue
                d, dp = best
                v = (d * 1e-6) / dt_s
                if not _velocity_ok(v, links):
                    continue
                links.append(
                    PeakLink(
                        src_peak_id=sp.peak_id,
                        dst_peak_id=dp.peak_id,
                        tracking_step=1,
                        link_kind="direct",
                        dt_frames=1,
                        distance_um=d,
                        velocity_m_per_s=v,
                        predicted_midpoint_xy_um=None,
                    )
                )
                used_src.add(sp.peak_id)
                used_dst.add(dp.peak_id)

        # (II) Skeletonization-assisted (consecutive, <=200 um)
        for fi in range(frames.shape[0] - 1):
            srcs = [p for p in by_frame.get(fi, []) if p.peak_id not in used_src]
            dsts = [p for p in by_frame.get(fi + 1, []) if p.peak_id not in used_dst]
            frame_avg = 0.5 * (frames[fi] + frames[fi + 1])
            support = self._build_skeleton_support(frame_avg=frame_avg, locs=locs, noise_std=noise_std)
            for sp in srcs:
                best: tuple[float, DetectedPeak] | None = None
                for dp in dsts:
                    d = float(np.hypot(dp.x_um - sp.x_um, dp.y_um - sp.y_um))
                    if d > float(self.params.skeleton_link_max_distance_um):
                        continue
                    if not self._skeleton_path_exists(
                        support=support,
                        src_xy=(sp.x_um, sp.y_um),
                        dst_xy=(dp.x_um, dp.y_um),
                    ):
                        continue
                    if best is None or d < best[0]:
                        best = (d, dp)
                if best is None:
                    continue
                d, dp = best
                v = (d * 1e-6) / dt_s
                if not _velocity_ok(v, links):
                    continue
                links.append(
                    PeakLink(
                        src_peak_id=sp.peak_id,
                        dst_peak_id=dp.peak_id,
                        tracking_step=2,
                        link_kind="skeleton_assisted",
                        dt_frames=1,
                        distance_um=d,
                        velocity_m_per_s=v,
                        predicted_midpoint_xy_um=None,
                    )
                )
                used_src.add(sp.peak_id)
                used_dst.add(dp.peak_id)

        # (III) Indirect interconnection (every other frame, <=400 um)
        for fi in range(frames.shape[0] - 2):
            srcs = [p for p in by_frame.get(fi, []) if p.peak_id not in used_src]
            dsts = [p for p in by_frame.get(fi + 2, []) if p.peak_id not in used_dst]
            frame_avg = (frames[fi] + frames[fi + 1] + frames[fi + 2]) / 3.0
            support = self._build_skeleton_support(frame_avg=frame_avg, locs=locs, noise_std=noise_std)
            for sp in srcs:
                best: tuple[float, DetectedPeak] | None = None
                for dp in dsts:
                    d = float(np.hypot(dp.x_um - sp.x_um, dp.y_um - sp.y_um))
                    if d > float(self.params.indirect_link_max_distance_um):
                        continue
                    if not self._skeleton_path_exists(
                        support=support,
                        src_xy=(sp.x_um, sp.y_um),
                        dst_xy=(dp.x_um, dp.y_um),
                    ):
                        continue
                    if best is None or d < best[0]:
                        best = (d, dp)
                if best is None:
                    continue
                d, dp = best
                v = (d * 1e-6) / (2.0 * dt_s)
                if not _velocity_ok(v, links):
                    continue
                predicted_mid = ((sp.x_um + dp.x_um) * 0.5, (sp.y_um + dp.y_um) * 0.5)
                links.append(
                    PeakLink(
                        src_peak_id=sp.peak_id,
                        dst_peak_id=dp.peak_id,
                        tracking_step=3,
                        link_kind="indirect",
                        dt_frames=2,
                        distance_um=d,
                        velocity_m_per_s=v,
                        predicted_midpoint_xy_um=(float(predicted_mid[0]), float(predicted_mid[1])),
                    )
                )
                used_src.add(sp.peak_id)
                used_dst.add(dp.peak_id)

        return links

    def _build_skeleton_support(
        self,
        *,
        frame_avg: np.ndarray,
        locs: np.ndarray,
        noise_std: float,
    ) -> dict[str, Any]:
        from scipy import interpolate, ndimage  # type: ignore[import-not-found]

        thr = -float(self.params.support_level_std) * float(noise_std)
        values = np.asarray(frame_avg, dtype=float)
        points = np.asarray(locs[:, :2], dtype=float)

        x_min = float(np.min(points[:, 0]))
        x_max = float(np.max(points[:, 0]))
        y_min = float(np.min(points[:, 1]))
        y_max = float(np.max(points[:, 1]))

        step_um = float(self.params.skeleton_grid_step_um)
        if not np.isfinite(step_um) or step_um <= 0:
            step_um = 10.0

        nx = int(np.ceil((x_max - x_min) / step_um)) + 3
        ny = int(np.ceil((y_max - y_min) / step_um)) + 3
        nx = max(nx, 5)
        ny = max(ny, 5)

        grid_x = x_min - step_um + np.arange(nx, dtype=float) * step_um
        grid_y = y_min - step_um + np.arange(ny, dtype=float) * step_um
        gx, gy = np.meshgrid(grid_x, grid_y, indexing="xy")

        zi_lin = interpolate.griddata(points, values, (gx, gy), method="linear")
        zi_near = interpolate.griddata(points, values, (gx, gy), method="nearest")
        if zi_lin is None or zi_near is None:
            raise RuntimeError("Failed to construct interpolated skeleton support map")
        img = np.where(np.isfinite(zi_lin), zi_lin, zi_near)

        binary = np.asarray(img <= thr, dtype=bool)
        structure = np.ones((3, 3), dtype=bool)
        binary = ndimage.binary_opening(binary, structure=structure)
        binary = ndimage.binary_closing(binary, structure=structure)

        skeleton = self._morphological_skeleton(binary)
        # Thicken slightly to tolerate rasterization mismatch from endpoint projection.
        skeleton_thick = ndimage.binary_dilation(skeleton, structure=structure, iterations=1)

        return {
            "skeleton": np.asarray(skeleton_thick, dtype=bool),
            "x0": x_min - step_um,
            "y0": y_min - step_um,
            "step_um": step_um,
            "nx": int(nx),
            "ny": int(ny),
        }

    def _morphological_skeleton(self, binary: np.ndarray) -> np.ndarray:
        from scipy import ndimage  # type: ignore[import-not-found]

        img = np.asarray(binary, dtype=bool)
        if img.ndim != 2:
            raise ValueError(f"Expected 2D binary image for skeletonization, got {img.shape}")

        structure = np.ones((3, 3), dtype=bool)
        skel = np.zeros_like(img, dtype=bool)
        work = img.copy()

        while np.any(work):
            eroded = ndimage.binary_erosion(work, structure=structure)
            opened = ndimage.binary_dilation(eroded, structure=structure)
            skel |= work & (~opened)
            work = eroded

        return skel

    def _xy_to_grid_ij(self, *, x: float, y: float, support: dict[str, Any]) -> tuple[int, int]:
        x0 = float(support["x0"])
        y0 = float(support["y0"])
        step = float(support["step_um"])
        nx = int(support["nx"])
        ny = int(support["ny"])

        j = int(np.clip(np.round((float(x) - x0) / step), 0, nx - 1))
        i = int(np.clip(np.round((float(y) - y0) / step), 0, ny - 1))
        return i, j

    def _skeleton_path_exists(
        self,
        *,
        support: dict[str, Any],
        src_xy: tuple[float, float],
        dst_xy: tuple[float, float],
    ) -> bool:
        from scipy import ndimage  # type: ignore[import-not-found]

        mask = np.asarray(support["skeleton"], dtype=bool).copy()
        if mask.ndim != 2 or mask.size == 0:
            return False

        src_i, src_j = self._xy_to_grid_ij(x=float(src_xy[0]), y=float(src_xy[1]), support=support)
        dst_i, dst_j = self._xy_to_grid_ij(x=float(dst_xy[0]), y=float(dst_xy[1]), support=support)
        mask[src_i, src_j] = True
        mask[dst_i, dst_j] = True

        labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
        a = int(labels[src_i, src_j])
        b = int(labels[dst_i, dst_j])
        return bool(a > 0 and a == b)

    def _build_trajectories(self, *, peaks: list[DetectedPeak], links: list[PeakLink]) -> list[list[int]]:
        if not peaks:
            return []

        out_map: dict[int, list[int]] = {}
        in_deg: dict[int, int] = {p.peak_id: 0 for p in peaks}
        for l in links:
            out_map.setdefault(l.src_peak_id, []).append(l.dst_peak_id)
            in_deg[l.dst_peak_id] = in_deg.get(l.dst_peak_id, 0) + 1

        starts = [pid for pid, deg in in_deg.items() if deg == 0]
        trajs: list[list[int]] = []

        for s in starts:
            stack: list[tuple[int, list[int]]] = [(s, [s])]
            while stack:
                cur, path = stack.pop()
                nxt = out_map.get(cur, [])
                if not nxt:
                    trajs.append(path)
                    continue
                for n in nxt:
                    if n in path:
                        trajs.append(path)
                        continue
                    stack.append((n, path + [n]))

        trajs_sorted = sorted(trajs, key=lambda p: (-len(p), p))
        return trajs_sorted


__all__ = [
    "DetectedPeak",
    "PeakLink",
    "Radivojevic2023Input",
    "Radivojevic2023Params",
    "Radivojevic2023Reconstructor",
    "Radivojevic2023Result",
]
