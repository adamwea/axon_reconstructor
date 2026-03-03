from __future__ import annotations

import math
from typing import Any


DEFAULT_QM_METRIC_NAMES: list[str] = [
    # Core counts/rates
    "num_spikes",
    "firing_rate",
    # Metrics used in downstream curation and/or for inspection
    "presence_ratio",
    "isi_violation",
    "rp_violation",
    "snr",
    "amplitude_cutoff",
    "amplitude_median",
    "amplitude_cv",
]


def build_quality_metrics_extension_params(*, min_duration_s: float | None, logger: Any | None) -> dict[str, Any]:
    """Build consistent `quality_metrics` extension parameters.

    Key goals:
    - Avoid SpikeInterface warnings when segments are short by shrinking
      `presence_ratio.bin_duration_s` when needed.
    - Keep params consistent across concat and per-segment analyzers.
    - Avoid metrics with additional extension dependencies (e.g. `drift` requires
      `spike_locations`) unless explicitly requested.
    """

    desired_bin_s = 60.0
    min_bin_s = 0.25

    # Choose a bin size that is guaranteed to be <= duration/2 when duration is known.
    chosen_bin_s = desired_bin_s
    if min_duration_s is not None:
        try:
            d = float(min_duration_s)
        except Exception:
            d = float("nan")

        if math.isfinite(d) and d > 0:
            chosen_bin_s = min(desired_bin_s, max(min_bin_s, d / 2.0))
            # Ensure it is strictly smaller than the recording/segment duration.
            if chosen_bin_s >= d:
                chosen_bin_s = max(min_bin_s, d / 2.0)

            if logger is not None:
                if chosen_bin_s < desired_bin_s:
                    try:
                        logger.warning(
                            "Short recording/segment detected (min_duration_s=%.6g). "
                            "Using presence_ratio.bin_duration_s=%.6g (was %.6g).",
                            d,
                            chosen_bin_s,
                            desired_bin_s,
                        )
                    except Exception:
                        pass

                # User requested a warning when bins get very small.
                if chosen_bin_s < 5.0:
                    try:
                        logger.warning(
                            "presence_ratio.bin_duration_s=%.6g is very small; presence_ratio may be noisy.",
                            chosen_bin_s,
                        )
                    except Exception:
                        pass

    metric_params: dict[str, dict[str, Any]] = {
        "presence_ratio": {
            "bin_duration_s": float(chosen_bin_s),
            "mean_fr_ratio_thresh": 0.0,
        },
        # Be explicit about key defaults so concat/segments match.
        "rp_violation": {
            "refractory_period_ms": 1.0,
            "censored_period_ms": 0.0,
        },
    }

    return {
        "metric_names": list(DEFAULT_QM_METRIC_NAMES),
        "metric_params": metric_params,
        "peak_sign": "neg",
        "seed": 0,
        "skip_pc_metrics": True,
        # Avoid keeping previously-computed metrics (e.g. `drift`) when re-running
        # `compute('quality_metrics', ...)` on an existing analyzer folder.
        "delete_existing_metrics": True,
    }
