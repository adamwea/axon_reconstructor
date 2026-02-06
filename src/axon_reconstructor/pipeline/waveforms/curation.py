from __future__ import annotations

from typing import Any


def apply_mea_analysis_curation(*, q_metrics: Any, user_thresholds: Any = None) -> tuple[Any, Any]:
    """Apply MEA_Analysis curation logic to a quality-metrics DataFrame.

    Returns:
        clean_metrics_df, rejection_log_df
    """

    try:
        import numpy as np  # type: ignore[import-not-found]
        import pandas as pd  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Curation requires numpy/pandas") from e

    # Apply the same curation logic as MEA_Analysis if available.
    try:
        from MEA_Analysis.IPNAnalysis.mea_analysis_routine import MEAPipeline  # type: ignore[import-not-found]

        dummy = MEAPipeline.__new__(MEAPipeline)
        clean_metrics, rejection_log = MEAPipeline._apply_curation_logic(dummy, q_metrics, user_thresholds)
        return clean_metrics, rejection_log
    except Exception:
        defaults = {
            "presence_ratio": 0.75,
            "rp_contamination": 0.15,
            "firing_rate": 0.05,
            "amplitude_median": -20,
            "amplitude_cv_median": 0.5,
        }
        if user_thresholds:
            try:
                defaults.update(dict(user_thresholds))
            except Exception:
                pass

        keep_mask = np.ones(len(q_metrics), dtype=bool)
        rejections: list[dict[str, Any]] = []
        for idx, row in q_metrics.iterrows():
            reasons: list[str] = []
            if row.get("presence_ratio", 1) < defaults["presence_ratio"]:
                reasons.append("Low Presence")
            if row.get("rp_contamination", 0) > defaults["rp_contamination"]:
                reasons.append("High Contam")
            if row.get("firing_rate", 0) < defaults["firing_rate"]:
                reasons.append("Low FR")
            if row.get("amplitude_median", -100) > defaults["amplitude_median"]:
                reasons.append("Low Amp")

            if reasons:
                keep_mask[q_metrics.index.get_loc(row.name)] = False
                rejections.append({"unit_id": row.name, "reasons": "; ".join(reasons)})

        clean_metrics = q_metrics[keep_mask]
        rejection_log = pd.DataFrame(rejections)
        return clean_metrics, rejection_log


__all__ = [
    "apply_mea_analysis_curation",
]
