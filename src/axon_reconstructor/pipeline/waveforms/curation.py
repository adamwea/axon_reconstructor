from __future__ import annotations

from pathlib import Path
from typing import Any


def _run_mea_analysis_style_curation(
    *,
    recording: Any,
    sorting: Any,
    output_dir: Path,
    n_jobs: int,
    ms_before: float,
    ms_after: float,
    force_restart: bool,
    logger: Any,
) -> tuple[list[Any], dict[str, Path]]:
    """Compute MEA_Analysis-like curation artifacts and return curated unit IDs.

    Writes the following files into output_dir:
    - qm_unfiltered.xlsx
    - tm_unfiltered.xlsx
    - metrics_curated.xlsx
    - rejection_log.xlsx
    - tm_curated.xlsx

    Returns (clean_units, paths).
    """

    try:
        import numpy as np  # type: ignore[import-not-found]
        import pandas as pd  # type: ignore[import-not-found]
        import spikeinterface.full as si  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Curation requires spikeinterface/numpy/pandas") from e

    output_dir.mkdir(parents=True, exist_ok=True)

    qm_unfiltered_xlsx = output_dir / "qm_unfiltered.xlsx"
    tm_unfiltered_xlsx = output_dir / "tm_unfiltered.xlsx"
    metrics_curated_xlsx = output_dir / "metrics_curated.xlsx"
    rejection_log_xlsx = output_dir / "rejection_log.xlsx"
    tm_curated_xlsx = output_dir / "tm_curated.xlsx"

    analyzer_dir = output_dir / "curation_analyzer"

    analyzer = None
    if analyzer_dir.exists() and not force_restart:
        try:
            analyzer = si.load_sorting_analyzer(analyzer_dir)
        except Exception:
            analyzer = None
    if analyzer is None:
        if analyzer_dir.exists():
            try:
                import shutil

                shutil.rmtree(analyzer_dir)
            except Exception:
                pass

        logger.info("Computing SortingAnalyzer for curation -> %s", analyzer_dir)

        sparsity = si.estimate_sparsity(
            sorting,
            recording,
            method="radius",
            radius_um=50,
            peak_sign="neg",
        )

        analyzer = si.create_sorting_analyzer(
            sorting,
            recording,
            format="binary_folder",
            folder=analyzer_dir,
            sparsity=sparsity,
            return_in_uV=True,
        )

        ext_list = [
            "random_spikes",
            "spike_amplitudes",
            "waveforms",
            "templates",
            "noise_levels",
            "quality_metrics",
            "template_metrics",
            "unit_locations",
        ]
        ext_params = {
            "waveforms": {"ms_before": float(ms_before), "ms_after": float(ms_after)},
            "unit_locations": {"method": "monopolar_triangulation"},
        }

        analyzer.compute(
            ext_list,
            extension_params=ext_params,
            verbose=False,
            n_jobs=int(n_jobs),
        )

    q_metrics = analyzer.get_extension("quality_metrics").get_data()
    t_metrics = analyzer.get_extension("template_metrics").get_data()
    locations = analyzer.get_extension("unit_locations").get_data()

    # Match MEA_Analysis: add unit locations into q_metrics.
    q_metrics = q_metrics.copy()
    q_metrics["loc_x"] = locations[:, 0]
    q_metrics["loc_y"] = locations[:, 1]

    q_metrics.to_excel(qm_unfiltered_xlsx)
    t_metrics.to_excel(tm_unfiltered_xlsx)

    # Apply the same curation logic as MEA_Analysis.
    clean_units: list[Any]
    try:
        from MEA_Analysis.IPNAnalysis.mea_analysis_routine import MEAPipeline  # type: ignore[import-not-found]

        # _apply_curation_logic does not depend on MEAPipeline instance state.
        dummy = MEAPipeline.__new__(MEAPipeline)
        clean_metrics, rejection_log = MEAPipeline._apply_curation_logic(dummy, q_metrics, None)
    except Exception:
        defaults = {
            "presence_ratio": 0.75,
            "rp_contamination": 0.15,
            "firing_rate": 0.05,
            "amplitude_median": -20,
            "amplitude_cv_median": 0.5,
        }
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

    clean_units = list(clean_metrics.index.values)
    clean_metrics.to_excel(metrics_curated_xlsx)
    rejection_log.to_excel(rejection_log_xlsx)

    try:
        t_metrics.loc[clean_units].to_excel(tm_curated_xlsx)
    except Exception:
        # If something goes wrong with indexing, still emit the unfiltered TM.
        pass

    paths = {
        "qm_unfiltered.xlsx": qm_unfiltered_xlsx,
        "tm_unfiltered.xlsx": tm_unfiltered_xlsx,
        "metrics_curated.xlsx": metrics_curated_xlsx,
        "rejection_log.xlsx": rejection_log_xlsx,
        "tm_curated.xlsx": tm_curated_xlsx,
        "curation_analyzer_dir": analyzer_dir,
    }
    return clean_units, paths


__all__ = [
    "_run_mea_analysis_style_curation",
]
