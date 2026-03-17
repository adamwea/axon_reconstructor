from __future__ import annotations

import logging
from typing import Any


def apply_standard_preprocessing(*, recording: Any, logger: logging.Logger | None = None) -> Any:
    """Apply MEA_Analysis preprocessing parity transforms to a recording.

    Sequence mirrors MEA_Analysis.IPNAnalysis.mea_analysis_routine.run_preprocessing:
    - unsigned_to_signed (only for unsigned dtypes)
    - highpass_filter(freq_min=300)
    - local median common_reference with global fallback
    - annotate(is_filtered=True)
    - cast to float32
    """

    try:
        import spikeinterface.preprocessing as spre
    except Exception as e:  # pragma: no cover
        raise RuntimeError("preprocessing requires `spikeinterface.preprocessing` installed") from e

    rec = recording

    # Keep this conversion first to avoid filtering/referencing uint data with offset bias.
    try:
        dtype_str = str(rec.get_dtype())
    except Exception:
        dtype_str = ""
    if dtype_str.startswith("uint"):
        rec = spre.unsigned_to_signed(rec)

    rec = spre.highpass_filter(rec, freq_min=300.0)

    try:
        # Keep radius parity with MEA_Analysis routine.
        rec = spre.common_reference(rec, reference="local", operator="median", local_radius=(250, 250))
    except Exception as e:
        if logger is not None:
            logger.warning("Local common_reference failed; falling back to global median reference (%s)", e)
        rec = spre.common_reference(rec, reference="global", operator="median")

    try:
        rec.annotate(is_filtered=True)
    except Exception:
        pass

    try:
        dtype_after = str(rec.get_dtype())
    except Exception:
        dtype_after = ""
    if dtype_after != "float32":
        rec = spre.astype(rec, "float32")

    return rec


__all__ = ["apply_standard_preprocessing"]
