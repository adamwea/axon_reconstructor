from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class PreprocessingDebugInputs:
    h5_path: Path
    stream_id: str
    n_jobs: int
    mea_output_root: Path
    break_before_run: bool
    force_restart: bool
    temporal_resample_factor: Optional[int]
    temporal_resample_rate_hz: Optional[int]
    temporal_resample_margin_ms: float
    temporal_resample_dtype: Optional[str]


def build_preprocessing_debug_inputs(*, args: Any, env: Any) -> PreprocessingDebugInputs:
    h5_path = Path(args.h5_path) if args.h5_path is not None else env.env_required_path("AXON_RECON_H5_PATH")
    stream_id = str(args.stream_id) if args.stream_id is not None else env.env_required_str("AXON_RECON_STREAM_ID")
    n_jobs = int(args.n_jobs) if args.n_jobs is not None else int(env.env_int("AXON_RECON_N_JOBS", default=8) or 8)
    mea_output_root = (
        Path(args.mea_output_root)
        if args.mea_output_root is not None
        else env.env_required_path("AXON_RECON_MEA_OUTPUT_ROOT")
    )

    break_before_run = (
        env.env_bool("AXON_RECON_BREAK_BEFORE_RUN", default=False)
        if args.break_before_run is None
        else bool(args.break_before_run)
    )
    force_restart = (
        env.env_bool("AXON_RECON_FORCE_RESTART", default=False)
        if args.force_restart is None
        else bool(args.force_restart)
    )

    temporal_resample_factor = (
        int(args.temporal_resample_factor)
        if args.temporal_resample_factor is not None
        else env.env_int("AXON_RECON_TEMPORAL_RESAMPLE_FACTOR", default=None)
    )
    temporal_resample_rate_hz = (
        int(args.temporal_resample_rate_hz)
        if args.temporal_resample_rate_hz is not None
        else env.env_int("AXON_RECON_TEMPORAL_RESAMPLE_RATE_HZ", default=None)
    )
    temporal_resample_margin_ms = (
        float(args.temporal_resample_margin_ms)
        if args.temporal_resample_margin_ms is not None
        else float(env.env_float("AXON_RECON_TEMPORAL_RESAMPLE_MARGIN_MS", default=100.0) or 100.0)
    )
    temporal_resample_dtype = (
        str(args.temporal_resample_dtype)
        if args.temporal_resample_dtype is not None
        else env.env_str("AXON_RECON_TEMPORAL_RESAMPLE_DTYPE", default=None)
    )

    return PreprocessingDebugInputs(
        h5_path=h5_path,
        stream_id=stream_id,
        n_jobs=int(n_jobs),
        mea_output_root=mea_output_root,
        break_before_run=bool(break_before_run),
        force_restart=bool(force_restart),
        temporal_resample_factor=temporal_resample_factor,
        temporal_resample_rate_hz=temporal_resample_rate_hz,
        temporal_resample_margin_ms=float(temporal_resample_margin_ms),
        temporal_resample_dtype=temporal_resample_dtype,
    )
