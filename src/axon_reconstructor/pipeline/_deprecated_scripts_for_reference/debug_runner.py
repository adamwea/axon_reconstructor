from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

def _ensure_maxwell_hdf5_plugin_env() -> None:
    """Configure the Maxwell HDF5 decompression plugin for host-side debugging.

    Maxwell-compressed `.raw.h5` files require the vendor HDF5 filter plugin (libcompression.so).
    In Shifter/Docker this is typically configured already; on the host we set `HDF5_PLUGIN_PATH`.

    This must run *before* importing `h5py` (directly or indirectly via SpikeInterface).
    """

    # Inside Shifter/Docker, /entrypoint.sh typically handles plugin setup.
    if Path("/entrypoint.sh").exists():
        return

    # If the user already configured it, don't override.
    if os.environ.get("HDF5_PLUGIN_PATH"):
        return

    # For debugging on Perlmutter host, use the same vendored plugin directory as the smoke tests.
    repo_root = Path(__file__).resolve().parents[3]
    plugin_dir = (
        repo_root / "tools/smoke_tests/perlmutter/_shared/vendor/maxwell_hdf5_plugin/Linux"
    )
    plugin_so = plugin_dir / "libcompression.so"

    if plugin_dir.is_dir() and plugin_so.exists():
        os.environ["HDF5_PLUGIN_PATH"] = str(plugin_dir)
        return

    # Avoid raising here: we want the debugger to start even if the dataset isn't Maxwell-compressed.
    # The subsequent HDF5 open will fail with a clearer message if this is required.
    print(
        "WARNING: Maxwell HDF5 plugin not configured; reading Maxwell-compressed .raw.h5 may fail. "
        "Expected vendored plugin at tools/smoke_tests/perlmutter/_shared/vendor/maxwell_hdf5_plugin/Linux "
        "or run inside the Shifter image.",
        flush=True,
    )


def run_debug_preprocess(
    *,
    h5_parent_dirs: list[str] | list[Path],
    mea_environment: str = "nersc",
    mea_analysis_output_root: Optional[str | Path] = None,
    mea_analysis_repo_root: Optional[str | Path] = None,
    mea_analysis_docker_image: Optional[str] = None,
    mea_auto_run_driver: bool = False,
) -> tuple[object, list[int]]:
    """Run preprocessing for debugger step-through.

    The private debug harness typically hardcodes dataset paths and calls this
    function with `AxonReconstructor` constructor-style args.

    It will:
    - Ensure the Maxwell HDF5 decompression plugin is configured on-host
    - Build an `AxonReconstructor`
    - Select the first discovered `.h5` under `h5_parent_dirs`
    - Run `preprocess_for_spikesorting()` for a chosen stream

    Stream + n_jobs are read from env for convenience:
    - STREAM_ID (default: well000)
    - N_JOBS (default: 8)

    Suggested breakpoints:
    - axon_reconstructor/pipeline/pipeline_driver.py : AxonReconstructor.preprocess_for_spikesorting
    - axon_reconstructor/pipeline/raw_preprocessing.py : build_concatenated_recording
    """

    _ensure_maxwell_hdf5_plugin_env()

    # Import is intentionally deferred until after the HDF5 plugin env is set.
    from axon_reconstructor.pipeline.pipeline_driver import AxonReconstructor

    out_root_path = (
        Path(mea_analysis_output_root).expanduser().resolve() if mea_analysis_output_root else None
    )
    if out_root_path is not None:
        out_root_path.mkdir(parents=True, exist_ok=True)

    repo_root_path = (
        Path(mea_analysis_repo_root).expanduser().resolve() if mea_analysis_repo_root else None
    )

    recon = AxonReconstructor(
        h5_parent_dirs=h5_parent_dirs,
        mea_environment=mea_environment,
        mea_analysis_output_root=str(out_root_path) if out_root_path else None,
        mea_analysis_repo_root=str(repo_root_path) if repo_root_path else None,
        mea_analysis_docker_image=mea_analysis_docker_image,
        mea_auto_run_driver=bool(mea_auto_run_driver),
    )

    raw_files = recon.iter_raw_h5_files()
    if not raw_files:
        raise RuntimeError(f"No .h5 files found under: {h5_parent_dirs}")
    h5_path = raw_files[0]

    stream_id = (os.environ.get("STREAM_ID") or "well000").strip()
    n_jobs_str = (os.environ.get("N_JOBS") or "8").strip()
    try:
        n_jobs = int(n_jobs_str)
    except ValueError as e:
        raise RuntimeError(f"Invalid N_JOBS={n_jobs_str!r}; expected int") from e

    multirec, common_el = recon.preprocess_for_spikesorting(
        h5_path=h5_path,
        stream_id=stream_id,
        n_jobs=n_jobs,
    )

    return multirec, common_el
