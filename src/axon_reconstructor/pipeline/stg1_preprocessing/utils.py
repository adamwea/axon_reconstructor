from __future__ import annotations

from pathlib import Path


def _ensure_maxwell_hdf5_plugin_path(*, prefix: str = "[axon_reconstructor]") -> None:
    """Best-effort fix for Maxwell HDF5 decompression plugin discovery.

    Some environments end up with `HDF5_PLUGIN_PATH` pointing at a non-existent
    directory, which causes HDF5 reads to fail when the Maxwell compression
    filter is encountered.

    We prefer the vendored plugin shipped with this repo when available.
    """

    import os

    env = os.environ.get("HDF5_PLUGIN_PATH")
    if env:
        try:
            if not Path(env).expanduser().exists():
                print(f"{prefix}[WARN] HDF5_PLUGIN_PATH points to missing dir: {env}; ignoring", flush=True)
                os.environ.pop("HDF5_PLUGIN_PATH", None)
        except Exception:
            pass

    if os.environ.get("HDF5_PLUGIN_PATH"):
        return

    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        cand_dir = parent / "vendor" / "maxwell_hdf5_plugin" / "Linux"
        if (cand_dir / "libcompression.so").exists():
            os.environ["HDF5_PLUGIN_PATH"] = str(cand_dir)
            print(f"{prefix}[DEBUG] set HDF5_PLUGIN_PATH={cand_dir}", flush=True)
            return


__all__ = ["_ensure_maxwell_hdf5_plugin_path"]
