from __future__ import annotations

import os
from pathlib import Path


def compute_mea_analysis_output_dir(
    *,
    output_root: Path,
    data_file: Path,
    well: str,
) -> Path:
    """Compute MEA_Analysis-style per-well output directory.

    Mirrors the effective contract used by MEA_Analysis metadata parsing.
    """

    output_root = Path(output_root).expanduser()
    data_file = Path(data_file).expanduser().resolve()

    try:
        relative_pattern = f"{data_file.parent.parent.name}/{data_file.parent.name}/{data_file.name}"
    except Exception:
        relative_pattern = str(data_file.name)

    parts = str(data_file).split(os.sep)
    if len(parts) > 5:
        relative_pattern = os.path.join(*parts[-6:-1])

    return output_root / relative_pattern / str(well)
