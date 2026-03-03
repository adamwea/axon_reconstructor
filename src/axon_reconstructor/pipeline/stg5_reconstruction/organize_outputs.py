"""Organize reconstruction outputs into the canonical on-disk layout.

This is a lightweight, on-disk reorganization utility meant for existing runs.
It does *not* recompute reconstruction or regenerate plots; it only moves known
PNG/PDF artifacts into the subfolders used by the reconstruction stage.

Example:
    python -m axon_reconstructor.pipeline.reconstruction.organize_outputs \
        --unit-dir /path/to/<well>/stg5_reconstruction_outputs/by_unit/unit_119

"""

from __future__ import annotations

import argparse
from pathlib import Path

from .plotting import (
    _compute_unit_output_layout,
    _ensure_unit_output_layout,
    _maybe_migrate_legacy_unit_outputs,
)


def organize_unit_outputs(*, unit_dir: Path) -> None:
    unit_dir = Path(unit_dir)
    if not unit_dir.exists() or not unit_dir.is_dir():
        raise FileNotFoundError(f"Unit dir not found: {unit_dir}")

    layout = _compute_unit_output_layout(out_unit_dir=unit_dir)
    _ensure_unit_output_layout(layout)
    _maybe_migrate_legacy_unit_outputs(out_unit_dir=unit_dir, layout=layout)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Organize reconstruction outputs on disk")
    p.add_argument("--unit-dir", type=Path, required=True, help="Path to .../stg5_reconstruction_outputs/by_unit/unit_<id>")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    organize_unit_outputs(unit_dir=args.unit_dir)


if __name__ == "__main__":
    main()
