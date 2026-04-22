#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from axon_reconstructor.pipeline.analysis.cross_well import main as run_cross_well_analysis


def main() -> None:
    default_config = Path(__file__).resolve().parent / "cross_well_config.yml"
    run_cross_well_analysis(default_config_path=default_config)


if __name__ == "__main__":
    main()
