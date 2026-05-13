#!/usr/bin/env python3
"""Detect spikesort datasets where at least one included well is incomplete.

Reads a runtime yml + its referenced data yml, walks each (dataset, well)
pair, and prints a comma-separated list of dataset indices that have at
least one well missing the spikesort completion marker:

    <output_root>/<rel_pattern>/<well_id>/spikesort_outputs/merge_SLAy/merge_stage_summary.json

The <rel_pattern> mirrors src/axon_recon/pipeline/output_paths.py:
compute_mea_analysis_output_dir — last 5 path components of the raw H5 file
(excluding `data.raw.h5` itself).

merge_SLAy is the final enabled phase of the spikesort stage; its
merge_stage_summary.json is written at phase end (src/axon_recon/pipeline/
stages/spikesort/runner.py:8472/8551/9326). Presence of the file implies
all upstream phases (bootstrap_concat_binary, sort, snapshot_sorter_output,
concat_analyzer, bombcell_label) also completed for that well.

Output: a single line with the CSV list (empty string if every included
dataset/well combination already has the marker).

Usage:
    detect_incomplete_spikesort.py <runtime.yml>
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml


SPIKESORT_COMPLETION_MARKER = (
    "spikesort_outputs",
    "merge_SLAy",
    "merge_stage_summary.json",
)


def _resolve_data_yml(runtime_yml: Path, data_ref: str | Path) -> Path:
    data_path = Path(data_ref)
    if data_path.is_absolute():
        return data_path
    return (runtime_yml.parent / data_path).resolve()


def _rel_pattern_from_h5(raw_h5: Path) -> str:
    """Mirror compute_mea_analysis_output_dir's rel-pattern: last 5 path parts
    of the raw H5 file (excluding data.raw.h5 itself)."""
    parts = str(raw_h5).split("/")
    if len(parts) > 5:
        return "/".join(parts[-6:-1])
    return raw_h5.name


def main(runtime_yml_path: Path) -> int:
    runtime_cfg = yaml.safe_load(runtime_yml_path.read_text())
    data_yml_path = _resolve_data_yml(runtime_yml_path, runtime_cfg.get("data", "./debug.data.yml"))
    data_cfg = yaml.safe_load(data_yml_path.read_text())

    output_root = Path(data_cfg["output_root"])
    incomplete: list[int] = []

    for i, dataset in enumerate(data_cfg.get("datasets", []) or []):
        if not isinstance(dataset, dict):
            continue
        if not dataset.get("include_in_runtime", True):
            continue
        raw_h5 = Path(dataset["raw_data_h5_path"])
        rel_pattern = _rel_pattern_from_h5(raw_h5)

        for well in dataset.get("wells", []) or []:
            if not isinstance(well, dict):
                continue
            if not well.get("include_in_runtime", True):
                continue
            well_id = str(well["well_id"])
            marker = output_root / rel_pattern / well_id
            for piece in SPIKESORT_COMPLETION_MARKER:
                marker = marker / piece
            if not marker.is_file():
                incomplete.append(i)
                break  # one incomplete well is enough to flag the dataset

    print(",".join(str(i) for i in incomplete))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: detect_incomplete_spikesort.py <runtime.yml>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))
