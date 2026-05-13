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

Output (stdout): a single line with the CSV list of incomplete dataset
indices. Empty string if every included dataset/well combination already
has the marker.

With --verbose, a per-dataset status table is also written to stderr:
each row shows the dataset index, DIV, raw H5 short path, total included
wells, complete well count, and the list of missing wells (if any).

Usage:
    detect_incomplete_spikesort.py [--verbose] <runtime.yml>
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


def _short_h5(raw_h5: Path) -> str:
    """Compact dataset label: <date>/<chip>/<run>."""
    parts = str(raw_h5).split("/")
    if len(parts) >= 5:
        return f"{parts[-5]}/{parts[-4]}/{parts[-2]}"
    return raw_h5.name


def main(runtime_yml_path: Path, *, verbose: bool = False) -> int:
    runtime_cfg = yaml.safe_load(runtime_yml_path.read_text())
    data_yml_path = _resolve_data_yml(runtime_yml_path, runtime_cfg.get("data", "./debug.data.yml"))
    data_cfg = yaml.safe_load(data_yml_path.read_text())

    output_root = Path(data_cfg["output_root"])
    incomplete: list[int] = []
    rows: list[tuple[int, int, str, int, int, list[str]]] = []

    for i, dataset in enumerate(data_cfg.get("datasets", []) or []):
        if not isinstance(dataset, dict):
            continue
        if not dataset.get("include_in_runtime", True):
            continue
        raw_h5 = Path(dataset["raw_data_h5_path"])
        rel_pattern = _rel_pattern_from_h5(raw_h5)
        div = dataset.get("DIV", -1)

        wells_included: list[str] = []
        wells_missing: list[str] = []
        for well in dataset.get("wells", []) or []:
            if not isinstance(well, dict):
                continue
            if not well.get("include_in_runtime", True):
                continue
            well_id = str(well["well_id"])
            wells_included.append(well_id)
            marker = output_root / rel_pattern / well_id
            for piece in SPIKESORT_COMPLETION_MARKER:
                marker = marker / piece
            if not marker.is_file():
                wells_missing.append(well_id)

        if wells_missing:
            incomplete.append(i)
        rows.append((i, int(div) if div is not None else -1, _short_h5(raw_h5),
                     len(wells_included), len(wells_included) - len(wells_missing), wells_missing))

    if verbose:
        print(f"# spikesort completeness scan vs {output_root}", file=sys.stderr)
        print(f"# marker: <well>/{'/'.join(SPIKESORT_COMPLETION_MARKER)}", file=sys.stderr)
        print(f"{'idx':>3}  {'DIV':>3}  {'dataset':<30}  {'wells (ok/total)':<18}  missing_wells", file=sys.stderr)
        print(f"{'---':>3}  {'---':>3}  {'-'*30:<30}  {'-'*18:<18}  -------------", file=sys.stderr)
        for idx, div, short, total, ok, missing in rows:
            div_str = str(div) if div >= 0 else "-"
            status = f"{ok}/{total}"
            missing_str = ",".join(missing) if missing else "-"
            tag = "  COMPLETE" if not missing else ""
            print(f"{idx:>3}  {div_str:>3}  {short:<30}  {status:<18}  {missing_str}{tag}", file=sys.stderr)
        print(f"# {len(incomplete)} of {len(rows)} datasets have at least one missing well", file=sys.stderr)

    print(",".join(str(i) for i in incomplete))
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    verbose = False
    if args and args[0] == "--verbose":
        verbose = True
        args = args[1:]
    if len(args) != 1:
        print("usage: detect_incomplete_spikesort.py [--verbose] <runtime.yml>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(Path(args[0]), verbose=verbose))
