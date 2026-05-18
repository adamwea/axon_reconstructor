"""Backfill SLAy's auxiliary cluster_*.tsv files (cluster_KSLabel.tsv,
cluster_Amplitude.tsv, cluster_ContamPct.tsv) for every well whose SLAy ran
PRE-SLAy-aux-tsv-sync-patch.

Why: SLAy's auto_accept_merges writes new merged cluster_ids to
spike_clusters.npy and cluster_group.tsv only. cluster_KSLabel,
cluster_Amplitude, cluster_ContamPct retain pre-merge content. SI's
read_kilosort inner-joins all cluster_*.tsv on cluster_id and drops every
new merged id missing from the per-metric TSVs → downstream pipelines
(recon templates phase) see only pre-merge survivors, not the full
post-merge unit roster. The SLAy patch landed 2026-05-18 (digest
32638ea26b in shifter pipeline-v2); wells whose SLAy ran before that don't
have the synced rows. This script applies the same sync logic to their
on-disk TSVs.

Idempotent: a `.pre_slay_sync_bk` sibling file marks a well as already
synced; those wells are skipped.

Usage (run inside shifter for pandas):
  shifter --image=adammwea/axon-recon:pipeline-v2 python3 \\
    dev/debug_NERSC/jobs/sync_slay_aux_tsvs.py \\
    --analyzed-root /pscratch/sd/a/adammwea/analyzed_data/Media_Density_T5_02182026_AR_axon_analysis_AW \\
    [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd


AUX_SPECS = (
    ("cluster_KSLabel.tsv", "KSLabel"),
    ("cluster_Amplitude.tsv", "Amplitude"),
    ("cluster_ContamPct.tsv", "ContamPct"),
)


def find_slay_sorter_outputs(analyzed_root: Path) -> list[Path]:
    """Find every canonical sorter_output dir whose automerge/new2old.json
    indicates SLAy applied merges in place. Path shape:
      <well>/spikesort_outputs/sorter_output/automerge/new2old.json
    We want the .../sorter_output/ dir (3 levels up from new2old.json).
    SLAy also drops a copy under .../merge_SLAy/SLAy_outputs/automerge/ —
    those aren't the canonical sorter_output, so filter them out.
    """
    results: list[Path] = []
    for new2old in analyzed_root.rglob("automerge/new2old.json"):
        sorter_output = new2old.parent.parent  # automerge -> sorter_output
        if sorter_output.name != "sorter_output":
            continue
        results.append(sorter_output)
    return sorted(set(results))


def sync_one_well(sorter_output: Path, *, dry_run: bool) -> tuple[str, dict]:
    new2old_path = sorter_output / "automerge" / "new2old.json"
    if not new2old_path.exists():
        return ("skipped_no_new2old", {})
    merges = {int(k): v for k, v in json.load(open(new2old_path)).items()}
    if not merges:
        return ("skipped_no_merges", {})

    backup_present = (sorter_output / "cluster_KSLabel.tsv.pre_slay_sync_bk").exists()
    if backup_present:
        return ("skipped_already_synced", {"merges": len(merges)})

    cl_group_path = sorter_output / "cluster_group.tsv"
    if not cl_group_path.exists():
        return ("error_no_cluster_group", {"merges": len(merges)})
    cl_group = pd.read_csv(cl_group_path, sep="\t")

    added: dict[str, int] = {}
    for filename, value_col in AUX_SPECS:
        path = sorter_output / filename
        if not path.exists():
            added[filename] = -1
            continue
        df = pd.read_csv(path, sep="\t")
        if "cluster_id" not in df.columns or value_col not in df.columns:
            added[filename] = -2
            continue
        existing_ids = set(df["cluster_id"].tolist())

        new_rows = []
        for new_id, old_ids in merges.items():
            if new_id in existing_ids:
                continue
            if value_col == "KSLabel":
                row = cl_group.loc[cl_group["cluster_id"] == new_id]
                label_val = ""
                if len(row) and "label" in row.columns:
                    label_val = str(row["label"].iloc[0])
                new_rows.append({"cluster_id": new_id, value_col: label_val})
            else:
                parent_val = float("nan")
                for old in old_ids:
                    try:
                        old_int = int(old)
                    except Exception:
                        continue
                    m = df.loc[df["cluster_id"] == old_int, value_col]
                    if len(m) > 0:
                        parent_val = float(m.iloc[0])
                        break
                new_rows.append({"cluster_id": new_id, value_col: parent_val})

        added[filename] = len(new_rows)
        if not new_rows:
            continue
        if dry_run:
            continue

        backup = path.with_suffix(path.suffix + ".pre_slay_sync_bk")
        if not backup.exists():
            shutil.copy2(path, backup)

        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df = df.sort_values("cluster_id").reset_index(drop=True)
        df.to_csv(path, sep="\t", index=False)

    return ("synced" if not dry_run else "would_sync", {"merges": len(merges), **added})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--analyzed-root",
        type=Path,
        default=Path("/pscratch/sd/a/adammwea/analyzed_data/Media_Density_T5_02182026_AR_axon_analysis_AW"),
        help="Root under which to find <date>/<chip>/AxonTracking/<rec>/<well>/spikesort_outputs/sorter_output dirs",
    )
    p.add_argument("--dry-run", action="store_true", help="Report what would change without modifying anything")
    args = p.parse_args(argv)

    if not args.analyzed_root.exists():
        print(f"ERROR: analyzed-root not found: {args.analyzed_root}", file=sys.stderr)
        return 2

    sorter_outputs = find_slay_sorter_outputs(args.analyzed_root)
    print(f"discovered {len(sorter_outputs)} sorter_output dirs with automerge/new2old.json")

    summary: dict[str, int] = {}
    for so in sorter_outputs:
        status, info = sync_one_well(so, dry_run=args.dry_run)
        summary[status] = summary.get(status, 0) + 1
        rel = so.relative_to(args.analyzed_root)
        per_file = " ".join(f"{k}=+{v}" for k, v in info.items() if k not in {"merges"})
        merges_str = f"merges={info.get('merges', 0)}" if info else "-"
        print(f"  [{status}] {rel}  {merges_str}  {per_file}")

    print()
    print("=== summary ===")
    for k in sorted(summary):
        print(f"  {k}: {summary[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
