#!/usr/bin/env python3
"""Print incomplete-spikesort dataset indices via the shared status module.

Thin wrapper around ``axon_recon.pipeline.status.scan_status`` that preserves
the original contract (CSV of incomplete dataset indices on stdout; optional
per-dataset table on stderr with ``--verbose``). Kept as a standalone script
so the pipeline chain script can call it without booting the full
``axon-recon`` CLI inside Shifter.

For the full per-stage status table, prefer ``axon-recon status``.

Usage:
    detect_incomplete_spikesort.py [--verbose] <runtime.yml>
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly without an editable install: src/ is one
# level up from examples/, so add it to sys.path before importing axon_recon.
_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_REPO_SRC) not in sys.path:
	sys.path.insert(0, str(_REPO_SRC))

from axon_recon.pipeline.status import (  # noqa: E402  (import after sys.path tweak)
	STAGE_WELL_MARKER,
	format_default_tables,
	incomplete_dataset_indices,
	scan_status,
)


def main(runtime_yml_path: Path, *, verbose: bool = False) -> int:
	report = scan_status(runtime_yml_path, stages=["spikesort"])
	incomplete = incomplete_dataset_indices(report, stage="spikesort")

	if verbose:
		print(format_default_tables(report), file=sys.stderr)

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
