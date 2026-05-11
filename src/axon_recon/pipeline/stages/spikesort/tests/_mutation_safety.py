"""Helpers for mutation-safety regression tests (slice 7).

After slices 3-5, every label/merge phase honors a `dry_run` knob that
keeps the canonical sorter_output byte-identical. The helpers here
let regression tests hash a directory tree before and after a phase
invocation and assert no unexpected mutation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping


def hash_directory(root: Path) -> dict[str, str]:
	"""Return ``{posix_relative_path: sha256_hex}`` for every file under ``root``.

	Sorted by relative path so the dict ordering is deterministic. Empty
	directories are skipped (only files contribute to the hash). Returns
	an empty dict if ``root`` does not exist.
	"""
	root = Path(root)
	if not root.exists():
		return {}
	root = root.resolve()
	out: dict[str, str] = {}
	for p in sorted(root.rglob("*")):
		if not p.is_file():
			continue
		rel = p.resolve().relative_to(root).as_posix()
		out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
	return dict(sorted(out.items()))


def assert_directory_unchanged(root: Path, baseline: Mapping[str, str]) -> None:
	"""Re-hash ``root`` and assert equality with ``baseline``.

	Raises AssertionError with a diff summary on failure. Use after a
	phase that promised not to mutate ``root`` (e.g., dry_run=True for
	bombcell_label / merge_SLAy).
	"""
	current = hash_directory(root)
	if current == dict(baseline):
		return

	added = sorted(set(current) - set(baseline))
	removed = sorted(set(baseline) - set(current))
	changed = sorted(rel for rel in (set(current) & set(baseline)) if current[rel] != baseline[rel])
	parts: list[str] = [f"directory {root} mutated unexpectedly"]
	if added:
		parts.append(f"  added: {added}")
	if removed:
		parts.append(f"  removed: {removed}")
	if changed:
		parts.append(f"  changed: {changed}")
	raise AssertionError("\n".join(parts))
