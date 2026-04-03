#!/usr/bin/env python3
"""Prune a dataset tree down to AxonTracking-only.

Keeps:
- Any directory that is an ancestor of at least one `AxonTracking/` directory (so the path still exists)
- Any `AxonTracking/` directory and all of its contents

Deletes:
- All other files
- All other directories (recursively)

Safety:
- Default mode is dry-run (prints what would be deleted)
- Does NOT follow symlinks when walking the tree

Example:
  python prune_to_axontracking.py --base "$HOME/adamm/local_RBS_data/raw_data/possibly_incomplete_data_sets" --dry-run
  python prune_to_axontracking.py --base "$HOME/adamm/local_RBS_data/raw_data/possibly_incomplete_data_sets" --execute
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_BASE = os.path.expanduser("~/adamm/local_RBS_data/raw_data/possibly_incomplete_data_sets")


def _is_descendant(path: Path, parent: Path) -> bool:
    """True if `path` is within `parent` (or equal), without resolving symlinks."""
    try:
        path_abs = path.absolute()
        parent_abs = parent.absolute()
        return path_abs == parent_abs or parent_abs in path_abs.parents
    except Exception:
        return False


def _iter_ancestors(child: Path, stop_at: Path) -> Iterable[Path]:
    """Yield child, its parents, ... up to and including stop_at."""
    child_abs = child.absolute()
    stop_abs = stop_at.absolute()

    p = child_abs
    while True:
        yield p
        if p == stop_abs:
            return
        if p.parent == p:
            # Hit filesystem root before stop_at.
            return
        p = p.parent


def _find_axontracking_dirs(base: Path) -> list[Path]:
    base = base.absolute()
    found: list[Path] = []

    for root, dirnames, _filenames in os.walk(base, topdown=True, followlinks=False):
        if "AxonTracking" in dirnames:
            found.append(Path(root) / "AxonTracking")
            # Don’t prune dirnames here; we only *discover* first.

    # De-dup while preserving order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in found:
        pa = p.absolute()
        if pa not in seen:
            uniq.append(pa)
            seen.add(pa)
    return uniq


def _build_keep_ancestor_dirs(*, base: Path, axon_dirs: list[Path]) -> set[Path]:
    """Directories to keep as path scaffolding from base to each AxonTracking dir."""
    keep: set[Path] = set()
    base_abs = base.absolute()
    keep.add(base_abs)

    for ax in axon_dirs:
        for anc in _iter_ancestors(ax.parent, base_abs):
            keep.add(anc.absolute())
    return keep


def _inside_any_axontracking(path: Path, axon_dirs: list[Path]) -> bool:
    p = path.absolute()
    for ax in axon_dirs:
        if _is_descendant(p, ax.absolute()):
            return True
    return False


def _format_path(p: Path) -> str:
    try:
        return str(p)
    except Exception:
        return repr(p)


def _rmtree_onerror(func, path, exc_info):
    """Best-effort handler to delete read-only files on Windows-mounted drives."""
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    except Exception:
        pass
    try:
        func(path)
    except Exception:
        # Let caller record failure.
        raise


def _unlink_with_retry(p: Path) -> None:
    try:
        p.unlink()
        return
    except PermissionError:
        try:
            os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
        except Exception:
            pass
        p.unlink()


def _rmtree_with_retry(p: Path) -> None:
    shutil.rmtree(p, onerror=_rmtree_onerror)


def _list_survivors(*, base: Path, axon_dirs: list[Path], keep_anc_dirs: set[Path]) -> tuple[list[Path], list[Path]]:
    """Return (files, dirs) that remain but are not allowed by the keep rules."""
    bad_files: list[Path] = []
    bad_dirs: list[Path] = []
    base_abs = base.absolute()

    for root, dirnames, filenames in os.walk(base_abs, topdown=False, followlinks=False):
        root_p = Path(root).absolute()
        for fn in filenames:
            p = (root_p / fn).absolute()
            if _inside_any_axontracking(p, axon_dirs):
                continue
            bad_files.append(p)
        for dn in dirnames:
            p = (root_p / dn).absolute()
            if p == base_abs:
                continue
            if p in keep_anc_dirs:
                continue
            if _inside_any_axontracking(p, axon_dirs):
                continue
            bad_dirs.append(p)

    bad_files.sort(key=lambda p: len(p.parts), reverse=True)
    bad_dirs.sort(key=lambda p: len(p.parts), reverse=True)
    return bad_files, bad_dirs


def _find_empty_dirs_to_remove(*, base: Path, axon_dirs: list[Path], keep_anc_dirs: set[Path]) -> list[Path]:
    """Find empty directories eligible for removal.

    We remove empty directories that are:
    - not the base directory
    - not in the kept ancestor scaffolding
    - not inside any AxonTracking directory

    This is intentionally conservative: it won’t remove empty directories under
    AxonTracking (since those are part of the kept subtree).
    """
    base_abs = base.absolute()
    empty: list[Path] = []

    for root, dirnames, filenames in os.walk(base_abs, topdown=False, followlinks=False):
        root_p = Path(root).absolute()

        if root_p == base_abs:
            continue
        if root_p in keep_anc_dirs:
            continue
        if _inside_any_axontracking(root_p, axon_dirs):
            continue

        # os.walk lists current state; if there are no entries, it's empty.
        if (not dirnames) and (not filenames):
            empty.append(root_p)

    empty.sort(key=lambda p: len(p.parts), reverse=True)
    return empty


def _remove_empty_dirs(*, base: Path, axon_dirs: list[Path], keep_anc_dirs: set[Path]) -> int:
    """Remove empty dirs in repeated passes until stable; returns count removed."""
    removed_total = 0
    # Cascading empties can appear after removing a child directory.
    while True:
        empty = _find_empty_dirs_to_remove(base=base, axon_dirs=axon_dirs, keep_anc_dirs=keep_anc_dirs)
        if not empty:
            break
        for d in empty:
            try:
                os.rmdir(d)
                removed_total += 1
            except FileNotFoundError:
                continue
            except OSError:
                # Not empty anymore, permission issue, etc.
                continue
    return removed_total


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete everything except AxonTracking trees")
    parser.add_argument(
        "--base",
        type=Path,
        default=Path(DEFAULT_BASE),
        help=f"Base directory to prune (default: {DEFAULT_BASE})",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete files/dirs (default is dry-run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted (default behavior)",
    )
    parser.add_argument(
        "--limit-print",
        type=int,
        default=200,
        help="Limit number of delete actions printed (default: 200)",
    )
    parser.add_argument(
        "--require-axontracking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort if no AxonTracking directories are found (default: true)",
    )
    parser.add_argument(
        "--verify-after",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After --execute, scan and report any remaining paths that should have been deleted (default: true)",
    )

    args = parser.parse_args()

    base: Path = args.base.expanduser().absolute()
    if not base.exists():
        print(f"ERROR: base does not exist: {base}", file=sys.stderr)
        return 2
    if not base.is_dir():
        print(f"ERROR: base is not a directory: {base}", file=sys.stderr)
        return 2

    do_execute = bool(args.execute)
    do_dry = bool(args.dry_run) or not do_execute

    axon_dirs = _find_axontracking_dirs(base)
    if not axon_dirs:
        msg = f"No AxonTracking directories found under: {base}"
        if bool(args.require_axontracking):
            print(f"ERROR: {msg} (aborting)", file=sys.stderr)
            return 3
        print(f"WARN: {msg}; nothing to do")
        return 0

    keep_anc_dirs = _build_keep_ancestor_dirs(base=base, axon_dirs=axon_dirs)

    print(f"Base: {base}")
    print(f"AxonTracking dirs found: {len(axon_dirs)}")
    if do_dry:
        print("Mode: DRY RUN (no deletions will occur)")
    else:
        print("Mode: EXECUTE (deletions will occur)")

    # Walk bottom-up so deleting directories is safe.
    delete_files: list[Path] = []
    delete_dirs: list[Path] = []

    for root, dirnames, filenames in os.walk(base, topdown=False, followlinks=False):
        root_p = Path(root).absolute()

        # Files: keep only if inside any AxonTracking dir.
        for fn in filenames:
            p = (root_p / fn).absolute()
            if _inside_any_axontracking(p, axon_dirs):
                continue
            delete_files.append(p)

        # Directories: keep if (a) ancestor scaffolding, or (b) inside any AxonTracking dir.
        for dn in dirnames:
            p = (root_p / dn).absolute()
            if p in keep_anc_dirs:
                continue
            if _inside_any_axontracking(p, axon_dirs):
                continue
            delete_dirs.append(p)

    # Sort deepest-first for nicer output and safe deletes.
    delete_files.sort(key=lambda p: len(p.parts), reverse=True)
    delete_dirs.sort(key=lambda p: len(p.parts), reverse=True)

    total_actions = len(delete_files) + len(delete_dirs)
    print(f"Will delete: {len(delete_files)} files, {len(delete_dirs)} dirs (total actions: {total_actions})")

    limit = max(0, int(args.limit_print))
    if limit > 0:
        printed = 0
        for p in delete_files:
            if printed >= limit:
                break
            print(f"DEL file: {_format_path(p)}")
            printed += 1
        for p in delete_dirs:
            if printed >= limit:
                break
            print(f"DEL dir : {_format_path(p)}")
            printed += 1
        if total_actions > limit:
            print(f"... (suppressed {total_actions - limit} more delete actions; increase --limit-print to see all)")

    if do_dry:
        empty = _find_empty_dirs_to_remove(base=base, axon_dirs=axon_dirs, keep_anc_dirs=keep_anc_dirs)
        print(f"Will also remove empty dirs after prune: {len(empty)}")
        if limit > 0 and empty:
            shown = 0
            for d in empty:
                if shown >= limit:
                    break
                print(f"RMDIR empty: {_format_path(d)}")
                shown += 1
            if len(empty) > limit:
                print(
                    f"... (suppressed {len(empty) - limit} more empty dirs; increase --limit-print to see all)"
                )
        return 0

    # Execute deletions.
    # Files first, then dirs.
    errors = 0
    error_samples: list[str] = []

    for p in delete_files:
        try:
            # Remove symlinks as symlinks; unlink is fine for both files and symlinks.
            _unlink_with_retry(p)
        except FileNotFoundError:
            continue
        except IsADirectoryError:
            # In rare cases (e.g. race / unusual FS), fall back.
            try:
                _rmtree_with_retry(p)
            except Exception:
                errors += 1
                if len(error_samples) < 50:
                    error_samples.append(f"file-as-dir delete failed: {_format_path(p)}")
        except Exception:
            errors += 1
            if len(error_samples) < 50:
                error_samples.append(f"file delete failed: {_format_path(p)}")

    for p in delete_dirs:
        try:
            _rmtree_with_retry(p)
        except FileNotFoundError:
            continue
        except Exception:
            errors += 1
            if len(error_samples) < 50:
                error_samples.append(f"dir delete failed: {_format_path(p)}")

    # Final cleanup: remove any empty directories left behind.
    try:
        removed_empty = _remove_empty_dirs(base=base, axon_dirs=axon_dirs, keep_anc_dirs=keep_anc_dirs)
        if removed_empty:
            print(f"Removed empty dirs: {removed_empty}")
    except Exception:
        print("WARN: empty-dir cleanup failed; continuing.", file=sys.stderr)

    # Verify that nothing outside the keep rules remains.
    if bool(args.verify_after):
        bad_files, bad_dirs = _list_survivors(base=base, axon_dirs=axon_dirs, keep_anc_dirs=keep_anc_dirs)
        if bad_files or bad_dirs:
            print(
                f"VERIFY FAILED: still present (should be deleted): {len(bad_files)} files, {len(bad_dirs)} dirs",
                file=sys.stderr,
            )
            limit = max(0, int(args.limit_print))
            shown = 0
            for p in bad_files:
                if shown >= limit:
                    break
                print(f"SURVIVOR file: {_format_path(p)}", file=sys.stderr)
                shown += 1
            for p in bad_dirs:
                if shown >= limit:
                    break
                print(f"SURVIVOR dir : {_format_path(p)}", file=sys.stderr)
                shown += 1
            if (len(bad_files) + len(bad_dirs)) > limit and limit > 0:
                print(
                    f"... (suppressed {(len(bad_files) + len(bad_dirs)) - limit} more survivors; increase --limit-print)",
                    file=sys.stderr,
                )
            return 5

    if errors:
        print(f"Completed with {errors} deletion errors (see filesystem permissions / mounts).", file=sys.stderr)
        for s in error_samples:
            print(f"ERROR: {s}", file=sys.stderr)
        return 4

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
