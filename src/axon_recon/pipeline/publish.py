from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Iterable


def map_active_path_to_final(*, path: Path, active_root: Path, final_root: Path) -> Path | None:
    src = Path(path).expanduser().resolve()
    active = Path(active_root).expanduser().resolve()
    final = Path(final_root).expanduser().resolve()
    try:
        rel = src.relative_to(active)
    except Exception:
        return None
    return final / rel


def remap_path_string_to_final(*, raw: str | Path, active_root: Path, final_root: Path) -> str:
    mapped = map_active_path_to_final(path=Path(raw), active_root=active_root, final_root=final_root)
    return str(mapped if mapped is not None else Path(raw))


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _sync_dir_with_rsync(*, src_dir: Path, dst_dir: Path) -> bool:
    if shutil.which("rsync") is None:
        return False

    dst_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync",
        "-a",
        "--delete-delay",
        "--partial",
        "--inplace",
        "--append-verify",
        f"{src_dir}/",
        f"{dst_dir}/",
    ]
    proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return int(proc.returncode) == 0


def _sync_file_with_rsync(*, src_file: Path, dst_file: Path) -> bool:
    if shutil.which("rsync") is None:
        return False

    _ensure_dir(dst_file)
    cmd = [
        "rsync",
        "-a",
        "--partial",
        "--inplace",
        "--append-verify",
        str(src_file),
        str(dst_file),
    ]
    proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return int(proc.returncode) == 0


def _prune_empty_dirs(*, root_dir: Path) -> None:
    if not root_dir.exists() or not root_dir.is_dir():
        return
    for candidate in sorted((p for p in root_dir.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)):
        try:
            candidate.rmdir()
        except OSError:
            continue


def publish_path_to_final(
    *,
    path: Path,
    active_root: Path,
    final_root: Path,
    mode: str = "copy",
) -> Path | None:
    src = Path(path).expanduser().resolve()
    mapped = map_active_path_to_final(path=src, active_root=active_root, final_root=final_root)
    if mapped is None:
        return None

    if not src.exists():
        return mapped

    mode_norm = str(mode or "copy").strip().lower()
    move_mode = mode_norm == "move"

    if src.is_dir():
        synced = _sync_dir_with_rsync(src_dir=src, dst_dir=mapped)
        if not synced:
            mapped.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, mapped, dirs_exist_ok=True)

        if move_mode:
            shutil.rmtree(src, ignore_errors=True)
            _prune_empty_dirs(root_dir=Path(active_root).expanduser().resolve())
        return mapped

    synced = _sync_file_with_rsync(src_file=src, dst_file=mapped)
    if not synced:
        _ensure_dir(mapped)
        shutil.copy2(src, mapped)

    if move_mode:
        src.unlink(missing_ok=True)
        _prune_empty_dirs(root_dir=Path(active_root).expanduser().resolve())
    return mapped


def _dedupe_parent_paths(paths: list[Path]) -> list[Path]:
    unique = sorted({p.resolve() for p in paths}, key=lambda p: len(p.parts))
    out: list[Path] = []
    for candidate in unique:
        if any(parent == candidate or parent in candidate.parents for parent in out):
            continue
        out.append(candidate)
    return out


def publish_paths_to_final(
    *,
    paths: Iterable[Path],
    active_root: Path,
    final_root: Path,
    mode: str = "copy",
) -> list[Path]:
    candidates: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        mapped = map_active_path_to_final(path=path, active_root=active_root, final_root=final_root)
        if mapped is None:
            continue
        candidates.append(path)

    published: list[Path] = []
    for candidate in _dedupe_parent_paths(candidates):
        dst = publish_path_to_final(
            path=candidate,
            active_root=active_root,
            final_root=final_root,
            mode=mode,
        )
        if dst is not None:
            published.append(dst)
    return published
