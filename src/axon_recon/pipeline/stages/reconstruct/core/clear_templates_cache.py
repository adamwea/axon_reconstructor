from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any


def _templates_out_candidates(
	well_out_dir: Path,
	templates_output_rel_root: str | None = None,
) -> tuple[Path, ...]:
	candidates: list[Path] = []
	if templates_output_rel_root:
		configured = well_out_dir / Path(str(templates_output_rel_root)).expanduser()
		candidates.append(configured)
	for candidate in (
		well_out_dir / "template_outputs",
		well_out_dir / "templates_outputs",
		well_out_dir / "stg4_templates_outputs",
	):
		if candidate not in candidates:
			candidates.append(candidate)
	return tuple(candidates)


def _resolve_templates_out_dir(well_out_dir: Path, templates_output_rel_root: str | None = None) -> Path:
	candidates = _templates_out_candidates(well_out_dir, templates_output_rel_root)
	for candidate in candidates:
		if (candidate / "cache").exists():
			return candidate
	for candidate in candidates:
		if candidate.exists():
			return candidate
	return candidates[0]


def _is_relative_to(path: Path, parent: Path) -> bool:
	try:
		path.relative_to(parent)
		return True
	except ValueError:
		return False


def _count_delete_payload(path: Path) -> tuple[int, int]:
	bytes_deleted = 0
	files_deleted = 0
	if path.is_file() or path.is_symlink():
		try:
			return int(path.lstat().st_size), 1
		except FileNotFoundError:
			return 0, 0
	if not path.exists():
		return 0, 0
	for item in path.rglob("*"):
		if not (item.is_file() or item.is_symlink()):
			continue
		try:
			bytes_deleted += int(item.lstat().st_size)
			files_deleted += 1
		except FileNotFoundError:
			continue
	return int(bytes_deleted), int(files_deleted)


def _delete_tree(path: Path) -> tuple[int, int]:
	bytes_deleted, files_deleted = _count_delete_payload(path)
	if path.is_dir() and not path.is_symlink():
		shutil.rmtree(path)
	elif path.exists() or path.is_symlink():
		path.unlink(missing_ok=True)
	return int(bytes_deleted), int(files_deleted)


def run_clear_templates_cache_phase(
	*,
	well_out_dir: Path,
	enabled: bool,
	keep_merged_per_unit_outputs: bool,
	keep_full_channels_templates: bool,
	templates_output_rel_root: str | None = None,
	logger: logging.Logger | None = None,
) -> dict[str, Any]:
	if not bool(enabled):
		return {"phase": "clear_templates_cache", "skipped": True, "reason": "disabled"}

	active_logger = logger or logging.getLogger("axon_recon.reconstruct.clear_templates_cache")
	templates_out_dir = _resolve_templates_out_dir(Path(well_out_dir), templates_output_rel_root)
	cache_dir = templates_out_dir / "cache"
	preserve_dirs: list[Path] = []
	if bool(keep_merged_per_unit_outputs):
		merged_dir = cache_dir / "templates" / "merged"
		if merged_dir.exists():
			preserve_dirs.append(merged_dir)
	if bool(keep_full_channels_templates):
		full_dir = cache_dir / "templates" / "full"
		if full_dir.exists():
			preserve_dirs.append(full_dir)

	preserve_dirs = [path.resolve() for path in preserve_dirs]
	paths_preserved = [str(path) for path in preserve_dirs]
	bytes_deleted = 0
	files_deleted = 0

	def _is_preserved(path: Path) -> bool:
		resolved = path.resolve()
		return any(_is_relative_to(resolved, preserved) for preserved in preserve_dirs)

	def _is_preserved_ancestor(path: Path) -> bool:
		resolved = path.resolve()
		return any(_is_relative_to(preserved, resolved) for preserved in preserve_dirs)

	def _delete_unpreserved(path: Path) -> None:
		nonlocal bytes_deleted, files_deleted
		if not path.exists() and not path.is_symlink():
			return
		if _is_preserved(path):
			return
		if path.is_dir() and not path.is_symlink() and _is_preserved_ancestor(path):
			for child in sorted(path.iterdir()):
				_delete_unpreserved(child)
			return
		deleted_bytes, deleted_files = _delete_tree(path)
		bytes_deleted += int(deleted_bytes)
		files_deleted += int(deleted_files)

	if cache_dir.exists():
		for child in sorted(cache_dir.iterdir()):
			_delete_unpreserved(child)
	else:
		active_logger.info("Templates cache directory not found: %s", str(cache_dir))

	active_logger.info(
		"Cleared templates cache: templates_out_dir=%s files_deleted=%d bytes_deleted=%d paths_preserved=%d",
		str(templates_out_dir),
		int(files_deleted),
		int(bytes_deleted),
		int(len(paths_preserved)),
	)
	return {
		"phase": "clear_templates_cache",
		"skipped": False,
		"templates_out_dir": str(templates_out_dir),
		"bytes_deleted": int(bytes_deleted),
		"files_deleted": int(files_deleted),
		"paths_preserved": paths_preserved,
	}


__all__ = ["run_clear_templates_cache_phase"]
