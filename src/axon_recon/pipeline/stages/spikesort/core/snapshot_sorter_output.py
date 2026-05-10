from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SNAPSHOT_SUMMARY_FILENAME = "snapshot_summary.json"


def _iter_files(root: Path) -> list[Path]:
	root = Path(root)
	return sorted(p for p in root.rglob("*") if p.is_file())


def _hash_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
	h = hashlib.sha256()
	with path.open("rb") as fh:
		for chunk in iter(lambda: fh.read(chunk_size), b""):
			h.update(chunk)
	return h.hexdigest()


def hash_directory(root: Path) -> dict[str, str]:
	"""Return {posix_relative_path: sha256_hex} for every file under root."""
	root = Path(root).resolve()
	if not root.exists():
		return {}
	mapping: dict[str, str] = {}
	for p in _iter_files(root):
		rel = p.resolve().relative_to(root).as_posix()
		mapping[rel] = _hash_file(p)
	return dict(sorted(mapping.items()))


def _summary_payload(
	*,
	source_dir: Path,
	snapshot_dir: Path,
	files: list[Path],
) -> dict[str, Any]:
	total_bytes = sum(p.stat().st_size for p in files)
	return {
		"source_dir": str(source_dir),
		"snapshot_dir": str(snapshot_dir),
		"file_count": len(files),
		"total_bytes": int(total_bytes),
		"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
	}


def run_snapshot_sorter_output_phase(
	*,
	sorter_output_dir: Path,
	snapshot_dir: Path,
	skip_if_exists: bool,
	logger: logging.Logger | None = None,
) -> dict[str, Any]:
	"""Recursively snapshot ``sorter_output_dir`` to ``snapshot_dir``.

	Idempotent when ``skip_if_exists`` is True and a prior summary is present.
	Returns a dict with ``status`` ("ok" | "skipped"), ``snapshot_dir``,
	``source_dir``, ``file_count``, ``total_bytes``, and ``summary_json``.
	"""
	source_dir = Path(sorter_output_dir).resolve()
	target_dir = Path(snapshot_dir).resolve()
	summary_json_path = target_dir / SNAPSHOT_SUMMARY_FILENAME

	if not source_dir.exists() or not source_dir.is_dir():
		raise FileNotFoundError(
			f"snapshot_sorter_output: source sorter_output dir not found: {source_dir}"
		)

	if bool(skip_if_exists) and summary_json_path.exists():
		try:
			existing = json.loads(summary_json_path.read_text(encoding="utf-8"))
		except Exception:
			existing = {}
		if logger is not None:
			logger.info(
				"snapshot_sorter_output: existing snapshot found; skipping (snapshot_dir=%s)",
				str(target_dir),
			)
		return {
			"status": "skipped",
			"reason": "snapshot_exists",
			"source_dir": str(source_dir),
			"snapshot_dir": str(target_dir),
			"file_count": int(existing.get("file_count", 0) or 0),
			"total_bytes": int(existing.get("total_bytes", 0) or 0),
			"summary_json": str(summary_json_path),
		}

	if target_dir.exists():
		if target_dir.is_dir():
			shutil.rmtree(target_dir, ignore_errors=False)
		else:
			target_dir.unlink()

	if logger is not None:
		logger.info(
			"snapshot_sorter_output: copying %s -> %s",
			str(source_dir),
			str(target_dir),
		)
	shutil.copytree(source_dir, target_dir)

	files = _iter_files(target_dir)
	payload = _summary_payload(source_dir=source_dir, snapshot_dir=target_dir, files=files)
	summary_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
	return {
		"status": "ok",
		"source_dir": str(source_dir),
		"snapshot_dir": str(target_dir),
		"file_count": int(payload["file_count"]),
		"total_bytes": int(payload["total_bytes"]),
		"summary_json": str(summary_json_path),
	}


def run_restore_sorter_output_from_snapshot(
	*,
	snapshot_dir: Path,
	sorter_output_dir: Path,
	logger: logging.Logger | None = None,
) -> dict[str, Any]:
	"""Restore ``sorter_output_dir`` from a prior snapshot directory.

	Removes the existing canonical sorter_output (if any) and copies the
	snapshot in its place. The ``snapshot_summary.json`` file at the snapshot
	root is excluded from the restore so the canonical tree stays clean.
	"""
	src = Path(snapshot_dir).resolve()
	dst = Path(sorter_output_dir).resolve()
	summary_json_path = src / SNAPSHOT_SUMMARY_FILENAME

	if not src.exists() or not src.is_dir():
		raise FileNotFoundError(
			f"restore_sorter_output: snapshot dir not found: {src}"
		)
	if not summary_json_path.exists():
		raise FileNotFoundError(
			f"restore_sorter_output: snapshot summary missing at {summary_json_path}; "
			"refusing to restore from an unverified snapshot directory"
		)

	if dst.exists():
		if dst.is_dir():
			shutil.rmtree(dst, ignore_errors=False)
		else:
			dst.unlink()

	if logger is not None:
		logger.info(
			"restore_sorter_output: copying %s -> %s",
			str(src),
			str(dst),
		)
	shutil.copytree(src, dst, ignore=shutil.ignore_patterns(SNAPSHOT_SUMMARY_FILENAME))

	files = _iter_files(dst)
	total_bytes = sum(p.stat().st_size for p in files)
	return {
		"status": "ok",
		"source_dir": str(src),
		"sorter_output_dir": str(dst),
		"file_count": int(len(files)),
		"total_bytes": int(total_bytes),
	}
