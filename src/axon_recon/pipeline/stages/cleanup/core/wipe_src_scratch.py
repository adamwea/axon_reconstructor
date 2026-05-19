from __future__ import annotations

from pathlib import Path
from typing import Any


def candidate_wipe_src_scratch_paths(*, h5_path: Path, copied_to_scratch: bool) -> list[Path]:
	if not bool(copied_to_scratch):
		return []
	resolved_h5_path = Path(h5_path).expanduser()
	out: list[Path] = [resolved_h5_path]
	try:
		out.extend(sorted(resolved_h5_path.parent.glob("*.cfg")))
	except Exception:
		pass
	unique: list[Path] = []
	seen: set[str] = set()
	for path in out:
		key = str(path)
		if key in seen:
			continue
		seen.add(key)
		unique.append(path)
	return unique


def run_wipe_src_scratch_core(
	*,
	h5_path: Path,
	source_h5_path: Path,
	copied_to_scratch: bool,
	dry_run: bool,
	requires_use_scratch_root: bool,
	active_shared_users_remaining: int,
) -> dict[str, Any]:
	payload: dict[str, Any] = {
		"phase": "wipe_src_scratch",
		"source_h5_path": str(Path(source_h5_path).expanduser()),
		"resolved_h5_path": str(Path(h5_path).expanduser()),
		"copied_to_scratch": bool(copied_to_scratch),
		"dry_run": bool(dry_run),
		"requires_use_scratch_root": bool(requires_use_scratch_root),
		"active_shared_users_remaining": int(active_shared_users_remaining),
	}
	if not bool(copied_to_scratch):
		payload.update(
			{
				"status": "skipped",
				"reason": "selected_target_did_not_use_scratch_input_root",
				"removed_paths": [],
				"would_remove_paths": [],
				"missing_paths": [],
			}
		)
		return payload
	if int(active_shared_users_remaining) > 0:
		payload.update(
			{
				"status": "deferred",
				"reason": "shared_scratch_input_still_in_use",
				"removed_paths": [],
				"would_remove_paths": [],
				"missing_paths": [],
			}
		)
		return payload

	removed_paths: list[str] = []
	would_remove_paths: list[str] = []
	missing_paths: list[str] = []
	errors: list[str] = []
	for path in candidate_wipe_src_scratch_paths(h5_path=Path(h5_path), copied_to_scratch=bool(copied_to_scratch)):
		try:
			if not path.exists() and not path.is_symlink():
				missing_paths.append(str(path))
				continue
			if bool(dry_run):
				would_remove_paths.append(str(path))
				continue
			path.unlink()
			removed_paths.append(str(path))
		except FileNotFoundError:
			missing_paths.append(str(path))
		except Exception as exc:
			errors.append(f"{path}: {type(exc).__name__}: {exc}")
	payload["removed_paths"] = list(removed_paths)
	payload["would_remove_paths"] = list(would_remove_paths)
	payload["missing_paths"] = list(missing_paths)
	if errors:
		raise RuntimeError("wipe_src_scratch failed: " + "; ".join(errors))
	if bool(dry_run):
		payload["status"] = "dry_run" if would_remove_paths else "already_missing"
		return payload
	payload["status"] = "ok" if removed_paths else "already_missing"
	return payload