from __future__ import annotations

import logging
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable


LOGGER = logging.getLogger("axon_recon.init.copy_src_to_scratch")

_SCRATCH_COPY_CHUNK_BYTES = 16 * 1024 * 1024
_SCRATCH_COPY_PROGRESS_BAR_WIDTH = 28
_SCRATCH_COPY_PROGRESS_MAX_UPDATES = 40
_SCRATCH_COPY_PROGRESS_MIN_UPDATE_BYTES = 128 * 1024 * 1024
_SCRATCH_COPY_PROGRESS_MIN_UPDATE_SECONDS = 2.0


def _relative_input_tree_path(source_path: Path) -> Path:
	parts = list(source_path.parts)
	lower_parts = [str(p).lower() for p in parts]
	anchor_indexes = [idx for idx, token in enumerate(lower_parts) if token == "raw_data"]
	if anchor_indexes:
		anchor_idx = anchor_indexes[-1]
		rel_parts = parts[anchor_idx + 1 :]
		if rel_parts:
			return Path(*rel_parts)

	if source_path.is_absolute():
		rel_parts = source_path.parts[1:]
		if rel_parts:
			return Path(*rel_parts)

	return Path(source_path.name)


def _copied_file_is_current(*, src_stat: os.stat_result, dst_stat: os.stat_result) -> bool:
	if int(src_stat.st_size) != int(dst_stat.st_size):
		return False
	try:
		if int(src_stat.st_mtime_ns) == int(dst_stat.st_mtime_ns):
			return True
	except Exception:
		pass
	return int(src_stat.st_mtime) == int(dst_stat.st_mtime)


def _copy_file_if_needed(
	*,
	src: Path,
	dst: Path,
	progress_callback: Callable[[int, int], None] | None = None,
	chunk_bytes: int = _SCRATCH_COPY_CHUNK_BYTES,
) -> bool:
	dst.parent.mkdir(parents=True, exist_ok=True)
	src_stat = src.stat()
	if dst.exists():
		try:
			dst_stat = dst.stat()
			if _copied_file_is_current(src_stat=src_stat, dst_stat=dst_stat):
				return False
		except Exception:
			pass
		try:
			dst.chmod(dst.stat().st_mode | 0o200)
		except Exception:
			pass
		dst.unlink()
	total_bytes = max(0, int(src_stat.st_size))
	if progress_callback is None:
		shutil.copyfile(src, dst)
	else:
		copied_bytes = 0
		buffer_size = max(1, int(chunk_bytes))
		with src.open("rb") as src_handle, dst.open("wb") as dst_handle:
			while True:
				chunk = src_handle.read(buffer_size)
				if not chunk:
					break
				dst_handle.write(chunk)
				copied_bytes += len(chunk)
				progress_callback(int(copied_bytes), int(total_bytes))
	os.utime(dst, ns=(src_stat.st_atime_ns, src_stat.st_mtime_ns))
	return True


def _render_copy_progress_bar(*, completed: int, total: int, width: int = 24) -> str:
	total_safe = max(1, int(total))
	completed_safe = min(max(0, int(completed)), total_safe)
	bar_width = max(8, int(width))
	filled = int(round((float(completed_safe) / float(total_safe)) * float(bar_width)))
	filled = min(max(0, int(filled)), bar_width)
	return f"[{'#' * filled}{'-' * (bar_width - filled)}]"


def _format_copy_size(num_bytes: float) -> str:
	size = max(0.0, float(num_bytes))
	units = ("B", "KiB", "MiB", "GiB", "TiB")
	unit_index = 0
	while size >= 1024.0 and unit_index < (len(units) - 1):
		size /= 1024.0
		unit_index += 1
	if unit_index == 0:
		return f"{int(round(size))} {units[unit_index]}"
	return f"{size:.1f} {units[unit_index]}"


def _format_copy_rate(bytes_per_second: float) -> str:
	if not math.isfinite(bytes_per_second) or float(bytes_per_second) <= 0.0:
		return "0 B/s"
	return f"{_format_copy_size(float(bytes_per_second))}/s"


def _format_copy_eta(seconds: float | None) -> str:
	if seconds is None or not math.isfinite(seconds) or float(seconds) < 0.0:
		return "--:--"
	total_seconds = int(round(float(seconds)))
	hours, remainder = divmod(total_seconds, 3600)
	minutes, secs = divmod(remainder, 60)
	if hours > 0:
		return f"{hours:d}:{minutes:02d}:{secs:02d}"
	return f"{minutes:02d}:{secs:02d}"


def resolve_copy_src_to_scratch_input_path(
	*,
	source_h5_path: Path,
	scratch_input_root: Path | None,
	dataset_id: str,
	materialize_scratch_inputs: bool,
) -> Path:
	normalized_source_h5_path = source_h5_path.expanduser()
	if scratch_input_root is None:
		return normalized_source_h5_path
	if bool(materialize_scratch_inputs):
		return _materialize_dataset_input_in_scratch(
			source_h5_path=normalized_source_h5_path,
			scratch_input_root=scratch_input_root,
			dataset_id=dataset_id,
		)
	return _resolve_existing_dataset_input_in_scratch(
		source_h5_path=normalized_source_h5_path,
		scratch_input_root=scratch_input_root,
		dataset_id=dataset_id,
	) or normalized_source_h5_path


def _materialize_dataset_input_in_scratch(*, source_h5_path: Path, scratch_input_root: Path, dataset_id: str) -> Path:
	source_h5_path = source_h5_path.expanduser().resolve()
	scratch_input_root = scratch_input_root.expanduser().resolve()
	if not source_h5_path.exists():
		raise FileNotFoundError(f"Dataset input H5 not found: {source_h5_path}")

	LOGGER.info(
		"Scratch input materialization start dataset_id=%s source_h5=%s scratch_input_root=%s",
		dataset_id,
		source_h5_path,
		scratch_input_root,
	)

	rel_h5 = _relative_input_tree_path(source_h5_path)
	target_h5 = (scratch_input_root / rel_h5).resolve()
	cfg_paths = sorted(source_h5_path.parent.glob("*.cfg"))
	copy_plan: list[tuple[str, Path, Path]] = [("h5", source_h5_path, target_h5)]
	copy_plan.extend(("cfg", cfg_path.resolve(), (target_h5.parent / cfg_path.name)) for cfg_path in cfg_paths)
	total_files = int(len(copy_plan))

	LOGGER.info(
		"Scratch input copy plan dataset_id=%s total_files=%d",
		dataset_id,
		total_files,
	)

	def _needs_copy(src_path: Path, dst_path: Path) -> bool:
		if not dst_path.exists():
			return True
		try:
			return not _copied_file_is_current(src_stat=src_path.stat(), dst_stat=dst_path.stat())
		except Exception:
			return True

	pending_plan: list[tuple[str, Path, Path, int]] = []
	for file_kind, src_path, dst_path in copy_plan:
		if _needs_copy(src_path=src_path, dst_path=dst_path):
			pending_plan.append((file_kind, src_path, dst_path, int(src_path.stat().st_size)))

	copied_files = 0
	skipped_files = int(total_files - len(pending_plan))
	pending_total_bytes = int(sum(file_size for _file_kind, _src_path, _dst_path, file_size in pending_plan))
	if pending_plan:
		LOGGER.info(
			"Scratch input copy required dataset_id=%s files_to_copy=%d skipped_existing=%d total_bytes=%s",
			dataset_id,
			int(len(pending_plan)),
			int(skipped_files),
			_format_copy_size(float(pending_total_bytes)),
		)

	progress_started_at = time.monotonic()
	last_progress_logged_at = float(progress_started_at)
	last_progress_logged_bytes = 0
	overall_bytes_processed = 0
	progress_update_bytes = max(
		1,
		max(
			int(_SCRATCH_COPY_PROGRESS_MIN_UPDATE_BYTES),
			int(pending_total_bytes // max(1, int(_SCRATCH_COPY_PROGRESS_MAX_UPDATES))),
		),
	)

	def _log_copy_progress(
		*,
		overall_bytes_completed: int,
		current_file_index: int,
		current_file_kind: str,
		current_file_name: str,
		current_file_bytes_completed: int,
		current_file_total_bytes: int,
		force: bool = False,
	) -> None:
		nonlocal last_progress_logged_at, last_progress_logged_bytes
		total_safe = max(1, int(pending_total_bytes))
		overall_clamped = min(max(0, int(overall_bytes_completed)), total_safe)
		now = time.monotonic()
		bytes_since_last = int(overall_clamped - last_progress_logged_bytes)
		seconds_since_last = float(now - last_progress_logged_at)
		if not force:
			if bytes_since_last < int(progress_update_bytes) and seconds_since_last < float(_SCRATCH_COPY_PROGRESS_MIN_UPDATE_SECONDS):
				return
		elapsed_s = max(0.001, float(now - progress_started_at))
		pct = (100.0 * float(overall_clamped)) / float(total_safe)
		rate_bytes_per_s = float(overall_clamped) / float(elapsed_s)
		remaining_bytes = max(0, int(pending_total_bytes - overall_clamped))
		eta_s = (float(remaining_bytes) / float(rate_bytes_per_s)) if rate_bytes_per_s > 0.0 else None
		LOGGER.info(
			"Scratch input copy progress dataset_id=%s %s %.1f%% overall=%s/%s rate=%s eta=%s file=%d/%d kind=%s name=%s file_progress=%s/%s copied=%d skipped_existing=%d",
			dataset_id,
			_render_copy_progress_bar(
				completed=int(overall_clamped),
				total=int(total_safe),
				width=int(_SCRATCH_COPY_PROGRESS_BAR_WIDTH),
			),
			float(pct),
			_format_copy_size(float(overall_clamped)),
			_format_copy_size(float(total_safe)),
			_format_copy_rate(float(rate_bytes_per_s)),
			_format_copy_eta(eta_s),
			int(current_file_index),
			int(len(pending_plan)),
			str(current_file_kind),
			str(current_file_name),
			_format_copy_size(float(current_file_bytes_completed)),
			_format_copy_size(float(max(0, int(current_file_total_bytes)))),
			int(copied_files),
			int(skipped_files),
		)
		last_progress_logged_at = float(now)
		last_progress_logged_bytes = int(overall_clamped)

	for file_idx, (file_kind, src_path, dst_path, file_size_bytes) in enumerate(pending_plan, start=1):
		file_size_display = _format_copy_size(float(file_size_bytes))
		if str(file_kind) == "h5":
			LOGGER.info(
				"Scratch input H5 copy start dataset_id=%s src=%s dst=%s size=%s file=%d/%d",
				dataset_id,
				src_path,
				dst_path,
				file_size_display,
				int(file_idx),
				int(len(pending_plan)),
			)
			_log_copy_progress(
				overall_bytes_completed=int(overall_bytes_processed),
				current_file_index=int(file_idx),
				current_file_kind=str(file_kind),
				current_file_name=str(src_path.name),
				current_file_bytes_completed=0,
				current_file_total_bytes=int(file_size_bytes),
				force=True,
			)

		def _progress_callback(copied_bytes: int, total_bytes: int) -> None:
			_log_copy_progress(
				overall_bytes_completed=int(overall_bytes_processed + copied_bytes),
				current_file_index=int(file_idx),
				current_file_kind=str(file_kind),
				current_file_name=str(src_path.name),
				current_file_bytes_completed=int(copied_bytes),
				current_file_total_bytes=int(total_bytes),
				force=(int(copied_bytes) >= int(total_bytes)),
			)

		was_copied = _copy_file_if_needed(
			src=src_path,
			dst=dst_path,
			progress_callback=(_progress_callback if str(file_kind) == "h5" else None),
		)
		if was_copied:
			copied_files += 1
		else:
			skipped_files += 1
		overall_bytes_processed += int(max(0, file_size_bytes))

		if str(file_kind) == "h5":
			LOGGER.info(
				"Scratch input H5 copy complete dataset_id=%s dst=%s size=%s",
				dataset_id,
				dst_path,
				file_size_display,
			)

		if LOGGER.isEnabledFor(logging.DEBUG):
			LOGGER.debug(
				"Scratch input file action dataset_id=%s action=%s kind=%s src=%s dst=%s",
				dataset_id,
				("copied" if was_copied else "skipped"),
				str(file_kind),
				src_path,
				dst_path,
			)
		if str(file_kind) != "h5" or int(file_size_bytes) == 0 or int(file_idx) == int(len(pending_plan)):
			_log_copy_progress(
				overall_bytes_completed=int(overall_bytes_processed),
				current_file_index=int(file_idx),
				current_file_kind=str(file_kind),
				current_file_name=str(src_path.name),
				current_file_bytes_completed=int(file_size_bytes),
				current_file_total_bytes=int(file_size_bytes),
				force=(int(file_idx) == int(len(pending_plan))),
			)

	if int(copied_files) == 0:
		LOGGER.info(
			"Scratch input already materialized in scratch_inputs; skipping copy dataset_id=%s target_h5=%s total_files=%d",
			dataset_id,
			target_h5,
			int(total_files),
		)

	LOGGER.info(
		"Scratch input materialization complete dataset_id=%s target_h5=%s cfg_files=%d copied=%d skipped=%d",
		dataset_id,
		target_h5,
		int(len(cfg_paths)),
		int(copied_files),
		int(skipped_files),
	)

	return target_h5


def _resolve_existing_dataset_input_in_scratch(
	*,
	source_h5_path: Path,
	scratch_input_root: Path,
	dataset_id: str,
) -> Path | None:
	source_h5_path = source_h5_path.expanduser()
	scratch_input_root = scratch_input_root.expanduser()
	target_h5 = scratch_input_root / _relative_input_tree_path(source_h5_path)
	if not target_h5.exists():
		return None
	LOGGER.info(
		"Using existing scratch input dataset_id=%s source_h5=%s scratch_h5=%s",
		dataset_id,
		source_h5_path,
		target_h5,
	)
	return target_h5


def run_copy_src_to_scratch_core(
	*,
	h5_path: Path,
	source_h5_path: Path,
	stream_id: str,
	copied_to_scratch: bool,
	requires_use_scratch_root: bool,
) -> dict[str, Any]:
	_ = stream_id
	return {
		"phase": "copy_src_to_scratch",
		"source_h5_path": str(Path(source_h5_path).expanduser()),
		"resolved_h5_path": str(Path(h5_path).expanduser()),
		"copied_to_scratch": bool(copied_to_scratch),
		"requires_use_scratch_root": bool(requires_use_scratch_root),
	}