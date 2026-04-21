from __future__ import annotations

from pathlib import Path
from typing import Any


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