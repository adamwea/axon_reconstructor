from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..io import as_float_list, as_int_list, as_list
from ..models.inputs import ReconstructionBranchColorsConfig


@dataclass(frozen=True)
class ReconstructBranchRecord:
	branch_id: int
	branch_index: int
	label: Any
	selected_channels: tuple[int, ...]
	color: str
	scope: str
	velocity: float | None = None
	offset: float | None = None
	r2: float | None = None
	peak_times: tuple[float, ...] = ()
	distances: tuple[float, ...] = ()

	def as_payload(self) -> dict[str, Any]:
		return {
			"branch_index": int(self.branch_index),
			"channels": [int(ch) for ch in self.selected_channels],
			"label": self.label,
			"color": self.color,
		}


@dataclass(frozen=True)
class ReconstructBranchSelection:
	records: tuple[ReconstructBranchRecord, ...]
	source_name: str
	source_collection_class: str
	selected_branch_class: str
	raw_branch_count: int
	clean_branch_count: int
	clean_path_count: int

	def as_payload(self) -> list[dict[str, Any]]:
		return [record.as_payload() for record in self.records]


def format_branch_short_label(branch_like: Any) -> str:
	label = getattr(branch_like, "label", branch_like)
	try:
		return f"b{int(label)}"
	except Exception:
		text = str(label).strip()
		if not text:
			return "b?"
		if text.lower().startswith("b"):
			return text
		return f"b{text}"


def _branch_item_class_name(branches: Any) -> str:
	items = as_list(branches)
	if len(items) == 0:
		return "None"
	return str(type(items[0]).__name__)


def _preferred_branch_ids(branch_like: Any) -> list[int]:
	if not isinstance(branch_like, dict):
		return []
	for key in ("electrode_ids", "channels", "node_indices", "nodes"):
		vals = as_int_list(branch_like.get(key, []))
		if vals:
			return vals
	return []


def _normalize_branch_color(color_like: Any) -> str | None:
	if color_like is None:
		return None
	try:
		from matplotlib import colors as mcolors  # type: ignore[import-not-found]

		return str(mcolors.to_hex(color_like, keep_alpha=False))
	except Exception:
		text = str(color_like).strip()
		return text or None


def _resolve_branch_palette(count: int, branch_colors: ReconstructionBranchColorsConfig) -> list[str]:
	if count <= 0:
		return []
	try:
		from matplotlib import colormaps  # type: ignore[import-not-found]
		from matplotlib import colors as mcolors  # type: ignore[import-not-found]

		cmap = colormaps.get_cmap(str(branch_colors.color_scheme or "tab20"))
		if not bool(branch_colors.unique_color_per_branch):
			color = str(mcolors.to_hex(cmap(0.0), keep_alpha=False))
			return [color for _ in range(count)]
		den = max(1, count - 1)
		return [str(mcolors.to_hex(cmap(float(idx) / float(den)), keep_alpha=False)) for idx in range(count)]
	except Exception:
		return ["#1f77b4" for _ in range(count)]


def _raw_branch_records(gtr: Any) -> list[dict[str, Any]]:
	raw_paths = as_list(getattr(gtr, "_paths_raw", None))
	out: list[dict[str, Any]] = []
	for raw_idx, raw_path in enumerate(raw_paths):
		try:
			channels = tuple(int(x) for x in list(raw_path)[::-1])
		except Exception:
			continue
		if len(channels) < 2:
			continue
		out.append(
			{
				"branch_id": int(raw_idx),
				"branch_index": int(raw_idx),
				"label": int(raw_idx),
				"selected_channels": channels,
				"scope": "raw",
			}
		)
	return out


def _clean_branch_records(gtr: Any) -> list[dict[str, Any]]:
	out: list[dict[str, Any]] = []
	for idx, branch in enumerate(as_list(getattr(gtr, "branches", None))):
		if not isinstance(branch, dict):
			continue
		channels = tuple(_preferred_branch_ids(branch))
		if len(channels) < 2:
			continue
		branch_id = int(branch.get("branch_index", idx))
		out.append(
			{
				"branch_id": branch_id,
				"branch_index": branch_id,
				"label": branch.get("branch_index", idx),
				"selected_channels": channels,
				"scope": "clean",
				"color": branch.get("color", None),
				"velocity": branch.get("velocity", None),
				"offset": branch.get("offset", None),
				"r2": branch.get("r2", None),
				"peak_times": tuple(as_float_list(branch.get("peak_times", []))),
				"distances": tuple(as_float_list(branch.get("distances", []))),
			}
		)
	return out


def _clean_path_records(gtr: Any) -> list[dict[str, Any]]:
	out: list[dict[str, Any]] = []
	for idx, path in enumerate(as_list(getattr(gtr, "_paths_clean", None))):
		try:
			channels = tuple(int(x) for x in list(path))
		except Exception:
			continue
		if len(channels) < 2:
			continue
		out.append(
			{
				"branch_id": int(idx),
				"branch_index": int(idx),
				"label": int(idx),
				"selected_channels": channels,
				"scope": "clean",
			}
		)
	return out


def select_reconstruct_branch_records(
	*,
	gtr: Any,
	branch_scope: str,
	branch_colors: ReconstructionBranchColorsConfig,
) -> ReconstructBranchSelection:
	raw_paths = as_list(getattr(gtr, "_paths_raw", None))
	clean_branch_records = as_list(getattr(gtr, "branches", None))
	clean_paths = as_list(getattr(gtr, "_paths_clean", None))

	scope = str(branch_scope or "raw").strip().lower()
	if scope == "clean":
		candidates = _clean_branch_records(gtr)
		source_name = "gtr.branches"
		source_collection = getattr(gtr, "branches", None)
		if len(candidates) == 0 and len(clean_paths) > 0:
			candidates = _clean_path_records(gtr)
			source_name = "gtr._paths_clean"
			source_collection = getattr(gtr, "_paths_clean", None)
	else:
		candidates = _raw_branch_records(gtr)
		source_name = "gtr._paths_raw"
		source_collection = getattr(gtr, "_paths_raw", None)

	palette = _resolve_branch_palette(len(candidates), branch_colors)
	records: list[ReconstructBranchRecord] = []
	for idx, candidate in enumerate(candidates):
		color = _normalize_branch_color(candidate.get("color", None))
		if color is None:
			color = palette[idx] if idx < len(palette) else "#1f77b4"
		records.append(
			ReconstructBranchRecord(
				branch_id=int(candidate.get("branch_id", idx)),
				branch_index=int(candidate.get("branch_index", idx)),
				label=candidate.get("label", candidate.get("branch_index", idx)),
				selected_channels=tuple(int(ch) for ch in candidate.get("selected_channels", ())),
				color=str(color),
				scope=str(candidate.get("scope", scope) or scope),
				velocity=(None if candidate.get("velocity", None) is None else float(candidate.get("velocity"))),
				offset=(None if candidate.get("offset", None) is None else float(candidate.get("offset"))),
				r2=(None if candidate.get("r2", None) is None else float(candidate.get("r2"))),
				peak_times=tuple(float(v) for v in candidate.get("peak_times", ())),
				distances=tuple(float(v) for v in candidate.get("distances", ())),
			)
		)

	return ReconstructBranchSelection(
		records=tuple(records),
		source_name=source_name,
		source_collection_class=str(type(source_collection).__name__),
		selected_branch_class=_branch_item_class_name(source_collection),
		raw_branch_count=len(raw_paths),
		clean_branch_count=len(clean_branch_records),
		clean_path_count=len(clean_paths),
	)


def selection_to_branch_payload(selection: ReconstructBranchSelection) -> list[dict[str, Any]]:
	return selection.as_payload()


__all__ = [
	"ReconstructBranchRecord",
	"ReconstructBranchSelection",
	"select_reconstruct_branch_records",
	"selection_to_branch_payload",
]
