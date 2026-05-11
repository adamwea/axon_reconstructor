"""Pre-merge analyzer cache for merge_SLAy comparability plotting.

Snapshots the values merge_SLAy plotting needs (templates, unit_locations,
per-unit spike counts) once, before SLAy runs, so the post-merge values can
be derived from ``automerge/new2old.json`` without building a second
analyzer. Keeping the pre-merge spike selection as the authoritative source
guarantees that pre- and post-merge plots reflect the same underlying
spikes — a merged unit's template is the weighted average of its
constituents, exactly what SI would compute if random_spikes had selected
the union upfront.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

CACHE_FORMAT_VERSION = "1"
TEMPLATES_FILENAME = "templates.npz"
META_FILENAME = "meta.json"


@dataclass
class PreMergeCache:
	unit_ids: np.ndarray
	templates: np.ndarray
	unit_locations: np.ndarray
	spike_counts: np.ndarray
	channel_ids: np.ndarray
	channel_locations: np.ndarray
	ms_before: float
	ms_after: float
	sampling_frequency: float

	@property
	def n_units(self) -> int:
		return int(self.unit_ids.shape[0])

	@property
	def n_samples(self) -> int:
		return int(self.templates.shape[1])

	@property
	def n_channels(self) -> int:
		return int(self.templates.shape[2])

	def index_by_unit_id(self) -> dict[str, int]:
		# new2old.json keys arrive as JSON strings; normalize to strings on
		# both sides so int64 / numpy unit ids round-trip cleanly.
		return {str(uid): int(i) for i, uid in enumerate(self.unit_ids.tolist())}


def write_pre_merge_cache(
	*,
	analyzer: Any,
	cache_dir: Path,
) -> dict[str, Any]:
	cache_dir = Path(cache_dir)
	cache_dir.mkdir(parents=True, exist_ok=True)

	templates_ext = analyzer.get_extension("templates")
	if templates_ext is None:
		raise RuntimeError("pre_merge_cache: analyzer is missing the templates extension")
	unit_locations_ext = analyzer.get_extension("unit_locations")
	if unit_locations_ext is None:
		raise RuntimeError("pre_merge_cache: analyzer is missing the unit_locations extension")

	templates = np.asarray(templates_ext.get_templates(), dtype=np.float32)
	unit_locations = np.asarray(unit_locations_ext.get_data(), dtype=np.float32)
	unit_ids = np.asarray(analyzer.unit_ids)
	channel_ids = np.asarray(analyzer.channel_ids)
	channel_locations = np.asarray(analyzer.get_channel_locations(), dtype=np.float32)

	counts_dict = analyzer.sorting.count_num_spikes_per_unit()
	spike_counts = np.asarray(
		[int(counts_dict.get(uid, 0)) for uid in unit_ids.tolist()],
		dtype=np.int64,
	)

	templates_params = getattr(templates_ext, "params", None) or {}
	ms_before = float(templates_params.get("ms_before", 0.0))
	ms_after = float(templates_params.get("ms_after", 0.0))
	sampling_frequency = float(analyzer.sampling_frequency)

	templates_path = cache_dir / TEMPLATES_FILENAME
	np.savez(
		templates_path,
		unit_ids=unit_ids,
		templates=templates,
		unit_locations=unit_locations,
		spike_counts=spike_counts,
		channel_ids=channel_ids,
		channel_locations=channel_locations,
	)

	meta_path = cache_dir / META_FILENAME
	meta_path.write_text(
		json.dumps(
			{
				"format_version": CACHE_FORMAT_VERSION,
				"ms_before": ms_before,
				"ms_after": ms_after,
				"sampling_frequency": sampling_frequency,
				"n_units": int(unit_ids.shape[0]),
				"n_samples": int(templates.shape[1]),
				"n_channels": int(templates.shape[2]),
				"written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
			},
			indent=2,
			sort_keys=True,
		)
		+ "\n",
		encoding="utf-8",
	)

	return {
		"cache_dir": str(cache_dir),
		"templates_path": str(templates_path),
		"meta_path": str(meta_path),
		"n_units": int(unit_ids.shape[0]),
		"n_samples": int(templates.shape[1]),
		"n_channels": int(templates.shape[2]),
	}


def read_pre_merge_cache(cache_dir: Path) -> PreMergeCache:
	cache_dir = Path(cache_dir)
	meta_path = cache_dir / META_FILENAME
	templates_path = cache_dir / TEMPLATES_FILENAME
	if not meta_path.exists():
		raise FileNotFoundError(f"pre_merge_cache: meta missing at {meta_path}")
	if not templates_path.exists():
		raise FileNotFoundError(f"pre_merge_cache: templates missing at {templates_path}")

	meta = json.loads(meta_path.read_text(encoding="utf-8"))
	version = str(meta.get("format_version", ""))
	if version != CACHE_FORMAT_VERSION:
		raise RuntimeError(
			f"pre_merge_cache: unsupported format_version={version!r}, expected {CACHE_FORMAT_VERSION!r}"
		)

	npz = np.load(templates_path)
	return PreMergeCache(
		unit_ids=np.asarray(npz["unit_ids"]),
		templates=np.asarray(npz["templates"], dtype=np.float32),
		unit_locations=np.asarray(npz["unit_locations"], dtype=np.float32),
		spike_counts=np.asarray(npz["spike_counts"], dtype=np.int64),
		channel_ids=np.asarray(npz["channel_ids"]),
		channel_locations=np.asarray(npz["channel_locations"], dtype=np.float32),
		ms_before=float(meta["ms_before"]),
		ms_after=float(meta["ms_after"]),
		sampling_frequency=float(meta["sampling_frequency"]),
	)
