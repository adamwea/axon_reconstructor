"""Chip-well group discovery + per-session path resolution for
analysis.unitmatch.

The unitmatch phase operates at chip-well GROUP granularity rather than
per (dataset, well): each group bundles the same (chip_id, well_id)
recorded across multiple DIVs (or other dataset-axis variation) so
``unitlink.match`` can compute cross-session unit correspondences.

This module ships two pure functions:

- ``discover_chip_well_groups(data_cfg, well_metadata=None)`` —
  inspects the data config (or a pre-built well_metadata lookup from
  ``build_well_metadata_lookup``) and returns a stable
  ``{(chip_id, well_id): [dataset_index, ...]}`` mapping. Wells without
  a parseable chip_id are omitted; the caller can detect dropouts by
  comparing the dict's keys to the original well set.

- ``resolve_session_inputs(...)`` — for one (dataset_index, well_id)
  pair, returns the canonical filesystem paths that ``unitlink.match``
  needs:
    - ``sorter_output``: the kssynth-shaped folder at
      ``<well_out_dir>/recon_outputs/synth_sorter_output``.
    - ``analyzers``: the per-segment SortingAnalyzer roots at
      ``<well_out_dir>/recon_outputs/cache/analyzers/segments``.
  When ``require_exists=True`` (default), the function raises
  ``UnitmatchSessionInputMissing`` with actionable suggestions
  ("run reconstruct.kssynth first") if expected paths don't exist.

Slice 3 of unitmatch_phase_plan will call these from the phase
runner; slice 4 will reuse ``discover_chip_well_groups`` for
``--targets`` mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from axon_recon.runtime_config import RuntimeConfig

from ....output_paths import compute_mea_analysis_output_dir
from ..config import build_well_metadata_lookup


class UnitmatchSessionInputMissing(FileNotFoundError):
	"""Raised when a session's required unitmatch inputs aren't on disk."""


@dataclass(frozen=True)
class SessionInputs:
	"""Canonical paths the unitmatch phase needs for one (dataset, well)."""

	sorter_output: Path
	analyzers: Path
	well_out_dir: Path
	dataset_index: int
	well_id: str

	def as_dict(self) -> dict[str, Any]:
		return {
			"sorter_output": str(self.sorter_output),
			"analyzers": str(self.analyzers),
			"well_out_dir": str(self.well_out_dir),
			"dataset_index": int(self.dataset_index),
			"well_id": str(self.well_id),
		}


def discover_chip_well_groups(
	data_cfg: RuntimeConfig | dict[str, Any] | None = None,
	*,
	well_metadata: dict[tuple[int, str], dict[str, Any]] | None = None,
) -> dict[tuple[str, str], list[int]]:
	"""Group dataset indices by (chip_id, well_id).

	One of ``data_cfg`` or ``well_metadata`` must be supplied. When both
	are provided, ``well_metadata`` wins (the caller has presumably
	post-processed the data config and wants the function to use the
	curated lookup).

	Wells whose chip_id can't be parsed (the h5 path doesn't match the
	canonical ``<project>/<YYMMDD>/<chip>/<scan_type>/<run>/data.raw.h5``
	shape) are omitted from the result. The per-group dataset_index list
	is sorted ascending for deterministic downstream ordering.

	Returns
	-------
	``{(chip_id, well_id): [dataset_index, ...]}``. Empty when no
	parseable chip-well pairs are found.
	"""

	if well_metadata is None:
		if data_cfg is None:
			raise ValueError(
				"discover_chip_well_groups: supply either data_cfg or well_metadata"
			)
		runtime_obj = data_cfg if isinstance(data_cfg, RuntimeConfig) else RuntimeConfig(dict(data_cfg))
		well_metadata = build_well_metadata_lookup(runtime_obj)

	groups: dict[tuple[str, str], list[int]] = {}
	for (dataset_index, well_id), info in well_metadata.items():
		chip_id = info.get("chip_id") if isinstance(info, dict) else None
		if chip_id is None or str(chip_id).strip() == "":
			continue
		key = (str(chip_id), str(well_id))
		groups.setdefault(key, []).append(int(dataset_index))

	# Deterministic ordering.
	return {key: sorted(set(values)) for key, values in groups.items()}


def _recon_output_root(
	*,
	output_root: Path,
	h5_path: Path,
	well_id: str,
	recon_output_rel_root: str = "recon_outputs",
) -> Path:
	well_dir = compute_mea_analysis_output_dir(
		output_root=Path(output_root),
		data_file=Path(h5_path),
		well=str(well_id),
	)
	return well_dir / str(recon_output_rel_root)


def resolve_session_inputs(
	*,
	dataset_index: int,
	well_id: str,
	h5_path: str | Path,
	output_root: str | Path,
	recon_output_rel_root: str = "recon_outputs",
	synth_relpath: str = "synth_sorter_output",
	analyzers_relpath: str = "cache/analyzers/segments",
	require_exists: bool = True,
) -> SessionInputs:
	"""Resolve the canonical paths the unitmatch phase needs for one session.

	Arguments
	---------
	dataset_index, well_id, h5_path
	    Identify the session.
	output_root
	    The ``output_root`` from the data config (the analyzed_data root).
	recon_output_rel_root, synth_relpath, analyzers_relpath
	    Canonical relpaths under the well output dir. Defaults match
	    the kssynth + axon_recon recon-stage conventions.
	require_exists
	    When True (default), raise ``UnitmatchSessionInputMissing`` if
	    either the sorter_output or analyzers path is absent. When
	    False, return the resolved paths regardless — useful for the
	    slice-2 unit tests + dry-run-style debugging.
	"""

	recon_root = _recon_output_root(
		output_root=Path(output_root),
		h5_path=Path(h5_path),
		well_id=str(well_id),
		recon_output_rel_root=recon_output_rel_root,
	)
	well_out_dir = recon_root.parent
	sorter_output = (recon_root / synth_relpath).resolve()
	analyzers = (recon_root / analyzers_relpath).resolve()

	if require_exists:
		missing: list[str] = []
		suggestions: list[str] = []
		if not sorter_output.is_dir():
			missing.append(f"sorter_output: {sorter_output}")
			suggestions.append(
				"Run `axon-recon stages reconstruct.kssynth` first to produce the "
				"synthetic sorter_output folder this phase consumes."
			)
		if not analyzers.is_dir():
			missing.append(f"analyzers: {analyzers}")
			suggestions.append(
				"Run `axon-recon stages reconstruct.analyzers` to build the per-segment "
				"SortingAnalyzer cache."
			)
		if missing:
			suggestion_block = "\n  " + "\n  ".join(suggestions) if suggestions else ""
			raise UnitmatchSessionInputMissing(
				f"unitmatch: session (dataset={dataset_index}, well={well_id!r}) "
				f"is missing required inputs:\n  " + "\n  ".join(missing) + suggestion_block
			)

	return SessionInputs(
		sorter_output=sorter_output,
		analyzers=analyzers,
		well_out_dir=well_out_dir,
		dataset_index=int(dataset_index),
		well_id=str(well_id),
	)


def resolve_group_session_inputs(
	*,
	group_dataset_indices: Iterable[int],
	well_id: str,
	data_cfg: RuntimeConfig | dict[str, Any] | None = None,
	well_metadata: dict[tuple[int, str], dict[str, Any]] | None = None,
	output_root: str | Path,
	recon_output_rel_root: str = "recon_outputs",
	require_exists: bool = True,
) -> list[SessionInputs]:
	"""Resolve session inputs for every dataset in a chip-well group.

	Convenience composition of ``resolve_session_inputs`` over a group's
	dataset indices. Reads each dataset's ``raw_data_h5_path`` from
	either the supplied ``data_cfg`` or from the ``well_metadata``
	lookup (whichever is provided; metadata-lookup wins when both are).

	Raises ``UnitmatchSessionInputMissing`` listing EVERY missing input
	across the group when ``require_exists=True``, so the operator gets
	the full picture in one error.
	"""

	if well_metadata is None and data_cfg is None:
		raise ValueError(
			"resolve_group_session_inputs: supply data_cfg or well_metadata"
		)

	def _h5_for_dataset(dataset_index: int) -> tuple[str | None, str | None]:
		"""Return (h5_path, error_message_if_unresolvable) for dataset_index."""

		if well_metadata is not None:
			entry = well_metadata.get((int(dataset_index), str(well_id)))
			if entry is not None and entry.get("raw_data_h5_path"):
				return str(entry["raw_data_h5_path"]), None
		if data_cfg is not None:
			runtime_obj = data_cfg if isinstance(data_cfg, RuntimeConfig) else RuntimeConfig(dict(data_cfg))
			datasets_raw = runtime_obj.get("datasets", []) or []
			if isinstance(datasets_raw, list):
				if dataset_index < 0 or dataset_index >= len(datasets_raw):
					return None, (
						f"dataset_index={dataset_index} out of range for "
						f"data_cfg.datasets (len={len(datasets_raw)})"
					)
				entry = datasets_raw[dataset_index]
				if isinstance(entry, dict):
					h5_path = entry.get("raw_data_h5_path")
					if h5_path:
						return str(h5_path), None
		return None, f"dataset_index={dataset_index}: missing raw_data_h5_path"

	session_inputs: list[SessionInputs] = []
	errors: list[str] = []
	for dataset_index in sorted(set(int(idx) for idx in group_dataset_indices)):
		h5_path, err = _h5_for_dataset(dataset_index)
		if err is not None:
			errors.append(err)
			continue
		try:
			session_inputs.append(
				resolve_session_inputs(
					dataset_index=dataset_index,
					well_id=well_id,
					h5_path=h5_path,
					output_root=output_root,
					recon_output_rel_root=recon_output_rel_root,
					require_exists=require_exists,
				)
			)
		except UnitmatchSessionInputMissing as exc:
			errors.append(str(exc))

	if errors and require_exists:
		raise UnitmatchSessionInputMissing(
			"unitmatch: one or more group sessions missing inputs:\n  "
			+ "\n  ".join(errors)
		)
	return session_inputs


__all__ = [
	"SessionInputs",
	"UnitmatchSessionInputMissing",
	"discover_chip_well_groups",
	"resolve_group_session_inputs",
	"resolve_session_inputs",
]
