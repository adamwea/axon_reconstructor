"""Walk the runtime target scope to find every `<well>/analysis_outputs/manifest.json`.

Reuses `select_execution_targets` so the dashboard sees the exact same well set
that `axon-recon stages analysis` would compute for the same scope flags.
Read-only — tolerates missing manifests (returns only the ones present on disk).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from axon_recon.pipeline.config import (
	PipelineRuntimeBundle,
	load_pipeline_runtime_bundle,
	select_execution_targets,
)
from axon_recon.pipeline.output_paths import compute_mea_analysis_output_dir


_DEFAULT_OUTPUT_REL_ROOT = "analysis_outputs"
_DEFAULT_MANIFEST_RELPATH = "manifest.json"


def _analysis_output_rel_root(runtime_config: Any) -> str:
	value = runtime_config.get("stages.analysis.output_rel_root", None)
	text = str(value or _DEFAULT_OUTPUT_REL_ROOT).strip()
	return text.lstrip("/") or _DEFAULT_OUTPUT_REL_ROOT


def _manifest_relpath(runtime_config: Any) -> str:
	value = runtime_config.get("stages.analysis.manifest_relpath", None)
	text = str(value or _DEFAULT_MANIFEST_RELPATH).strip()
	return text.lstrip("/") or _DEFAULT_MANIFEST_RELPATH


def iter_manifest_paths(
	bundle: PipelineRuntimeBundle,
	*,
	target_datasets: list[int] | None = None,
	limit_wells: int | None = None,
	limit_datasets: int | None = None,
	limit_wells_per_dataset: int | None = None,
) -> list[Path]:
	"""Return existing manifest.json paths within the runtime target scope.

	Missing manifests (no analysis stage run for that well yet) are skipped
	silently — the dashboard surfaces what's on disk and doesn't try to
	re-run the stage.
	"""
	targets = select_execution_targets(
		bundle=bundle,
		limit_datasets=limit_datasets,
		target_datasets=target_datasets,
		limit_wells=limit_wells,
		limit_wells_per_dataset=limit_wells_per_dataset,
	)
	output_rel_root = _analysis_output_rel_root(bundle.runtime_config)
	manifest_relpath = _manifest_relpath(bundle.runtime_config)

	found: list[Path] = []
	for target in targets:
		well_out_dir = compute_mea_analysis_output_dir(
			output_root=target.mea_output_root,
			data_file=target.h5_path,
			well=target.stream_id,
		)
		manifest = well_out_dir / output_rel_root / manifest_relpath
		if manifest.is_file():
			found.append(manifest)
	return found


def iter_manifest_paths_from_config(
	*,
	config_path: str,
	target_datasets: list[int] | None = None,
	limit_wells: int | None = None,
	limit_datasets: int | None = None,
	limit_wells_per_dataset: int | None = None,
) -> list[Path]:
	"""Convenience wrapper that loads the bundle then delegates."""
	bundle = load_pipeline_runtime_bundle(config_path=str(config_path))
	return iter_manifest_paths(
		bundle,
		target_datasets=target_datasets,
		limit_wells=limit_wells,
		limit_datasets=limit_datasets,
		limit_wells_per_dataset=limit_wells_per_dataset,
	)
