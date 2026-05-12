from __future__ import annotations

from pathlib import Path
from typing import Any

from .models.inputs import SpikesortInputs
from .models.results import SpikesortBombcellResult, SpikesortMergeResult, SpikesortResult
from .orchestrators import (
	run_spikesort_bombcell_label,
	run_spikesort_bombcell_label_pass2,
	run_spikesort_bootstrap_concat_binary,
	run_spikesort_cleanup_analyzers,
	run_spikesort_cleanup_concat_binary,
	run_spikesort_concat_analyzer,
	run_spikesort_merge_units,
	run_spikesort_restore_sorter_output,
	run_spikesort_snapshot_sorter_output,
	run_spikesort_sort,
	run_spikesort_summarize,
)


def run_spikesort(inputs: SpikesortInputs) -> SpikesortResult:
	return run_spikesort_sort(inputs)


def summarize_spikesort(inputs: SpikesortInputs) -> SpikesortResult:
	return run_spikesort_summarize(inputs)


def run_spikesort_merge(
	*,
	h5_path: Path,
	stream_id: str,
	mea_output_root: Path,
	output_rel_root: str,
	stage_config: Any,
	force_restart: bool,
	force_replot: bool = False,
) -> SpikesortMergeResult:
	return run_spikesort_merge_units(
		h5_path=h5_path,
		stream_id=stream_id,
		mea_output_root=mea_output_root,
		output_rel_root=output_rel_root,
		stage_config=stage_config,
		force_restart=force_restart,
		force_replot=force_replot,
	)


def run_spikesort_bombcell(
	*,
	h5_path: Path,
	stream_id: str,
	mea_output_root: Path,
	output_rel_root: str,
	stage_config: Any,
	force_restart: bool,
) -> SpikesortBombcellResult:
	return run_spikesort_bombcell_label(
		h5_path=h5_path,
		stream_id=stream_id,
		mea_output_root=mea_output_root,
		output_rel_root=output_rel_root,
		stage_config=stage_config,
		force_restart=force_restart,
	)


def run_spikesort_bombcell_pass2(
	*,
	h5_path: Path,
	stream_id: str,
	mea_output_root: Path,
	output_rel_root: str,
	stage_config: Any,
	force_restart: bool,
) -> SpikesortBombcellResult:
	return run_spikesort_bombcell_label_pass2(
		h5_path=h5_path,
		stream_id=stream_id,
		mea_output_root=mea_output_root,
		output_rel_root=output_rel_root,
		stage_config=stage_config,
		force_restart=force_restart,
	)


def bootstrap_spikesort_concat_binary(
	*,
	h5_path: Path,
	stream_id: str,
	mea_output_root: Path,
	output_rel_root: str,
	stage_config: Any,
	force_restart: bool,
) -> SpikesortResult:
	return run_spikesort_bootstrap_concat_binary(
		h5_path=h5_path,
		stream_id=stream_id,
		mea_output_root=mea_output_root,
		output_rel_root=output_rel_root,
		stage_config=stage_config,
		force_restart=force_restart,
	)


def cleanup_spikesort_concat_binary(
	*,
	h5_path: Path,
	stream_id: str,
	mea_output_root: Path,
	output_rel_root: str,
	stage_config: Any,
	force_restart: bool,
) -> SpikesortResult:
	return run_spikesort_cleanup_concat_binary(
		h5_path=h5_path,
		stream_id=stream_id,
		mea_output_root=mea_output_root,
		output_rel_root=output_rel_root,
		stage_config=stage_config,
		force_restart=force_restart,
	)


def cleanup_spikesort_analyzers(
	*,
	h5_path: Path,
	stream_id: str,
	mea_output_root: Path,
	output_rel_root: str,
	stage_config: Any,
	force_restart: bool,
) -> SpikesortResult:
	return run_spikesort_cleanup_analyzers(
		h5_path=h5_path,
		stream_id=stream_id,
		mea_output_root=mea_output_root,
		output_rel_root=output_rel_root,
		stage_config=stage_config,
		force_restart=force_restart,
	)


def snapshot_spikesort_sorter_output(
	*,
	h5_path: Path,
	stream_id: str,
	mea_output_root: Path,
	output_rel_root: str,
	stage_config: Any,
	force_restart: bool,
) -> SpikesortResult:
	return run_spikesort_snapshot_sorter_output(
		h5_path=h5_path,
		stream_id=stream_id,
		mea_output_root=mea_output_root,
		output_rel_root=output_rel_root,
		stage_config=stage_config,
		force_restart=force_restart,
	)


def restore_spikesort_sorter_output(
	*,
	h5_path: Path,
	stream_id: str,
	mea_output_root: Path,
	output_rel_root: str,
	stage_config: Any,
	force_restart: bool = False,
) -> SpikesortResult:
	return run_spikesort_restore_sorter_output(
		h5_path=h5_path,
		stream_id=stream_id,
		mea_output_root=mea_output_root,
		output_rel_root=output_rel_root,
		stage_config=stage_config,
		force_restart=force_restart,
	)


def build_spikesort_concat_analyzer(
	*,
	h5_path: Path,
	stream_id: str,
	mea_output_root: Path,
	output_rel_root: str,
	stage_config: Any,
	force_restart: bool,
) -> SpikesortResult:
	return run_spikesort_concat_analyzer(
		h5_path=h5_path,
		stream_id=stream_id,
		mea_output_root=mea_output_root,
		output_rel_root=output_rel_root,
		stage_config=stage_config,
		force_restart=force_restart,
	)
