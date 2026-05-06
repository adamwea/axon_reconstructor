from __future__ import annotations

import argparse
import json
from pathlib import Path

from ....execution.results import MultiTargetStageResult
from ..models.inputs import SpikesortInputs
from ..models.results import SpikesortResult
from ..runner import run_spikesort_summarize_sort
from .sort import _target_datasets_override_from_args


def run_spikesort_summarize(inputs: SpikesortInputs) -> SpikesortResult:
	return run_spikesort_summarize_sort(inputs)


def run_spikesort_summarize_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	from ....runner import (
		run_spikesort_summarize_sort_from_runtime as run_spikesort_summarize_runtime,
	)

	return run_spikesort_summarize_runtime(
		config_path=str(config_path),
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def _load_counts_by_label(summary_json: object) -> dict[str, int]:
	if summary_json is None:
		return {}
	try:
		summary_path = Path(str(summary_json)).resolve()
	except Exception:
		return {}
	if not summary_path.exists():
		return {}
	try:
		payload = json.loads(summary_path.read_text(encoding="utf-8"))
	except Exception:
		return {}
	counts_by_label = payload.get("counts_by_label", {}) if isinstance(payload, dict) else {}
	if not isinstance(counts_by_label, dict):
		return {}
	formatted: dict[str, int] = {}
	for raw_label, raw_count in counts_by_label.items():
		label = str(raw_label).strip()
		if not label:
			continue
		try:
			formatted[label] = int(raw_count)
		except Exception:
			continue
	return dict(sorted(formatted.items(), key=lambda item: item[0]))


def _print_spikesort_summarize_aggregate(agg: object) -> int:
	print(f"stage: {agg.stage}")
	print(f"targets_total: {agg.total_targets}")
	print(f"targets_succeeded: {agg.succeeded_targets}")
	print(f"targets_failed: {agg.failed_targets}")
	for item in agg.target_results:
		target = item.target
		if item.status == "ok" and item.result is not None:
			summary_json = getattr(item.result, "summary_json", None)
			counts_by_label = _load_counts_by_label(summary_json)
			counts_suffix = f" labels={counts_by_label}" if counts_by_label else ""
			print(
				f"target[{target.dataset_index}:{target.stream_id}] status=ok "
				f"summary={summary_json}{counts_suffix}"
			)
		else:
			print(
				f"target[{target.dataset_index}:{target.stream_id}] status=error "
				f"error={item.error or 'unknown'}"
			)
	return 0


def _run_summarize_sort_from_args(args: argparse.Namespace) -> int:
	target_datasets_override = _target_datasets_override_from_args(args)
	return _print_spikesort_summarize_aggregate(
		run_spikesort_summarize_from_runtime(
			config_path=str(args.config),
			limit_segments_override=getattr(args, "limit_segments", None),
			limit_datasets_override=getattr(args, "limit_datasets", None),
			target_datasets_override=target_datasets_override,
			limit_wells_per_dataset_override=getattr(args, "limit_wells_per_dataset", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)