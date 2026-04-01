from __future__ import annotations

import csv
import json
from pathlib import Path

from axon_recon.pipeline.execution.context import ExecutionTarget
from axon_recon.pipeline.execution.results import TargetStageResult
from axon_recon.pipeline.stages.analysis.cross_well import generate_cross_well_artifacts
from axon_recon.pipeline.stages.analysis.models.results import AnalysisResult


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _target_result_for_well(
    *,
    tmp_path: Path,
    dataset_dir: str,
    well_id: str,
    values: list[float],
    dataset_index: int = 0,
) -> TargetStageResult:
    base = tmp_path / dataset_dir / well_id / "analysis_outputs"
    per_unit_path = base / "unit_metrics" / "n_branches.csv"
    per_well_stats_path = base / "well_metrics" / "branches_per_unit_stats.csv"

    per_unit_rows = [
        {
            "h5_path": f"/tmp/{well_id}.raw.h5",
            "stream_id": well_id,
            "well_id": well_id,
            "unit_id": i,
            "status": "ok",
            "value": value,
        }
        for i, value in enumerate(values, start=1)
    ]
    _write_rows(per_unit_path, per_unit_rows)

    mean_value = sum(values) / len(values)
    _write_rows(
        per_well_stats_path,
        [
            {
                "h5_path": f"/tmp/{well_id}.raw.h5",
                "stream_id": well_id,
                "well_id": well_id,
                "n": len(values),
                "mean": mean_value,
                "std": 1.0,
                "sem": 0.5,
            }
        ],
    )

    target = ExecutionTarget(
        dataset_index=int(dataset_index),
        dataset_id=f"dataset:{dataset_dir}",
        h5_path=tmp_path / f"{dataset_dir}_{well_id}.raw.h5",
        stream_id=well_id,
        mea_output_root=tmp_path,
    )
    result = AnalysisResult(
        well_out_dir=base.parent,
        analysis_out_dir=base,
        summary_json=base / "analysis_summary.json",
        outputs={
            "per_unit.n_branches": str(per_unit_path),
            "per_well.branches_per_unit_stats": str(per_well_stats_path),
        },
    )
    return TargetStageResult(target=target, status="ok", result=result)


def test_generate_cross_well_artifacts_writes_stats_and_summary(tmp_path: Path) -> None:
    tr1 = _target_result_for_well(
        tmp_path=tmp_path,
        dataset_dir="dataset_a",
        well_id="well001",
        values=[2.0, 3.0, 5.0, 7.0],
    )
    tr2 = _target_result_for_well(
        tmp_path=tmp_path,
        dataset_dir="dataset_a",
        well_id="well002",
        values=[1.0, 1.5, 2.5, 3.5],
    )

    metrics_cfg = {
        "cross_well": {
            "ordering": "config",
            "statistical_testing": {
                "enable": True,
                "apply_to": "both",
                "alpha": 0.05,
                "min_samples_per_well": 3,
                "multiple_testing_correction": {"enable": True, "method": "holm"},
                "effect_size": {"compute": True, "nonparametric_metric": "cliffs_delta"},
                "annotation": {"mark_on_plot": False},
            },
            "box_and_whisker_plots": {
                "reldir": "box_and_whisker_plots/",
                "defaults": {"write_png": False, "write_pdf": False},
                "n_branches": {
                    "source_level": "per_unit",
                    "source_metric": "n_branches",
                    "write_png": False,
                    "write_pdf": False,
                },
            },
            "bar_plots": {
                "reldir": "bar_plots/",
                "defaults": {"write_png": False, "write_pdf": False},
                "mean_branches_per_unit": {
                    "source_level": "per_well_stats",
                    "source_metric": "branches_per_unit_stats",
                    "source_column": "mean",
                    "stats_source_level": "per_unit",
                    "stats_source_metric": "n_branches",
                    "write_png": False,
                    "write_pdf": False,
                },
            },
        }
    }

    outputs, warnings = generate_cross_well_artifacts(
        target_results=[tr1, tr2],
        metrics_cfg=metrics_cfg,
        output_rel_root="analysis_outputs",
    )

    assert "cross_well.summary_json" in outputs
    assert "cross_well.pairwise_tests_csv" in outputs
    assert Path(outputs["cross_well.summary_json"]).exists()
    assert Path(outputs["cross_well.pairwise_tests_csv"]).exists()
    assert not any("not implemented" in w.lower() for w in warnings)

    with open(outputs["cross_well.pairwise_tests_csv"], "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    metric_ids = {row.get("metric_id") for row in rows}
    assert "box_and_whisker_plots.n_branches" in metric_ids
    assert "bar_plots.mean_branches_per_unit" in metric_ids


def test_generate_cross_well_artifacts_uses_unique_keys_for_duplicate_stream_ids(tmp_path: Path) -> None:
    tr1 = _target_result_for_well(
        tmp_path=tmp_path,
        dataset_dir="dataset_a",
        well_id="well001",
        values=[2.0, 3.0, 4.0, 5.0],
        dataset_index=1,
    )
    tr2 = _target_result_for_well(
        tmp_path=tmp_path,
        dataset_dir="dataset_b",
        well_id="well001",
        values=[1.0, 1.5, 2.0, 2.5],
        dataset_index=2,
    )

    metrics_cfg = {
        "cross_well": {
            "ordering": "config",
            "statistical_testing": {
                "enable": True,
                "apply_to": "box_and_whisker_plots",
                "alpha": 0.05,
                "min_samples_per_well": 3,
            },
            "box_and_whisker_plots": {
                "reldir": "box_and_whisker_plots/",
                "defaults": {"write_png": False, "write_pdf": False},
                "n_branches": {
                    "source_level": "per_unit",
                    "source_metric": "n_branches",
                    "write_png": False,
                    "write_pdf": False,
                },
            },
            "bar_plots": {
                "reldir": "bar_plots/",
                "defaults": {"write_png": False, "write_pdf": False},
            },
        }
    }

    outputs, _warnings = generate_cross_well_artifacts(
        target_results=[tr1, tr2],
        metrics_cfg=metrics_cfg,
        output_rel_root="analysis_outputs",
    )

    summary = json.loads(Path(outputs["cross_well.summary_json"]).read_text(encoding="utf-8"))
    assert summary["well_order"] == ["1:well001", "2:well001"]

    with open(outputs["cross_well.pairwise_tests_csv"], "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    row = next(item for item in rows if item.get("metric_id") == "box_and_whisker_plots.n_branches")
    assert row["well_a"] == "1:well001"
    assert row["well_b"] == "2:well001"
