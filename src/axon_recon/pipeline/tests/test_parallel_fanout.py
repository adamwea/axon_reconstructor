from __future__ import annotations

from pathlib import Path

from axon_recon.pipeline.config import (
    load_pipeline_runtime_bundle,
    resolve_stage_parallelism,
    select_execution_targets,
)


def test_select_execution_targets_uses_all_include_in_runtime(tmp_path: Path) -> None:
    data_path = tmp_path / "debug.data.yml"
    data_path.write_text(
        """
output_root: /tmp/out
datasets:
  - raw_data_h5_path: /tmp/ds0.h5
    include_in_runtime: false
    wells:
      - well_id: well001
  - raw_data_h5_path: /tmp/ds1.h5
    include_in_runtime: true
    wells:
      - well_id: well001
  - raw_data_h5_path: /tmp/ds2.h5
    include_in_runtime: true
    wells:
      - well_id: well001
""".strip()
        + "\n",
        encoding="utf-8",
    )

    runtime_path = tmp_path / "debug.runtime.yml"
    runtime_path.write_text(
        f"""
data: {data_path}
stages:
  reconstruct:
    resources:
      max_stage_workers: 8
      well_workers: 2
""".strip()
        + "\n",
        encoding="utf-8",
    )

    bundle = load_pipeline_runtime_bundle(config_path=str(runtime_path))
    targets = select_execution_targets(bundle=bundle)
    assert len(targets) == 2
    assert targets[0].h5_path.name == "ds1.h5"
    assert targets[1].h5_path.name == "ds2.h5"


def test_stage_parallelism_derives_unit_workers_like_legacy(tmp_path: Path) -> None:
    data_path = tmp_path / "debug.data.yml"
    data_path.write_text(
        """
output_root: /tmp/out
datasets:
  - raw_data_h5_path: /tmp/ds1.h5
    include_in_runtime: true
    wells:
      - well_id: well001
""".strip()
        + "\n",
        encoding="utf-8",
    )

    runtime_path = tmp_path / "debug.runtime.yml"
    runtime_path.write_text(
        f"""
data: {data_path}
resources:
  max_workers: 24
stages:
  reconstruct:
    resources:
      max_stage_workers: 8
      well_workers: 2
""".strip()
        + "\n",
        encoding="utf-8",
    )

    bundle = load_pipeline_runtime_bundle(config_path=str(runtime_path))
    p = resolve_stage_parallelism(bundle=bundle, stage_name="reconstruct")
    assert p.max_workers == 24
    assert p.max_stage_workers == 8
    assert p.well_workers == 2
    assert p.unit_workers == 4


def test_select_execution_targets_prefers_scratch_root_for_active_output(tmp_path: Path) -> None:
    data_path = tmp_path / "debug.data.yml"
    data_path.write_text(
        """
output_root: /tmp/final_out
scratch_root: /tmp/global_scratch
datasets:
  - raw_data_h5_path: /tmp/ds1.h5
    include_in_runtime: true
    wells:
      - well_id: well001
  - raw_data_h5_path: /tmp/ds2.h5
    include_in_runtime: true
    scratch_root: /tmp/dataset_scratch
    wells:
      - well_id: well001
""".strip()
        + "\n",
        encoding="utf-8",
    )

    runtime_path = tmp_path / "debug.runtime.yml"
    runtime_path.write_text(
        f"""
data: {data_path}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    bundle = load_pipeline_runtime_bundle(config_path=str(runtime_path))
    targets = select_execution_targets(bundle=bundle)
    assert len(targets) == 2

    first = targets[0]
    assert first.final_output_root == Path("/tmp/final_out")
    assert first.scratch_output_root == Path("/tmp/global_scratch")
    assert first.mea_output_root == Path("/tmp/global_scratch")

    second = targets[1]
    assert second.final_output_root == Path("/tmp/final_out")
    assert second.scratch_output_root == Path("/tmp/dataset_scratch")
    assert second.mea_output_root == Path("/tmp/dataset_scratch")
