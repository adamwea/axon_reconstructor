from __future__ import annotations

from pathlib import Path

import pytest

from axon_recon.pipeline.config import (
    load_pipeline_runtime_bundle,
    resolve_stage_parallelism,
    select_execution_targets,
)
from axon_reconstructor.pipeline.scratch_layout import resolve_scratch_layout


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
        include_in_runtime: true
  - raw_data_h5_path: /tmp/ds1.h5
    include_in_runtime: true
    wells:
      - well_id: well001
        include_in_runtime: true
  - raw_data_h5_path: /tmp/ds2.h5
    include_in_runtime: true
    wells:
      - well_id: well001
        include_in_runtime: true
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


def test_select_execution_targets_requires_dataset_and_well_runtime_flags(tmp_path: Path) -> None:
    data_path = tmp_path / "debug.data.yml"
    data_path.write_text(
        """
output_root: /tmp/out
datasets:
  - raw_data_h5_path: /tmp/ds0.h5
    include_in_runtime: false
    wells:
      - well_id: well000
        include_in_runtime: true
  - raw_data_h5_path: /tmp/ds1.h5
    include_in_runtime: true
    wells:
      - well_id: well000
        include_in_runtime: false
      - well_id: well001
        include_in_runtime: true
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

    assert len(targets) == 1
    assert targets[0].h5_path.name == "ds1.h5"
    assert targets[0].stream_id == "well001"


def test_select_execution_targets_errors_when_no_dataset_enabled(tmp_path: Path) -> None:
    data_path = tmp_path / "debug.data.yml"
    data_path.write_text(
        """
output_root: /tmp/out
datasets:
  - raw_data_h5_path: /tmp/ds0.h5
    include_in_runtime: false
    wells:
      - well_id: well001
        include_in_runtime: true
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
    with pytest.raises(ValueError, match="No datasets enabled for runtime execution"):
        select_execution_targets(bundle=bundle)


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
        include_in_runtime: true
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


def test_stage_parallelism_uses_selected_target_count_when_provided(tmp_path: Path) -> None:
    data_path = tmp_path / "debug.data.yml"
    data_path.write_text(
        """
output_root: /tmp/out
datasets:
  - raw_data_h5_path: /tmp/ds1.h5
    include_in_runtime: true
    wells:
      - well_id: well001
        include_in_runtime: true
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
  templates:
    resources:
      max_stage_workers: 24
      well_workers: 2
""".strip()
        + "\n",
        encoding="utf-8",
    )

    bundle = load_pipeline_runtime_bundle(config_path=str(runtime_path))
    p = resolve_stage_parallelism(bundle=bundle, stage_name="templates", target_count=1)
    assert p.max_workers == 24
    assert p.max_stage_workers == 24
    assert p.well_workers == 1
    assert p.unit_workers == 24


def test_stage_parallelism_splits_stage_workers_across_active_well_workers(tmp_path: Path) -> None:
    data_path = tmp_path / "debug.data.yml"
    data_path.write_text(
        """
output_root: /tmp/out
datasets:
  - raw_data_h5_path: /tmp/ds1.h5
    include_in_runtime: true
    wells:
      - well_id: well001
        include_in_runtime: true
      - well_id: well002
        include_in_runtime: true
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
  preprocess:
    resources:
      max_stage_workers: 24
      well_workers: 2
""".strip()
        + "\n",
        encoding="utf-8",
    )

    bundle = load_pipeline_runtime_bundle(config_path=str(runtime_path))
    p = resolve_stage_parallelism(bundle=bundle, stage_name="preprocess", target_count=2)

    assert p.max_workers == 24
    assert p.max_stage_workers == 24
    assert p.well_workers == 2
    assert p.unit_workers == 12


def test_select_execution_targets_prefers_scratch_root_for_active_output(tmp_path: Path) -> None:
    ds1_h5 = tmp_path / "raw_data" / "ds1" / "data.raw.h5"
    ds1_h5.parent.mkdir(parents=True, exist_ok=True)
    ds1_h5.write_bytes(b"ds1")

    ds2_h5 = tmp_path / "raw_data" / "ds2" / "data.raw.h5"
    ds2_h5.parent.mkdir(parents=True, exist_ok=True)
    ds2_h5.write_bytes(b"ds2")

    data_path = tmp_path / "debug.data.yml"
    data_path.write_text(
        f"""
output_root: /tmp/final_out
scratch_root: /tmp/global_scratch
datasets:
  - raw_data_h5_path: {ds1_h5}
    include_in_runtime: true
    wells:
      - well_id: well001
        include_in_runtime: true
  - raw_data_h5_path: {ds2_h5}
    include_in_runtime: true
    scratch_root: /tmp/dataset_scratch
    wells:
      - well_id: well001
        include_in_runtime: true
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

    global_layout = resolve_scratch_layout("/tmp/global_scratch")
    dataset_layout = resolve_scratch_layout("/tmp/dataset_scratch")
    assert global_layout is not None
    assert dataset_layout is not None

    first = targets[0]
    assert first.final_output_root == Path("/tmp/final_out")
    assert first.scratch_output_root == global_layout.outputs_root
    assert first.mea_output_root == global_layout.outputs_root

    second = targets[1]
    assert second.final_output_root == Path("/tmp/final_out")
    assert second.scratch_output_root == dataset_layout.outputs_root
    assert second.mea_output_root == dataset_layout.outputs_root


def test_select_execution_targets_disables_scratch_when_use_scratch_root_false(tmp_path: Path) -> None:
    data_path = tmp_path / "debug.data.yml"
    data_path.write_text(
        """
output_root: /tmp/final_out
scratch_root: /tmp/global_scratch
use_scratch_root: false
datasets:
  - raw_data_h5_path: /tmp/ds1.h5
    include_in_runtime: true
    wells:
      - well_id: well001
        include_in_runtime: true
  - raw_data_h5_path: /tmp/ds2.h5
    include_in_runtime: true
    scratch_root: /tmp/dataset_scratch
    wells:
      - well_id: well001
        include_in_runtime: true
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
    assert first.scratch_output_root is None
    assert first.mea_output_root == Path("/tmp/final_out")

    second = targets[1]
    assert second.final_output_root == Path("/tmp/final_out")
    assert second.scratch_output_root is None
    assert second.mea_output_root == Path("/tmp/final_out")


def test_select_execution_targets_carries_output_root_2_lookup_fallback(tmp_path: Path) -> None:
    data_path = tmp_path / "debug.data.yml"
    data_path.write_text(
        """
output_root: /tmp/c_out
output_root_2: /tmp/h_out
use_scratch_root: false
datasets:
  - raw_data_h5_path: /tmp/ds1.h5
    include_in_runtime: true
    wells:
      - well_id: well001
        include_in_runtime: true
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
    assert len(targets) == 1

    target = targets[0]
    assert target.mea_output_root == Path("/tmp/c_out")
    assert target.artifact_lookup_roots == (Path("/tmp/h_out"),)


def test_select_execution_targets_materializes_inputs_under_canonical_scratch_layout(tmp_path: Path) -> None:
    source_h5 = tmp_path / "raw_data" / "batch_a" / "recording_001" / "data.raw.h5"
    source_h5.parent.mkdir(parents=True, exist_ok=True)
    source_h5.write_bytes(b"source h5 bytes")
    cfg_a = source_h5.parent / "well000.cfg"
    cfg_b = source_h5.parent / "well001.cfg"
    cfg_a.write_text("[well000]\n", encoding="utf-8")
    cfg_b.write_text("[well001]\n", encoding="utf-8")

    scratch_root = tmp_path / "scratch"
    scratch_layout = resolve_scratch_layout(scratch_root)
    assert scratch_layout is not None

    data_path = tmp_path / "debug.data.yml"
    data_path.write_text(
        f"""
output_root: /tmp/out
scratch_root: {scratch_root}
datasets:
  - raw_data_h5_path: {source_h5}
    include_in_runtime: true
    wells:
      - well_id: well000
        include_in_runtime: true
      - well_id: well001
        include_in_runtime: true
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
    targets = select_execution_targets(bundle=bundle, materialize_scratch_inputs=True)
    assert len(targets) == 2

    expected_h5 = scratch_layout.inputs_root / "batch_a" / "recording_001" / "data.raw.h5"
    assert targets[0].h5_path == expected_h5
    assert targets[1].h5_path == expected_h5
    assert expected_h5.exists()
    assert expected_h5.read_bytes() == b"source h5 bytes"
    assert (expected_h5.parent / "well000.cfg").exists()
    assert (expected_h5.parent / "well001.cfg").exists()


def test_select_execution_targets_does_not_materialize_inputs_by_default(tmp_path: Path) -> None:
    source_h5 = tmp_path / "raw_data" / "batch_a" / "recording_001" / "data.raw.h5"
    source_h5.parent.mkdir(parents=True, exist_ok=True)
    source_h5.write_bytes(b"source h5 bytes")

    scratch_root = tmp_path / "scratch"
    scratch_layout = resolve_scratch_layout(scratch_root)
    assert scratch_layout is not None

    data_path = tmp_path / "debug.data.yml"
    data_path.write_text(
        f"""
output_root: /tmp/out
scratch_root: {scratch_root}
datasets:
  - raw_data_h5_path: {source_h5}
    include_in_runtime: true
    wells:
      - well_id: well000
        include_in_runtime: true
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

    assert len(targets) == 1
    assert targets[0].h5_path == source_h5.resolve()
    expected_h5 = scratch_layout.inputs_root / "batch_a" / "recording_001" / "data.raw.h5"
    assert not expected_h5.exists()


def test_select_execution_targets_prefers_existing_current_scratch_input_by_default(tmp_path: Path) -> None:
    source_h5 = tmp_path / "raw_data" / "batch_a" / "recording_001" / "data.raw.h5"
    source_h5.parent.mkdir(parents=True, exist_ok=True)
    source_h5.write_bytes(b"source h5 bytes")

    scratch_root = tmp_path / "scratch"
    scratch_layout = resolve_scratch_layout(scratch_root)
    assert scratch_layout is not None
    scratch_h5 = scratch_layout.inputs_root / "batch_a" / "recording_001" / "data.raw.h5"
    scratch_h5.parent.mkdir(parents=True, exist_ok=True)
    scratch_h5.write_bytes(b"source h5 bytes")
    source_stat = source_h5.stat()
    scratch_h5.touch()
    scratch_h5.write_bytes(b"source h5 bytes")
    import os

    os.utime(scratch_h5, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))

    data_path = tmp_path / "debug.data.yml"
    data_path.write_text(
        f"""
output_root: /tmp/out
scratch_root: {scratch_root}
datasets:
  - raw_data_h5_path: {source_h5}
    include_in_runtime: true
    wells:
      - well_id: well000
        include_in_runtime: true
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

    assert len(targets) == 1
    assert targets[0].h5_path == scratch_h5.resolve()
    assert targets[0].source_h5_path == source_h5.resolve()


def test_select_execution_targets_keeps_legacy_dataset_input_scratch_overrides(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    ds1_h5 = tmp_path / "raw_data" / "set_1" / "data.raw.h5"
    ds1_h5.parent.mkdir(parents=True, exist_ok=True)
    ds1_h5.write_bytes(b"ds1")

    ds2_h5 = tmp_path / "raw_data" / "set_2" / "data.raw.h5"
    ds2_h5.parent.mkdir(parents=True, exist_ok=True)
    ds2_h5.write_bytes(b"ds2")

    global_scratch_input = tmp_path / "scratch_inputs_global"
    dataset_scratch_input = tmp_path / "scratch_inputs_ds2"

    data_path = tmp_path / "debug.data.yml"
    data_path.write_text(
        f"""
output_root: /tmp/out
scratch_input_root: {global_scratch_input}
use_scratch_input_root: true
datasets:
  - raw_data_h5_path: {ds1_h5}
    include_in_runtime: true
    use_scratch_input_root: false
    wells:
      - well_id: well000
        include_in_runtime: true
  - raw_data_h5_path: {ds2_h5}
    include_in_runtime: true
    scratch_input_root: {dataset_scratch_input}
    wells:
      - well_id: well000
        include_in_runtime: true
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
    targets = select_execution_targets(bundle=bundle, materialize_scratch_inputs=True)
    assert len(targets) == 2
    assert "scratch_input_root/use_scratch_input_root are deprecated" in caplog.text

    first = targets[0]
    assert first.h5_path == ds1_h5.resolve()

    second = targets[1]
    expected_ds2_h5 = dataset_scratch_input.resolve() / "set_2" / "data.raw.h5"
    assert second.h5_path == expected_ds2_h5
    assert expected_ds2_h5.exists()
