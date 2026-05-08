from __future__ import annotations

from pathlib import Path

import pytest

from axon_recon.pipeline.config import (
    load_pipeline_runtime_bundle,
    resolve_stage_parallelism,
    select_execution_targets,
)
from axon_recon.pipeline.scratch_layout import resolve_scratch_layout


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
""".strip()
        + "\n",
        encoding="utf-8",
    )

    bundle = load_pipeline_runtime_bundle(config_path=str(runtime_path))
    targets = select_execution_targets(bundle=bundle)
    assert len(targets) == 2
    assert targets[0].h5_path.name == "ds1.h5"
    assert targets[1].h5_path.name == "ds2.h5"


def test_select_execution_targets_applies_limits_before_scratch_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import axon_recon.pipeline.config as pipeline_config

    data_path = tmp_path / "debug.data.yml"
    data_path.write_text(
        f"""
output_root: {tmp_path / "outputs"}
scratch_root: {tmp_path / "scratch"}
use_scratch_root: true
datasets:
  - raw_data_h5_path: {tmp_path / "ds0.h5"}
    include_in_runtime: true
    wells:
      - well_id: well001
        include_in_runtime: true
      - well_id: well002
        include_in_runtime: true
  - raw_data_h5_path: {tmp_path / "ds1.h5"}
    include_in_runtime: true
    wells:
      - well_id: well001
        include_in_runtime: true
  - raw_data_h5_path: {tmp_path / "ds2.h5"}
    include_in_runtime: true
    wells:
      - well_id: well001
        include_in_runtime: true
""".strip()
        + "\n",
        encoding="utf-8",
    )

    runtime_path = tmp_path / "debug.runtime.yml"
    runtime_path.write_text(f"data: {data_path}\n", encoding="utf-8")

    materialized_dataset_ids: list[str] = []

    def _fake_resolve_copy_src_to_scratch_input_path(
        *,
        source_h5_path: Path,
        scratch_input_root: Path | None,
        dataset_id: str,
        materialize_scratch_inputs: bool,
    ) -> Path:
        assert scratch_input_root is not None
        if materialize_scratch_inputs:
            materialized_dataset_ids.append(str(dataset_id))
        return scratch_input_root / Path(source_h5_path).name

    monkeypatch.setattr(
        pipeline_config,
        "resolve_copy_src_to_scratch_input_path",
        _fake_resolve_copy_src_to_scratch_input_path,
    )

    bundle = load_pipeline_runtime_bundle(config_path=str(runtime_path))
    targets = select_execution_targets(
        bundle=bundle,
        materialize_scratch_inputs=True,
        limit_datasets=1,
        limit_wells_per_dataset=1,
    )

    assert materialized_dataset_ids == ["dataset_000:ds0.h5"]
    assert [(target.dataset_index, target.stream_id) for target in targets] == [(0, "well001")]


def test_select_execution_targets_applies_target_dataset_indices_before_scratch_materialization(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	import axon_recon.pipeline.config as pipeline_config

	data_path = tmp_path / "debug.data.yml"
	data_path.write_text(
		f"""
output_root: {tmp_path / "outputs"}
scratch_root: {tmp_path / "scratch"}
use_scratch_root: true
datasets:
  - raw_data_h5_path: {tmp_path / "ds0.h5"}
    include_in_runtime: true
    wells:
      - well_id: well001
        include_in_runtime: true
  - raw_data_h5_path: {tmp_path / "ds1.h5"}
    include_in_runtime: true
    wells:
      - well_id: well001
        include_in_runtime: true
  - raw_data_h5_path: {tmp_path / "ds2.h5"}
    include_in_runtime: true
    wells:
      - well_id: well001
        include_in_runtime: true
  - raw_data_h5_path: {tmp_path / "ds3.h5"}
    include_in_runtime: true
    wells:
      - well_id: well001
        include_in_runtime: true
""".strip()
		+ "\n",
		encoding="utf-8",
	)

	runtime_path = tmp_path / "debug.runtime.yml"
	runtime_path.write_text(f"data: {data_path}\n", encoding="utf-8")

	materialized_dataset_ids: list[str] = []

	def _fake_resolve_copy_src_to_scratch_input_path(
		*,
		source_h5_path: Path,
		scratch_input_root: Path | None,
		dataset_id: str,
		materialize_scratch_inputs: bool,
	) -> Path:
		assert scratch_input_root is not None
		if materialize_scratch_inputs:
			materialized_dataset_ids.append(str(dataset_id))
		return scratch_input_root / Path(source_h5_path).name

	monkeypatch.setattr(
		pipeline_config,
		"resolve_copy_src_to_scratch_input_path",
		_fake_resolve_copy_src_to_scratch_input_path,
	)

	bundle = load_pipeline_runtime_bundle(config_path=str(runtime_path))
	targets = select_execution_targets(
		bundle=bundle,
		materialize_scratch_inputs=True,
		target_datasets=[1, 3],
	)

	assert materialized_dataset_ids == ["dataset_001:ds1.h5", "dataset_003:ds3.h5"]
	assert [(target.dataset_index, target.stream_id) for target in targets] == [
		(1, "well001"),
		(3, "well001"),
	]


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


def test_select_execution_targets_preserves_symlinked_h5_path(tmp_path: Path) -> None:
    real_h5 = tmp_path / "real" / "data.raw.h5"
    real_h5.parent.mkdir(parents=True, exist_ok=True)
    real_h5.write_text("placeholder\n", encoding="utf-8")
    link_h5 = tmp_path / "links" / "data_link.raw.h5"
    link_h5.parent.mkdir(parents=True, exist_ok=True)
    link_h5.symlink_to(real_h5)

    data_path = tmp_path / "debug.data.yml"
    data_path.write_text(
        f"""
output_root: /tmp/out
datasets:
  - raw_data_h5_path: {link_h5}
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
    assert targets[0].h5_path == link_h5


def test_select_execution_targets_resolves_relative_h5_path_against_data_config(tmp_path: Path) -> None:
    relative_h5 = Path("relative/input.raw.h5")
    data_path = tmp_path / "configs" / "debug.data.yml"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(
        f"""
output_root: /tmp/out
datasets:
  - raw_data_h5_path: {relative_h5.as_posix()}
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
    assert targets[0].h5_path == data_path.parent / relative_h5


def test_select_execution_targets_preserves_symlinked_output_root(tmp_path: Path) -> None:
    real_output_root = tmp_path / "real_outputs"
    real_output_root.mkdir(parents=True, exist_ok=True)
    linked_output_root = tmp_path / "linked_outputs"
    linked_output_root.symlink_to(real_output_root)

    data_path = tmp_path / "debug.data.yml"
    data_path.write_text(
        f"""
output_root: {linked_output_root}
datasets:
  - raw_data_h5_path: /tmp/input.raw.h5
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
    assert targets[0].mea_output_root == linked_output_root
    assert targets[0].final_output_root == linked_output_root


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


# ---------------------------------------------------------------------------
# Slice 5 inner-worker derivation integration tests
# ---------------------------------------------------------------------------


def test_phase_budget_drives_inner_worker_count_for_unit_workers_phase() -> None:
    """A phase with no cpus_per_task clamp inherits the slot CPU count."""
    from axon_recon.pipeline.cpu_allocation import (
        TaskSlot,
        phase_budgets_context,
        resolve_inner_worker_count,
        task_slot_context,
    )
    from axon_recon.pipeline.resources import PhaseResourceClassConfig

    slot = TaskSlot(
        slot_id=0,
        logical_cpus=tuple(range(10)),
        core_ids=tuple(range(10)),
        package_ids=(0,),
    )
    budgets = {
        "reconstruct.extract_partial_templates": PhaseResourceClassConfig(
            nested_shape="segment_workers",
            cpus_per_task=None,  # inherit
        ),
    }
    captured: dict[str, int] = {}
    with task_slot_context(slot), phase_budgets_context(budgets):
        # Simulate the call-site lookup pattern used by the 12 patched ThreadPoolExecutor sites.
        from axon_recon.pipeline.cpu_allocation import current_phase_budget

        budget = current_phase_budget("reconstruct", "extract_partial_templates")
        captured["max_workers"] = resolve_inner_worker_count(
            nested_shape=str(getattr(budget, "nested_shape", "segment_workers")),
            phase_cpus_per_task=getattr(budget, "cpus_per_task", None),
            yaml_n_jobs_override=None,
            work_item_count=None,
        )
    # No clamp, no override, no work item count => inherits slot.cpu_count = 10.
    assert captured["max_workers"] == 10


def test_phase_budget_clamps_inner_worker_count_for_plot_recons() -> None:
    """A phase with explicit cpus_per_task=4 caps the inner thread count at 4."""
    from axon_recon.pipeline.cpu_allocation import (
        TaskSlot,
        phase_budgets_context,
        resolve_inner_worker_count,
        task_slot_context,
    )
    from axon_recon.pipeline.resources import PhaseResourceClassConfig

    slot = TaskSlot(
        slot_id=0,
        logical_cpus=tuple(range(10)),
        core_ids=tuple(range(10)),
        package_ids=(0,),
    )
    budgets = {
        "reconstruct.plot_recons": PhaseResourceClassConfig(
            nested_shape="unit_workers",
            cpus_per_task=4,  # explicit clamp (matplotlib RAM)
        ),
    }
    captured: dict[str, int] = {}
    with task_slot_context(slot), phase_budgets_context(budgets):
        from axon_recon.pipeline.cpu_allocation import current_phase_budget

        budget = current_phase_budget("reconstruct", "plot_recons")
        captured["max_workers"] = resolve_inner_worker_count(
            nested_shape=str(getattr(budget, "nested_shape", "unit_workers")),
            phase_cpus_per_task=getattr(budget, "cpus_per_task", None),
            yaml_n_jobs_override=None,
            work_item_count=None,
        )
    # cpus_per_task=4 clamps the slot's 10 cpus down to 4.
    assert captured["max_workers"] == 4


def test_phase_budget_missing_entry_inherits_slot_for_unconfigured_phase() -> None:
    """When no entry is configured for stage.phase, the helper still returns slot cpu count."""
    from axon_recon.pipeline.cpu_allocation import (
        TaskSlot,
        phase_budgets_context,
        resolve_inner_worker_count,
        task_slot_context,
        current_phase_budget,
    )

    slot = TaskSlot(
        slot_id=0,
        logical_cpus=tuple(range(8)),
        core_ids=tuple(range(8)),
        package_ids=(0,),
    )
    with task_slot_context(slot), phase_budgets_context({}):
        budget = current_phase_budget("reconstruct", "nonexistent_phase")
        assert budget is None
        result = resolve_inner_worker_count(
            nested_shape="si_njobs",
            phase_cpus_per_task=None,
            yaml_n_jobs_override=None,
            work_item_count=None,
        )
    assert result == 8


def test_keyed_h5_serializes_same_source() -> None:
    """Two targets sharing the same source H5 run serially; different H5 runs concurrently."""
    import threading
    from axon_recon.pipeline.execution.distributor import distribute_targets
    from axon_recon.pipeline.execution.context import ExecutionTarget

    shared_h5 = Path("/tmp/shared.h5")
    other_h5 = Path("/tmp/other.h5")

    def _target(stream_id: str, h5: Path) -> ExecutionTarget:
        return ExecutionTarget(
            dataset_index=0,
            dataset_id="test",
            h5_path=h5,
            stream_id=stream_id,
            mea_output_root=Path("/tmp"),
            source_h5_path=h5,
        )

    active: set[str] = set()
    lock = threading.Lock()
    max_concurrent: list[int] = [0]
    release = threading.Event()

    def worker(target: ExecutionTarget) -> None:
        with lock:
            active.add(target.stream_id)
            max_concurrent[0] = max(max_concurrent[0], len(active))
        assert release.wait(timeout=5)
        with lock:
            active.discard(target.stream_id)

    targets = [
        _target("well-a", shared_h5),
        _target("well-b", shared_h5),
    ]
    release.set()
    results = distribute_targets(
        targets=targets,
        well_workers=2,
        worker_fn=worker,
        max_simultaneous_well_reads_per_dataset=1,
    )
    assert all(r.status == "ok" for r in results)
    assert max_concurrent[0] == 1, f"Expected serial execution, saw {max_concurrent[0]} concurrent"
