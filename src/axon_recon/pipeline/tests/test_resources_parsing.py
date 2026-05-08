"""Tests for resources.py parsing — nested_shape, cpus_per_task, Profile schema, and back-compat."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from axon_recon.pipeline.resources import (
	PhaseResourceClassConfig,
	_parse_phase_resource_class,
	get_active_profile,
	get_keyed_resource_limit_config,
	parse_resources_config,
)


def test_nested_shape_si_njobs_parses():
    cfg = _parse_phase_resource_class({"nested_shape": "si_njobs"})
    assert cfg.nested_shape == "si_njobs"


def test_nested_shape_segment_workers_parses():
    cfg = _parse_phase_resource_class({"nested_shape": "segment_workers"})
    assert cfg.nested_shape == "segment_workers"


def test_nested_shape_unit_workers_parses():
    cfg = _parse_phase_resource_class({"nested_shape": "unit_workers"})
    assert cfg.nested_shape == "unit_workers"


def test_nested_shape_serial_parses():
    cfg = _parse_phase_resource_class({"nested_shape": "serial"})
    assert cfg.nested_shape == "serial"


def test_nested_shape_invalid_raises():
    with pytest.raises(ValueError, match="nested_shape"):
        _parse_phase_resource_class({"nested_shape": "lolwut"})


def test_nested_shape_absent_defaults_to_serial():
    cfg = _parse_phase_resource_class({})
    assert cfg.nested_shape == "serial"


def test_cpus_per_task_parses():
    cfg = _parse_phase_resource_class({"cpus_per_task": 6})
    assert cfg.cpus_per_task == 6


def test_cpus_per_task_absent_defaults_to_none():
    cfg = _parse_phase_resource_class({})
    assert cfg.cpus_per_task is None


def test_nested_shape_field_on_dataclass():
    assert "nested_shape" in PhaseResourceClassConfig.__dataclass_fields__
    assert "cpus_per_task" in PhaseResourceClassConfig.__dataclass_fields__


def _make_runtime_config(raw: dict):
    """Minimal RuntimeConfig stand-in that supports .get() and .has()."""
    rc = MagicMock()
    rc.get.side_effect = lambda key, default=None: raw.get(key, default)
    rc.has.side_effect = lambda key: key in raw
    return rc


def test_new_profile_shape_round_trips():
    """New schema: profiles[name].{capacity, task_allocation, keyed_resource_limits}."""
    rc = _make_runtime_config({
        "resources": {
            "active_profile": "lab",
            "profiles": {
                "lab": {
                    "capacity": {"cpu_cores": 16, "ram_gb": 64},
                    "task_allocation": {"enabled": True, "backend": "local_affinity"},
                    "keyed_resource_limits": {
                        "source_h5_path": {"max_concurrent": 2},
                    },
                },
            },
            "phase_budgets": {},
        },
    })
    cfg = parse_resources_config(runtime_config=rc)
    assert cfg.active_profile == "lab"
    profile = get_active_profile(cfg)
    assert profile is not None
    assert profile.capacity.cpu_cores == 16
    assert profile.task_allocation.enabled is True
    assert profile.task_allocation.backend == "local_affinity"
    limit = get_keyed_resource_limit_config(cfg, "source_h5_path")
    assert limit is not None
    assert limit.max_concurrent == 2


# ── Slice 4: per-machine selectable profiles and CLI override ──────────────


def _make_two_profile_bundle(tmp_path: Path):
    """Build a minimal PipelineRuntimeBundle with lab_server + perlmutter_cpu profiles."""
    from axon_recon.pipeline.config import PipelineRuntimeBundle
    from axon_recon.runtime_config import RuntimeConfig

    data_path = tmp_path / "data.yml"
    data_path.write_text(
        "output_root: /tmp/out\ndatasets: []\n",
        encoding="utf-8",
    )
    runtime_path = tmp_path / "runtime.yml"
    runtime_path.write_text(
        f"""
data: {data_path}
resources:
  active_profile: lab_server
  profiles:
    lab_server:
      capacity:
        cpu_cores: 36
      task_allocation:
        enabled: true
        backend: local_affinity
        task_unit: well
        cpus_per_task: 10
      keyed_resource_limits:
        source_h5_path:
          max_concurrent: 1
    perlmutter_cpu:
      capacity:
        cpu_cores: 128
      task_allocation:
        enabled: true
        backend: mpi
        task_unit: well
        cpus_per_task: 16
        set_thread_env: true
        nested_thread_policy: match_cpus_per_task
      keyed_resource_limits:
        source_h5_path:
          max_concurrent: 1
  phase_budgets: {{}}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    rc = RuntimeConfig.load(str(runtime_path))
    return PipelineRuntimeBundle(
        runtime_config_path=runtime_path,
        data_config_path=data_path,
        runtime_config=rc,
        data_config=RuntimeConfig({"output_root": "/tmp/out", "datasets": []}),
    )


def _make_parallelism(**kwargs):
    from axon_recon.pipeline.execution.context import StageParallelism
    defaults = dict(
        well_workers=3,
        unit_workers=8,
    )
    defaults.update(kwargs)
    return StageParallelism(**defaults)


def test_active_profile_cli_override_switches_to_perlmutter(tmp_path: Path):
    """_attach_task_allocation_plan with active_profile_override='perlmutter_cpu' uses mpi backend."""
    from axon_recon.pipeline.runner import _attach_task_allocation_plan

    bundle = _make_two_profile_bundle(tmp_path)
    parallelism = _make_parallelism()
    # perlmutter_cpu has backend=mpi → build_task_allocation_plan returns None (no local slots)
    # but the function still propagates set_thread_env and nested_thread_policy
    result = _attach_task_allocation_plan(
        bundle=bundle,
        parallelism=parallelism,
        target_count=2,
        active_profile_override="perlmutter_cpu",
    )
    # MPI backend → plan is None, thread env propagates
    assert getattr(result, "set_thread_env", False) is True
    assert getattr(result, "nested_thread_policy", None) == "match_cpus_per_task"


def test_active_profile_cli_override_invalid_raises(tmp_path: Path):
    """_attach_task_allocation_plan with an unknown profile raises ValueError."""
    from axon_recon.pipeline.runner import _attach_task_allocation_plan

    bundle = _make_two_profile_bundle(tmp_path)
    parallelism = _make_parallelism()

    with pytest.raises(ValueError, match="undefined profile"):
        _attach_task_allocation_plan(
            bundle=bundle,
            parallelism=parallelism,
            target_count=1,
            active_profile_override="nonexistent_machine",
        )
