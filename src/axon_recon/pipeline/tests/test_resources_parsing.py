"""Tests for resources.py parsing — nested_shape, cpus_per_task, Profile schema, and back-compat."""
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


def test_legacy_profile_shape_back_compat():
    """Old schema: profiles[name] IS capacity; task_allocation/keyed_resource_limits at top-level."""
    rc = _make_runtime_config({
        "resources": {
            "active_profile": "lab",
            "profiles": {
                "lab": {"cpu_cores": 16, "ram_gb": 64},
            },
            "task_allocation": {"enabled": True, "backend": "local_affinity"},
            "keyed_resource_limits": {
                "source_h5_path": {"max_concurrent": 3},
            },
            "phase_budgets": {},
        },
    })
    cfg = parse_resources_config(runtime_config=rc)
    profile = get_active_profile(cfg)
    assert profile is not None
    assert profile.capacity.cpu_cores == 16
    assert profile.task_allocation.enabled is True
    limit = get_keyed_resource_limit_config(cfg, "source_h5_path")
    assert limit is not None
    assert limit.max_concurrent == 3


def test_no_profile_legacy_keyed_limits_fallback():
    """No profiles, no active_profile: top-level keyed_resource_limits must still be reachable."""
    rc = _make_runtime_config({
        "resources": {
            "keyed_resource_limits": {
                "source_h5_path": {"max_concurrent": 4},
            },
            "phase_budgets": {},
        },
    })
    cfg = parse_resources_config(runtime_config=rc)
    assert cfg.active_profile is None
    limit = get_keyed_resource_limit_config(cfg, "source_h5_path")
    assert limit is not None
    assert limit.max_concurrent == 4


def test_phase_resource_classes_legacy_name_still_parses():
    """phase_resource_classes key is back-compat alias for phase_budgets."""
    rc = _make_runtime_config({
        "resources": {
            "phase_resource_classes": {
                "heavy": {"cpu_cores": 8, "nested_shape": "unit_workers"},
            },
        },
    })
    cfg = parse_resources_config(runtime_config=rc)
    assert "heavy" in cfg.phase_budgets
    assert cfg.phase_budgets["heavy"].nested_shape == "unit_workers"
