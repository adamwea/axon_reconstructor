"""Tests for resources.py parsing — nested_shape and cpus_per_task on PhaseResourceClassConfig."""
import pytest

from axon_recon.pipeline.resources import PhaseResourceClassConfig, _parse_phase_resource_class


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
