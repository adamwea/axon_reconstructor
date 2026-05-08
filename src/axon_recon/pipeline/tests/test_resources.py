from __future__ import annotations

from types import SimpleNamespace

import pytest

from axon_recon.pipeline.resources import (
	get_active_profile,
	get_keyed_resource_limit_config,
	get_resource_default,
	parse_resources_config,
)
from axon_recon.pipeline.stages.preprocess.config import parse_preprocess_stage_config
from axon_recon.pipeline.stages.reconstruct.config import parse_reconstruction_stage_config
from axon_recon.pipeline.stages.reconstruct.templates.config import parse_reconstruct_templates_config
from axon_recon.pipeline.stages.spikesort.config import parse_spikesort_stage_config
from axon_recon.runtime_config import RuntimeConfig


def _resource_payload() -> dict[str, object]:
	return {
		"resources": {
			"active_profile": "lab_server_safe",
			"profiles": {
				"lab_server_safe": {
					"capacity": {
						"cpu_cores": 18,
						"ram_gb": 96,
						"gpu_sort_slots": 1,
						"h5_read_slots": 1,
						"disk_heavy_slots": 1,
						"plot_slots": 1,
						"analyzer_slots": 1,
						"default_chunk_duration": "1s",
					},
					"keyed_resource_limits": {
						"source_h5_path": {
							"description": "Limit concurrent well-workers touching the same source H5 file.",
							"max_concurrent": 1,
							"applies_to": ["h5_metadata", "h5_to_binary", "preprocess_segments"],
						},
					},
				},
			},
			"phase_budgets": {
				"h5_metadata": {
					"description": "Light HDF5 metadata and small file reads.",
					"bottleneck": "h5_read",
					"cpus_per_task": 1,
					"ram_gb": 4,
					"h5_read_slots": 1,
					"keyed_resources": {"source_h5_path": 1},
				},
				"h5_to_binary": {
					"description": "HDF5/source read plus large binary write.",
					"bottleneck": "h5_read_and_disk_write",
					"cpus_per_task": 4,
					"ram_gb": 16,
					"h5_read_slots": 1,
					"disk_heavy_slots": 1,
					"keyed_resources": {"source_h5_path": 1},
				},
				"preprocess_segments": {
					"description": "Segment-level preprocessing with HDF5/source reads and disk output.",
					"bottleneck": "h5_read_and_disk_io",
					"cpus_per_task": 6,
					"ram_gb": 24,
					"h5_read_slots": 1,
					"disk_heavy_slots": 1,
					"keyed_resources": {"source_h5_path": 1},
				},
				"plot_unit": {
					"description": "Per-unit plotting or matplotlib-heavy rendering.",
					"bottleneck": "plotting_memory",
					"cpus_per_task": 2,
					"ram_gb": 24,
					"plot_slots": 1,
				},
				"kilosort4": {
					"description": "GPU-backed Kilosort4 sorting through SpikeInterface.",
					"bottleneck": "gpu_and_disk_io",
					"cpus_per_task": 8,
					"ram_gb": 32,
					"gpu_sort_slots": 1,
					"disk_heavy_slots": 1,
				},
				"spikeinterface_analyzer": {
					"description": "SpikeInterface analyzer, waveform, template, sparsity, or curation work.",
					"bottleneck": "ram_cpu_disk_io",
					"cpus_per_task": 8,
					"ram_gb": 48,
					"analyzer_slots": 1,
					"disk_heavy_slots": 1,
				},
				"axon_reconstruction": {
					"description": "Axon reconstruction, GTR generation, or unit-level reconstruction math.",
					"bottleneck": "cpu_memory_light",
					"cpus_per_task": 2,
					"ram_gb": 8,
				},
				"template_build": {
					"description": "Template merge/build or lightweight unit-level compute.",
					"bottleneck": "cpu_memory_light",
					"cpus_per_task": 2,
					"ram_gb": 8,
				},
				"plot_report_grid": {
					"description": "Large report, grid, PDF, summary, or full-chip plotting.",
					"bottleneck": "plotting_memory_heavy",
					"cpus_per_task": 4,
					"ram_gb": 48,
					"plot_slots": 1,
				},
			},
			"defaults": {
				"chunk_duration": "3s",
				"max_workers": 7,
			},
		},
	}


def test_parse_resources_config_parses_typed_profiles_and_keyed_limits():
	cfg = RuntimeConfig(_resource_payload())
	parsed = parse_resources_config(runtime_config=cfg)
	active = get_active_profile(parsed)

	assert parsed.active_profile == "lab_server_safe"
	assert active is not None
	assert active.capacity.cpu_cores == 18
	assert active.capacity.default_chunk_duration == "1s"
	assert parsed.defaults["chunk_duration"] == "3s"
	assert get_keyed_resource_limit_config(parsed, "source_h5_path") is not None
	assert get_keyed_resource_limit_config(parsed, "source_h5_path").max_concurrent == 1
	assert parsed.phase_budgets["h5_metadata"].keyed_resources["source_h5_path"] == 1
	assert parsed.phase_budgets["h5_to_binary"].keyed_resources["source_h5_path"] == 1
	assert parsed.phase_budgets["kilosort4"].description is not None
	assert parsed.phase_budgets["kilosort4"].gpu_sort_slots == 1
	assert get_resource_default(runtime_config=cfg, key="max_workers", default=None) == 7



def test_parse_resources_config_defaults_task_allocation_to_disabled_schema():
	parsed = parse_resources_config(runtime_config=RuntimeConfig(_resource_payload()))
	active = get_active_profile(parsed)
	assert active is not None
	ta = active.task_allocation

	assert ta.enabled is False
	assert ta.backend == "none"
	assert ta.task_unit == "well"
	assert ta.cpus_per_task == "auto"
	assert ta.tasks_per_node == "auto"
	assert ta.bind == "none"
	assert ta.use_hyperthreads is False
	assert ta.reserve_cpus == 0
	assert ta.set_thread_env is False
	assert ta.nested_thread_policy == "preserve_existing"
	assert ta.ram_gb_per_task is None
	assert ta.shm_gb_per_task is None


def test_parse_resources_config_parses_explicit_task_allocation_block():
	payload = _resource_payload()
	resources = payload["resources"]
	assert isinstance(resources, dict)
	profiles = resources["profiles"]
	assert isinstance(profiles, dict)
	profiles["lab_server_safe"]["task_allocation"] = {
		"enabled": True,
		"backend": "local_affinity",
		"task_unit": "well",
		"cpus_per_task": 4,
		"tasks_per_node": "auto",
		"bind": "physical_cores",
		"use_hyperthreads": False,
		"reserve_cpus": 2,
		"set_thread_env": True,
		"nested_thread_policy": "match_cpus_per_task",
		"ram_gb_per_task": 12.5,
		"shm_gb_per_task": 8,
	}

	parsed = parse_resources_config(runtime_config=RuntimeConfig(payload))
	active = get_active_profile(parsed)
	assert active is not None
	ta = active.task_allocation

	assert ta.enabled is True
	assert ta.backend == "local_affinity"
	assert ta.task_unit == "well"
	assert ta.cpus_per_task == 4
	assert ta.tasks_per_node == "auto"
	assert ta.bind == "physical_cores"
	assert ta.use_hyperthreads is False
	assert ta.reserve_cpus == 2
	assert ta.set_thread_env is True
	assert ta.nested_thread_policy == "match_cpus_per_task"
	assert ta.ram_gb_per_task == pytest.approx(12.5)
	assert ta.shm_gb_per_task == pytest.approx(8.0)


@pytest.mark.parametrize(
	("field_name", "field_value"),
	[
		("backend", "bogus"),
		("task_unit", "dataset"),
		("cpus_per_task", 0),
		("tasks_per_node", -1),
		("bind", "socket"),
		("reserve_cpus", -1),
		("nested_thread_policy", "inherit"),
		("ram_gb_per_task", 0),
		("shm_gb_per_task", "bad"),
	],
)
def test_parse_resources_config_rejects_invalid_task_allocation_values(
	field_name: str,
	field_value: object,
) -> None:
	payload = _resource_payload()
	resources = payload["resources"]
	assert isinstance(resources, dict)
	profiles = resources["profiles"]
	assert isinstance(profiles, dict)
	profiles["lab_server_safe"]["task_allocation"] = {field_name: field_value}

	with pytest.raises(ValueError, match=field_name):
		parse_resources_config(runtime_config=RuntimeConfig(payload))


def test_preprocess_stage_parses_phase_resource_class_and_default_chunk_duration():
	payload = _resource_payload()
	payload["stages"] = {
		"preprocess": {
			"phases": {
				"prepare_raw_binaries": {"resource_class": "h5_to_binary"},
				"plot_segment_traces": {"resource_class": "plot_unit"},
			},
		}
	}

	parsed = parse_preprocess_stage_config(runtime_config=RuntimeConfig(payload))

	assert parsed.save_chunk_duration == "3s"
	assert parsed.phases.prepare_raw_binaries.resource_class == "h5_to_binary"
	assert parsed.phases.plot_segment_traces.resource_class == "plot_unit"


def test_spikesort_stage_parses_phase_resource_class_and_default_chunk_duration():
	payload = _resource_payload()
	payload["stages"] = {
		"spikesort": {
			"phases": {
				"sort": {"resource_class": "kilosort4"},
				"summarize_sort": {"resource_class": "template_build"},
			},
		}
	}

	parsed = parse_spikesort_stage_config(runtime_config=RuntimeConfig(payload))

	assert parsed.chunk_duration == "3s"
	assert parsed.sort_resource_class == "kilosort4"
	assert parsed.summarize_sort_resource_class == "template_build"


def test_spikesort_runtime_phase_plan_preserves_phase_resource_class():
	from axon_recon.pipeline.runner import _enabled_spikesort_runtime_phase_plan

	stage_config = SimpleNamespace(
		phase_sequence=("sort", "merge_si_auto", "cleanup_concat_binary"),
		bootstrap_concat_binary_enabled=False,
		sort_enabled=True,
		sort_resource_class="kilosort4",
		summarize_sort_enabled=False,
		bombcell_label_enabled=False,
		merge_slay_enabled=False,
		merge_si_auto_enabled=True,
		merge_si_auto_resource_class="spikeinterface_analyzer",
		merge_unitmatch_enabled=False,
		cleanup_concat_binary_enabled=True,
		cleanup_concat_binary_resource_class="disk_cleanup",
	)

	phase_plan = _enabled_spikesort_runtime_phase_plan(stage_config)

	assert [(phase.phase_label, phase.resource_class) for phase in phase_plan] == [
		("sort", "kilosort4"),
		("merge_si_auto", "spikeinterface_analyzer"),
		("cleanup_concat_binary", "disk_cleanup"),
	]


def test_reconstruct_and_templates_parsers_preserve_phase_resource_class():
	payload = _resource_payload()
	payload["stages"] = {
		"reconstruct": {
			"phases": {
				"generate_gtrs": {"resource_class": "axon_reconstruction"},
				"report_recons": {"resource_class": "plot_report_grid"},
				"resolve_sources": {"resource_class": "h5_metadata"},
				"build_templates": {"resource_class": "template_build"},
			},
		}
	}

	reconstruct_cfg = parse_reconstruction_stage_config(runtime_config=RuntimeConfig(payload))
	templates_cfg = parse_reconstruct_templates_config(runtime_config=RuntimeConfig(payload))

	assert reconstruct_cfg.phases.generate_gtrs.resource_class == "axon_reconstruction"
	assert reconstruct_cfg.phases.report_recons.resource_class == "plot_report_grid"
	assert templates_cfg.resolve_sources_phase.resource_class == "h5_metadata"
	assert templates_cfg.phases.build_templates.resource_class == "template_build"
	assert templates_cfg.phases.per_unit_processing.build_templates.resource_class == "template_build"


def test_get_resource_default_uses_active_profile_default_chunk_duration_when_defaults_absent():
	payload = _resource_payload()
	resources = payload["resources"]
	assert isinstance(resources, dict)
	defaults = resources.get("defaults", {})
	assert isinstance(defaults, dict)
	defaults.pop("chunk_duration", None)

	assert get_resource_default(runtime_config=RuntimeConfig(payload), key="chunk_duration", default=None) == "1s"


@pytest.mark.parametrize(
	"payload",
	[
		{"resources": {"active_profile": "missing", "profiles": {}, "phase_budgets": {}}},
		{
			**_resource_payload(),
			"stages": {
				"preprocess": {
					"phases": {"prepare_raw_binaries": {"resource_class": "missing"}},
				}
			},
		},
	],
)
def test_invalid_resource_configuration_raises(payload):
	with pytest.raises(ValueError):
		parse_preprocess_stage_config(runtime_config=RuntimeConfig(payload))
