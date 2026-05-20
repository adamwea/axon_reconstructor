from __future__ import annotations

import pytest

from axon_recon.runtime_config import RuntimeConfig

from ..config import (
	DEFAULT_ANALYSIS_PHASE_SEQUENCE,
	_ANALYSIS_PHASE_ALIASES,
	AnalysisStageConfig,
	build_well_metadata_lookup,
	normalize_analysis_phase_name,
	parse_analysis_stage_config,
)


def test_default_phase_sequence_is_compute_metrics_then_unitmatch() -> None:
	# Per unitmatch_phase_plan.md slice 1: the unitmatch phase joins the
	# default analysis sequence, but defaults to enabled:false so it's a
	# no-op until the user opts in via YAML.
	assert DEFAULT_ANALYSIS_PHASE_SEQUENCE == (
		"compute_metrics",
		"unitmatch",
		"propagation_video",
	)


def test_parse_defaults() -> None:
	parsed = parse_analysis_stage_config(runtime_config=RuntimeConfig({"stages": {"analysis": {}}}))
	assert isinstance(parsed, AnalysisStageConfig)
	assert parsed.output_rel_root == "analysis_outputs"
	assert parsed.phase_sequence == ("compute_metrics", "unitmatch", "propagation_video")
	assert parsed.compute_metrics_enabled is True
	assert parsed.compute_metrics_resource_class is None
	assert parsed.unitmatch_enabled is False
	assert parsed.unitmatch_resource_class is None
	assert parsed.unitmatch_rel_output_root == "unitmatch"
	assert parsed.propagation_video_enabled is False
	assert parsed.propagation_video_resource_class is None
	assert parsed.propagation_video_rel_output_root == "propagation_video"
	assert parsed.manifest_relpath == "manifest.json"
	assert parsed.tables_relpath == "tables"
	assert parsed.recon_output_rel_root == "recon_outputs"
	assert parsed.well_metadata_lookup == {}


def test_parse_phase_block() -> None:
	cfg = RuntimeConfig(
		{
			"stages": {
				"analysis": {
					"output_rel_root": "alt_analysis",
					"phase_sequence": ["compute_metrics"],
					"phases": {
						"compute_metrics": {"enabled": False},
					},
				}
			}
		}
	)
	parsed = parse_analysis_stage_config(runtime_config=cfg)
	assert parsed.output_rel_root == "alt_analysis"
	assert parsed.phase_sequence == ("compute_metrics",)
	assert parsed.compute_metrics_enabled is False


def test_phase_alias_normalization() -> None:
	for alias, canonical in _ANALYSIS_PHASE_ALIASES.items():
		assert normalize_analysis_phase_name(alias) == canonical
		assert normalize_analysis_phase_name(f"analysis.{alias}") == canonical


def test_phase_alias_rejects_unknown() -> None:
	with pytest.raises(ValueError):
		normalize_analysis_phase_name("totally_made_up")


def test_build_well_metadata_lookup_threads_div_and_attributes() -> None:
	data_cfg = RuntimeConfig(
		{
			"datasets": [
				{
					"raw_data_h5_path": (
						"/srv/raw/Media_Density_T5_02182026_AR/260319/M06804/AxonTracking/000174/data.raw.h5"
					),
					"include_in_runtime": True,
					"DIV": 36,
					"wells": [
						{
							"well_id": "well000",
							"include_in_runtime": True,
							"attributes": {
								"plating_density": 80000,
								"media": "DMEM",
								"genotype": "WT",
							},
						},
						{
							"well_id": "well001",
							"attributes": {"genotype": "KO"},
						},
					],
				}
			]
		}
	)
	lookup = build_well_metadata_lookup(data_cfg)
	assert (0, "well000") in lookup
	entry = lookup[(0, "well000")]
	assert entry["DIV"] == 36
	assert entry["project"] == "Media_Density_T5_02182026_AR"
	assert entry["recording_date"] == "2026-03-19"
	assert entry["chip_id"] == "M06804"
	assert entry["scan_type"] == "AxonTracking"
	assert entry["run_id"] == "000174"
	assert entry["well_attributes"] == {
		"plating_density": 80000,
		"media": "DMEM",
		"genotype": "WT",
	}
	assert lookup[(0, "well001")]["well_attributes"] == {"genotype": "KO"}


def test_parse_uses_data_config_for_well_metadata_lookup() -> None:
	data_cfg = RuntimeConfig(
		{
			"datasets": [
				{
					"raw_data_h5_path": (
						"/srv/raw/Media_Density_T5_02182026_AR/260326/M08073/AxonTracking/000208/data.raw.h5"
					),
					"DIV": 6,
					"wells": [
						{"well_id": "well000", "attributes": {"genotype": "WT"}},
					],
				}
			]
		}
	)
	parsed = parse_analysis_stage_config(
		runtime_config=RuntimeConfig({"stages": {"analysis": {}}}),
		data_config=data_cfg,
	)
	assert (0, "well000") in parsed.well_metadata_lookup
	entry = parsed.well_metadata_lookup[(0, "well000")]
	assert entry["DIV"] == 6
	assert entry["recording_date"] == "2026-03-26"
	assert entry["chip_id"] == "M08073"
	assert entry["run_id"] == "000208"


def test_phase_alias_for_analysis_short_form() -> None:
	assert normalize_analysis_phase_name("metrics") == "compute_metrics"
	assert normalize_analysis_phase_name("compute") == "compute_metrics"
	assert normalize_analysis_phase_name("analysis.metrics") == "compute_metrics"
