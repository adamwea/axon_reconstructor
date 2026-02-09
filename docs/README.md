# Docs

This folder is for project documentation.

Suggested next steps:
- Start here: [developer_setup.md](developer_setup.md)
- Stage docs (recommended reading order):
	- [stage_01_preprocessing.md](stage_01_preprocessing.md)
	- [stage_02_spikesorting.md](stage_02_spikesorting.md)
	- [stage_03_waveforms.md](stage_03_waveforms.md)
	- [stage_04_templates.md](stage_04_templates.md)
	- [stage_05_reconstruction.md](stage_05_reconstruction.md)
	- [stage_06_analysis.md](stage_06_analysis.md)
- Legacy/WIP methods note (superseded by the stage docs): [methods/methods_preprocess_spikesort.md](methods/methods_preprocess_spikesort.md)
- Add `architecture.md` (high-level design)
- Add `data_formats.md` (inputs/outputs)
- Add `developer_setup.md` (environments, HPC notes)

Current methods notes:
- [methods/methods_preprocess_spikesort.md](methods/methods_preprocess_spikesort.md)

Stepwise methods docs:
- Preprocessing
	- [methods/methods_preprocessing_step_01_setup_plan_and_output_layout.md](methods/methods_preprocessing_step_01_setup_plan_and_output_layout.md)
	- [methods/methods_preprocessing_step_02_common_electrodes_and_segment_normalization.md](methods/methods_preprocessing_step_02_common_electrodes_and_segment_normalization.md)
	- [methods/methods_preprocessing_step_03_concatenation_times_and_epoch_markers.md](methods/methods_preprocessing_step_03_concatenation_times_and_epoch_markers.md)
	- [methods/methods_preprocessing_step_04_temporal_resampling_and_epoch_scaling.md](methods/methods_preprocessing_step_04_temporal_resampling_and_epoch_scaling.md)
	- [methods/methods_preprocessing_step_05_persistence_resume_and_debugging.md](methods/methods_preprocessing_step_05_persistence_resume_and_debugging.md)

- Spikesorting
	- [methods/methods_spikesorting_step_01_path_contract_and_output_layout.md](methods/methods_spikesorting_step_01_path_contract_and_output_layout.md)
	- [methods/methods_spikesorting_step_02_building_driver_commands.md](methods/methods_spikesorting_step_02_building_driver_commands.md)
	- [methods/methods_spikesorting_step_03_in_process_debug_harness.md](methods/methods_spikesorting_step_03_in_process_debug_harness.md)
	- [methods/methods_spikesorting_step_04_outputs_validation_and_handoff.md](methods/methods_spikesorting_step_04_outputs_validation_and_handoff.md)
	- [methods/methods_spikesorting_step_05_troubleshooting_and_hpc_notes.md](methods/methods_spikesorting_step_05_troubleshooting_and_hpc_notes.md)

- Waveforms
	- [methods/methods_waveforms_step_01_setup_and_filtering.md](methods/methods_waveforms_step_01_setup_and_filtering.md)
	- [methods/methods_waveforms_step_02_concat_analyzer_extraction.md](methods/methods_waveforms_step_02_concat_analyzer_extraction.md)
	- [methods/methods_waveforms_step_03_per_segment_waveforms_extraction.md](methods/methods_waveforms_step_03_per_segment_waveforms_extraction.md)
	- [methods/methods_waveforms_step_04_cross_source_qc_reporting_and_persistence.md](methods/methods_waveforms_step_04_cross_source_qc_reporting_and_persistence.md)
	- [methods/methods_waveforms_step_05_metrics_curation_plotting_and_checkpoint.md](methods/methods_waveforms_step_05_metrics_curation_plotting_and_checkpoint.md)

- Templates
	- [methods/methods_templates_step_01_setup_resume_and_outputs.md](methods/methods_templates_step_01_setup_resume_and_outputs.md)
	- [methods/methods_templates_step_02_load_analyzers_and_select_units.md](methods/methods_templates_step_02_load_analyzers_and_select_units.md)
	- [methods/methods_templates_step_03_extract_templates_and_build_merged_contributing.md](methods/methods_templates_step_03_extract_templates_and_build_merged_contributing.md)
	- [methods/methods_templates_step_04_persist_templates_and_reconstruction_handoff.md](methods/methods_templates_step_04_persist_templates_and_reconstruction_handoff.md)
	- [methods/methods_templates_step_05_plotting_summary_and_checkpoint.md](methods/methods_templates_step_05_plotting_summary_and_checkpoint.md)

- Reconstruction
	- [methods/methods_reconstruction_README.md](methods/methods_reconstruction_README.md)
	- [methods/methods_reconstruction_step_01_setup_resume_and_outputs.md](methods/methods_reconstruction_step_01_setup_resume_and_outputs.md)
	- [methods/methods_reconstruction_step_02_load_templates_and_select_units.md](methods/methods_reconstruction_step_02_load_templates_and_select_units.md)
	- [methods/methods_reconstruction_step_03_run_axon_velocity_and_persist_outputs.md](methods/methods_reconstruction_step_03_run_axon_velocity_and_persist_outputs.md)
	- [methods/methods_reconstruction_step_04_plotting.md](methods/methods_reconstruction_step_04_plotting.md)
	- [methods/methods_reconstruction_step_05_summary_and_checkpoint.md](methods/methods_reconstruction_step_05_summary_and_checkpoint.md)

- Analysis
	- [methods/methods_analysis_step_01_setup_checkpoint_and_outputs.md](methods/methods_analysis_step_01_setup_checkpoint_and_outputs.md)
	- [methods/methods_analysis_step_02_unit_discovery_and_panel_contract.md](methods/methods_analysis_step_02_unit_discovery_and_panel_contract.md)
	- [methods/methods_analysis_step_03_rendering_backend_and_dependencies.md](methods/methods_analysis_step_03_rendering_backend_and_dependencies.md)
	- [methods/methods_analysis_step_04_resume_overwrite_and_summary_json.md](methods/methods_analysis_step_04_resume_overwrite_and_summary_json.md)
	- [methods/methods_analysis_step_05_extending_panels_and_custom_runs.md](methods/methods_analysis_step_05_extending_panels_and_custom_runs.md)
