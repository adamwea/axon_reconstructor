# Stage 03 — Waveforms

This stage extracts waveforms into SpikeInterface `SortingAnalyzer` artifacts and performs spike-level filtering/QC.

High-level entry point:

- `axon_reconstructor.pipeline.waveforms.extract_waveforms(...)`

Key feature that depends on Stage 01 preprocessing:

- `filter_by_maxwell_epochs`: drops spikes whose waveform window would cross Maxwell snippet boundaries.
  - This consumes `preprocess_outputs/maxwell_contiguous_epochs_<stream_id>.json`.

## Detailed stepwise docs

- Channel sets: [methods/methods_waveforms_channel_sets.md](methods/methods_waveforms_channel_sets.md)
- Step 01: [methods/methods_waveforms_step_01_setup_and_filtering.md](methods/methods_waveforms_step_01_setup_and_filtering.md)
- Step 02: [methods/methods_waveforms_step_02_concat_analyzer_extraction.md](methods/methods_waveforms_step_02_concat_analyzer_extraction.md)
- Step 03: [methods/methods_waveforms_step_03_per_segment_waveforms_extraction.md](methods/methods_waveforms_step_03_per_segment_waveforms_extraction.md)
- Step 04: [methods/methods_waveforms_step_04_cross_source_qc_reporting_and_persistence.md](methods/methods_waveforms_step_04_cross_source_qc_reporting_and_persistence.md)
- Step 05: [methods/methods_waveforms_step_05_metrics_curation_plotting_and_checkpoint.md](methods/methods_waveforms_step_05_metrics_curation_plotting_and_checkpoint.md)
