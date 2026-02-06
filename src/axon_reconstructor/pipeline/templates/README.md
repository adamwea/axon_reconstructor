# Templates

Extracts per-unit templates from waveforms analyzers and builds a **merged_contributing** template per unit across sources.

This step produces QC PDFs and a JSON summary.

## Scientific methods (data handling)

This stage is designed to produce *scientifically defensible* per-unit template waveforms for downstream reconstruction.

- **Source of truth**: templates are derived from the waveforms-stage `SortingAnalyzer` artifacts.
  This stage does **not** re-run spike detection/sorting.
- **No downstream spike-level filtering**: spike-level exclusions belong to the waveforms stage.
  This stage only selects which unit ids are processed (by default using spikesorting-stage curation from `<well>/spikesorting_outputs/qm_unfiltered.xlsx`).
- **Multi-source rationale (concat vs per-segment)**:
  - The concat analyzer provides templates on the stable channel set used for sorting.
  - Optional per-segment analyzers can recover templates on channels absent from concat (e.g., not present in all segments).
- **Contributing channel set (across sources)**: for each unit, a `merged_contributing` template is constructed by merging
  the unit's contributing channels observed across sources.
- **Overlap handling**: when multiple sources contribute the *same physical channel* (matched by electrode id/channel id
  or spatial location), the waveform for that channel is merged using a mean-of-waveforms strategy (best-effort) to avoid
  keep-first bias.
- **Footprints**: the peak-to-peak amplitude (PTP) across time is computed per channel for the `merged_contributing` template.
  Both linear and log-scaled visualizations are written as quick-look QC.

## Primary API

- Inputs: `axon_reconstructor.pipeline.templates.TemplateExtractInputs`
- Outputs: `axon_reconstructor.pipeline.templates.TemplateExtractOutputs`
- Runner: `axon_reconstructor.pipeline.templates.extract_and_merge_templates(inputs=...)`

## Inputs

- `h5_path`, `stream_id`, `mea_output_root`
- include sources: `include_concat`, `include_segments`
- unit selection: `unit_ids`, `unit_limit`
- auto-merge (optional): `run_unit_merging`, `merge_presets`, `merge_recursive`
- plotting controls: `plot_templates_grid_pdf`, `plot_multi_source_templates_pdf`, `top_channels_per_template`
- footprints: `plot_merged_contributing_footprints_linear_and_log`
- reconstruction handoff: `save_full_channels_templates`
- extra plots: `plot_topo_unit_footprints`, `plot_propagation_plots`, `propagation_top_channels`, `propagation_n_waveforms`
- `n_jobs`, `force_restart`

This step reads waveforms analyzers from:

- `<well>/waveforms_outputs/concat_waveforms/`
- `<well>/waveforms_outputs/segment_waveforms/*/`

## Outputs (artifacts)

Under `<well>/templates_outputs/`:

- Extracted templates
  - `extracted_templates/` (per-source `.npy` templates + metadata)
- Merged templates
  - `merged_units/unit_<id>/` (data-only)
    - `merged_contributing_template.npy`
    - `merged_contributing_channel_locations.npy`
    - `merged_contributing_template_meta.json`
    - `merged_contributing_footprint_ptp.npy` (peak-to-peak amplitude per channel)
    - `axon_velocity_inputs.npz` (convenience bundle for plotting/reconstruction)
- Merged template QC (plots)
  - `footprints/`
    - `unit_<id>_merged_contributing_footprint_ptp_linear.png`
    - `unit_<id>_merged_contributing_footprint_ptp_log.png`
  - `svgs/`
    - `unit_<id>_merged_contributing_template_footprint_linear.svg`
    - `unit_<id>_merged_contributing_template_footprint_log.svg`
  - `full_chip_maps/`
    - `unit_<id>_template_amplitude_map_full_chip.png`
    - `unit_<id>_template_peak_latency_map_full_chip.png`
- QC PDFs
  - `templates_grid.pdf`
  - `unit_segment_grids/` (optional per-unit overlays)
    - `unit_<id>_templates.pdf`
    - `unit_<id>_footprints.pdf`
- Reconstruction handoff (dense)
  - `full_channels_templates/unit_<id>/`
    - `full_template.npy` (zeros on non-contributing channels)
    - `full_channel_locations_xy.npy`
    - `full_channel_ids.npy`
    - `full_electrode_ids.npy` (when available)
    - `contributing_full_channel_indices.npy`
- Additional plots
  - `topo_unit_footprints/unit_<id>.png` (3D PTP height map)
  - `propagation_plots/unit_<id>.pdf` (merged template waveforms, top channels ordered by peak latency)
- Optional real axon-velocity bundle (opt-in)
  - `axon_velocity_outputs/unit_<id>/...`
- Summaries
  - `templates_summary.json`

## Exclusions + curation

- `wf_exclusions.npz` is deprecated and not required.
- This step does not apply spike-level exclusions.
- By default, this step processes only curated units derived from spikesorting-stage quality metrics
  (from `<well>/spikesorting_outputs/qm_unfiltered.xlsx`).
  To override, pass `unit_ids=[...]` explicitly or set `require_curated_units=False`.

## Mermaid flow

```mermaid
flowchart TD
  A[waveforms analyzers] --> B[extract_and_merge_templates]
  B --> C[extracted_templates/*.npy]
  B --> D[merged_units/unit_*/merged_contributing_template.npy]
  B --> E[templates_grid_*.pdf]
  B --> F[templates_summary.json]
  D --> G[reconstruction]
```

## Notes

- The merged_contributing outputs are the canonical handoff for reconstruction.
