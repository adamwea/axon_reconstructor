# Templates

Extracts per-unit templates from waveforms analyzers and builds a **merged_union** template per unit across sources.

This step produces QC PDFs and a JSON summary.

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
- `n_jobs`, `force_restart`

This step reads waveforms analyzers from:

- `<well>/waveforms_outputs/concat_waveforms/`
- `<well>/waveforms_outputs/segment_waveforms/*/`

## Outputs (artifacts)

Under `<well>/templates_outputs/`:

- Extracted templates
  - `extracted_templates/` (per-source `.npy` templates + metadata)
- Merged templates
  - `merged_union_by_unit/unit_<id>/`
    - `merged_union_template.npy`
    - `merged_union_channel_locations.npy`
    - `merged_union_template_meta.json`
    - `merged_union_footprint_ptp.npy` (peak-to-peak amplitude per channel)
    - `axon_velocity_inputs.npz` (convenience bundle for plotting/reconstruction)
- QC PDFs
  - `templates_grid.pdf`
  - `multi_source_by_unit/` (optional per-unit overlays)
- Summaries
  - `templates_summary.json`

## Exclusions + curation

- `wf_exclusions.npz` is deprecated and not required.
- This step does not apply spike-level exclusions; curation here refers only to which unit ids are processed (waveforms-stage curation when available).

## Mermaid flow

```mermaid
flowchart TD
  A[waveforms analyzers] --> B[extract_and_merge_templates]
  B --> C[extracted_templates/*.npy]
  B --> D[merged_union_by_unit/unit_*/merged_union_template.npy]
  B --> E[templates_grid_*.pdf]
  B --> F[templates_summary.json]
  D --> G[reconstruction]
```

## Notes

- The merged_union outputs are the canonical handoff for reconstruction.
