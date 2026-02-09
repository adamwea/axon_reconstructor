# Reconstruction

Runs axon reconstruction / velocity estimation using `axon_velocity`, consuming merged contributing-channels templates from the templates step.

Produces per-unit and all-units PDF summaries plus a JSON summary.

## Primary API

- Inputs: `axon_reconstructor.pipeline.reconstruction.ReconstructionInputs`
- Outputs: `axon_reconstructor.pipeline.reconstruction.ReconstructionOutputs`
- Runner: `axon_reconstructor.pipeline.reconstruction.reconstruct_from_templates(inputs=...)`

## Inputs

- `h5_path`, `stream_id`, `mea_output_root`
- unit selection: `unit_ids`, `unit_limit`
- `axon_velocity_params`: overrides for the axon_velocity call (best-effort filtered to accepted kwargs)
- plotting controls: `write_unit_pdfs`, `write_all_units_overview_pdf`
- `verbose`, `force_restart`

## Inputs consumed (from templates)

axon_velocity is most robust when given a *dense full-channel template* on a deterministic geometry.
By default, reconstruction consumes the full-channel templates emitted by the templates stage.

From `<well>/templates_outputs/full_channels_templates/unit_<id>/`:

- `full_template.npy`
- `full_channel_locations_xy.npy`
- `full_template_meta.json` (best-effort)

Reconstruction also reads (best-effort) sampling frequency metadata from the merged-contributing meta file:

From `<well>/templates_outputs/merged_units/unit_<id>/`:

- `merged_contributing_template_meta.json`

## Outputs (artifacts)

Under `<well>/reconstruction_outputs/`:

- `reconstruction_summary.json`
- `all_units_morphology.pdf` (optional)
- `by_unit/unit_<id>/`
  - per-unit JSON outputs and per-unit PDFs (when enabled)

## Mermaid flow

```mermaid
flowchart TD
  A[merged_contributing templates] --> B[reconstruct_from_templates]
  B --> C[reconstruction_outputs/reconstruction_summary.json]
  B --> D[reconstruction_outputs/all_units_morphology.pdf]
  B --> E[reconstruction_outputs/by_unit/unit_*/...]
```

## Notes

- This step uses a dedicated reconstruction checkpoint file so it can resume independently of earlier stages.
