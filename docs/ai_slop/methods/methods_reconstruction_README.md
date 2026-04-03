# Reconstruction Methods Docs (Stepwise)

This folder contains a step-by-step description of the reconstruction stage. It is intended to match the style used for waveforms and templates.

## Files

1. `methods_reconstruction_step_01_setup_resume_and_outputs.md`
   - runner entry point, logging, checkpoints, output layout, resume shortcuts

2. `methods_reconstruction_step_02_load_templates_and_select_units.md`
   - templates outputs discovery, unit id selection, full-channel vs merged-contributing templates

3. `methods_reconstruction_step_03_run_axon_velocity_and_persist_outputs.md`
   - axon_velocity invocation, parameter filtering, and persisted JSON artifacts

4. `methods_reconstruction_step_04_plotting.md`
   - per-unit PDFs and all-units overview PDF

5. `methods_reconstruction_step_05_summary_and_checkpoint.md`
   - summary JSON schema and final checkpoint update

## Notes

- Reconstruction defaults to consuming templates-stage full-channel templates under `stg4_templates_outputs/full_channels_templates/` because `axon_velocity` is most robust with dense templates on a deterministic geometry.
- Sampling frequency is currently sourced from the merged-contributing meta JSON when available.
