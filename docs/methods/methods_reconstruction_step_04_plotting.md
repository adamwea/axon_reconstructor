# Reconstruction Step — Part 4: Plotting (Per-Unit PDFs + All-Units Overview)

Scope: this document covers the plotting artifacts produced by reconstruction.

Primary code paths:
- `axon_reconstructor.pipeline.reconstruction.plotting.write_unit_reconstruction_pdfs`
- `axon_reconstructor.pipeline.reconstruction.plotting.write_all_units_overview_pdf`

---

## 1. Per-unit PDFs

If `ReconstructionInputs.write_unit_pdfs=True`, reconstruction attempts to write per-unit PDFs under:

- `<well>/stg5_reconstruction_outputs/by_unit/unit_<id>/`

These plots are best-effort.

Inputs include:

- `gtr` object returned by axon_velocity
- `locs_xy` used in the axon_velocity call (full-channel geometry by default)

Scientific intent:
- provide quick-look QC for morphology, selected channels, and velocity fits.

---

## 2. All-units overview PDF

If `ReconstructionInputs.write_all_units_overview_pdf=True`, reconstruction attempts to write:

- `<well>/stg5_reconstruction_outputs/all_units_morphology.pdf`

This plot is built from:

- per-unit branch polylines (XY)
- per-unit channel locations

It is intended as a dataset-level morphology overview.

---

## End of Part 4

At this point, reconstruction has:

- written per-unit QC PDFs (if enabled)
- written an all-units morphology overview (if enabled)

Next, the stage writes the summary JSON and completes checkpointing. That is covered in **Part 5**.
