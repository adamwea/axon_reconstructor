# Reconstruction Step — Part 2: Load Templates Artifacts and Select Units

Scope: this document covers how reconstruction locates templates-stage artifacts on disk and determines the unit list.

Primary code path:
- `axon_reconstructor.pipeline.reconstruction.runner.reconstruct_from_templates`

---

## 1. Templates output roots

Reconstruction expects templates to have written under:

- `<well>/stg4_templates_outputs/`

Key subdirectories:

- `merged_units/`
  - contains the per-unit merged-contributing artifacts
- `full_channels_templates/` (preferred)
  - contains the per-unit dense templates on a deterministic “full channels” axis

---

## 2. Why reconstruction prefers full-channel templates

`axon_velocity.compute_graph_propagation_velocity(template_ch_by_t, locs_xy, fs_hz, ...)` is most robust when given:

- a template defined over a stable, deterministic geometry
- and a dense channel axis where non-contributing/quiet channels can be represented as zeros

Therefore, reconstruction defaults to consuming per-unit artifacts under:

- `<well>/stg4_templates_outputs/full_channels_templates/unit_<id>/`

This is controlled by `ReconstructionInputs`:

- `use_full_channels_templates=True` (default)
- `require_full_channels_templates=True` (default)

If the directory is missing and `require_full_channels_templates=True`, reconstruction raises with guidance to re-run templates.

---

## 3. Artifact paths per unit

For each unit `uid`, reconstruction tracks both:

### 3.1 Dense full-channel template (preferred)

From `<well>/stg4_templates_outputs/full_channels_templates/unit_<uid>/`:

- `full_template.npy`
  - shape `(n_samples, n_full_channels)`
- `full_channel_locations_xy.npy`
  - shape `(n_full_channels, 2)`
- `full_template_meta.json`
  - best-effort mapping diagnostics; not required to run

### 3.2 Merged contributing-channels metadata (best-effort)

From `<well>/stg4_templates_outputs/merged_units/unit_<uid>/`:

- `merged_contributing_template_meta.json`
  - used primarily to get `sampling_frequency_hz`

Note:
- Reconstruction does not need the merged-contributing waveform array when full-channel templates are available; it only uses the metadata as a convenient source of `fs_hz`.

---

## 4. Unit discovery and selection

Reconstruction discovers unit ids from:

- `<well>/stg4_templates_outputs/merged_units/unit_*` directory names

Selection rules:

- If `inputs.unit_ids` is provided, use that list.
- Else use the discovered unit ids.
- If `inputs.unit_limit` is provided, truncate to the first `unit_limit` units.

---

## End of Part 2

At this point, reconstruction has:

- located the templates outputs directories
- determined a unit list
- established which template representation it will pass into axon_velocity

Next, the stage runs axon_velocity per unit and persists results. That is covered in **Part 3**.
