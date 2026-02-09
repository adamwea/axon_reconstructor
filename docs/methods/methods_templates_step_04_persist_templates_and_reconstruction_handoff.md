# Templates Step — Part 4: Persistence, Full-Channel Templates, and Reconstruction Handoff

Scope: this document covers what the templates stage writes to disk for each unit, with emphasis on the files consumed downstream by reconstruction.

Primary code path:
- `axon_reconstructor.pipeline.templates.extraction._persist_unit_templates(...)`

Key outputs live under:
- `<well>/templates_outputs/`

---

## 1. Two persistence families: per-source vs merged-contributing

For each unit, templates writes:

1. **Per-source template arrays** (data extracted from each analyzer)
2. **Merged-contributing outputs** (the canonical handoff used downstream)

### 1.1 Per-source templates (`extracted_templates/`)

For a unit `uid` and a source `src_name` (e.g. `concat`, `seg_...`):

- `<well>/templates_outputs/extracted_templates/<src_name>/unit_<uid>.npy`
  - numpy array shaped `(n_samples, n_channels_for_that_source)`

- `<well>/templates_outputs/extracted_templates/<src_name>/unit_<uid>_meta.json`
  - JSON metadata, including channel ids, electrode ids, sampling frequency, and plotting window hints.

These are primarily for debugging/QC and for multi-source overlays.

### 1.2 Per-unit merged-contributing outputs (`merged_units/unit_<id>/`)

For each `uid`, templates writes into:

- `<well>/templates_outputs/merged_units/unit_<uid>/`

Core data:

- `merged_contributing_template.npy`
  - array shaped `(n_samples, n_contributing_channels)`

- `merged_contributing_channel_locations.npy`
  - array shaped `(n_contributing_channels, 2)` (XY)

- `merged_contributing_channel_ids.npy` (best-effort)
  - best-effort channel identifiers for the merged-contributing channels.
  - typically an object array shaped `(n_contributing_channels,)`, but may be missing/degenerate (e.g. saved as a scalar `None`) if the underlying analyzer/recording did not provide stable channel ids.

- `merged_contributing_electrode_ids.npy` (best-effort)
  - best-effort electrode identifiers for the merged-contributing channels.
  - typically an object array shaped `(n_contributing_channels,)`, but may be missing/degenerate (e.g. saved as a scalar `None`) if electrode ids could not be extracted from recording properties.

Derived convenience data:

- `merged_contributing_footprint_ptp.npy` (best-effort)
  - peak-to-peak amplitude across time, shape `(n_contributing_channels,)`

- `axon_velocity_inputs.npz` (best-effort)
  - convenience bundle expected by downstream tooling:
    - `template_ch_by_t` (channels × time)
    - `locations_xy`
    - `sampling_frequency_hz`
    - `channel_ids` / `electrode_ids` when available

Metadata:

- `merged_contributing_template_meta.json`
  - includes pointers to the arrays above plus overlap diagnostics (`stats`, `overlap`) when present.

Scientific intent:
- `merged_contributing_template.npy` + `merged_contributing_channel_locations.npy` are the minimal sufficient “footprint definition” for reconstruction.

---

## 2. Full-chip QC maps written for merged-contributing templates

In addition to the unit-local `merged_units/unit_<id>/` outputs, templates writes “full chip” visualizations under:

- `<well>/templates_outputs/full_chip_maps/`

Per unit:

- `unit_<uid>_template_amplitude_map_full_chip.png`
  - shows template amplitude (typically PTP) on a full Maxwell grid

- `unit_<uid>_template_peak_latency_map_full_chip.png`
  - shows per-electrode peak latency (ms) on a full Maxwell grid

These plots use a tri-state rendering:

- **quiet electrodes**: not present in any recording/analyzer electrode universe
- **non-contributing electrodes**: present in the recording universe but not in this template
- **contributing electrodes**: electrodes with a template value

To support that distinction, templates tracks the **all recorded electrodes** set (union over analyzers) and passes it into plotting:
- electrode ids that appear in at least one templates-stage recording/analyzer (union over sources)

---

## 3. Optional: dense “full channel” templates for reconstruction

If `TemplateExtractInputs.save_full_channels_templates=True`, templates will also write a *dense* template in a deterministic “full channels order” so downstream reconstruction can index directly into a fixed geometry.

These outputs live under:

- `<well>/templates_outputs/full_channels_templates/`

### 3.1 Recording electrode universe

Once per run (best-effort), templates writes:

- `all_recorded_electrode_ids.npy`
- `all_recorded_electrode_ids_meta.json`

Definition:
- electrode ids that appear in **at least one** templates-stage recording/analyzer (union over sources).

This is used to distinguish “quiet” vs “present but noncontributing” in later disk-only replotting.

### 3.2 Per-unit dense outputs

For each unit `uid`:

- `<well>/templates_outputs/full_channels_templates/unit_<uid>/full_template.npy`
  - array shaped `(n_samples, n_full_channels)`
  - the merged-contributing waveform placed into the chosen full-channel order
  - all non-contributing channels are zero

- `full_channel_locations_xy.npy`
- `full_channel_ids.npy` (best-effort)
- `full_electrode_ids.npy` (best-effort)
- `contributing_full_channel_indices.npy`
  - indices into the full channel axis that were filled by the merged-contributing mapping

- `full_template_meta.json`
  - includes mapping strategy used and counts

### 3.3 Full geometry selection (Maxwell-aware)

Templates attempts to detect when we are on a Maxwell full-chip electrode id scheme.

If yes:
- it uses a deterministic 220×120 grid geometry (17.5 µm pitch)
- and uses electrode ids as the canonical identity.

If not:
- it falls back to the reference analyzer’s recording geometry.

### 3.4 Channel mapping strategy (preferred order)

When mapping merged-contributing channels into the full template axis, templates tries:

1. map by `electrode_ids` (most stable)
2. else map by `channel_ids`
3. else map by (x, y) locations with a tolerance-derived key

Channels that cannot be mapped are skipped.

Scientific rationale:
- Reconstruction benefits from a stable “global” indexing scheme and the ability to represent missing channels as zeros.

---

## 4. Optional: axon_velocity plot bundle

If `TemplateExtractInputs.plot_axon_velocity_outputs=True`, templates will attempt to generate an axon_velocity plot bundle under:

- `<well>/templates_outputs/axon_velocity_outputs/unit_<uid>/...`

This is best-effort and requires extra dependencies.

---

## End of Part 4

At this point, templates has produced:

- extracted per-source templates
- merged-contributing templates and per-unit reconstruction handoff files
- (optionally) dense full-channel templates and axon_velocity outputs

Next, templates produces the QC PDFs/PNGs/SVGs and final stage summary + checkpoint completion.

That is covered in **Part 5**.
