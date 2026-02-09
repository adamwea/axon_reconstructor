# Templates Step — Part 3: Extract Per-Source Templates and Build `merged_contributing`

Scope: this document covers the scientific/algorithmic core of the templates stage:

- extracting per-unit templates from each waveforms-stage analyzer source
- handling sparse analyzers/templates
- building a per-unit **merged contributing-channels** template across sources
- resolving overlaps for channels present in multiple sources

Primary code path:
- `axon_reconstructor.pipeline.templates.processing.process_unit_list(...)`

Related modules:
- `axon_reconstructor.pipeline.templates.extraction` (per-source template gathering)
- `axon_reconstructor.pipeline.templates.utils._build_merged_contributing_template_for_unit` (merge + overlap handling)
- `axon_reconstructor.pipeline.templates.overlaps` (mean-waveform overlap resolution utilities)

---

## 1. Ensure templates extension exists (best-effort)

Before iterating units, `process_unit_list` ensures each analyzer has a `templates` extension:

- For each analyzer `an`:
  - if `not an.has_extension("templates")`, attempt `an.compute(["templates"], n_jobs=...)`
  - failures are swallowed (best-effort)

Scientific intent:
- templates should be the *authoritative representation* of the unit footprint for that analyzer.

---

## 2. Gather template sources for a unit

For each unit id `uid`, we call:

- `_gather_template_sources_for_unit(uid, analyzers, ...)`

It returns `sources_for_unit: list[dict]` with entries like:

- `name`: source name (e.g., `concat`, `seg_0`, ...)
- `template`: array shaped `(n_samples, n_channels_for_this_source)`
- `channel_locations`: array shaped `(n_channels, 2 or 3)` from `recording.get_channel_locations()`
- `channel_ids`: best-effort list from `recording.get_channel_ids()` (may be None)
- `electrode_ids`: best-effort list from recording properties (may be None)
- `_analyzer`: runtime-only pointer to the analyzer (used for overlap resolution; not serialized)

### 2.1 Template extraction from SpikeInterface

The function prefers:

- `an.get_extension("templates")` then `templates_ext.get_unit_template(unit_id)`

Compatibility fallbacks exist in `multi_source_utils._get_unit_template_from_extension`.

If a template can’t be loaded for a given source+unit, that source is skipped for that unit.

### 2.2 Sparse analyzers and sparse templates

Waveforms analyzers may be sparse (unit-specific channel subsets). SpikeInterface versions vary:

- some versions return dense templates with zeros outside sparsity
- some return already-sparse templates

To avoid false “overlap” and to keep merging meaningful, `_gather_template_sources_for_unit` tries to enforce sparsity-aware channel selection:

1. Detect a `ChannelSparsity` object via:
   - `an.sparsity` or `an.get_extension("waveforms").sparsity`

2. Use `multi_source_utils._sparsity_unit_channel_indices(sparsity, unit_id)` to obtain channel indices.

3. If the template looks dense (template channels equals recording channels) and sparsity indices are smaller:
   - subset template columns and the matching channel metadata arrays.

4. If the template is sparse but locations are still full:
   - subset locations/ids so `template.shape[1] == locs.shape[0]`.

If after best-effort alignment `template.shape[1] != channel_locations.shape[0]`, the source is skipped.

Scientific rationale:
- merging is meaningful only if the channel identities are correct.

---

## 3. Build the `merged_contributing` template (contributing channels across sources)

After gathering sources for a unit, templates constructs a merged template across sources:

- `merged_contributing = _build_merged_contributing_template_for_unit(sources_for_unit, unit_id, logger)`

The intent is that `merged_contributing` contains one waveform per *contributing* channel for that unit, where “contributing” means:

- the channel appears in at least one per-source template for that unit.

### 3.1 Channel identity keys

Each channel in each source is assigned a key.

Preference order:

1. **electrode id** key: `("electrode", int(electrode_id))`
2. else **channel id** key: `("channel", str(channel_id))`
3. else **location** key: `("loc", loc_key(xy, tol))`

Where:

- `tol` is inferred from channel locations (nearest-neighbor spacing / 4, best-effort)
- `loc_key` bins locations to a grid by rounding `x/tol` and `y/tol`

This is what defines “the same physical channel” across sources.

### 3.2 Baseline centering per channel

When a channel is first added to the merged set, its per-channel waveform is baseline-centered:

- `wf = wf - robust_baseline_pre_negative_peak(wf)`

This is intended to make merged templates baseline-consistent across sources.

(Implementation lives in `axon_reconstructor.pipeline.templates.overlaps`.)

### 3.3 Overlap detection

If a channel key is seen again (same electrode/channel/location from a different source), it is considered an overlap.

Rather than simply skipping or overwriting, the code records “contributions” so it can attempt a mean-of-waveforms merge:

- A `WaveformContribution` stores:
  - source name
  - analyzer pointer
  - unit id
  - channel reference (electrode id or channel id if available, else index)

### 3.4 Overlap resolution strategy: mean of contributing waveforms

For each key with multiple contributions:

- call `mean_waveform_from_contributions(contributions, logger)`

The intent is:

- pull the underlying waveforms for that unit+channel from each contributing analyzer
- stack those waveforms and take a mean

This mimics template computation and avoids “keep-first” bias.

If this fails, the code falls back to keeping the original waveform from the first source.

The merged waveform is baseline-centered again after merging.

### 3.5 What `merged_contributing` contains

If successful, `merged_contributing` is a dict entry compatible with the per-source entries, plus diagnostics:

- `name: "merged_contributing"`
- `template: (n_samples, n_contributing_channels)`
- `channel_locations: (n_contributing_channels, 2)`
- `channel_ids: list[... ]` (best-effort)
- `electrode_ids: list[... ]` (best-effort)
- `channel_source_names: list[str]` for provenance per channel
- `stats`: overlap counts and strategy
- `overlap`: per-channel overlap details (if any)

---

## 4. What the main templates grid plots

`process_unit_list` uses `merged_contributing` as the canonical representation for the main grid:

- grid entries use `merged_contributing["template"]` and `merged_contributing["channel_locations"]`

If `merged_contributing` is `None`, the unit is typically skipped for the grid output.

---

## End of Part 3

At this point, for each unit we have:

- per-source templates (concat + segment sources that contain the unit)
- a merged contributing-channels template that unifies channel identities across sources
- overlap-resolution diagnostics

Next, templates persists these artifacts to disk in a reconstruction-friendly structure and generates plots.

Persistence details are covered in **Part 4**.
