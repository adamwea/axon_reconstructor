# Reconstruction Step — Part 3: Run axon_velocity and Persist Outputs

Scope: this document covers the per-unit axon_velocity call and the core JSON persistence performed by reconstruction.

Primary code path:
- `axon_reconstructor.pipeline.reconstruction.runner.reconstruct_from_templates`

---

## 1. Parameters passed into axon_velocity

Reconstruction builds axon_velocity params as:

1. Start from defaults:

- `params = axon_velocity.get_default_graph_velocity_params()`

2. Merge user overrides:

- `params.update(inputs.axon_velocity_params)` (if provided)

3. Filter to only kwargs accepted by the target callable:

- `params = _filter_kwargs_for_callable(axon_velocity.compute_graph_propagation_velocity, params)`

4. Force verbosity:

- `params.setdefault("verbose", inputs.verbose)`

Scientific intent:
- keep reproducibility while allowing controlled tuning.

---

## 2. Building axon_velocity inputs from templates artifacts

For each unit:

1. Load template + locations.

Preferred (full-channel):

- `tmpl = np.load(full_template.npy)` with shape `(n_samples, n_full_channels)`
- `locs_xy = np.load(full_channel_locations_xy.npy)[:, :2]` with shape `(n_full_channels, 2)`

Fallback (merged-contributing):

- `tmpl = np.load(merged_contributing_template.npy)` with shape `(n_samples, n_contributing_channels)`
- `locs_xy = np.load(merged_contributing_channel_locations.npy)[:, :2]`

2. Convert template orientation for axon_velocity:

- pipeline saves templates as `(time, channels)`
- axon_velocity expects `(channels, time)`

So we use:

- `tmpl_ch_by_t = tmpl.T`

3. Sampling frequency (`fs_hz`)

Reconstruction reads `sampling_frequency_hz` from:

- `<well>/templates_outputs/merged_units/unit_<id>/merged_contributing_template_meta.json`

If unavailable, it falls back to `10_000.0` Hz.

---

## 3. The axon_velocity call

Per unit:

- `gtr = axon_velocity.compute_graph_propagation_velocity(tmpl_ch_by_t, locs_xy, fs_hz, **params)`

The returned object (`gtr`) is treated as a black box and accessed via best-effort attribute reads.

---

## 4. Persisted JSON outputs per unit

Under:

- `<well>/reconstruction_outputs/by_unit/unit_<id>/`

Reconstruction writes:

### 4.1 `branches.json`

A simplified, stable representation of per-branch results:

- channels list
- a polyline in XY coordinates (derived by indexing `locs_xy` by channel)
- fit quantities (velocity, r2, pval, etc.)
- optional timing vectors

### 4.2 `heuristics.json`

A simplified representation of axon_velocity channel selection / heuristics:

- `init_channel`
- `selected_channels`
- counts and best-effort node-heuristic vector

These JSONs are intended to be easy to diff, inspect, and re-use in downstream analysis.

---

## End of Part 3

At this point, reconstruction has:

- run axon_velocity for each unit
- written per-unit JSON artifacts

Next, reconstruction writes per-unit and all-units PDFs. That is covered in **Part 4**.
