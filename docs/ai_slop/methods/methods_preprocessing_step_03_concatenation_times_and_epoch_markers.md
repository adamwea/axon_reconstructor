# Preprocessing Step — Part 3: Concatenation, Time Vectors, and Epoch Marker Generation

Scope: this document covers how preprocessing creates a single concatenated recording and attaches timing metadata that preserves triggered/snippet gaps when available.

Primary code path:
- `axon_reconstructor.pipeline.preprocessing.runner.build_concatenated_recording(...)`

Related modules:
- `axon_reconstructor.pipeline.preprocessing.h5_helpers`
  - `_read_well_rec_frame_nos_and_trigger_settings(...)`
  - assay/data_store stats helpers (diagnostics)

---

## 1. Concatenation

After per-segment conditioning, preprocessing concatenates recordings:

- `multirecording = spikeinterface.full.concatenate_recordings(rec_list)`

Important contract:
- The concatenated recording defines the **authoritative sample index space** used by later stages.
  - Sorter outputs are interpreted in concat sample coordinates.
  - Epoch markers are recorded in concat sample coordinates.

---

## 2. Concatenation stitch epochs (`concat_epochs`)

Preprocessing computes segment stitch epochs in concat sample coordinates:

- `seg_lengths = [r.get_num_samples() for r in rec_list]`
- `seg_offsets = cumulative_sum(seg_lengths)`

Then it emits a list of dictionaries (one per segment) with:

- `segment_index`
- `rec_name`
- `start_sample` / `end_sample`
- `n_samples`

These become:

- `concatenation_stitch_epochs_<stream_id>.json`

Usage downstream:
- waveforms stage uses these stitch epochs when it needs to reason about per-segment extraction in segment-local coordinates.

---

## 3. Time vector reconstruction from `frame_nos`

Maxwell triggered/snippet acquisitions can store samples that are *discontinuous in wall-clock time*. In that case, `num_samples / fs` underestimates the wall-clock span.

To preserve gaps for plotting and sanity-checking, preprocessing attempts to reconstruct times from:

- `/wells/<stream>/<rec>/groups/routed/frame_nos`

Algorithm (high-level):

1. For each segment `rec_name`, read `frame_nos` and trigger settings.
2. Compute per-sample relative times:
   - `times_rel = (frame_nos - frame_nos[0]) / fs`
3. Compute per-sample “absolute-ish” times aligned to the first segment’s start:
   - `times_abs = (segment_start_s - t0_start_s) + times_rel`
4. Concatenate `times_abs` across segments into `concat_times`.
5. If `len(concat_times)` matches `multirecording.get_num_samples()`, set it via:
   - `multirecording.set_times(concat_times)`

Important detail:
- Segment-local plots (if generated) use `times_rel` so each segment plot spans roughly `0..segment_duration`.

---

## 4. Maxwell contiguous epochs (`maxwell_epochs`)

Triggered/snippet gaps are also represented explicitly as epoch markers.

Per segment:

1. Identify runs of contiguous saved frames:
   - `diff(frame_nos) == 1` defines contiguity
2. Each contiguous run becomes an epoch with:
   - `start_sample` / `end_sample` in **concat** coordinates (segment offset + run start/end)
   - also includes `segment_start_sample` / `segment_end_sample` for segment-local debugging

These epochs become:

- `maxwell_contiguous_epochs_<stream_id>.json`

Downstream use:
- the waveforms stage uses these epochs to drop spikes whose waveform windows would cross snippet boundaries.

---

## 5. Persistence

When `epoch_markers_output_dir` is set, preprocessing writes:

- `maxwell_contiguous_epochs_<stream_id>.json`
- `concatenation_stitch_epochs_<stream_id>.json`

to `<well_out_dir>/stg1_preprocess_outputs/`.
