# Preprocessing Step — Part 4: Temporal Resampling (Upsampling) and Epoch Scaling

Scope: this document covers the optional temporal upsampling feature (paper-like “upsample to 200 kHz”) and the key invariants required to keep later stages correct.

Primary code path:
- `axon_reconstructor.pipeline.preprocessing.runner.build_concatenated_recording(...)`

Related modules:
- `spikeinterface.preprocessing.resample(...)` (temporal resampling)

---

## 0. Resampling intent and constraints

The goal is to support *temporal interpolation / upsampling* of the concatenated recording (e.g. 20 kHz → 200 kHz) **without changing spatial channel layout**.

Key constraints:

- Sorting and waveforms logic depend on sample indices. If the recording is resampled, **every sample-based annotation must be scaled** accordingly.
- In this pipeline, the resampled recording is still treated as the authoritative time base for downstream stages.

---

## 1. Choosing the target sample rate

Resampling can be requested via either:

- `temporal_resample_rate_hz` (explicit target Hz), or
- `temporal_resample_factor` (multiplicative factor; must be >= 2)

The runner computes:

- `old_fs = multirecording.get_sampling_frequency()`
- `new_fs_int = round(target_fs)`
- `ratio = new_fs_int / old_fs`

---

## 2. SpikeInterface resample call

Resampling is applied *after* concatenation:

- `multirecording = spikeinterface.preprocessing.resample(multirecording, resample_rate=new_fs_int, margin_ms=..., dtype=..., skip_checks=False)`

Notes on parameters:

- `margin_ms` is an implementation detail used by SpikeInterface to reduce edge effects at chunk boundaries.
- `dtype` can be used to force output sample dtype.

Failure mode:
- any resampling error raises a `RuntimeError` so the pipeline fails fast rather than producing subtly inconsistent metadata.

---

## 3. Scaling epoch markers

Two epoch marker sets are scaled:

- `concat_epochs` (segment stitch epochs)
- `maxwell_epochs` (contiguous snippet epochs)

Scaling rule:

- every sample index field is multiplied by `ratio` and then rounded:
  - `start_sample`, `end_sample`
  - `segment_start_sample`, `segment_end_sample`
  - `n_samples`

This is critical for waveforms filtering:

- waveforms drops spikes whose waveform window would cross `maxwell_epochs` boundaries.
- if epochs are not scaled, spikes would be incorrectly accepted/rejected.

---

## 4. Post-resample bounds assertion

After scaling, preprocessing checks that epoch indices still fit inside the resampled recording:

- compute `n_new = multirecording.get_num_samples()`
- ensure `max(end_sample)` across `maxwell_epochs` and `concat_epochs` is `<= n_new`

If the assertion fails, preprocessing raises an error with diagnostic context:

- `n_samples`, `maxwell_max_end`, `concat_max_end`, and the `ratio`

This guards against subtle off-by-one/rounding drift.

---

## 5. Time vector handling under resampling

If preprocessing successfully built `concat_times` from `frame_nos`, it attempts to keep a useful time vector after resampling by interpolating the original time vector onto the new sample grid.

This is a pragmatic plotting convenience:

- downstream stage correctness is driven by sample indices + epoch markers
- time vectors are primarily used for inspection plots
