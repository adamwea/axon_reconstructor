# Waveforms Step — Part 2: Persist Params + Extract Concat Waveforms Analyzer (Runner)

Scope: This document covers the **next chunk** of the waveforms stage as implemented in the runner. It begins immediately after Part 1 ends (we already have `filtered_sorting`) and ends right after the **concat** `SortingAnalyzer` has been computed and we have `concat_best` (best-channel-by-PTP summary computed from concat templates).

Primary code paths:
- `axon_reconstructor.pipeline.waveforms.runner.extract_waveforms(...)`
- `axon_reconstructor.pipeline.waveforms.artifacts._write_waveform_extraction_params(...)`
- `axon_reconstructor.pipeline.waveforms.extraction._extract_concat_waveforms(...)`

---

## Preconditions (inputs produced by Part 1)

At this point in the runner we already have:

- `recording`: the **preprocessed concatenated recording** used for sorting (authoritative time base)
- `fs_hz`: sampling rate of the concat recording
- `window`: waveform cutout window (`ms_before`, `ms_after`, `pre_samples`, `post_samples`)
- `sorter_output_dir`: resolved sorter output folder
- `sorting`: loaded sorter output (best-effort cleaned against recording)
- `epochs`: epoch marker inputs (maxwell epochs + concat stitch epochs)
- `quality_metrics_params`: parameters for SpikeInterface `quality_metrics` extension
- `filtering_summary`: initialized summary dict
- `wf_rejection_rows` + `base_rej_fields`: spike-level rejection logging scaffolding
- `filtered_sorting`: output of epoch-aware filtering (used from here onward)

---

## 1. Persist waveform extraction parameters (JSON)

1. Call `_write_waveform_extraction_params(...)`.

   - Function: `axon_reconstructor.pipeline.waveforms.artifacts._write_waveform_extraction_params`
   - Output file: `<waveforms_out_dir>/waveform_extraction_params.json`

2. The JSON payload includes (best-effort reproducibility snapshot):

   1. Dataset identity
      1. `h5_path`
      2. `stream_id`
      3. `sorter`
      4. `sorter_output_dir`

   2. Waveform window
      1. `ms_before`
      2. `ms_after`

   3. Compute controls
      1. `n_jobs`
      2. `max_spikes_per_unit`

   4. Per-segment mode flags (even though this part only computes concat)
      1. `per_segment`
      2. `per_segment_recording_source` (set to `raw_maxwell_full_channels` when per-segment is enabled)
      3. `per_segment_only_additional_channels`

Why this is written *here*:
- This is the earliest point where we have all critical context resolved (window, sorter_output_dir, flags).
- If later steps fail, we still have a persistent description of what was intended.

---

## 2. Determine the concat (common/intersection) channel set

1. Attempt to define the “common channel ids” present in the concat recording:

   - `common_channel_ids = set(int(x) for x in recording.get_channel_ids())`

2. If channel IDs cannot be cast to `int` (or something fails), fall back to `common_channel_ids = set()`.

Why this is computed:
- Later, per-segment waveform extraction can optionally exclude these channels and focus only on **additional channels** that were dropped during concatenation.
- Doing it here ensures a single consistent definition of “common channels” derived from the concat analyzer’s recording.

---

## 3. Extract concat waveforms analyzer (SortingAnalyzer folder)

1. Call `_extract_concat_waveforms(...)`.

   - Function: `axon_reconstructor.pipeline.waveforms.extraction._extract_concat_waveforms`
   - Inputs:
     - `filtered_sorting` (epoch-filtered concat spike trains)
     - `recording` (concat preprocessed recording)
     - `concat_waveforms_dir` (folder under `<waveforms_out_dir>/concat_waveforms/`)
     - `window` (ms_before/ms_after)
     - `quality_metrics_params` (from `qm_config.build_quality_metrics_extension_params`)
     - `inputs` (for `n_jobs`, `max_spikes_per_unit`, and `force_restart`)

2. Handle force-restart directory semantics:

   1. If `concat_waveforms_dir` exists and `inputs.force_restart=True`, delete it (recursive) before re-creating outputs.

3. Create the SpikeInterface `SortingAnalyzer`:

   1. `concat_analyzer = si.create_sorting_analyzer(
        filtered_sorting,
        recording,
        format="binary_folder",
        folder=concat_waveforms_dir,
        return_in_uV=True,
      )`

Notes:
- `format="binary_folder"` means the analyzer persists to disk and can be re-loaded later.
- `return_in_uV=True` standardizes waveform/template amplitudes into microvolts in downstream extensions.

---

## 4. Compute extensions: random spike selection + waveforms

1. Compute the core waveform data using a bounded random subset of spikes per unit:

   1. `concat_analyzer.compute(["random_spikes", "waveforms"], ...)`

2. Extension parameters:

   1. `random_spikes`:
      - `method="uniform"`
      - `max_spikes_per_unit = inputs.max_spikes_per_unit`
      - `seed = 0`

   2. `waveforms`:
      - `ms_before = window.ms_before`
      - `ms_after = window.ms_after`

3. Execution controls:

   - `verbose=False`
   - `n_jobs = inputs.n_jobs`

Scientific / computational rationale:
- The waveform extension extracts waveform snippets only for the spikes chosen by `random_spikes`.
- Limiting to `max_spikes_per_unit` keeps runtime and storage bounded while still supporting:
  - template estimation (mean waveforms)
  - QC plots
  - quality/template metrics

---

## 5. Compute extensions: templates, metrics, and unit locations

1. Compute the remaining extensions needed for QC + downstream stages:

   1. `concat_analyzer.compute([
        "spike_amplitudes",
        "templates",
        "noise_levels",
        "quality_metrics",
        "template_metrics",
        "unit_locations",
      ], ...)`

2. Extension parameters:

   1. `unit_locations`:
      - `method="monopolar_triangulation"`

   2. `quality_metrics`:
      - Uses the `quality_metrics_params` dict computed in the runner.
      - This includes metric names + `presence_ratio.bin_duration_s` tuning and a few explicit defaults.

3. Execution controls:

   - `verbose=False`
   - `n_jobs = inputs.n_jobs`

Outputs created inside the analyzer folder:
- Persisted extension data for waveforms, templates, unit locations, and metric tables.

---

## 6. QC warning: detect possible “clipped at start” templates

1. Immediately after templates exist, run:

   - `_warn_early_negative_peaks_from_templates(analyzer=concat_analyzer, window=window, logger=logger)`

2. What this does (high-level):

   - Loads the mean templates and checks, **per unit**, whether a non-trivial fraction of *active* channels have their negative peak at sample 0..1.

3. Important implementation nuance:

   - Templates can be sparse (many channels are near-zero). Near-zero channels trivially appear to “peak” at the first sample.
   - The function therefore only evaluates channels with non-trivial PTP relative to the unit’s best channel.

4. If suspicious early peaks are detected, it logs a warning suggesting:

   - `ms_before` might be too small (waveform peak clipped)
   - or spike-time alignment is off

---

## 7. Compute `concat_best`: best PTP channel per unit (from templates)

1. `_extract_concat_waveforms` returns:

   - `concat_best = _best_ptp_channel_by_unit_from_templates(analyzer=concat_analyzer)`

2. What `_best_ptp_channel_by_unit_from_templates` does:

   1. Load channel IDs from the analyzer/recording.
   2. Load templates from the analyzer (`templates` extension).
   3. For each unit:
      1. Compute per-channel $\mathrm{PTP} = \max(template) - \min(template)$ across time.
      2. Choose channel index with max PTP.
      3. Map that index back to a channel ID.

3. Output format:

   - `concat_best: {unit_id: (best_ptp_uv, best_channel_id)}`

Why this is computed now:
- It is used later for cross-source comparisons (segment vs concat) to detect whether the concat common-channel intersection dropped a unit’s strongest electrode.
- It also feeds reporting (`best_channel_sources.xlsx`) later in the runner.

---

## Output of Part 2

At the end of this part, we have:

- `waveform_extraction_params.json` written under `<waveforms_out_dir>/`
- `concat_waveforms_dir` populated with a SpikeInterface `SortingAnalyzer` folder containing:
  - `random_spikes`, `waveforms`, `templates`, `unit_locations`, `quality_metrics`, `template_metrics`, etc.
- `concat_best`: best-channel-by-PTP summary per unit from concat templates

Next part (Part 3):
- Per-segment waveform extraction on raw recordings (optional) + segment-local edge filtering and per-segment analyzers.
