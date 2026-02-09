# Spikesorting Step — Part 3: In-Process Debug Harness (Inject Preprocessed Recording)

Scope: this document covers the in-repo spikesorting debug harness that runs MEA_Analysis *in-process* (Python import) while using the Stage 01 saved recording.

Primary script:
- `tools/stepwise_debug_scripts/spikesorting_debug.py`
  - `run_spikesorting_only(inputs=..., logger=...)`

---

## 0. Why this harness exists

Running MEA_Analysis inside Shifter/HPC is the production path, but it’s not ideal for iterative debugging.

This harness provides a way to:

- load the already-saved `preprocessed_recording/`
- create a MEA_Analysis `MEAPipeline` object
- **skip MEA_Analysis preprocessing** by injecting the loaded recording
- run sorting/analyzer/reports with tight control over output locations

---

## 1. Inputs and required preconditions

The harness assumes Stage 01 has already produced:

- `<well_out_dir>/preprocess_outputs/preprocessed_recording/`

It requires:

- `mea_analysis_repo_root` (so `import MEA_Analysis...` works in ad-hoc sessions)
- `mea_output_root` (to compute the per-well output folder)

---

## 2. Making MEA_Analysis importable

The harness modifies `sys.path` so that `MEA_Analysis` can be imported from a repo checkout:

- it inserts `mea_analysis_repo_root.parent` onto `sys.path`

This is a pragmatic development feature; it avoids forcing a formal install in every debug session.

---

## 3. Re-homing MEA_Analysis outputs under `spikesorting_outputs/`

MEA_Analysis typically writes under the per-well output directory.

axon_reconstructor keeps it stage-scoped by relocating:

- `pipeline.output_dir = <well_out_dir>/spikesorting_outputs/`
- `pipeline.checkpoint_file = <well_out_dir>/spikesorting_outputs/checkpoints/<...>_checkpoint.json`

It then attempts to reload MEA_Analysis state from that new checkpoint location.

This keeps the per-well folder tidy and makes it easier for downstream stages to find `sorter_output/`.

---

## 4. Injecting the preprocessed recording

After constructing `MEAPipeline(...)`, the harness sets:

- `pipeline.recording = <loaded preprocessed recording>`

and bumps stage state to indicate preprocessing is already complete:

- `pipeline.state["stage"] = max(existing_stage, PREPROCESSING_COMPLETE)`

Then it runs:

- `pipeline.run_sorting()`
- optional `pipeline.run_analyzer()`
- optional `pipeline.generate_reports(...)`

---

## 5. Outputs

The harness returns a small struct with key paths:

- `recording_dir` (the input recording folder)
- `sorter_output_dir` (the produced `<well>/spikesorting_outputs/sorter_output`)
- `analyzer_dir` (typically `<well>/spikesorting_outputs/analyzer_output`)

This is intended to make it easy to hand off to waveforms/templates stages.
