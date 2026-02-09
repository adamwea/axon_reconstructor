# Spikesorting Step — Part 4: Expected Artifacts, Validation, and Downstream Handoff

Scope: this document describes what spikesorting is expected to produce on disk, how axon_reconstructor validates that output at a lightweight level, and how downstream stages consume it.

---

## 1. The canonical handoff directory

Downstream stages expect a sorter output folder at:

- `<well_out_dir>/spikesorting_outputs/sorter_output/`

This folder is treated as the contract boundary. It is typically produced by MEA_Analysis Phase 2.

---

## 2. Common artifacts in `sorter_output/`

Exact contents vary by sorter and MEA_Analysis version, but common Kilosort/SpikeInterface outputs include:

- `spike_times.npy`
- `spike_clusters.npy`
- `channel_map.npy`
- `ops.npy`
- `params.py`
- `cluster_info.tsv`

axon_reconstructor does not hard-require a specific file list at this stage; it treats it as an external tool output.

---

## 3. Lightweight validation

For a fast “does this look real?” check, use:

- `axon_reconstructor.integrations.mea_analysis.validate_sorter_output_dir(sorter_output_dir)`

This intentionally avoids importing SpikeInterface. It checks for common expected filenames, and falls back to “non-empty directory”.

Stronger validation happens implicitly when later stages try to load the sorter output into a SpikeInterface `Sorting` object.

---

## 4. Analyzer and report outputs

If MEA_Analysis Phase 3/4 are run, you will also see:

- `<well>/spikesorting_outputs/analyzer_output/`
- plots/reports under `<well>/spikesorting_outputs/` (structure varies)

These are useful for debugging sorting quality and for producing curated unit lists.

---

## 5. Handoff to waveforms/templates

Waveforms stage typically does:

- load the Stage 01 saved recording (`preprocess_outputs/preprocessed_recording/`)
- load the sorter output (`spikesorting_outputs/sorter_output/`)
- optionally load quality-metric outputs to decide curated vs uncurated units

The waveforms step then performs its own spike-level boundary filtering using the preprocessing epoch JSONs.
