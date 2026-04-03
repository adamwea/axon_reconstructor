# Spikesorting Step — Part 5: Troubleshooting + HPC Notes

Scope: this document collects common failure modes and practical debugging strategies for the spikesorting integration.

---

## 1. “Sorter output not found”

Symptoms:
- downstream stages can’t find `stg2_spikesorting_outputs/sorter_output/`

Checks:
- confirm the per-well output directory is what you expect (relative_pattern + well)
- confirm MEA_Analysis was launched with the same `--output-dir` (output root)
- confirm MEA_Analysis actually ran Phase 2 sorting (not skipped)

---

## 2. HDF5 plugin / decompression errors

Symptoms:
- MEA_Analysis fails reading the raw `.h5`

Likely cause:
- missing Maxwell compression plugin.

Fix:
- ensure `HDF5_PLUGIN_PATH` is set appropriately inside the runtime environment (Shifter/Docker/host).

---

## 3. GPU availability

Symptoms:
- Kilosort4 is extremely slow or fails

Checks:
- ensure you are on a GPU node and the container/runtime can see the GPU
- in NERSC Shifter workflows, ensure `CUDA_VISIBLE_DEVICES` is set correctly

---

## 4. Scratch usage and staging back outputs

On HPC systems, it’s common to run heavy I/O in `$SLURM_TMPDIR`.

The MEA_Analysis driver supports:

- `--scratch-dir <path>`
- `--stage-back none|sorter|all`
- `--stage-back-mode copy|move`

axon_reconstructor CLI subcommands can pass these through.

---

## 5. Checkpoint confusion (MEA_Analysis vs axon_reconstructor)

Both toolchains have checkpointing:

- MEA_Analysis has its own per-run checkpoint JSON under `stg2_spikesorting_outputs/checkpoints/`.
- axon_reconstructor has a per-well pipeline checkpoint under the well root.

If you suspect MEA_Analysis is resuming when you don’t want it to:

- use MEA_Analysis `--force-restart` (or set `force_restart=True` in the in-process harness).

---

## 6. Version skew

Because MEA_Analysis evolves, output folder shapes can change.

When diagnosing issues:

- prefer checking for `sorter_output/` existence first
- then try loading via SpikeInterface reader functions to see a concrete error
