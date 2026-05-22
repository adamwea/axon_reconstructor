# Notes

Free-form scratch. Debugging trails, half-formed ideas, breadcrumbs for the next session. Prune aggressively — anything durable belongs in guardrails / plans / trackers; anything shipped belongs in current_state.md or commit_log.md.

(initially empty)

## 2026-05-22 session-end pickup (radivojevic algo + axon_recon wiring)

**Where we are**: 6/6 radivojevic algorithm-correctness slices shipped in sibling repo (`~/dev/pkgs/radivojevic2023_recon_algo`, commits 2e1caa7 → af4bdea); 171/171 tests pass. End-to-end smoke on unit_598 shows "a bit better but still wrong" per user — selected channels still cluster centrally instead of tracing the axon arbor.

**Root cause of remaining gap**: noise estimation. Template-derived noise (mad/window on the merged_template) is √n_spikes too small vs the paper's spec (raw-recording noise during inactive periods). The hack threshold-scaling (n_std=90 etc) over-corrects and pushes detection too tight — distal axon channels with weak signal get filtered out.

**Concrete next-session tasks** (priority order):

1. **Slice 11 axon_recon-side wiring** — get per-channel raw-recording noise into radivojevic input:
   - YAML: add `noise_levels: {}` to `reconstruct.analyzers.segments.extensions` (per-segment block currently has NO extensions block; only concat_analyzer at line 733 does)
   - `templates/integrations/spikeinterface_extract.py`: when building per-segment analyzers, also call `analyzer.compute("noise_levels")` per the configured extension list
   - Per-segment cached analyzer's `extensions/noise_levels/` dir should then contain noise per channel
   - Load-side helper: read noise_levels from cache; merge across segments (e.g. median across segments per channel)
   - Pass to `radivojevic2023_recon_algo.reconstruct(..., noise_std_per_channel=<merged>)` via the new slice-11 API

2. **Re-run slice 15 integration smoke** with proper noise + paper defaults (n_std 9/2/1, k_stage2=1.0, no scaling hacks). Acceptance: selected channels trace axon arbor (not central blob).

3. **`reconstruct.radivojevic_recon` phase wire-in** — radivojevic plan slice 5. Need a new recon-stage phase that consumes kssynth's merged_template + per-channel noise + invokes radivojevic.reconstruct. Output gets rendered via plot_recons machinery (need adapter — slice 4 of original plan).

4. **resources_profiles slice 6** (deferred) — investigate why srun -c 128 still yielded n_jobs=16 in the kssynth salloc smoke. Needs the user's salloc stdout log (not in dev_outputs/).

5. **kssynth slice 8** (deferred) — defer-to-analyzer-policy + add upsample step. Audit was in brain/refs/old_pipeline_analyzer_window.md.

**Useful artifacts to re-open tomorrow**:
- `dev_outputs/.../slice13_comparison/slice13_compare.png` — side-by-side legacy union vs paper pair-averaged skeleton
- `dev_outputs/.../v5_comparison/comparison_knobs.png` — knob-by-knob sweep
- `dev_outputs/.../unit_598_radivojevic_v4_scaled_all/` — last "working" v4 baseline (700 peaks, 30 skel/frame)
- `brain/refs/old_pipeline_analyzer_window.md` — paper audit + bug findings
- `plans/active/radivojevic_recon_algo_plan.md` — slices 10-15 spec (now all shipped except wiring)
