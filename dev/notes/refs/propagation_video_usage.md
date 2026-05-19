# `analysis.propagation_video` — usage + opt-in instructions

Slice 9 of `analysis_propagation_video_plan.md`: user-facing docs for
the analysis-stage `propagation_video` phase. Generates a per-unit GIF
of extracellular template propagation along each unit's reconstructed
axon branches.

## Quickstart

```bash
# 1. Make sure the reconstruct stage has run for the target wells —
# the phase needs merged templates + GTR objects produced by
# `reconstruct.axon_velocity_gtrs`.
axon-recon stages reconstruct --config dev/debug_NERSC/debug.runtime.yml \
  --target-wells well000 --limit-datasets 1

# 2. Enable `analysis.propagation_video` in your YAML (see "YAML
# config" below) — it defaults to disabled because each video is
# minutes of compute.

# 3. Dry-run first to confirm wiring + check inputs are present:
axon-recon stages analysis.propagation_video \
  --config dev/debug_NERSC/debug.runtime.yml \
  --dry-run --target-wells well000 --limit-datasets 1

# 4. Real run when dry-run is clean. Outputs land at
# <output_root>/<chip>/<run>/<well>/analysis_outputs/propagation_video/unit_<NNNN>.gif
axon-recon stages analysis.propagation_video \
  --config dev/debug_NERSC/debug.runtime.yml \
  --target-wells well000 --limit-datasets 1
```

## Why opt-in (default disabled)

Each video is **N minutes of compute** (per-frame matplotlib render +
PillowWriter GIF assembly). A typical well has dozens of units; a
campaign of dozens of wells × dozens of DIVs would generate hours of
compute. The phase defaults to `enabled: false` so it doesn't burn
cycles unless the operator explicitly asks for it.

Opt-in usage patterns:
- **Single-unit deep dive**: `--targets <ds>:<well>` then visually
  inspect specific unit GIFs.
- **Per-DIV sanity check**: enable only on the dataset of interest,
  not the whole roster.
- **First-time validation**: run on 1-2 units to verify branch
  propagation looks correct, then disable again.

## YAML config

Under `stages.analysis.phases.propagation_video`:

```yaml
stages:
  analysis:
    phase_sequence:
      - compute_metrics
      - unitmatch
      - propagation_video        # add to the sequence
    phases:
      propagation_video:
        enabled: true             # opt in (default false)
        resource_class: disk_cleanup
        rel_output_root: propagation_video   # default — output subdir
        fps: 20                   # PillowWriter frame rate (default 20)
        skip_frames: 2            # 1 = every frame; 2 = halve (default 2)
        cmap: coolwarm            # matplotlib cmap name (default)
```

Recommended overrides:
- `cmap: viridis` if you prefer Plotly's standard perceptual cmap over
  the v1-era coolwarm.
- `skip_frames: 4` for faster previews when you don't need full
  temporal resolution.
- `fps: 30` for smoother playback at the cost of larger files.

## Output layout

Per (dataset, well), one GIF per unit:

```
<output_root>/<project>/<date>/<chip>/AxonTracking/<run>/<well_id>/analysis_outputs/
  context/propagation_video_summary.json   # per-target summary marker
  propagation_video/                       # one GIF per unit
    unit_0005.gif
    unit_0007.gif
    unit_0094.gif
    …
```

The per-target summary marker contains:
- `status`: `ok` (every unit rendered or skipped idempotently),
  `partial` (some units failed), `error` (all failed / no units
  found), `noop` (phase disabled), or `dry_run_ok` (dry-run).
- `units_processed`: per-unit list of `{unit_id, status, reason,
  out_path, frames, cmap, fps, skip_frames}`.
- `n_units_ok` / `n_units_skipped` / `n_units_error`: aggregate counts.

## Prerequisites

The phase needs:
1. **Merged templates** at
   `<recon_outputs>/cache/templates/merged/unit_<id>/merged_template.npy`
   (v2) or `merged_contributing_template.npy` (legacy). Produced by
   the recon stage's template-extraction phases.
2. **Channel locations** at the same dir,
   `merged_channel_locations.npy` / `merged_contributing_channel_locations.npy`.
3. **GTR pickle** at `<recon_outputs>/units/<unit_id_zero_padded>/gtr.pkl`
   produced by `reconstruct.axon_velocity_gtrs`.

Missing prerequisites surface in the per-target summary's
`validation.missing_prerequisites` section (under dry-run) or as
`status: error` per-unit entries (under real run).

## Idempotency + force-restart

- **Default**: skips per-unit render when `<out_path>.gif` already
  exists. Re-running is a no-op.
- **`--force-restart`**: overwrites existing GIFs.

The orchestrator-level skip (slice 14c) ALSO fires when every analysis
phase's per-target summary shows ok — so re-running
`axon-recon stages analysis` after a successful pass short-circuits
without re-loading the renderer.

## Soft dep on `axon_velocity`

The renderer wraps `axon_velocity.plotting.play_template_map`. The
import happens at call time, so the analysis stage's other phases
(`compute_metrics`, `unitmatch`) keep working when `axon_velocity`
isn't installed.

Inside the shifter image (or a `pip install .[full]` conda env),
`axon_velocity==0.1.2` is preinstalled. Running outside those (e.g. a
fresh conda env that only has `[dev]` extras) will raise
`PropagationVideoRenderUnavailable` with installation instructions.

## v1 elaborations not yet ported

The minimal slice-4 port covers the central
`play_template_map → PillowWriter` chain. The v1 implementation
included several optional knobs that deferred until after the slice-8
HARD-gate diagnostic confirms the basic output is correct:

- ROI cropping around contributing channels (smaller files; faster
  render).
- Saturation clipping via quantile (e.g. clip top/bottom 0.5%).
- Colorbar overlay.
- Time-counter overlay.

If you need these before they ship, see the slice-1 archeology audit
at `dev/notes/refs/propagation_video_audit.md` for the v1 source
(commit `57e343f`).

## See also

- Plan: `dev/notes/plans/active/analysis_propagation_video_plan.md`
- Slice-1 archeology: `dev/notes/refs/propagation_video_audit.md`
- Soft dep contract: `pyproject.toml` `[full]` extra includes
  `axon_velocity==0.1.2`.
- Visual diagnostics gate: `dev/notes/memory/diagnostics_to_review.md`
  (the slice-8 HARD-gate goes here).
