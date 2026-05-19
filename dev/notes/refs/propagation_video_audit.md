# Propagation-video plan slice 1 — archeology + API survey

Slice 1 of `analysis_propagation_video_plan.md` — locate the old v1
implementation in git history, survey the current `axon_velocity`
video API, decide whether slice 4 wraps the API fresh or migrates the
old code.

## Archeology: old v1 implementation

**Location**: `src/axon_reconstructor/pipeline/reconstruction/plotting.py`
in the retired `axon_reconstructor` v1 package. Deleted in commit
`630d689` ("ai: delete retired v1 package").

**Last live snapshot** (before deletion): commit `57e343f` ("some
updates to fix regen of gif in recon", 2026-02-20). At that point,
`plotting.py` was ~1800 lines covering many recon-stage plots; the
template-movie-gif block was lines 1655-1810 (~155 lines).

**How it worked** (from the snapshot):
- Imported `axon_velocity.plotting.play_template_map` (same module +
  function name as today's API).
- Wrapped the AV call with:
  - Output gating: `AXON_RECON_RECON_WRITE_TEMPLATE_MOVIE_GIF` env var
    (default on).
  - Cropping (`crop_template_movie_gif`): computed a square ROI around
    contributing channels via `_compute_zoom_limits_from_xy`, remapped
    branches into the cropped index space so AV could draw them.
  - Colormap (`template_movie_gif_cmap`, default coolwarm).
  - Quantile clip (`template_movie_gif_clip_quantile`, default 0.995)
    to keep saturated colors readable.
  - Colorbar overlay (`write_template_movie_gif_colorbar`).
  - Time-counter overlay (`write_template_movie_gif_time_counter`).
  - Zoom toggle (`zoom_template_movie_gif`).
- Used `matplotlib.animation.PillowWriter` to write the GIF (no
  ffmpeg dep needed).
- Force-restart honored: regen when `force_restart` OR the gif file
  doesn't yet exist.

**Configuration surface**: ~10 env-var knobs. Slice 4 (core impl)
should re-expose these as YAML config so they're discoverable + per-
run scopable.

## Current `axon_velocity` API

```python
axon_velocity.plotting.play_template_map(
    template,          # (n_channels, n_samples) waveform array
    locations,         # (n_channels, 2) XY positions
    gtr=None,          # optional GTR object with .branches + .locations
    elec_size=8,
    cmap='viridis',
    log=False,
    ax=None,
    skip_frames=1,
    interval=10,
    **to_image_kwargs, # forwarded to _get_image (probe rasterizer)
)
```

Returns a `matplotlib.animation.ArtistAnimation`. The caller wraps it
with `ArtistAnimation.save(out_path, writer=PillowWriter(fps=...))` to
materialize the GIF.

**Key observations**:
- Same module + function name + arg shape as the v1-era code.
  Migration cost: low — slice 4 can lift the v1 wrapper essentially
  verbatim, just relocate it under the analysis stage.
- AV's spec includes `gtr` (branch overlay) as an opt-in argument —
  matches the user's "branch propagation" goal.
- `to_image_kwargs` is the escape hatch for unusual probe geometries.
- No native MP4 export; PillowWriter writes GIF, FFMpegWriter would
  do MP4 if `imageio[ffmpeg]` is added (small env_install_unification
  follow-up if MP4 ever becomes a requirement).

## Slice 4 implementation recommendation

The audit findings make slice 4 mostly a port + place-relocation:

1. **Lift** the ~150-line template-movie-gif block from v1's
   `plotting.py:1655-1810` into a new
   `pipeline/stages/analysis/core/propagation_video.py`.
2. **Replace** env-var knobs with YAML config fields under
   `stages.analysis.phases.propagation_video.*` (slice 5 wires the
   YAML; slice 4 sets sensible defaults inline).
3. **Format**: GIF default via PillowWriter — no new deps beyond what
   `[full]` already pulls. MP4 deferred.
4. **Inputs resolver** (slice 3) needs:
   - Per-unit reconstructed template (already at
     `<well_out_dir>/recon_outputs/units/<unit_id>/`).
   - Per-unit channel locations (already in the merged GTR output).
   - Per-unit GTR object (already in `axon_velocity_gtrs_summary.json`
     consumers).

All three inputs are already on disk after the recon-stage
`axon_velocity_gtrs` phase. **No new recon-stage work required.**

## Per-unit targeting (slice 5)

The user wants this on specific `(dataset, well, unit)` tuples, not
whole campaigns. Existing `--targets` (post-slice-4 of
`unitmatch_phase_plan.md`) is `<ds>:<well>` OR
`chip-well:<chip>:<well>`. Neither addresses individual units.

Options for slice 5 to choose:
1. **New `--targets-units <ds>:<well>:<unit>` triplet form** —
   parallel to chip-well: pattern; clean grammar.
2. **Reuse `--limit-units N` + a new `--unit-ids 5,7,12` knob** — less
   uniform with existing `--targets` shape but reuses scope-flags
   machinery from `guardrails/scope_flags.md`.

Lean toward option 1 — keeps `--targets` as the single per-task scope
flag and matches the `chip-well:` precedent. Per-unit triplet token
parses as `<ds_int>:<well>:<unit_int>` and stays orthogonal to the
existing `<ds>:<well>` pair form.

## Frame rate / duration defaults

The v1 code didn't expose frame-rate config — relied on
PillowWriter's default (~50ms per frame ≈ 20 FPS). At
`skip_frames=2` and a typical 60-sample spike width, that yields
~1.5 second gifs. Reasonable default; slice 4 should expose
`fps` + `skip_frames` as YAML knobs.

## Out of scope follow-ups (not this plan)

1. **MP4 export**: requires `imageio[ffmpeg]` in `[full]`. Defer until
   a user asks.
2. **Full-chip animation**: the user explicitly excluded this in
   `## Out of scope (v1)` of the plan.
3. **Interactive playback / streaming**: defer until a user-domain
   need surfaces.
