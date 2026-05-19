# Chip-layout phase split + move to analysis plan

## Motivation

The current `plot_full_chip_layout` phase lives in the recon stage —
one plot per (dataset, well) showing per-unit reconstructions on the
chip layout. The user wants three things:

1. **Move to analysis stage** — gives the plot global awareness of
   unit identities + locations across sessions on the same chip
   (needed for the color-consistency story below).
2. **Split into two phases** with different visualization purposes:
   - **Phase A — chip timeline grid (white background)**: multi-session
     overview. Grid of chip plots, one cell per recording session for
     the same (chip, well). Simple white background, color-coded
     reconstructions. Purpose: track how reconstructions change over
     time at a glance.
   - **Phase B — chip layout detailed (black background)**: viewed in
     isolation, one (dataset, well) at a time. Black background,
     colored reconstructions, **selected electrodes drawn with the
     reconstruction's color at lower opacity**. **Electrode-overlap
     blending**: where selected-electrode sets overlap between
     reconstructions, blend colors weighted by per-reconstruction
     signal strength contribution. Optional gridding mode for
     viewing multiple sessions or wells in one figure.
3. **Cross-session color consistency**: same unit (across sessions on
   the same chip/well) gets the same color. Source: unitlink
   per-(chip,well) match table (UIDs). Fallback when unitlink output
   isn't available: extremum-electrode-location-based heuristic.

User direction (2026-05-19):
- "Make reconstruction colors consistent across repeated recordings on
  the same chip so it's easier to track reconstructions changing with
  time."
- "Ideally we base this off of the info gathered from unitlink, but we
  can also do this more crudely with extremum electrode locations if
  that's not immediately available."
- "Phase A: white background and color coded reconstructions. Phase B:
  black background, colored reconstructions, also plotting the
  selected electrodes for that reconstruction with the same color as
  the reconstruction, but lower opacity. In the event selected
  electrodes for more than one reconstruction overlaps, have the
  colors blend and bias one color or another based on the signal
  strength from the contributing reconstructions."

## Out of scope (v1)

- New reconstruction logic — plotting only.
- Cross-chip color consistency (different chips can have independent
  palettes; intra-chip is the requirement).
- Animation / video versions (covered by
  `analysis_propagation_video_plan.md`).
- Interactive viewers — static PNG/PDF outputs only.

## Target shape

Two new analysis-stage phases; names TBD in slice 2 — candidates:
- **Phase A**: `chip_timeline_grid` / `cross_session_chip_grid`
- **Phase B**: `chip_layout_detailed` / `chip_layout_overlay` /
  `full_chip_layout` (inheriting the existing name, just relocated)

Both:
- Live in `pipeline/stages/analysis/`.
- `enabled: false` default; opt-in per run.
- All scope flags per `guardrails/scope_flags.md`.
- `--dry-run` per `guardrails/dry_run.md`.
- `--replot` (these are plot phases) per `guardrails/force_restart.md`.
- Color palette + per-unit color assignment shared between both phases
  via a common helper module.

Phase A inputs:
- Recon outputs across multiple (date, well) recordings for the same
  (chip, well) group (the 80k DMEM well000 cohort in v1).
- unitlink match table for that group (preferred unit-identity source).
- Extremum-electrode positions from recon outputs (fallback when
  unitlink output is absent).

Phase B inputs:
- Recon outputs for a single (dataset, well) — per-unit selected
  electrodes + per-electrode signal strengths.
- Same unit-identity source as Phase A.

Outputs:
- Phase A: one grid plot per (chip, well) showing all available DIVs.
- Phase B: one detailed plot per (dataset, well); plus optional grid
  combining multiple if a YAML option enables it.

## Slices

### Slice 1 — Audit existing `plot_full_chip_layout` (no code changes)

Locate the current phase impl (probably under
`pipeline/stages/reconstruct/phases/` or `pipeline/stages/reconstruct/templates/core/`),
read its inputs / outputs / config knobs, document in this plan.

### Slice 2 — Scaffold two analysis-stage phases (disabled noops)

- Two phase impls per the analysis-phase template; both `enabled: false`,
  both noop with `status: skipped, reason: not_implemented_yet`.
- YAML scaffolding in `debug.runtime.yml` + `debug.data.yml` per
  YAML-hygiene injection.
- Disable the OLD `plot_full_chip_layout` in the recon stage's default
  phase_sequence (keep code per the "bombcell / SLAy code deletion
  timing" pattern — code stays, just disabled; we may want to compare
  outputs during slice 5).
- 5-7 wiring tests per phase.

### Slice 3 — Cross-session color-key helper

New module `pipeline/stages/analysis/core/cross_session_colors.py` (or
similar) with:

- `assign_unit_colors_from_unitlink(match_table) -> {uid: rgba}`
- `assign_unit_colors_from_extremum_fallback(per_session_recons)
  -> {(session, unit_id): rgba}`
- `resolve_color_key(...)` — auto-picks unitlink output when present;
  falls back to extremum heuristic with a logged warning.

Palette: perceptually-uniform + categorical-friendly (candidates:
`glasbey`, `tab20`, viridis-derived). Pick during slice 3 visual
diagnostic — render ~50 units in each palette and compare.

Tests: golden tests for color assignment from fixture match tables;
fallback path exercised; palette wrap-around behavior under high unit
counts.

### Slice 4 — Phase A (chip timeline grid)

White-background multi-session grid.

- Consumes all (date, well) recordings for one (chip, well) group.
- Each subplot: chip outline + per-unit reconstructions colored via
  slice-3 color key.
- Auto-grid layout (rows × cols ≈ sqrt(N)) or YAML-configurable.
- Tests: golden test using synthetic small grid (2 sessions × few
  units each).

### Slice 5 — Phase B core (chip layout detailed — single session)

Black-background detailed plot.

- One (dataset, well) per output.
- Per-unit reconstruction colored via slice-3 color key.
- **Electrode overlay**: for each unit, plot its selected electrodes
  as scatter points with the unit's color at reduced opacity
  (~0.3-0.5; pick via visual diagnostic).
- **Overlap blending (v1 interpretation)**: at each electrode that
  appears in more than one unit's selected set, compute a
  signal-strength-weighted RGB blend:
  ```
  rgb_blend = Σ (w_i · rgb_i) / Σ w_i
  ```
  where `w_i` is unit i's signal strength at that electrode.
  Empirically: peak-to-peak template amplitude at that electrode is
  the most likely candidate for `w_i`; confirm or revise during
  slice 5 visual diagnostic.

**HARD-gate visual diagnostic**: first real-data render on one
(dataset, well) of 80k DMEM well000. User confirms the
electrode-overlay opacity + the signal-strength-weighted blend look
right before slice 6.

### Slice 6 — Phase B grid mode (optional)

Extend Phase B with optional gridding (multiple sessions or wells in
one figure), gated by a YAML toggle (`grid_mode: {none, by_session,
by_well}`). Tests + visual diagnostic.

### Slice 7 — Real-data smoke (both phases) + HARD-gate diagnostics

Login-node smoke on the 80k DMEM well000 cohort across all available
DIVs (read-only inputs from `analyzed_data/`; outputs to
`/pscratch/.../dev_outputs/chip_layout_phase_split/`).

Both phases produce their plots. HARD-gate entries in
`memory/diagnostics_to_review.md`:

1. Phase A timeline grid — does color consistency look right across
   sessions? Are reconstructions visually trackable over time?
2. Phase B detailed plot at multi-session scope — electrode-overlay +
   blending across the cohort?

### Slice 8 — Docs + opt-in instructions + old-phase fate

- README blurb on each phase + when to use which.
- Note on color-consistency dependence on unitlink running first.
- Decide fate of old recon-stage `plot_full_chip_layout`: delete
  (now superseded) or keep disabled (for future comparison)? User
  decision at slice 7 HARD-gate review.

## Smoke tests

- After slice 4: Phase A renders multi-session grid on synthetic
  input.
- After slice 5: Phase B renders one detailed plot with overlay +
  blending; visual diagnostic flagged HARD-gate.
- After slice 7: real-data both phases; both HARD-gate diagnostics
  filed.

## Done criteria

- `axon-recon analysis chip_timeline_grid` produces a multi-session
  grid for a (chip, well) group.
- `axon-recon analysis chip_layout_detailed --targets <ds>:<well>`
  produces a single detailed plot with electrode overlay + weighted
  blending.
- Both phases use unitlink output when available; fall back to
  extremum-electrode heuristic with a logged notice.
- Both HARD-gate visual diagnostics approved.
- Old recon-stage `plot_full_chip_layout` decision made (delete vs.
  keep-disabled).

## Open decisions

- **Phase names** (slice 2): pick from candidates.
- **Palette** (slice 3): perceptually-uniform vs. categorical; visual
  diagnostic.
- **Electrode opacity** (slice 5): empirical — 0.3 / 0.4 / 0.5 sweep.
- **Signal-strength metric for blending** (slice 5): per-electrode
  peak-to-peak template amplitude is the v1 default; alternatives are
  mean amplitude, SNR, or normalized signal-mass. Confirm via visual
  diagnostic.
- **Grid auto-layout for Phase A** (slice 4): row-major / col-major /
  YAML-configurable.
- **Old recon-stage phase fate** (slice 8): delete or keep disabled?

## Tier + sequencing

**Tier 4 (gated, kickoff-after-integration)** — same trigger as the
Radivojevic plan: **kssynth slice 9 AND unitmatch_phase slice 5 BOTH
shipped**. Reasons:

- unitlink output is the *preferred* color key, which requires
  unitmatch_phase to be running on real data.
- The extremum-electrode fallback would let us start earlier, but the
  user-stated preference is unitlink-based color consistency.
- The analysis-stage move pattern is more mature once unitmatch_phase
  is live (slice 14c approach A landed for analysis on 2026-05-19,
  so the runner shape is ready).

Once both gates clear, slice 1 (audit, no code changes) is a good
first slice; slices 2-8 sequence normally. Plan is contained — 2
USER GATEs (slices 5 + 7 visual diagnostics).

## Relationship to other plans

- **Depends on**: unitlink + unitmatch_phase delivering per-(chip,well)
  match tables (`kssynth_plan` slice 9 + `unitmatch_phase_plan` slice 5).
- **Lives next to**: future `propagation_video` phase
  (`analysis_propagation_video_plan.md`) and the planned comparison
  phase from the Radivojevic plan. All analysis-stage visualizations
  benefit from the cross-session color key from slice 3.
- **Shared helper**: `cross_session_colors.py` from slice 3 is likely
  reused by Radivojevic comparison plotting + propagation-video
  cohort-overview frames (not v1 scope but worth keeping the API
  general).
