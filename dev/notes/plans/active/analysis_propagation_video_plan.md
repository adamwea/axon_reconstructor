# Analysis-stage propagation-video plan

## Motivation

`axon_velocity` ships code to generate **video / GIF of signal propagation
along the branches of reconstructed units** — turns abstract template +
footprint data into something interpretable at a glance. An older version
of this lived somewhere inside the recon stage but is no longer wired into
the current pipeline (git archeology required to locate it — see slice 1;
fine if it's gone or unfindable, we wrap the current `axon_velocity` API
directly).

This plan re-implements the feature as a new phase in the **analysis stage**.
Reasons for that placement:

- **Computationally expensive**: each video bakes per-unit per-frame
  rendering; running it on every unit in every well of every dataset would
  blow up routine pipeline runtime. Analysis is where the user reaches for
  targeted, on-demand work — exactly the right home.
- **Downstream of reconstruction**: videos need merged per-unit recon
  outputs (templates + footprints + per-branch propagation timing from
  `axon_velocity_gtrs`). Those land at the end of the reconstruct stage;
  analysis consumes them.
- **Per-unit targeting**: the user wants this on specific
  `(dataset, well, unit)` tuples, not whole campaigns. Analysis already
  hosts per-unit (`compute_metrics`) and group-scoped (`unitmatch`) phases,
  so unit-level targeting fits.

## Out of scope (v1)

- Animating the full chip at once (would be a separate kind of plot —
  raster-style channel-level activity, not branch-level propagation).
- Real-time / streaming output (always offline batch).
- New propagation logic beyond what `axon_velocity` already supports —
  this plan is a wrapper around its API.

## Target shape

A new analysis-stage phase, **name TBD during slice 2** — candidates:
`propagation_video`, `axon_velocity_video`, `branch_propagation_video`.
Draft uses `propagation_video`.

- **Stage**: `analysis`
- **Inputs**: per-unit recon outputs (merged templates + axon_velocity GTRS)
  per `(dataset, well, unit)` tuple.
- **Outputs**: one video / GIF per unit at
  `<output_root>/<dataset>/<well>/analysis/propagation_video/unit_<N>.gif`.
- **Default state**: `enabled: false` in YAML. Opt-in per run.
- **Scope flags**: all existing `--target-*` / `--limit-*` / `--targets`
  per `guardrails/scope_flags.md`. Needs **unit-level targeting** — see
  Open decisions.
- **Status enum + in_progress marker**: per
  `guardrails/stage_phase_architecture.md`.
- **`--dry-run`**: short-circuits at input resolution per
  `guardrails/dry_run.md`. Reports `(dataset, well, unit)` it would
  process; doesn't render.
- **`--replot`**: videos are render outputs, so `--replot` re-runs this
  phase per `guardrails/force_restart.md`.

## Slices

1. **Audit + API survey** — git archeology to locate the old recon-stage
   impl (`git log --all --diff-filter=D -- '**/animate*' '**/video*' '**/propagation*' '**/gif*'`);
   survey current `axon_velocity` video-generation API (likely
   `axon_velocity.plotting.animate_propagation` or similar). Document
   findings inline in this plan. If the old impl is unfindable, fine —
   slice 4 wraps the current API from scratch. NO code changes in this
   slice.

2. **Scaffold the analysis-stage phase** — wire `propagation_video` into
   `pipeline/stages/analysis/` per the existing analysis-phase template
   (compare to `compute_metrics` + `unitmatch`). Phase impl is a noop that
   writes `propagation_video_summary.json` with
   `status: skipped, reason: not_implemented_yet`. YAML scaffolding in
   debug.runtime.yml + debug.data.yml per YAML-hygiene injection.
   `enabled: false` default. 5-7 unit tests for wiring.

3. **Inputs resolver** — per-target input collection: walk recon outputs
   for each `(dataset, well, unit)`, locate the GTRS files
   `axon_velocity` needs. Define an input dataclass. Tests for resolution
   including missing-input handling (raise an
   `UnitmatchSessionInputMissing`-style exception with actionable hints).

4. **Core impl — single-unit video render** — call `axon_velocity`'s
   video API for one unit; write to per-unit path. Decide format: GIF
   (no extra deps; universal) vs. MP4 (needs ffmpeg / `imageio[ffmpeg]`
   — if new pip dep, mirror per `env_parity` guardrail). Decide
   default frame rate + duration. 1-2 mocked-API tests + a golden-path
   test that runs the real `axon_velocity` against tiny synthetic input.

5. **CLI + YAML config** — verify whether `--target-units` exists
   (currently only `--limit-units` per `guardrails/scope_flags.md`); if
   not, add it OR extend `--targets` to accept `<ds>:<well>:<unit>`
   triplet form (the latter is more consistent with the existing per-pair
   grammar). Plus per-phase YAML block for knobs (frame rate, duration,
   format). Smoke: `--targets 13:0:5` (or equivalent) on a known unit
   produces one video.

6. **`--dry-run` support** — short-circuit at input resolution;
   `propagation_video_summary.json` with `status: dry_run_ok`.

7. **Orchestrator wire-in** — `run_analysis_propagation_video`
   orchestrator; per-unit fan-out using the standard `n_jobs` resolver
   per `guardrails/parallelism.md`. Tests for orchestrator behavior.

8. **First real-data smoke + HARD-GATE visual diagnostic** — login-node
   smoke on **one (dataset, well, unit)** of the 80k DMEM well000
   cohort. Output: one GIF / MP4. **MUST add HARD-gate entry to
   `dev/notes/memory/diagnostics_to_review.md`** — user inspects the
   video and confirms the propagation visualization is correct before
   downstream slices ship. This is exactly the case CLAUDE.md
   §"Visual diagnostics" describes: a new phase whose claim of
   correctness depends on the user looking at the picture.

9. **Doc + opt-in instructions** — README / docs blurb on enabling the
   phase + scoping. Note the cost (one video ≈ N minutes of compute) so
   users don't accidentally run it on full campaigns.

## Smoke tests

- After slice 4: golden-path test renders one frame from synthetic input
  using real `axon_velocity`.
- After slice 7: dry-run on a small target list (3 units across 2 wells)
  reports correct paths; no rendering.
- After slice 8: real login-node smoke renders one unit; visual diagnostic
  saved + flagged HARD-gate.

## Done criteria

- `axon-recon analysis propagation_video --targets <ds>:<well>:<unit>`
  produces a GIF/MP4 at the expected path.
- Phase wired into the analysis-stage default phase_sequence as
  `enabled: false`.
- All scope flags + `--dry-run` + `--replot` work per relevant guardrails.
- User has approved the first real-data video output via the
  diagnostics_to_review entry.

## Open decisions

- **Output format**: GIF (no new deps, universal) vs. MP4 (smaller for
  longer sequences, needs ffmpeg). Default GIF; MP4 via per-phase YAML
  knob.
- **Unit-level targeting**: extend `--targets` to triplet form
  (`<ds>:<well>:<unit>`) vs. add a new `--target-units` flag. Triplet
  form is more consistent. Decide in slice 5.
- **Phase naming**: `propagation_video` vs. `axon_velocity_video` vs.
  `branch_propagation_video`. Decide in slice 2.
- **Frame rate + duration defaults**: empirical — pick during slice 4
  (itself a visual diagnostic gate, lightly).
- **Cumulative vs. instantaneous propagation animation**: if
  `axon_velocity` supports both, pick one for v1; expose the other
  later.

## Relationship to existing artifacts

- **Depends on**: reconstruct stage producing per-unit GTRS outputs
  (already shipped; `axon_velocity_gtrs` phase post-rename in
  phase_roster_cleanup slice 3).
- **Lives next to**: `analysis.compute_metrics` (per-unit) and
  `analysis.unitmatch` (per-group). Pattern-matches the per-unit form.
- **Future consumer**: nothing in pipeline — videos are end-user
  artifacts. Dashboard could surface them.
- **Tier**: Tier 5 / chip-away. Not blocking anything in current queue;
  ships when other work winds down. The slice 8 real-data smoke + first
  video for user review would be a nice end-of-day deliverable for some
  future loop run.

## Slice-1 archeology hints

The old impl probably lived under something like
`src/axon_recon/.../recon/.../animate*`,
`src/axon_recon/.../plot/.../video*`, or a script in `dev/` or `tools/`.
Search patterns to try first:

```bash
git log --all --diff-filter=D --name-only -- '**/animate*' '**/video*' '**/propagation*' '**/gif*' | sort -u
git log --all --oneline --grep -i 'video\|animate\|gif\|propagation' | head -20
git grep -nE 'animate_propagation|imageio|matplotlib\.animation' $(git rev-list --all) 2>/dev/null | head -30
```

Surfacing old code is a starting point — fine to rewrite cleanly against
the current `axon_velocity` API if the archeology turns up little.
