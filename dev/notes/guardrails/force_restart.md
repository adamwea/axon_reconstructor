# Force-restart guardrail

## Contract

The pipeline has THREE invocation modes for running a stage:

1. **No flag** (auto-restart-from-first-broken). The standard / default mode. The stage walks its `phase_sequence` per target, finds the first phase whose checkpoint status is not `ok` and not `skipped`, force-restarts THAT phase and all downstream phases. Phases that are `ok` are reused. See `guardrails/stage_phase_architecture.md` §"Checkpoint status enum + auto-restart-from-first-broken" for the status definitions and rules.

2. **`--force-restart`** (nuclear). Bypass auto-restart entirely; rmtree the whole output dir and rebuild from scratch:
   - Stage scope: `rmtree(<well>/<stage_output_root>/)` BEFORE the stage runs.
   - Phase scope (when invoked on a single phase): `rmtree(<well>/<stage_output_root>/<phase_output_dir>/)` BEFORE the phase runs.

3. **`--replot`** (plot/report rebuild only, orthogonal to auto-restart). Run plot/report phases in the stage's `phase_sequence`, regardless of any other phase's status. Skip non-plot phases entirely. Useful when the user wants fresh plots from existing computed data. The caller is responsible for ensuring upstream data is valid; this flag does NOT auto-restart anything.

Anything more granular — partial restart options, per-step cache preservation across force-restart, `*_delete_outputs_on_force_restart` YAML knobs that selectively retain things — is **forbidden going forward** and on the deletion list. No fallbacks. No backwards compat. No "but this saves time".

**`--force-replot` is dead.** Two flags exist: `--force-restart` (nuclear) and `--replot` (plot phases only). The old `--force-replot` is being eliminated entirely — "replot" already implies "do it again", no need for "force".

## Why

The pipeline accumulated several different cleanup-on-force-restart helpers — `_cleanup_spikesort_outputs_for_force_restart`'s hard-coded allowlist, the reconstruct-stage templates-cache stash-and-restore dance, the per-phase `bombcell_label_delete_outputs_on_force_restart` / `slay_delete_outputs_on_force_restart` / `merge_delete_outputs_on_force_restart` / `slay_force_restart_retrain_model` YAML knobs. Each one solved a real problem at the time and silently broke later (the SLAy aux-tsv bug, the bombcell stale-labels contamination, the recon templates-cache leaking pre-SLAy unit IDs). The complexity earned no benefit users could feel, and added a stream of "force-restart didn't actually start over" issues.

The collapse to `rmtree` is the destructive cleanup `trackers/tech_debt.md` §"Collapse `--force-restart` semantics" tracks. The reconstruct stage's templates-cache-preserve was removed in commit `5e2b883` as the first piece. The spikesort stage's allowlist (`_cleanup_spikesort_outputs_for_force_restart`) was expanded in this week's work as a stopgap; full replacement by `rmtree(stage_output_root)` is the next step.

## Concrete sub-rules

1. **No `*_delete_outputs_on_force_restart` YAML knobs in new code.** Existing ones are scheduled for deletion. Don't add new ones.

2. **No "preserve cache across force-restart" branches.** The reconstruct-stage templates cache used to be preserved through a rename-aside-and-restore dance; that's gone. Same shape (and same removal) applies anywhere else it shows up.

3. **The rmtree happens FIRST.** Before any phase runs. Not inside the phase's resource-gate-acquired body. The order is: parse force_restart flag → confirm scope (stage vs phase) → rmtree the target → start running phases as if the folder didn't exist.

4. **Idempotent.** Re-running with `--force-restart` after a successful run produces the same outputs as the first run. No caching artifacts surviving across.

5. **Test contract**: write garbage into the affected output dir, run with `--force-restart=true`, assert the garbage is gone and the legitimate outputs are present. One such test per stage. See `trackers/issues.md` §"`--force-restart` does not reliably clean prior artifacts".

## Tests / verification

- Per-stage cleanup tests must use the "write garbage, force-restart, assert clean" pattern.
- Smoke-test trigger: any change to a stage's cleanup helper requires a real-data smoke run with `--force-restart`, then a `find` audit of the well's output dir to confirm nothing unexpected survived.
- Visual log markers to watch for:
  - `Reconstruct full restart: clearing output root <path>` — should appear once per well when stage force-restart is requested.
  - Similar markers for other stages once the collapse lands.

## Open exceptions / follow-ups

- The current `_cleanup_spikesort_outputs_for_force_restart` allowlist is a stopgap. Long-term `rmtree(stage_output_root)` replaces it. Tracked in `trackers/tech_debt.md` §"Collapse `--force-restart` semantics".
- `force_replot` is separate today. Its collapse into `--force-restart` with a flag (or its removal) is tracked in the same tech_debt entry.
- The `phase_roster_cleanup_plan` is the prerequisite that unblocks the destructive collapse — every phase deleted there is one fewer cleanup helper to migrate.
