# Force-restart guardrail

## Contract

Two rules, no exceptions:

1. **Stage `--force-restart` for a well = `rmtree(<well>/<stage_output_root>/)` BEFORE the stage runs.** Whole output dir gone. Then the stage rebuilds whatever it needs.

2. **Phase `--force-restart` (when invoked on a single phase) = `rmtree(<well>/<stage_output_root>/<phase_output_dir>/)` BEFORE the phase runs.** Just that phase's folder.

Anything more granular — partial restart options, per-step cache preservation across force-restart, `*_delete_outputs_on_force_restart` YAML knobs that selectively retain things — is **forbidden going forward** and on the deletion list. No fallbacks. No backwards compat. No "but this saves time".

`--force-replot` is a separate flag today: redo plot / report work without recomputing data. It's planned to collapse into `--force-restart` with a flag (or be eliminated entirely). Until then, treat it as legacy-but-supported.

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
