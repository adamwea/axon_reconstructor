# Dry-run guardrail

## Contract

**Every phase exposes a `--dry-run` mode that does as little compute as possible while still verifying the phase's wiring is intact.** In dry-run, the phase:

1. Resolves all inputs (paths, configs, prerequisites) the same way it would in a real run.
2. Validates that prerequisites are present and readable (fail fast with actionable errors if not).
3. **Skips the expensive work** — no spike sorting, no template extraction, no waveform compute, no large-array allocations, no model loading, no plotting/reporting.
4. Writes a `<phase>_summary.json` with `status: dry_run_ok`, a list of resolved input paths, a list of would-be-output paths, and any input-validation findings.
5. Exits successfully (return code 0) for the phase invocation.

`--dry-run` is the cheapest possible smoke check. It should complete in seconds for almost every phase. It's the first step in the smoke-test scoping ladder defined in `CLAUDE.md` §"Smoke test scoping ladder".

## Why

Heavy smoke tests are expensive: one well of recon templates at `n_jobs=1` is ~30-60 minutes. We can't run that after every slice. But most pipeline bugs are wiring bugs — wrong paths, missing prerequisites, stale config references — and those break LONG before the expensive step. `--dry-run` catches them in seconds.

Plus, `--dry-run` doubles as a deployment-readiness check: before kicking off a 4-hour interactive allocation, run `--dry-run` on every well in the planned scope. If they all return `status: dry_run_ok`, the wiring is right; if any fail, fix and re-test cheaply before burning the allocation.

## Concrete sub-rules

1. **Universal flag.** `--dry-run` is wired in the shared argparse setup at the CLI entry point (`pipeline/cli.py`). Every stage / phase subcommand inherits it without needing per-phase argparse additions.

2. **Carried via stage_config**: the parsed `--dry-run` flag flows to phases as `stage_config.dry_run: bool`. Phase implementations check this attribute at the top of their main work block:
   ```python
   if getattr(stage_config, "dry_run", False):
       return _write_dry_run_summary(...)
   ```

3. **Dry-run summary schema** (`<phase>_summary.json`):
   ```json
   {
     "status": "dry_run_ok",
     "well_out_dir": "/path/to/well",
     "stage_output_root_dir": "/path/to/stage_output",
     "phase": "<phase_name>",
     "inputs_resolved": [
       {"name": "concat_recording", "path": "...", "exists": true},
       {"name": "sorter_output", "path": "...", "exists": true}
     ],
     "outputs_would_produce": [
       {"name": "merged_template", "path": "..."},
       {"name": "summary_json", "path": "..."}
     ],
     "validation": {
       "missing_prerequisites": [],
       "warnings": []
     }
   }
   ```
   Same path as the real summary_json so `axon-recon status` reads it cleanly.

4. **Fail-fast on missing prerequisites.** If a required upstream output doesn't exist, dry-run raises with a clear error message (e.g. "spike_clusters.npy not found at <path>; did `reconstruct.kssynth` run for this well?"). Don't silently skip — silent skip defeats the wiring-check purpose.

5. **Test contract for new phases**: every phase has a dry-run test that:
   - Builds a tmp-path fixture with the minimal prerequisites the phase needs.
   - Invokes the phase with `dry_run=True`.
   - Asserts the summary_json was written with `status: dry_run_ok`.
   - Asserts the phase's expensive function was NOT called (mock the expensive entry points and check call count = 0).
   - Asserts the phase emits a clear error when a prerequisite is missing.

6. **No partial dry-runs.** "Skip just the GPU step" or "run template extraction but skip plotting" is a `--no-plot` / phase-disabled situation, not dry-run. Dry-run is binary: resolve inputs, write summary, exit.

7. **Dry-run works under `--force-restart`**: combining `--dry-run --force-restart` resolves inputs, identifies what would be wiped, lists those in the summary's `outputs_would_produce`, and exits without rmtree'ing anything. The rmtree is part of "expensive work" that dry-run skips.

## Tests / verification

- Per-phase dry-run unit tests (per §sub-rule 5).
- After every CLI flag change touching `--dry-run` wiring: confirm `axon-recon stages <stage>.<phase> --dry-run --config <yaml>` returns 0 for a known-good fixture and non-zero for a fixture with a missing prerequisite.
- Smoke-test enabler: `axon-recon stages <stage> --dry-run --config <yaml> --targets <ds>:<well>` should complete in <30 seconds on a fully-set-up well. If it takes longer, the phase is doing expensive work in dry-run mode and the guardrail is violated.

## Open exceptions / follow-ups

- **Dry-run does not exist today.** This guardrail describes the contract a future "add `--dry-run` to every phase" slice (or set of slices) will land. The slice itself isn't yet planned; it's a candidate to add as a new entry in `phase_roster_cleanup_plan.md` or as its own plan once the phase roster settles.
- Plan ordering: dry-run is most useful AFTER the phase roster is settled (otherwise we wire it into phases we're about to delete). Schedule the dry-run rollout slice after `phase_roster_cleanup_plan` completes.
- The dry-run summary_json schema may need extension fields per phase. The base schema in §sub-rule 3 is the minimum; phases can add fields (e.g. spike count estimates from the resolved sorter_output, segment count from the manifest) as long as the base fields stay present.
