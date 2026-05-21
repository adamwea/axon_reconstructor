# Scope-flag guardrail

## Contract

Every stage / phase CLI accepts the following flags with stable semantics:

| Flag | Type | Meaning |
|---|---|---|
| `--target-dataset` / `--target-datasets` | comma-sep int list OR repeated | 0-based dataset indices into `debug.data.yml`'s `datasets[*]`. Filters which datasets run. |
| `--target-well` / `--target-wells` | comma-sep mixed | well IDs as `wellNNN` strings OR plain ints (zero-padded internally). Applied across all selected datasets. |
| `--targets` | `<ds>:<well>[,<ds>:<well>...]` | Per-pair filter. Takes precedence over `--target-datasets` + `--target-wells` when set. The well part accepts int or wellNNN. |
| `--limit-datasets` | positive int | Cap on dataset count after target filtering (debug runs). |
| `--limit-wells` | positive int | Cap on total well count across all datasets. |
| `--limit-wells-per-dataset` | positive int | Cap per dataset. |
| `--limit-units` | positive int | Cap on units per well (debug). |
| `--limit-segments` | positive int | Cap on segments per well (debug). |
| `--profile` / `--task-profile` | string | Override `resources.active_profile`. Takes precedence over YAML. |
| `--task-backend` | `mpi` / `local_affinity` / `none` | Override `resources.profiles.<active>.task_allocation.backend`. |
| `--cpus-per-task` | positive int | Override `cpus_per_task` of the active profile. |
| `--force-restart` | bool | Per `guardrails/force_restart.md` — rmtree the affected output. Bypasses auto-restart-from-first-broken. |
| `--replot` | bool | Per `guardrails/force_restart.md` — run plot/report phases only, regardless of upstream phase statuses. Orthogonal to auto-restart. **Replaces the old `--force-replot` (which is eliminated).** |
| `--output-root` | path | Override `data_config.output_root` for this run. Used to write iteration outputs to `/pscratch/sd/a/adammwea/dev_outputs/<plan_slice>/` without mutating the reference `analyzed_data/` tree. |
| `--dry-run` | bool | Per `guardrails/dry_run.md` — every phase short-circuits at input resolution. |
| `--no-plot` | bool | Skip plot/report generation across plot-heavy phases. Tracked via `_NO_PLOT_OVERRIDE` ContextVar in `pipeline/config.py`. |

## Why

These flags are the user's interface for scoping any run — debug, smoke, production. They need to mean the same thing everywhere. The post-status-flag-debacle reality:

- `--target-wells well007` and `--target-wells 7` must produce identical output (commit `c205d14`).
- `--targets 6:1,12:4` must NOT cross-product into `(6,1), (6,4), (12,1), (12,4)` — it's exactly two pairs.
- `--target-datasets 4` and `--target-dataset 4` are aliases (singular for ergonomics).
- `--profile perlmutter_gpu` MUST be honored by `_build_stage_resource_budget_manager`, not silently ignored (the ds4 timeout fix).

The parsing helpers live in `src/axon_recon/pipeline/cli.py`:
- `_parse_target_dataset_indices_from_args`
- `_parse_target_well_ids_from_args` + `_normalize_well_token`
- `_parse_targets_pairs_from_args`

Process-wide overrides set at CLI entry via:
- `set_target_wells_override(...)` → `_TARGET_WELLS_OVERRIDE`
- `set_target_pairs_override(...)` → `_TARGET_PAIRS_OVERRIDE`
- `set_active_profile_override(...)` → `_ACTIVE_PROFILE_OVERRIDE`
- `set_no_plot_override(...)` → `_NO_PLOT_OVERRIDE`

All four are cleared in the `finally` block of `pipeline/cli.py`'s main() so in-process re-invocations don't carry state.

## Sub-rules for new phases

1. **Don't reinvent flag parsing.** New stage / phase subcommands inherit `--target-*`, `--limit-*`, `--targets`, `--profile`, `--task-backend`, `--force-restart`, `--replot`, `--output-root`, `--no-plot`, `--dry-run` from the shared argparse setup (currently `_register_debug_limit_arguments` + `_register_status_parser` + per-stage parsers in `cli.py`). If your phase needs a NEW flag, it goes in the per-stage parser AFTER you've checked the shared one doesn't already cover it.

2. **Stage-level resolution**: target filtering happens in `select_execution_targets` (`pipeline/config.py`). Phase-level code receives a filtered list of `ExecutionTarget`s; it doesn't re-apply the filter.

3. **Pair-override wins**: when `--targets` is set, `--target-datasets` and `--target-wells` are ignored (per the precedence rule in `select_execution_targets`).

4. **Well-id normalization**: integers always zero-pad to `well{NNN}` via `_normalize_well_token`. Anything else stays verbatim (case-sensitive, no folding). Don't strip leading zeros, don't lowercase.

## Tests / verification

- `src/axon_recon/pipeline/tests/test_cli.py` (and per-subcommand test files) cover the parser semantics. Smoke-test the help output:
  ```bash
  shifter --image=adammwea/axon-recon:pipeline-v2 axon-recon stages spikesort --help | grep -E "target|limit|profile|task-backend"
  ```
- Smoke-test trigger: any change touching the flag parsers, the override setters/getters, or `select_execution_targets` requires running one of the existing smoke runs against a known fixture and confirming the targets resolve as expected (e.g. `--targets 13:0` selects only `260326/M08073/AxonTracking/000208/well000`).

## Open exceptions / follow-ups

- `--force-replot` is DEAD. Eliminated. Use `--replot` instead.
- `--profile` is on the kill list once `resources.profiles` is gone (`trackers/tech_debt.md` §"Minimize / eliminate `resources.profiles`"). The CLI flag's existence is a workaround; long-term the resolver reads srun/cgroup state.
- `--task-backend` may shrink in scope once the profile elimination lands.
