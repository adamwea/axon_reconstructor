# `dev/notes/` layout

This directory holds non-source development notes for the axon_recon pipeline:
plans for in-flight work, guardrails that bound agent behavior, trackers for
ideas/bugs/cleanup that haven't graduated to plans yet, **memory** that carries
Claude's working state across sessions, and the append-only commit log.

**See also**: the root `CLAUDE.md` is the canonical session entry point. It
points here and defines the loop protocol (model selection, context-window
management, smoke-test discipline).

Runtime/data configs and launch wrappers live in sibling directories:

- `dev/debug_local/debug.runtime.yml` + `debug.data.yml` — canonical lab-server config
- `dev/debug_NERSC/debug.runtime.yml` + `debug.data.yml` — NERSC-tuned mirror
- `src/axon_recon/default.runtime.yml` — hermetic template (bundled with package)
- `examples/example.data.yml` — schema-only data config example
- `examples/` — `mpirun.sh`, `localrun.sh`, `containrun.sh`, `smoketest_sort_and_recon.sh`,
  `perlmutter_*.sbatch.example`. All accept `RUNTIME_CFG=<path>` to point at any
  of the configs above.

## Layout

```
dev/notes/
  README.md                       ← you are here
  commit_log.md                   ← append-only commit log (each PR/slice adds an entry)
  Ammara_MaxTwo Tracking Sheet_*  ← lab metadata workbook (genotype/DIV/condition reference)
  archive/                        ← older brainstorming dumps, transcripts, working notes

  plans/
    active/                       ← plans whose Definition of Done has not landed yet
    completed/                    ← plans whose DoD has landed (commit hash recorded in commit_log.md)
    abandoned/                    ← plans we decided not to pursue (each carries a "why dropped" preamble)

  guardrails/                     ← read-only-ish law: locked code contracts. Treat as policy when working in src/.
                                    Update in a dedicated `claude:` commit when a contract genuinely changes.
                                    Topics: parallelism, scope_flags, force_restart, stage_phase_architecture,
                                    package_contracts, output_locations, dry_run. See guardrails/README.md.

  memory/                         ← Claude's cross-session working memory. Flexible, current, frequently refined.
                                    Distinct from guardrails (which are locked contracts).
                                    Files: current_state.md, open_questions.md, notes.md. See memory/README.md.

  trackers/                       ← living backlog. Entries are ideas/bugs/debt that don't yet
                                    justify their own plan doc. When an entry matures into a plan,
                                    it collapses to a one-liner pointing at the plan.
    roadmap.md                    ← future feature ideas (curation GUI, Milos axon-tracking, etc.)
    issues.md                     ← known bugs / specific fixes (`--force-restart`, plot bboxes, …)
    tech_debt.md                  ← cleanup / refactor / repo-footprint backlog
```

## Conventions

### When a plan starts

1. Create a `<topic>_plan.md` under `dev/notes/plans/active/`.
2. If it'll be driven autonomously, write a sibling `<topic>_loop_prompt.md` next to it.
3. Add (or update) a one-line entry in `roadmap.md` pointing at the new plan; bump its
   status from `idea` → `in-plan`.

### When a plan lands

1. Verify the plan's §8 Definition of Done is satisfied.
2. Append a `## <date> - <PLAN NAME> COMPLETE` entry to `commit_log.md` summarizing test
   counts, smoke results, and any deviations.
3. `git mv dev/notes/plans/active/<topic>_plan.md dev/notes/plans/completed/`. Move the sibling
   `_loop_prompt.md` with it.
4. Update the roadmap entry (or remove it if it was a one-liner pointing at the plan):
   bump status to `landed` with a link to the merge commit.

### When a plan is abandoned

1. Append a 1-paragraph `## Why dropped` preamble to the plan file explaining the
   decision and what we learned.
2. `git mv` it into `dev/notes/plans/abandoned/`.
3. Update the roadmap entry's status to `dropped`.

### When a tracker entry matures into a plan

The roadmap/issues/tech_debt entry shrinks to a one-liner pointing at the plan
doc. **Tracker entries point AT plans; plans do not duplicate tracker content.**

### Where configs and scripts live now

- Tests use the placeholder string `default.runtime.yml` (cosmetic — they don't
  open the file) or build their own YAML via `tmp_path`. Only one test
  (`stages/reconstruct/tests/test_config.py`) actually opens a real config; it
  parent-traverses to `dev/debug_local/debug.runtime.yml`.
- Launch wrappers in `examples/` accept `RUNTIME_CFG=<path>` so the same script
  works against `dev/debug_local/`, `dev/debug_NERSC/`, or any custom config.
- The `STOP_AUTONOMOUS_LOOP` sentinel (when it exists) is checked at the repo root.

### Cross-references

When a plan or tracker entry references another file in `dev/notes/`, it should use a
**full repo-relative path** (e.g. `dev/notes/plans/active/foo.md`). Don't introduce
relative references like `../plans/` because they break when a file is moved between
active/completed/abandoned.
