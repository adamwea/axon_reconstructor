# `debug/` layout

This directory holds non-source assets for the axon_reconstructor pipeline:
runtime/data configs, smoketest scripts, plans for in-flight work, guardrails
that bound agent behavior, and trackers for ideas/bugs/cleanup that haven't
graduated to plans yet.

## Layout

```
debug/
  README.md                       ← you are here
  commit_log.md                   ← append-only commit log (each PR/slice adds an entry)
  debug.data.yml                  ← canonical data config (raw_data paths, well attributes, DIV)
  debug.runtime.yml               ← canonical runtime config (stages, phases, resource_classes)
  smoketest_sort_and_recon.sh     ← local end-to-end smoketest script
  mpirun.sh / localrun.sh / containrun.sh   ← canonical launch wrappers
  Ammara_MaxTwo Tracking Sheet_*  ← lab metadata workbook (genotype/DIV/condition reference)
  ai_notes/                       ← brainstorming dumps, transcripts, working notes
  notes/                          ← misc human notes (not agent-consumed)
  outputs/                        ← ad-hoc output dumps (gitignored / transient)

  plans/
    active/                       ← plans whose Definition of Done has not landed yet
    completed/                    ← plans whose DoD has landed (commit hash recorded in commit_log.md)
    abandoned/                    ← plans we decided not to pursue (each carries a "why dropped" preamble)

  guardrails/                     ← read-only law: behavior constraints, naming conventions,
                                    stage/phase contracts. Treat as policy when working in src/.

  trackers/                       ← living backlog. Entries are ideas/bugs/debt that don't yet
                                    justify their own plan doc. When an entry matures into a plan,
                                    it collapses to a one-liner pointing at the plan.
    roadmap.md                    ← future feature ideas (curation GUI, Milos axon-tracking, etc.)
    issues.md                     ← known bugs / specific fixes (`--force-restart`, plot bboxes, …)
    tech_debt.md                  ← cleanup / refactor / repo-footprint backlog
```

## Conventions

### When a plan starts

1. Create a `<topic>_plan.md` under `debug/plans/active/`.
2. If it'll be driven autonomously, write a sibling `<topic>_loop_prompt.md` next to it.
3. Add (or update) a one-line entry in `roadmap.md` pointing at the new plan; bump its
   status from `idea` → `in-plan`.

### When a plan lands

1. Verify the plan's §8 Definition of Done is satisfied.
2. Append a `## <date> - <PLAN NAME> COMPLETE` entry to `commit_log.md` summarizing test
   counts, smoke results, and any deviations.
3. `git mv debug/plans/active/<topic>_plan.md debug/plans/completed/`. Move the sibling
   `_loop_prompt.md` with it.
4. Update the roadmap entry (or remove it if it was a one-liner pointing at the plan):
   bump status to `landed` with a link to the merge commit.

### When a plan is abandoned

1. Append a 1-paragraph `## Why dropped` preamble to the plan file explaining the
   decision and what we learned.
2. `git mv` it into `debug/plans/abandoned/`.
3. Update the roadmap entry's status to `dropped`.

### When a tracker entry matures into a plan

The roadmap/issues/tech_debt entry shrinks to a one-liner pointing at the plan
doc. **Tracker entries point AT plans; plans do not duplicate tracker content.**

### Files that stay at `debug/` root (do NOT move)

- `debug.runtime.yml` and `debug.data.yml` — referenced by tooling, container CLI,
  loop prompts, and user-facing CLI examples via literal `debug/<file>.yml` paths.
  Moving them is high blast radius for low gain.
- `*.sh` smoketest scripts — same reason.
- `commit_log.md` — append-only log, treated as the single global notes file.
  Kept at root for fast discovery and short pathing in agent prompts.
- `STOP_AUTONOMOUS_LOOP` (when it exists) — sentinel for halting in-flight loops.
  Loop prompts check for it at the root path.

### Cross-references

When a plan or tracker entry references another file in `debug/`, it should use a
**full repo-relative path** starting at `debug/`. The sed pass that did the original
reorg standardized this — don't introduce relative references like `../plans/` because
they break when a file is moved between active/completed/abandoned.
