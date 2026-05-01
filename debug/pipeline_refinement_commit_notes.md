# Pipeline Refinement Commit Notes

Living review log for AI-assisted refinement commits.

Use this file after Adam starts the longer iteration/refinement run. The instruction file is `debug/pipeline_refinement_instructions.md`; review both markdown files before each change slice. Once iteration begins, do not edit the instruction file unless Adam explicitly asks. Update this notes file after each AI commit and whenever Adam asks for running notes.

## How To Use

For each commit, append a new entry at the top of `Commit Log` or directly below the most recent entry.

Each entry should let Adam quickly review what changed after an unattended run:

- what was changed
- why it was changed
- acceptance criteria used
- what was expected to run and what was confirmed not to run
- tests and smoke checks run
- smoke timeout decisions and log inspection when a smoke is extended
- storage/cache impact
- resume and force-restart impact
- CLI impact
- any risks, follow-ups, or rollback notes

## Entry Template

```markdown
## YYYY-MM-DD HH:MM - <short_sha> - ai: <commit subject>

Status: accepted | needs follow-up | reverted

Summary:
-

Acceptance Criteria:
-

Expected To Run:
-

Confirmed Not Run:
-

Files/Modules Changed:
-

Validation:
- Pytest:
- Smoke (20 min max unless Adam approves longer):
- Smoke extension to 1 hour:
- Logs inspected:
- Not run:

Resume / Force-Restart Impact:
- Resume behavior:
- Force-restart cleanup:
- Partial-output handling:

Storage/Cache Impact:
- Created:
- Cleaned:
- Persisted:
- Size check:

CLI Impact:
-

Retired Code/Tests:
-

Risks And Follow-Ups:
-

Rollback Notes:
-
```

## Commit Log

No refinement commits logged yet.
