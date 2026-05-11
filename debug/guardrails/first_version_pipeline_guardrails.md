# First-Version Pipeline Guardrails

Status: temporary first-version guardrail. This document is intentionally more aggressive than a mature public pipeline policy. Once the pipeline rolls out and external compatibility matters, revisit, soften, or retire this file.

The active `axon_recon` pipeline is still the first real version of the pipeline. During this phase, the priority is to make the active workflow clean, predictable, and easy to maintain, not to preserve every historical knob, alias, fallback, or half-used path.

## Core Principle

Optimize for the pipeline that is actually being used now.

Before rollout, compatibility with unused internal history is less valuable than a small, clear, well-tested active path. Prefer deleting stale behavior over carrying it forward as a confusing option.

## YAML Knob Discipline

- Minimize runtime YAML knobs to behavior that is actively used, tested, and understood.
- Prefer one canonical knob for one behavior.
- Remove unused aliases instead of keeping parallel names indefinitely.
- Remove knobs that only preserve old config layouts or abandoned experiments.
- Do not add a new knob for hypothetical future flexibility unless the current workflow needs it.
- If a knob is retained, it should have a clear owner, default, parser test, behavior test, and smoke-test expectation when it affects runtime behavior.
- Active `debug/debug.runtime.yml` should demonstrate the desired canonical shape, not a museum of legacy alternatives.

## Fallback Code Discipline

- Remove fallback code that is not the desired behavior.
- Prefer clear validation errors over silent fallback chains.
- Do not keep fallback branches only because an old path, old stage, old config block, or old artifact layout once existed.
- If a fallback is temporarily needed for migration, mark it explicitly, test it, log when it is used, and give it a removal condition.
- Do not let fallback behavior hide missing artifacts, stale outputs, invalid config, or wrong stage/phase selection.

## Alias Discipline

- Remove unused CLI, YAML, stage, phase, and function aliases after confirming active configs and tests do not use them.
- Keep aliases only when they serve a current workflow or a deliberate migration path.
- Alias bridges should be shallow: parse once, normalize to the canonical name, and keep downstream code canonical.
- Do not add new behavior to legacy alias paths.
- Commit notes must identify which aliases were removed and what active command or config replaces them.

## Legacy Code Discipline

- Eliminate unused legacy code aggressively after reference audits.
- Delete retired modules, wrappers, tests, comments, and docs that only protect inactive behavior.
- Move still-needed logic into active v2 modules before deleting old surfaces.
- Keep runners thin and delete helper stacks that only exist to support retired execution paths.
- Do not preserve old behavior solely because it might be useful someday.

## Pre-Deletion Checklist

Before removing a knob, alias, fallback, or legacy path:

- Search active source, tests, docs, debug scripts, runtime YAML, data YAML, and container assets.
- Confirm active stage and phase selectors do not rely on it.
- Confirm direct phase execution still has a canonical path.
- Add or update tests for the desired canonical behavior.
- Run a focused pytest target.
- Run a limited real-data smoke test when the change touches CLI dispatch, config parsing, paths, phase wiring, logging, parallelism, resume, force-restart, or real-data IO.
- Record the deletion rationale, replacement path, validation, and rollback note in `debug/commit_log.md`.

## Acceptance Criteria

A first-version cleanup slice is acceptable only if:

- The active runtime YAML is simpler or more canonical after the change.
- Removed knobs, aliases, or fallbacks are not used by active configs or tests.
- Active behavior has equal or better focused test coverage.
- Any real-data behavior change is validated with CLI debug flags and a limited smoke scope.
- Logs and summaries remain clear enough to diagnose what ran, skipped, failed, or reused cache.
- Commit notes identify what was removed, why it was safe to remove now, and when the rule may need revisiting after rollout.

## Rollout Revisit

After the pipeline becomes a stable rolled-out workflow, revisit this document.

At that point, some priorities may change:

- Backward compatibility may matter more.
- Migration warnings may be preferable to immediate deletion.
- Public config schemas may need deprecation windows.
- Alias removal may need release notes.

Until then, keep the first-version pipeline lean.
