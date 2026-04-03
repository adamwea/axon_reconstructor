# CLI Parity Validation (3.2f)

Date: 2026-03-01

## Scope

Validation artifacts for stage-first CLI consolidation:

- canonical `stage` help surface
- canonical `analysis-deck` and `scope-run` help surfaces
- canonical `--env-file` acceptance path
- retired debug alias rejection signaling

## Repeatable command

From repo root:

```bash
./tools/debug/validate_stage_cli_parity.sh
```

## Result

- Status: PASS
- Output: `CLI parity checks passed`

## Notes

- `debug-*` aliases are retired and no longer registered.
- thin stage wrapper scripts were removed; canonical package commands are the active surface.
- `--env-file` is optional; direct CLI flags remain sufficient to run without env files.
