# Unified Heatmap Migration Rollout Notes

## Runtime Hierarchy

The shared heatmap resolution order in pipeline_v2 is:

1. output-level override
2. stage-level override
3. global defaults (`global_heatmap_defaults`)
4. dataclass defaults

This hierarchy now applies to templates and reconstruct/recon parsing paths used in pipeline_v2.

## Compatibility Policy

- Legacy stage-local keys remain supported during transition.
- Global defaults are additive; existing runtime files using legacy `global_heatmap_plotting` continue to work via parser alias.
- New parser wiring merges nested blocks to avoid losing inherited defaults when partial output-level blocks are provided.

## v1 Immutability

No implementation work for this migration should modify files under `src/axon_reconstructor`.
All migration code belongs under `src/axon_recon`.

## Smoke Validation Snapshot (unit 94)

Manual smoke runs with unit 94 succeeded for both stages after linear topographical mapping fix:

- Templates stage: success, expected per-unit artifacts present including topographical and propagation outputs.
- Reconstruct stage: success, expected per-unit artifacts present and stage summary JSON generated.

Note: stage-level reconstruct summary/report artifacts are feature-gated by runtime flags (`write_summary`, `write_report_md`).

## Remaining Cleanup (Phase 9)

- Remove redundant stage-local duplication only after parity is confirmed on representative datasets.
- Keep legacy aliases until deprecation window closes.
- Keep focused precedence and compatibility tests in CI to prevent regression.

## Ready-for-Merge Checklist

Completed:

- Shared plotting layer exists under `src/axon_recon/pipeline/shared/plotting` and is consumed by templates/reconstruct paths.
- Global/stage/output precedence behavior is covered by tests and validated in smoke runs.
- Templates + reconstruct smoke runs pass on unit 94 using `tools/debug/debug.runtime.yml`.
- Backward-compatible parser behavior is tested for reconstruct stage-local settings without global defaults.
- v1 immutability constraint is satisfied (no edits under `src/axon_reconstructor`).

Optional before merge:

- Review `tools/debug/debug.runtime.yml` to prune duplicate heatmap settings now superseded by shared defaults.
- If desired, enable `write_summary` / `write_report_md` in reconstruct runtime for production smoke snapshots.
- Add a short changelog entry linking this rollout note for downstream users.
