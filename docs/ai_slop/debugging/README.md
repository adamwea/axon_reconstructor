# Debug Harness (Package-Owned)

This folder documents the package-owned debug harness in [tools/debug](../../tools/debug).

## Purpose

- Preserve stage-by-stage developer debugging workflows inside `axon_reconstructor`.
- Provide user-facing examples for environment defaults and cross-well configs.
- Keep project-level analysis wrappers minimal and delegate heavy logic to package code.

## Entry points

- Canonical stage commands: `axon-reconstructor stage <stage> ...`
- Analysis deck command: `axon-reconstructor analysis-deck ...`
- Scope orchestration (canonical): `axon-reconstructor scope-run --config <scope.json|yml>`

## Defaults and examples

- Full migrated defaults (local): [tools/debug/debug.env](../../tools/debug/debug.env)
- Full migrated cross-well config (local): [tools/debug/cross_well_config.yml](../../tools/debug/cross_well_config.yml)
- Editable stage orchestrator script: [tools/debug/run_stage_combo.sh](../../tools/debug/run_stage_combo.sh)
- Editable scope orchestrator script: [tools/debug/run_scope_combo.sh](../../tools/debug/run_scope_combo.sh)
- User-editable generic examples:
  - [docs/examples/debug.env.example](../examples/debug.env.example)
  - [docs/examples/cross_well_config.example.yml](../examples/cross_well_config.example.yml)

## Typical usage

From repo root:

```bash
python -m axon_reconstructor.cli stage preprocess \
  --env-file tools/debug/debug.env
```

Run a stage using manual CLI args only (no env file required):

```bash
python -m axon_reconstructor.cli stage preprocess \
  --h5-path /path/to/data.raw.h5 \
  --stream-id well003 \
  --mea-output-root /path/to/outputs
```

Run reconstruction stage directly:

```bash
python -m axon_reconstructor.cli stage reconstruct --env-file tools/debug/debug.env
```

Run any stage combination via editable project-style runner:

```bash
bash tools/debug/run_stage_combo.sh
STAGES_CSV="preprocess,spikesort,analysis" bash tools/debug/run_stage_combo.sh
DRY_RUN=1 bash tools/debug/run_stage_combo.sh
```

Build analysis deck from existing stage outputs:

```bash
python -m axon_reconstructor.cli analysis-deck --env-file tools/debug/debug.env
```

Run pipeline-native stage barriers over full scope:

```bash
python -m axon_reconstructor.cli scope-run \
  --config docs/examples/scope_config.example.json \
  --env-file tools/debug/debug.env \
  --dry-run
```

Build scope config from cross-well config + env defaults:

```bash
python -m axon_reconstructor.cli scope-config-build \
  --cross-well-config tools/debug/cross_well_config.yml \
  --env-file tools/debug/debug.env \
  --out tools/debug/logs/scope_combo_generated.json \
  --stage-order preprocess,spikesort,unit_match,merge_update,waveforms,templates,reconstruct,analysis
```

Run editable scope combo script (project-style):

```bash
bash tools/debug/run_scope_combo.sh
STAGE_ORDER_CSV="preprocess,spikesort" bash tools/debug/run_scope_combo.sh
DRY_RUN=1 bash tools/debug/run_scope_combo.sh
```

## Multi-dataset usage

- Preferred path: run `scope-run` directly with an explicit scope config.
- Use [docs/examples/scope_config.example.json](../examples/scope_config.example.json) as a template.
