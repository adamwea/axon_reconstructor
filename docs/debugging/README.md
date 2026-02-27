# Debug Harness (Package-Owned)

This folder documents the package-owned debug harness in [tools/debug](../../tools/debug).

## Purpose

- Preserve stage-by-stage developer debugging workflows inside `axon_reconstructor`.
- Provide user-facing examples for environment defaults and cross-well configs.
- Keep project-level analysis wrappers minimal and delegate heavy logic to package code.

## Entry points

- Main staged runner: [tools/debug/debug_steps.py](../../tools/debug/debug_steps.py)
- Individual stages:
  - [tools/debug/debug_preprocessing_step.py](../../tools/debug/debug_preprocessing_step.py)
  - [tools/debug/debug_spikesorting_step.py](../../tools/debug/debug_spikesorting_step.py)
  - [tools/debug/debug_waveforms_step.py](../../tools/debug/debug_waveforms_step.py)
  - [tools/debug/debug_templates_step.py](../../tools/debug/debug_templates_step.py)
  - [tools/debug/debug_reconstruction_step.py](../../tools/debug/debug_reconstruction_step.py)
  - [tools/debug/debug_analysis_step.py](../../tools/debug/debug_analysis_step.py)
- Multi-dataset shell runners: [tools/debug](../../tools/debug)

## Defaults and examples

- Full migrated defaults (local): [tools/debug/debug.env](../../tools/debug/debug.env)
- Full migrated cross-well config (local): [tools/debug/cross_well_config.yml](../../tools/debug/cross_well_config.yml)
- User-editable generic examples:
  - [docs/examples/debug.env.example](../examples/debug.env.example)
  - [docs/examples/cross_well_config.example.yml](../examples/cross_well_config.example.yml)

## Typical usage

From repo root:

```bash
python tools/debug/debug_steps.py --env-file tools/debug/debug.env
```

Run one stage directly:

```bash
python tools/debug/debug_reconstruction_step.py --env-file tools/debug/debug.env
```
