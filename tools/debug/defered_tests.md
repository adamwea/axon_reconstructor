# Deferred Tests

## Reconstruct Multi-Unit Threading Validation

Status: deferred

Reason:
- We want to run the multi-unit threading stress/behavior test after additional reconstruct migration steps are complete.

Planned test run later:

```bash
PYTHONPATH=src /home/adamm/miniconda3/envs/axon_recon/bin/python -m axon_recon stages recon --config /home/adamm/dev/pkgs/axon_reconstructor/tools/debug/debug.runtime.yml --force-restart
```

Validation goals:
- Confirm per-unit threaded execution path is exercised when multiple units are selected.
- Confirm summary ordering remains deterministic.
- Confirm outputs are complete and correct across both include_in_runtime datasets.
- Compare runtime and stability versus single-unit behavior.
