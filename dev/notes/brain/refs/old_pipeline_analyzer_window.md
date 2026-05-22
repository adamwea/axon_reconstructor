# OLD pipeline analyzer window + upsample audit

Audit for `kssynth_recon_integration_plan.md` slice 7. Goal: document what
the OLD `extract_partial_templates` + `build_templates` phases did to
analyzer-derived templates (analyzer window + upsample) so kssynth slice 8
can match it.

**Status**: PARTIAL — covers code-level + YAML-level findings. Empirical
verification (re-running OLD pipeline + comparing to gtr.template shape)
pending.

---

## Key data points

| Source | Value |
|---|---|
| `gtr.pkl` for unit_0598 (reference data, M08073/000208/well000 DIV 36) | `template.shape=(13439, 300)`, `fs=10000.0`, `_upsample=1` |
| kssynth output for same unit (2026-05-21 salloc smoke) | `merged_template.shape=(13439, 70)`, no fs attr in npy |
| `debug.runtime.yml` `execution_upsampling` (line 1424) | `enabled=true, method=sinc, factor=10, mismatch_tolerance_hz=0.5` |
| `debug.runtime.yml` analyzers phase `segments.waveforms` (line 1303) | `ms_before=2, ms_after=5` (= 7ms window = 70 samples at 10 kHz) |
| `debug.runtime.yml` analyzers phase `defaults.waveforms` (line 1323) | `ms_before=2, ms_after=5` (same) |
| `debug.runtime.yml` reports `waveforms` block (line 743) | `ms_before=2.0, ms_after=3.0` (5ms — different context) |

## What the math says

- 300 samples / 10 = 30 samples (pre-upsample) at 10 kHz raw = 3ms window
- BUT YAML analyzer ms_before+ms_after = 7ms (70 samples raw)
- **Mismatch**: gtr.template's effective pre-upsample window (3ms) doesn't match the YAML analyzer config (7ms).

## Hypotheses for the mismatch

1. **OLD pipeline trims the analyzer template** in `extract_partial_templates` or `build_templates` before upsampling (e.g. crops to ms_before=0.5+ms_after=2.5=3ms). Then 30 samples × 10x upsample = 300 samples. **Most likely.**
2. **gtr.template was built with a different YAML config** — the analyzer policy that produced the existing reference data may have used `ms_before=1, ms_after=2` (3ms). Reference data is from May 12; current YAML may have evolved.
3. **axon_velocity_gtrs internally rebuilds the template** from cached analyzers with its own window, ignoring what the templates substage produced. (Less likely — gtr stores template, suggesting it was passed in.)

## Where upsampling lives (current code)

- `templates/core/merge.py:391` — `merge_unit_templates_from_payloads(..., execution_upsampling: TimeUpsampleConfig)` applies upsample during merge.
- `templates/core/render.py:1251` — `_time_upsample_template(template, upsample)` is the actual sinc-based upsampler.
- `templates/config.py:997` — `_build_time_upsample_config` reads YAML.
- YAML knobs: `enabled`, `factor` (int), `method` (default "sinc"), `mismatch_tolerance_hz`, `raw_rate_fallback_hz`.

## Where analyzer-window lives (current code)

- `phases/analyzers.py` consumes `inputs.phases.analyzers.segments.waveforms.ms_before/ms_after` (from YAML line 1303).
- `phases/build_templates.py` + `templates/core/build_templates.py` build templates from cached analyzers — uses the analyzer's existing window (no separate ms config at the build_templates layer).

## What kssynth does today

- `phases/kssynth.py` calls `_load_templates_phase_analyzers` (which goes through `iter_spikeinterface_analyzers`) with the YAML analyzer policy.
- BUT: kssynth's `_iter_templates_phase_analyzers` call at runner.py:1685 explicitly forces some defaults. Per the salloc log: `Segment analyzer policy: ms_before=2 ms_after=5 random_spikes_percentage=100.0%`. So kssynth IS reading the YAML config (7ms window), not hardcoded.
- **kssynth does NOT upsample** — the `execution_upsampling` config is not applied in `phases/kssynth.py`'s merge call. This is the bug.
- Per user (2026-05-21): "kssynth shouldn't be running its own parameters or code." The user's concern may also be that kssynth re-builds analyzers internally instead of consuming cached ones from `reconstruct.analyzers`. Worth verifying — kssynth as currently shipped DOES rebuild analyzers each run (uses `build_if_missing=true` and the segment cache_dir on the same well).

## Open audit items (next slice-7 iteration or escalate to user)

1. **Find where the trim happens** (if hypothesis 1) — search `extract_partial_templates.py` or `templates/core/build_templates.py` for ms-based or sample-based trimming logic.
2. **Confirm gtr.template's effective rate** — load gtr + look for any hint of the actual upsample applied (might be stored in `_init_frames` or similar).
3. **Check May-12 YAML state** — `git show ...debug.runtime.yml` at the commit that built the reference data, to see if analyzer config differed.
4. **Verify execution_upsampling is reached by build_templates** — trace from YAML → config dataclass → `merge_unit_templates_from_payloads` call site in build_templates.py.

## Slice 8 prerequisites

Once items 1-4 above are resolved, slice 8 can:
- (a) Make kssynth read AND apply `execution_upsampling` from YAML (factor=10 sinc by default).
- (b) Make kssynth apply the same trim/window logic that build_templates does.
- (c) Optionally: make kssynth consume cached analyzers from `reconstruct.analyzers` rather than rebuilding (cheaper + guarantees same source data).

## DIRECTIVE — copy timing from recon stage (USER 2026-05-21)

"you need to copy the timing in the recon stage."

The principle: kssynth must NOT have its own timing parameters. It should adopt the recon stage's effective timing end-to-end — analyzer window (ms_before/ms_after), trim, and upsample (factor/method). Whatever values the recon stage uses to produce its templates, kssynth must use the same values to produce its templates. Otherwise downstream `axon_velocity_gtrs` (and any other consumer) will see different inputs depending on which path produced the template.

Practically for slice 8:
- Read the recon stage's effective config (analyzer policy + execution_upsampling + any trim).
- Apply ALL of it in kssynth's template path.
- Tests verify that for the same analyzer source, kssynth + the OLD pipeline produce byte-equivalent (or shape-equivalent + numerically-close) merged templates.
