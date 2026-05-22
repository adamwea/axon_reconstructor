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

1. **OLD pipeline trims the analyzer template** in `extract_partial_templates` or `build_templates` before upsampling (e.g. crops to ms_before=0.5+ms_after=2.5=3ms). Then 30 samples × 10x upsample = 300 samples. **Plausible.**
2. **gtr.template was built with a different YAML config** — the analyzer policy that produced the existing reference data may have used `ms_before=1, ms_after=2` (3ms). Reference data is from May 12; current YAML may have evolved. **Verified: May-12 YAML had the same `ms_before=2, ms_after=5` (7ms) — so not config drift.**
3. **axon_velocity_gtrs internally rebuilds the template** from cached analyzers with its own window, ignoring what the templates substage produced. (Less likely — gtr stores template, suggesting it was passed in.)
4. **★ MOST LIKELY (per user 2026-05-21)**: the YAML waveform params (ms_before=2, ms_after=5) were NOT being applied in the OLD `extract_partial_templates`/`build_templates` path — SpikeInterface's default (ms_before=1, ms_after=2 = 3ms) was silently used instead. Math: 30 raw samples × 10x upsample = 300 samples ✓ matches gtr.template exactly. So gtr was built from SI-default-extracted templates that then got upsampled by the configured factor=10. The wider YAML config was non-functional in that path.

## What this means

- kssynth's output IS using the YAML config correctly (70 samples = ms_before=2 + ms_after=5 at 10 kHz). So kssynth is the RIGHT behavior; the OLD pipeline had a config-not-applied bug.
- "Copy the timing in the recon stage" (user directive) means: copy what the recon stage SHOULD be doing, not what it currently does. The recon stage's CURRENT effective behavior (gtr being built from SI defaults) reflects the bug, not the design.
- gtr.template (and any visual reconstruction based on it) is therefore from a 3ms-windowed STA, not the configured 7ms. That's narrower than intended.
- The radivojevic apples-to-apples comparison must decide: (a) compare against gtr.template's de-facto 3ms behavior (=run radivojevic on SI-default-windowed template), OR (b) wait for a new gtr built from the CORRECT 7ms config + match that.

## ★ VERIFIED EVIDENCE (2026-05-21 iter 2)

Inspected actual artifacts to test hypotheses:

| Source | Shape / value | Notes |
|---|---|---|
| `cache/analyzers/segments/000_rec0000/extensions/templates/average.npy` | `(335, 70, 992)` | Per-segment analyzer template. **70 samples ✓** — YAML ms_before=2 + ms_after=5 = 7ms at 10 kHz IS applied at analyzer extraction. |
| `units/0598/unit_templates_summary.json` `upsampling` block | `factor=10, method=sinc, applied=true, target_hz=100000, effective_sampling_rate_hz=100000` | Upsampling DID run per unit. Expected post-upsample = 700 samples. |
| `units/0598/gtr.pkl` `template` | `(13439, 300)` | 300 ≠ 700. There IS a trim step somewhere between per-segment-analyzer + upsample + gtr storage. |
| `units/0598/*.npy` | none exist | The merged-template npy file the OLD pipeline produced is NOT in this dir. Stored elsewhere (still unfound). |
| `load_templates_for_unit` (`core/reconstruct.py:9`) | reads `merged_template.npy` or `merged_contributing_template.npy` from `merged_units_dir/<unit_tok>/`; `fs_hz` from meta `effective_sampling_rate_hz` (would be 100000), defaults to 10000 on read failure | gtr.fs=10000 suggests meta read fell back to default at gtr-build-time. |

**Trim location — RESOLVED (user 2026-05-21)**: "They're just getting trimmed in axon_velocity." So the 700→300 trim happens inside `axon_velocity.GraphAxonTracking` (vendor code), NOT in axon_recon. axon_recon's contract: pass the full upsampled template; axon_velocity trims to its expected window internally. Slice 8 doesn't need a trim step — just the upsample.

**Comparison verdict (user 2026-05-21)**: "the comparison ends up being basically 1-to-1." Window/upsample differences between kssynth's (13439, 70) and gtr.template's effective input are minor at the reconstruction level. The radivojevic diagnostic can proceed with kssynth's current output.

## LESSON LEARNED — validate user hypotheses before propagating them

In this audit pass I accepted user-suggested theories (e.g. "SI defaults silently used", "trim happens in extract_partial_templates") and propagated them as findings without running the data inspection to confirm. User correctly called this out: "You need to do a better job validating my theories. I was wrong like 3 times."

Forward rule: when user hypothesizes a code/data behavior, treat it as a hypothesis to verify by reading the data or code FIRST, before updating audit docs or plan slices to reflect it as fact. The audit doc evolved through three wrong theories before landing on "trim is in axon_velocity" — wasted user time and my context. Adding anti-pattern entry.

## USER DIRECTIVE 2026-05-21 (post-finding)

"let's not reproduce the old bug... let's just compare the old 300 sample template to the new 700 sample template. that's good enough. we want those params wired up correctly."

Path forward:
- Slice 7 (this audit): document where the YAML waveform params get silently dropped in the OLD path. **Don't fix the OLD path** (it's being retired anyway by the kssynth integration). This is "know thy enemy" docs.
- Slice 8 (kssynth fix): add execution_upsampling application to kssynth's per-unit template output → produces (13439, 70) × 10x sinc = (13439, 700). Wire up the params correctly going forward.
- **Immediate diagnostic** (radivojevic apples-to-apples): proceed with the 300-vs-700 comparison even though the input templates differ in window/upsample. Document the methodological caveat clearly. The visual comparison is "good enough" per user.

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
