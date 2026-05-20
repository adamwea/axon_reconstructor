# kssynth recon-stage integration plan

> **Status (2026-05-20)**: scaffolded; slice 0 (audit) only. Companion to
> `ks_synthesizer_package_plan.md` slice 9 (which delegated this work
> here). Sibling to `unitmatch_phase_plan.md` — kssynth's output is the
> contract bridge that feeds `unitmatch_phase_plan` slice 5 (the
> per-(chip, well) match table).

## Motivation

Replace `extract_partial_templates` + `build_templates` (1,538 LoC across
two recon-stage phases) with a single `kssynth` phase that calls
`kssynth.synthesize(segment_analyzers, …)`. Output lands in
`recon_outputs/synth_sorter_output/` (a KS-shaped folder) which becomes
the canonical artifact downstream recon phases read.

The kssynth library is now installed and importable in the shifter image
(DIRECTIVE D SHIPPED, 2026-05-20 00:25 PDT; commit `d441154`). All
imports verified via in-container smoke. This plan executes the consumer
side: wire kssynth into the recon stage as a new phase, repoint
downstream consumers, retire the two replaced phases.

## Constraints carried over from sibling plans

- **Working-data scope** stays 80k DMEM well000 of M08073, all DIVs.
- **Iteration outputs** go to `/pscratch/sd/a/adammwea/dev_outputs/kssynth_recon_integration/<slice>/`.
- **Reference** under `/pscratch/sd/a/adammwea/analyzed_data/...` is
  read-only and stays as the regression baseline (pre-kssynth recon
  templates count + reconstructed unit count are the regression
  targets).
- **YAML hygiene as you go** (USER INJECTION #1): every slice that
  touches phase wiring updates both `debug_NERSC/debug.runtime.yml` and
  `debug_local/debug.runtime.yml`.
- **env_parity**: no conda dep additions needed (kssynth already in
  pyproject `[full]`/`[full-cuda]` via DIRECTIVE D commit `a85a3e5`).

## Audit findings (slice 0, this commit)

**Phases being replaced**:
- `phases/extract_partial_templates.py` (260 LoC): runs per-segment
  partial-template extraction; output landed in
  `recon_outputs/templates_cache/` (per-segment subdirs).
- `phases/build_templates.py` (1,278 LoC): merges per-segment partial
  templates into a unified template set; consumes the cache from
  `extract_partial_templates`; output is the canonical "templates" set
  used by `plot_templates_v2`, `report_templates`, `axon_velocity_gtrs`,
  etc.

**Phase consumers** (need repointing post-kssynth):
- `plot_templates_v2` — reads merged templates output to render unit
  template overlays.
- `report_templates` — reads merged templates output for summary tables.
- `axon_velocity_gtrs` — reads merged templates to extract propagation
  branches.
- (Downstream `plot_recons` / `plot_branch_*` / `plot_unit_summary` /
  `report_recons*` chain.)

**`analyzers` phase stays unchanged**: it builds per-segment
`SortingAnalyzer`s under `cache/analyzers/segments/` and publishes a
manifest. kssynth's input contract is "give me these analyzers"; the
phase itself doesn't need to know about kssynth.

**Current recon phase_sequence** (`debug_NERSC/debug.runtime.yml:1167-1185`):
```yaml
- analyzers
- extract_partial_templates    # ← REPLACE
- build_templates              # ← REPLACE
- plot_templates_v2            # ← REPOINT
- report_templates             # ← REPOINT
- axon_velocity_gtrs           # ← REPOINT
- plot_recons
- plot_branch_propagations
- plot_branch_velocities
- plot_unit_summary
- report_recons
- report_recon_grid
```

**Target phase_sequence** (post-slice-5):
```yaml
- analyzers
- kssynth                      # ← NEW (replaces 2 phases)
- plot_templates_v2            # ← reads recon_outputs/synth_sorter_output/
- report_templates             # ← reads recon_outputs/synth_sorter_output/
- axon_velocity_gtrs           # ← reads recon_outputs/synth_sorter_output/
- plot_recons
- plot_branch_propagations
- plot_branch_velocities
- plot_unit_summary
- report_recons
- report_recon_grid
```

**kssynth library API** (audited from sibling repo):
- Entry: `kssynth.synthesize(segment_analyzers, …) -> SynthResult`
- Per-step modules consumed: `cluster_tsv_sync`, `channel_grid`,
  `rasterize`, `partial_templates`, `merge_templates`,
  `io.ks_folder_writer`.
- Output: KS-shaped `sorter_output/` folder (spike_times.npy,
  spike_clusters.npy, cluster_KSLabel.tsv, cluster_group.tsv,
  cluster_Amplitude.tsv, cluster_ContamPct.tsv, channel_map.npy,
  channel_positions.npy, params.py, templates.npy, …).

## Slice execution order

### Slice 1 — `phases/kssynth.py` scaffold

**Split into two sub-slices to de-risk the analyzer-loading contract.**

#### Slice 1a — SHIPPED 2026-05-20 (commit `70021da`)
- Created `src/axon_recon/pipeline/stages/reconstruct/phases/kssynth.py`
  with entry `run_reconstruct_kssynth_phase(inputs: TemplatesInputs) -> dict[str, Any]`.
- Output constants pinned: `KSSYNTH_OUTPUT_RELDIR = "synth_sorter_output"`,
  `KSSYNTH_SUMMARY_RELPATH = "synth_sorter_output/kssynth_summary.json"`.
- `_resolve_kssynth_output_dirs` wraps existing `_resolve_build_templates_context`.
- v1a body: validates the `kssynth` import path, writes a stub summary JSON
  marked `status: scaffold_only` (or `error` on ImportError), returns the
  dict. No analyzer loading, no `kssynth.synthesize` call yet.
- 3 unit tests in `tests/test_kssynth_phase.py` (monkey-patch
  `_resolve_kssynth_output_dirs` to avoid the heavy `TemplatesInputs`
  fixture machinery): scaffold_only happy path, summary-JSON-on-disk
  contract, ImportError → error status.
- Phase NOT wired into `phase_sequence`; unreachable from CLI.

#### Slice 1b — SHIPPED 2026-05-20 (commit `4957ae0`)
- `_resolve_kssynth_output_dirs` returns `(well_out_dir, templates_out_dir,
  synth_out_dir)` — third element added so the output path is computed
  once at the top of the entry function.
- New helper `_load_segment_analyzers` wraps
  `templates.runner._load_templates_phase_analyzers` (the same loader
  `build_templates` uses) and drops the `(source_name, analyzer)`
  tuples → list[Any] matching kssynth.synthesize's signature.
- `run_reconstruct_kssynth_phase` calls `kssynth.api.synthesize` with
  default options (channel_grid="union",
  aggregation="spike_count_weighted_mean"). Returns the translated
  `WriterResult` → summary JSON: n_units (from unit_ids), n_channels,
  files_written, channel_grid_mode, policy, n_analyzers.
- Three error paths: ImportError, analyzer-load exception, synthesize
  exception — each writes `status: error` with the cause + partial
  state.
- 5 tests in `test_kssynth_phase.py` (all pass): ok path summary
  translation, summary persistence, synthesize exception, loader
  exception, ImportError path.
- Phase still unwired in YAML; unreachable from CLI. Real-data smoke is
  slice 3.

**Original spec (preserved for traceability):**
- Create `src/axon_recon/pipeline/stages/reconstruct/phases/kssynth.py`
  with a single `run_kssynth_phase(inputs: TemplatesInputs) -> dict[str, Any]`
  entry point that, in v1, calls `kssynth.synthesize(...)` with the
  analyzers resolved from `inputs.analyzers_cache_dir`.
- Output dir: `<well_out_dir>/recon_outputs/synth_sorter_output/`.
- Summary JSON: `<well_out_dir>/recon_outputs/synth_sorter_output/kssynth_summary.json`
  with `{status: ok|error, n_units: int, n_segments: int, params: {...}}`.
- No YAML wiring yet (phase remains unreachable from `phase_sequence`).
- Tests: 2-3 unit tests using stub analyzers (mock `kssynth.synthesize`)
  asserting the I/O contract.

### Slice 2 — wire `kssynth` into the recon-stage config + CLI dispatch

Split into 2a (config dataclass + YAML parser) + 2b (CLI dispatch +
runner wiring) for safer integration.

#### Slice 2a — SHIPPED 2026-05-20 (commit `d06a51c`)
- `models/inputs.py`: new `ReconstructionKssynthPhaseConfig` with
  `enabled`, `summary_json_relpath`, `resource_class`, plus the
  `kssynth.synthesize` knobs (`channel_grid`, `aggregation`,
  `tolerance_um`, `dtype`, `treat_zero_as_missing`, `clobber`).
  Defaults match the kssynth library spec. Field added to
  `ReconstructionPhasesConfig` between `clear_templates_cache` and
  `axon_velocity_gtrs`.
- `config.py`: import + new `kssynth_cfg = …` block parallels the other
  phase-block helpers; constructor reads each knob with `_as_bool` /
  float / str coercions and defaults.
- 2 new tests in `test_config.py`: populated YAML block sets all knobs,
  absent block keeps defaults (enabled=False, opt-in).
- 686 tests pass (108 recon + 574 pipeline + 4 skipped).

#### Slice 2b — SHIPPED 2026-05-20 (commit `a1d62a4`)
- 6-file plumbing chain mirroring `reconstruct.build_templates`:
  - `stages/reconstruct/runner.py`: bridge wrapper
    `run_reconstruct_templates_kssynth_phase` validates
    `inputs.templates_inputs` + dispatches to inner phase fn.
  - `stages/reconstruct/api.py`: `run_reconstruct_templates_kssynth`
    wraps the bridge with `_run_with_quiet_unexpected_plot_logs`.
  - `pipeline/runner.py`: `run_reconstruct_templates_kssynth_from_runtime`
    routes through `_run_reconstruct_substage_from_runtime` with
    `stage_name="reconstruct.kssynth"`.
  - `stages/reconstruct/cli.py`: `_run_reconstruct_kssynth_from_args`
    routes the parsed Namespace.
  - `pipeline/cli.py`: dispatch table maps `reconstruct.kssynth`;
    5 aliases (`recon.kssynth`, `recon.templates_kssynth`,
    `reconstruction.kssynth`, `reconstruction.templates_kssynth`,
    `reconstruct.templates_kssynth`).
- Tests: 6 new alias↔canonical pairs in `test_cli_stage_sequence.py`
  parametrize block. All pass.
- The CLI subcommand IS now invokable directly. With `enabled: false`
  the runtime should treat it as a no-op; slice 3 verifies on real
  data.

### Slice 3 — YAML wire-in + smoke

Split into 3a (YAML wire-in) and 3b (login-node smoke; needs a real
analyzer cache).

#### Slice 3a — SHIPPED 2026-05-20 (commit `6f519b8`)
- `debug_NERSC/debug.runtime.yml` + `debug_local/debug.runtime.yml`:
  added `stages.reconstruct.phases.kssynth: { enabled: false, ... }`
  blocks with all slice-2a knobs (channel_grid, aggregation,
  tolerance_um, dtype, treat_zero_as_missing, clobber) and
  resource_class=template_build + summary_json_relpath defaults.
  `phase_sequence` untouched — the phase is opt-in via the
  `enabled` knob.
- Status + phase_tuning tests stay green. Broader recon + pipeline
  sweep also green.
- Pre-existing analyzed_data on pscratch
  (`/pscratch/.../260326/M08073/AxonTracking/000208/well000/recon_outputs/`)
  has an empty `cache/` dir — no segment-analyzer cache available
  for an immediate smoke. Slice 3b is deferred until `reconstruct.analyzers`
  has run on at least one well (the smoke is a two-step: analyzers first,
  then kssynth).

#### Slice 3b — pending (needs analyzer cache or user-initiated run)
- Confirm an analyzer cache exists at
  `<well>/recon_outputs/cache/analyzers/segments/`. If not: user runs
  `axon-recon stages reconstruct.analyzers --config dev/debug_NERSC/debug.runtime.yml
  --target-dataset N --limit-wells 1 --task-backend local_affinity`
  on M08073/well000/DIV36 first.
- Then run `axon-recon stages reconstruct.kssynth --config dev/debug_NERSC/debug.runtime.yml
  --target-dataset N --limit-wells 1 --task-backend local_affinity`
  (requires temporarily setting `kssynth.enabled: true` OR adding a
  `--force-enable kssynth` CLI override — design TBD).
- Confirm `<well>/recon_outputs/synth_sorter_output/` materializes
  with the expected files (spike_times.npy, spike_clusters.npy,
  cluster_KSLabel.tsv, cluster_group.tsv, channel_map.npy,
  channel_positions.npy, params.py, templates.npy, kssynth_summary.json).
- Visual diagnostic: capture the directory listing + `kssynth_summary.json`
  contents into
  `/pscratch/sd/a/adammwea/dev_outputs/kssynth_recon_integration/slice3b/`.
  Add a SOFT-gate entry to `memory/diagnostics_to_review.md` for
  user review.

### Slice 4 — downstream phase repointing

**Design audit (2026-05-20, loop iteration after slice 3a)**:
- `plot_templates_v2`'s analyzer-side code (e.g. `phases/plot_templates_v2.py:243`)
  resolves templates dirs via `templates_runner._resolve_templates_dirs(...)`
  which expects the `merged_units_dir/unit_<id>/merged_template.npy` layout
  produced by `build_templates`. kssynth produces a KS-shaped
  `sorter_output/` (spike_times.npy + spike_clusters.npy + templates.npy
  + channel_map.npy + ...) — a DIFFERENT shape entirely.
- Two viable repointing strategies:
  - **(S4-A) Conversion layer in downstream phases**: each consumer
    gets a `templates_source` YAML toggle. When `kssynth` is selected,
    the phase reads `synth_sorter_output/templates.npy` (shape
    `(n_units, n_samples, n_channels)`) and demuxes it into per-unit
    slices on the fly. Adds complexity to each consumer.
  - **(S4-B) Postprocess step in kssynth**: extend the recon-stage
    `kssynth` phase to ALSO write per-unit `merged_template.npy` in
    the build_templates-compatible layout (just `templates.npy` sliced
    + saved per-unit). Then downstream phases need no changes; just
    repoint their input dir to `synth_sorter_output/per_unit/`.
- **Recommendation**: S4-B is lower-risk for the downstream code.
  The per-unit demux is mechanical (slicing a numpy array + writing
  N files). The kssynth WriterResult already provides `unit_ids` and
  the templates path; this is just an io helper.
- Either strategy means slice 4 is multi-file substantial work and
  benefits from its own iteration with a fresh context.

**Original spec (preserved)**:
- `plot_templates_v2`, `report_templates`, `axon_velocity_gtrs` each have
  a "where to read templates from" config field. Add a YAML toggle
  `templates_source: build_templates | kssynth` (default
  `build_templates` for back-compat) and wire each phase to read from
  either `templates_cache/` (old) or `synth_sorter_output/` (new).
- Tests: parametrized unit tests cover both modes.

### Slice 5 — enable in YAML + retire predecessor phases
- Flip `kssynth.enabled: true` in both debug YAMLs.
- Update both `phase_sequence` blocks: replace
  `extract_partial_templates, build_templates` with `kssynth`.
- Flip `templates_source: kssynth` on plot_templates_v2 +
  report_templates + axon_velocity_gtrs (per slice 4).
- Run full reconstruct stage smoke on M08073/well000/DIV36 (one of the
  known-good baselines): expect 176 reconstructed templates as the
  regression target (matches the 2026-05-18 pre-fix count). HARD-gate
  diagnostic — the templates count + visual overlay are the regression
  signal.
- After approval: delete `extract_partial_templates.py` +
  `build_templates.py`; remove their config dataclasses + parser logic;
  remove their CLI dispatchers; remove their tests. Touch is L.

### Slice 6 — unblock unitmatch_phase slice 5
- `unitmatch_phase_plan` slice 5 (enable + smoke) was gated on kssynth
  producing the per-(chip, well) `synth_sorter_output/` directories.
  Once slice 5 of this plan lands and produces those outputs on the
  80k DMEM cohort's well000, `unitmatch_phase` can proceed: flip
  `unitmatch.enabled: true`, run on a chip-well group with ≥3 DIVs,
  verify match table + UID chains land.
- This slice is in `unitmatch_phase_plan.md`; mentioned here as the
  downstream unlock.

## Tests + verification

- Per-slice: unit tests + (where slice changes runtime behavior) one
  login-node smoke on M08073/well000.
- Regression target: 176 reconstructed templates on
  `260326/M08073/000208/well000` (DIV 36) — matches the post-SLAy good-
  label count from the 2026-05-18 status replay.
- Container smoke: confirm `import kssynth` succeeds in the shifter image
  (DIRECTIVE D SHIPPED already verified this; baseline established).

## What this UNBLOCKS

- `unitmatch_phase_plan` slice 5 (enable + smoke).
- `radivojevic_recon_algo_plan` slice 3 (core algorithm impl — gated on
  kssynth slice 9 + unitmatch_phase slice 5 BOTH shipped, the "first
  real end-to-end smoke through the new sibling packages" trigger).
- `chip_layout_phase_split_plan` slice 1 (audit) — same kick-off
  trigger as Radivojevic.

## Risk + rollback

- The two phases being retired produced visually verifiable templates
  on real data. Slice 5's HARD-gate diagnostic compares pre/post images
  side by side; if kssynth's output differs visually from the
  reference, the user reviews before the retirement commit lands.
- Each slice is its own commit; `git revert` of any slice rolls back
  cleanly. The retirement commit (slice 5b) is the only "destructive"
  one — splittable into a separate revert-friendly commit.

## Open questions (logged to memory/open_questions.md as we hit them)

- Does `kssynth.synthesize` need any options that `build_templates` had
  but extract_partial_templates didn't (or vice versa)? Audit those
  config knobs during slice 1.
- Are there phases ELSEWHERE in the codebase (not in recon stage) that
  also read the templates_cache or build_templates output? Slice 1
  audit answers this.
- How does `--replot` interact with the new phase? `--replot` was
  defined as "run plot/report phases only"; `kssynth` is a computation
  phase, not a plot phase, so it's skipped by `--replot`. Verify in
  slice 3.
