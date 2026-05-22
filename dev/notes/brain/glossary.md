# Glossary — terminology lock

Project-specific terms with locked definitions. Read this when a term in a plan / commit / question feels ambiguous; cite this when authoring new docs. Prevents fresh-loop instances from drifting on the same words.

**Reading rule for the loop**: when ambiguity arises between two interpretations of the same term, the entry here wins. If no entry exists, propose one via `open_questions.md` rather than picking unilaterally.

**Writing rule**: USER-approved additions only (via `open_questions.md` multiple-choice). Loop CAN propose new entries or refinements; user approves before they land.

---

## Pipeline structure

- **Stage** — one of the 6 top-level pipeline divisions: `init` / `preprocess` / `spikesort` / `reconstruct` / `analysis` / `cleanup`. Each stage has its own `runner.py`, CLI subparser, and a `phase_sequence` in `dev/debug_NERSC/debug.runtime.yml`.

- **Phase** — a unit of work WITHIN a stage. E.g. `reconstruct.analyzers`, `reconstruct.kssynth`, `analysis.unitmatch`. Each phase has: a Python implementation (`pipeline/stages/<stage>/phases/<phase>.py` for most; some inline in `runner.py`), a YAML config block (`stages.<stage>.phases.<phase>:`), an entry in `phase_sequence`, an `enabled: true/false` flag, and a `summary_json_relpath` for output marker.

- **Slice** — a single COMMIT-sized unit of plan execution. Plans are decomposed into slices; each slice ships in one or more commits with a slice_contracts entry. NOT the same as a phase (a slice often modifies code that implements a phase, but slices are about WORK while phases are about RUNTIME).

- **Stage_config / phase_config** — Python dataclasses (`pipeline/stages/<stage>/config.py`) holding parsed YAML state. The dataclass shape is itself a junction (J7) — changing it propagates to every parser + runner that consumes it.

- **phase_sequence** — ordered list of phase names a stage runs through. YAML-defined per stage. Comments-out-to-disable pattern is common (with the per-phase `enabled: false` as the secondary off-switch).

- **Resource class** — per-phase RAM/slot DEMANDS (e.g. `analyzer_slots: 1`, `ram_gb: 14`, `cpus_per_task: 16`). Distinct from RESOURCE PROFILE (the supply-side YAML block that resources_profiles_elimination_plan deletes). Classes stay; profiles go.

## Outputs

- **merged_template** — `merged_template.npy` at `<well>/recon_outputs/units/unit_<id>/merged_template.npy`. Shape: `(n_active_channels, n_samples)` — **sparsified** (only channels with non-zero samples survive). NOT the same as `templates.npy` (which is dense and shape `(n_units, n_samples, n_channels)`). J1 in the dependency graph.

- **merged_channel_locations** — `merged_channel_locations.npy` paired with `merged_template.npy`. Shape `(n_active_channels, 2)`. J4.

- **gtr.pkl** — `axon_velocity`'s pickled `GraphTracking` result per unit. Contains: axon trajectory, branch tree, velocity per segment, soma channel, selected channels. J2.

- **gtr-shape** / **gtr-equivalent** — shorthand for "output structurally compatible with `gtr.pkl` for downstream consumers." When the radivojevic_recon plan says "produces a gtr-equivalent shape", it means radivojevic must emit something that the existing `plot_recons` phase can read alongside the axon_velocity_gtrs `gtr.pkl`.

- **analyzer cache** — `<well>/recon_outputs/cache/analyzers/segments/<seg-id>/` — per-segment SpikeInterface `SortingAnalyzer` saved dirs. J3.

- **synth_sorter_output** — kssynth's output dir: `<well>/recon_outputs/synth_sorter_output/`. KS-shaped (spike_times/clusters, channel_map, params.py, templates.npy) PLUS slice-4-postprocess per_unit/ subdir matching merged_template layout. J5.

- **sorter_output / sorter_output_snapshot** — kilosort's raw output (`sorter_output/`) and the immutable snapshot (`sorter_output_snapshot/`). J13 / J6.

## Trust / verification

- **Trusted output** — Tier 1. USER-anchored; loop verifies against. Failure = blocking. Entries live in `brain/trusted_outputs.md` as TR-NNN.

- **Advisory** — Tier 2. Existing tests + outputs that weren't formally user-verified. Loop runs them but failure ≠ blocking, pass ≠ green. Cannot be edited by loop to make passing.

- **Provisional** — Tier 3. New output from new code, no reference yet. Loop tags downstream as "rests on provisional X" until user promotes.

- **TR-NNN** — entries in `brain/trusted_outputs.md`. TR-000 = blanket pin on reference tree. TR-001..TR-NNN = specific pinned outputs.

- **TR-CAND-NNN** — proposed-for-promotion entries (loop ranks; user triages).

- **DC-NNN** — differential checks in `brain/trusted_outputs.md` for cross-algorithm comparisons (e.g. DC-001 = Radivojevic vs axon_velocity_gtrs).

- **Junction (J1-J13)** — a contract that many phases reach for. Documented in `brain/dependency_graph.md` §3. The "high-leverage" surface — changes propagate widely.

- **Reference data tree** — `/pscratch/sd/a/adammwea/analyzed_data/Media_Density_T5_02182026_AR_axon_analysis_AW/`. Tier 1 trusted by blanket pin (D-002). Read-only; loop must never write into it.

- **Anchor cohort / Anchor well** — `260326/M08073/AxonTracking/000208/well000` (DIV 36). The most-frequently-validated well; specific count invariants in TR-001.

## Process

- **DOD** — Definition Of Done. Per-objective in `brain/objectives.md`. The completion criteria that end an objective's execution.

- **Objective (O1-O5)** — top-level goal for the current iteration cycle. Lives in `brain/objectives.md`. Slow-changing; user-anchored.

- **USER INJECTION** — user-authored directive in `brain/current_state.md` ⚡ USER INJECTIONS section. Overrides plan tier order until satisfied. Authoritative on a per-iteration basis.

- **PRE-DIAGNOSTIC GATE** — multiple-choice question in `brain/open_questions.md` that the loop MUST resolve before generating a diagnostic. Per the B2 discipline (next 3 diagnostics gated).

- **Plan-audit finding** — loop-surfaced issue in active plans / trackers. Filed under `## 🔎 Plan-audit findings (loop-surfaced)` section of `brain/open_questions.md`. Multiple-choice format. Capped at 5 open.

- **STOP AND ASK** — discipline: when blocked on explicit user instruction, loop surfaces a multiple-choice question rather than improvising. Per `brain/current_state.md` USER INJECTION B1.

- **Multiple-choice format** — every loop-surfaced question MUST be 2-4 numbered options with `label / touch-size / tradeoff` per option + a Recommended option. Mirrors the assistant's `AskUserQuestion` tool. Per B1/B2.

- **Critic separation** — every code-shipping slice spawns a separate Explore subagent as verifier (see `brain/guardrails/critic_separation.md`). Actor doesn't grade its own work.

- **Perseveration limit** — N=3 same-fix retries; M=5 stalled-slice iterations; K=20 stalled-plan iterations; T=50 loop-wide thrash. Per `brain/escalation.md`. Counters in `brain/notes.md`.

- **Slice contract** — append-only entry in `brain/slice_contracts.md` per shipped slice: `Produces / Assumes / Propagates / Trusted-output impact / Metric impact / Prediction / Actual / Delta`. Compressed-return-with-contract discipline.

## Modes

- **extended_autonomous** — full execution mode. Ships code, runs smokes, etc. Per `brain/mode.md`.

- **collaborative** — question-curator mode. Loop reads + writes brain/plans/notes only. NO src/ changes, NO smokes. Surfaces ONE multiple-choice question per iteration; executes non-destructive user-picked answers. Per `brain/mode.md` + `loop_prompts/collaborative.md`.

- **paused** — refinement-only mode. Loop can do meta work but no code / smoke / phase-impl / loop-prompt-bump.

## Devices + recordings

- **MaxTwo** — MaxWell HD-MEA at 10 kHz native sample rate. Current 80k DMEM cohort. 26,400 electrodes / ~17.5 μm pitch.

- **MaxOne** — MaxWell HD-MEA at 20 kHz native sample rate. Matches Radivojevic 2023 paper. Same electrode geometry as MaxTwo.

- **ThreeBrain / Sony HD-MEA** — future device families; NOT yet integrated. Pipeline modularity goal tracked in `trackers/tech_debt.md` §"Device-agnostic source data".

- **Anchor cohort** — see above (M08073/well000 family).

- **80k DMEM** — the well-type currently in scope. "80k" refers to ~80,000 cells/cm² seeding density on DMEM medium.

## Pipeline tools / siblings

- **axon_velocity** / **axon_velocity_gtrs** — Buccino 2022 axon-reconstruction library + the axon_recon phase that wraps it. Consumes J1; produces J2.

- **SLAy** — spike-sort auto-merge library. Consumed by `spikesort.merge_SLAy` (currently disabled). Repo: `adamwea/SLAy` (public).

- **UnitMatchPy** — upstream cross-session unit-matching library (Carcea et al.). Consumed by unitlink. Pinned in Dockerfile via `--no-deps`.

- **kssynth** — sibling pkg that synthesizes a Kilosort-style sorter_output from analyzers + a templates source. Consumed by `reconstruct.kssynth`. Repo: `adamwea/kssynth` (public).

- **unitlink** — sibling pkg that wraps UnitMatchPy with axon_recon-friendly inputs. Consumed by `analysis.unitmatch`. Repo: `adamwea/unitlink` (public).

- **radivojevic2023_recon_algo** — sibling pkg implementing the Radivojevic 2023 axon-reconstruction algorithm clean-room. Will be consumed by (planned) `reconstruct.radivojevic_recon`. Local only at `~/dev/pkgs/radivojevic2023_recon_algo/`.

- **bombcell** — Matlab QC tool used by `spikesort.bombcell_label` (currently disabled).

## Infrastructure

- **shifter** — NERSC's containerization tool. We use `adammwea/axon-recon:pipeline-v2`.

- **podman** — what we build shifter images with on Perlmutter. NEVER use `docker` on Perlmutter — it's not available.

- **salloc smoke** — a smoke test requiring an interactive Slurm allocation (login-node smokes hit walltime / oom limits). Queued in `trackers/salloc_smokes_queued.md`.

- **GraphRoot** — podman's image storage root. We've overridden to `/pscratch/sd/a/adammwea/podman_storage/` (DIRECTIVE A; the ONLY scoped pscratch overlay exception).

## Anti-confused-with table

Common confusions to head off:

| If you wrote | You probably meant |
|---|---|
| `templates.npy` (in a recon-stage context) | `merged_template.npy` per unit (recon-stage shape) — the dense `templates.npy` is the kilosort-level output |
| `phase` (in slice-tracking context) | `slice` (the work unit; phase is the runtime unit) |
| `profile` (as an active concept) | `resource_class` — profiles are being eliminated; classes stay |
| `memory/` (file path) | `brain/` — folded 2026-05-21 (commit `77d38fc`) |
| `guardrails/` (top-level file path) | `brain/guardrails/` — same fold |
| `trusted test` | `trusted output` — we trust OUTPUTS not TESTS (per D-002 / `feedback_real_data_smoke_log`) |
| `the looper` | `the loop` or "/loop session" |
| `recon stage` (as plural-phase) | the stage that has many phases; do not confuse `reconstruct.X` (a phase) with `reconstruct` (the stage) |
