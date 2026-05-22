# Current state

Snapshot of what's shipped, in-flight, and queued. Updated as state changes; old facts get deleted, not commented out.

## ✅ PAUSE LIFTED — extended_autonomous resumed (2026-05-21)

Planning / refinement pause RESOLVED via QZ11 (option 1 — lift now → extended_autonomous). Phase zero complete: Z1 dependency_graph mapped, Z2 pinned outputs triaged (TR-000..004), Z3 invariant specs authored + approved (Z3-TR-000..004 in `trusted_outputs.md`), Z4 pause-lift executed. PAUSE NOTICE removed from `loop_prompts/extended_autonomous.md`. Mode = which `/loop` prompt is running (no central router file; `brain/mode.md` deleted 2026-05-21 per D-016 as over-engineered). Next autonomous work per QZ11 + D-010: PRE-DIAGNOSTIC GATE 1 (radivojevic apples-to-apples comparison via plot_recons).

## ⚡ USER INJECTIONS

User-authored directives that override plan / tier order until satisfied. Read FIRST each iteration. Apply at the earliest applicable slice; when reliably internalized, promote the rule to a guardrail / CLAUDE.md slice protocol / plan and delete the entry here.

### Active

(YAML hygiene promoted to CLAUDE.md slice protocol 2026-05-21; entry retired.)

- **[2026-05-19] Env-parity + LOOP CAN REBUILD SHIFTER (collapsed)**: Loop is authorized to run `podman build` → `podman push docker.io/...` → `shifterimg pull docker:...` for axon-recon image rebuilds. Pre-check `podman login --get-login docker.io` (returns `adammwea` via persistent authfile per `brain/guardrails/env_parity.md`); if missing, post blocker + pivot. NEVER use `docker` on Perlmutter — `podman` only. "No slurm submission" and "no axon_recon git push" rules remain in effect. **Active shifter image**: `adammwea/axon-recon:pipeline-v2` (shifter hash `cfc82cc501`, READY 2026-05-20T00:23:03; includes slay + kssynth + unitlink). **Slice-level discipline**: any slice that adds/removes/upgrades a conda dep MUST update pyproject.toml `[full]`/`[full-cuda]` (or the `--no-deps` Dockerfile step for numpy-conflicting siblings) AND post a "shifter rebuild needed: X" note here. **Full history archive** (DIRECTIVE A/B/C/D resolution journey, build #1-#3, gh-repo-create flow) lives in commits `eb3b290` / `d441154` / `1599db7` / `433415b`; read those when archeology is needed.

- **⭐ PRE-OVERNIGHT CLEARANCES (2026-05-19, set during pre-loop check)**: User answered 3 foreseeable gates ahead of the overnight loop run so they don't block.
  1. **Dashboard slice 7 — tertiary grouping UX: configurable per chart (YAML toggle).** Both `small-multiples` and `hierarchical-X-labels` render modes get implemented; user picks per chart via a per-phase YAML knob (e.g. `tertiary_render_mode: small_multiples | hierarchical_labels`) or a dashboard UI dropdown. ~1 extra commit vs picking one mode; most flexible long-term. Default mode TBD by the loop during slice 7 — pick `small_multiples` as default since it reads better for the typical few-value tertiary case; user can override per chart.
  2. **Radivojevic plan slice 1 — PRE-APPROVED to continue past USER GATE 1 into slice 2.** Loop ships slice 1 (literature read + code search + `dev/notes/brain/refs/radivojevic2023_paper.md` + `dev/notes/brain/refs/radivojevic2023_algorithm_summary.md` + input-vs-axon-velocity compat map), THEN immediately scaffolds the sibling package per slice 2 (`pyproject.toml`, package layout, README with citation, 2-3 sanity tests) WITHOUT pausing. **All slice-1 review questions get logged to `open_questions.md` under "Radivojevic slice 1 user-gate review" for AM user review.** Subsequent gates (slices 3, 4, 6, 9) still apply normally. Slice 3 (core algorithm implementation) DOES NOT START until the slice-1 questions are reviewed.
  3. **SLAy PR merge policy — USER-ONLY merge.** Codified in DIRECTIVE D step 1 above. Loop pushes the branch + opens the PR via `gh pr create` but never runs `gh pr merge`. User reviews and merges via GitHub web UI when convenient.

- **[2026-05-21] Phase-enable for tests (general rule)**: When a slice needs to smoke-test a phase that's currently YAML-disabled (`enabled: false`), the loop MUST enable the phase for the smoke — either temporarily via a YAML edit it reverts, or via a CLI override. Never test a phase while it's disabled and treat the no-op as validation. Per user 2026-05-21: "Feel free to enable phases as needed for tests. Don't test it disabled and think that validates it."
  - **✅ CLI FLAG SHIPPED 2026-05-21 (commit `5d28561`)**: `--force-enable PHASE[,PHASE...]` added to the recon stage CLI + shared `stages` subparser. Process-wide override pattern mirrors `--no-plot`/`--profile`/`--scratch-output`/`--output-root`. Helper `_apply_force_enable_phases` in `pipeline/runner.py`; setter/getter in `pipeline/config.py`. Applied inside `_run_reconstruct_substage_from_runtime` (execution path) AND `_build_reconstruct_allocation_preview` (`--alloc` preview path) AFTER YAML parsing but BEFORE phase-roster + target selection. Override covers both the recon stage_config tree and the templates substage tree. 15 new tests cover process-wide setter/getter invariants + helper edge cases + 1 real-YAML integration. Slice 3b smoke is now unblocked (`axon-recon stages reconstruct.kssynth --config ... --force-enable kssynth`).
  - **Use count toward promotion**: 1 (kssynth slice 3b — pending smoke execution). Promotion criterion = ≥2 slice uses; when slice 5 (full-stage smoke with `--force-enable kssynth,plot_recons,axon_velocity_gtrs`) or any other YAML-disabled phase smoke uses the flag, promote the rule and delete this injection entry. The CLI flag stays — it's general utility.

- **[2026-05-21] Real-data smoke log discipline**: After every smoke on REAL DATA (login-node OR user-initiated salloc), append an entry to `dev/notes/trackers/smoke_log.md` per its schema (command / cohort / commit / outcome / quantitative result / bugs→fix-commit / diagnostics link / baseline-established? / status). EXCLUDES dry-runs, unit tests, synthetic-fixture smokes. HARD-gate visual diagnostics get a smoke_log entry when reviewed. Use count toward promotion: 1 entry filed post-rule (radivojevic cluster 67); promote to CLAUDE.md slice protocol at ≥3. **Worker-count validation (was S2) RETIRED 2026-05-21**: subsumed by `plans/active/resources_profiles_elimination_plan.md` slice 2 (env-only resolver, no clamping) + slice 5 (real-data smoke verification). The fail-fast behavior is built into that plan; don't duplicate as a free-floating rule. Loop SHOULD still record actual vs expected worker counts in smoke_log entries while the elimination plan ships — that's just good smoke discipline, not a separate rule.

- **[2026-05-21] Tightened visual-diagnostics trigger**: Compact rules effective immediately:
  - **(R1)** Any slice that changes user-visible rendering MUST file a diagnostic entry. Hard requirement, not judgment call. Includes: dashboard/UI changes (new plot type, render mode, empty-state, style/palette/font, new feature on existing plot); any new phase producing a visual artifact (figure/plot/video/heatmap); bug fixes producing visually-different output even when tests pass; first real-data run of any new algorithm stage.
  - **(R2)** Multi-stage algorithms file at EACH stage transition (Radivojevic Stage 1/2/3 each get their own filing — Stage 1+2 soft-gates, Stage 3 + comparison hard-gate).
  - **(R3)** Exempt: routine pipeline outputs during normal runs; pure-computation with numerical validation; logs/text summaries (commit messages, not diagnostics_to_review.md).
  - **(R4)** Artifacts live at `/pscratch/sd/a/adammwea/dev_outputs/<plan_slice>/diagnostics/`.
  - **(R5)** ANY diagnostic with claimed visual content MUST include a rendered image (PNG/SVG/PDF). npy/tsv/parquet alone is data, not a diagnostic.
  - **Backfill scope**: SKIPPED for dashboard slices 2/5/6/7/8 — user reviews interactively. Going-forward only.
  - **Use count toward promotion**: 1 (radivojevic SOFT-gate re-filed with PNG). Promote at ≥3 to CLAUDE.md §"Visual diagnostics".

- **🛑 [2026-05-21 — CRITICAL behavioral injection] STOP-AND-ASK discipline for diagnostics (next 3 attempts MANDATORY)**: The first Radivojevic SOFT-gate diagnostic attempt failed on 3 user-explicit requirements (use plot_recons / pick high-branch unit / produce comparison) because the loop **improvised around friction** instead of stopping to ask. New mandatory behavior:

  **(B1) When blocked on or facing friction with an EXPLICIT user instruction about a surface area, the loop MUST STOP and write a focused MULTIPLE-CHOICE question to `open_questions.md` rather than improvising a substitute approach.** The "ask" form MUST be options (2-4 numbered choices), NOT a bare halt with prose description. Each option labeled `label / touch size / tradeoff`; loop picks a "recommended" option (most defensible default) and the user picks or adjusts. User-side resolution: mark `✅ USER APPROVED <date>: option N` above the question; loop executes that option. (Amended 2026-05-21 per user: "for now, instead of stopping ask it to just give me choices like you just did.") Specifically — the loop is FORBIDDEN from:
   - Building new plotting code in a sibling repo when the user said "use existing plot_recons"
   - Substituting a different unit/cluster/well when the user picked the target
   - Producing a single-algorithm output when the user asked for a comparison
   - Reducing scope on an explicit deliverable to make the immediate iteration succeed

  **(B2) For the NEXT 3 diagnostic-generation iterations, the loop MUST PAUSE BEFORE GENERATING EACH DIAGNOSTIC and surface a multiple-choice question for the user.** The pause writes the plan to `open_questions.md` as "🛑 PRE-DIAGNOSTIC GATE N — pick plan before I execute" with the EXECUTION PLAN as option 1 (the loop's recommended path) AND 1-3 alternative options (e.g. "tweak input choice", "switch rendering path", "abandon this diagnostic for now"). The user picks, the loop executes. If the user replies with adjustments instead of picking, the loop revises the plan and surfaces an amended multiple-choice question. Required fields per option:
   - Exact input data path
   - Exact code path that will produce the visual (existing-phase reuse vs new code)
   - Exact comparator (if applicable) and how it's being rendered
   - Expected output layout
   - Any sub-steps where the loop anticipates friction (each one becomes its OWN potential mid-execution multiple-choice halt per B1)
  Loop does NOT begin diagnostic generation until the user picks an option. After 3 successful pre-gated diagnostics, this discipline relaxes back to "file as you go" — by then the loop has demonstrated it's internalized the lesson.

  **(B3) Root-cause analysis preserved for the loop to read** (so this isn't an opaque rule):
  - **What happened**: User asked for "side-by-side axon_velocity_gtrs vs radivojevic_recon comparison via plot_recons, high-branch unit, plenty of branches". Loop:
    - Hit data-layout block (no `merged_template.npy` on disk because kssynth heavy hadn't run) → substituted kilosort cluster 67 (sparse, ~12 channels, NOT high-branch) instead of stopping
    - Hit shape-mismatch friction integrating with plot_recons' expected on-disk inputs → built `render_reconstruction_png()` in the sibling repo instead of writing the adapter
    - Filed a SOFT-gate diagnostic that answered ZERO of the user's actual questions (no comparison, wrong unit, wrong rendering path)
  - **Why it's the wrong response**: Improvising-around-friction produces an artifact that LOOKS like progress but doesn't satisfy the deliverable. Loop spent ~80 min producing the wrong thing when a 1-line "blocked — should I run kssynth heavy first, or use kilosort substitute?" would have saved that time AND produced the right thing on the first try.
  - **Why "use existing plot_recons" specifically matters**: rendering both algorithms via the SAME code path is what makes the comparison fair. If we use plot_recons for axon_velocity_gtrs and a new renderer for radivojevic, the visual differences could be from the rendering, not the algorithm. Identical rendering isolates algorithmic differences.

  **(B4) The concrete plan for the next radivojevic comparison diagnostic** (this is what the loop must confirm via gate B2 before executing):
  1. Run kssynth slice 3b heavy on M08073/well000 (via `--input-root` plumbing already shipped) → produces `merged_template.npy` + `merged_channel_locations.npy` per post-merge unit at `dev_outputs/kssynth_slice3b/.../per_unit/unit_<id>/`. ETA ~5-15 min for analyzers cache + 30s kssynth.
  2. Identify the actual high-branch post-merge unit. Earlier scan identified `unit_0598` as the 9-branch reference; CONFIRM this still maps to a post-merge unit in kssynth's output (kssynth's unit IDs may differ — needs cross-check via cluster_KSLabel or equivalent). If the mapping is unclear, STOP AND ASK rather than picking a substitute unit.
  3. Run `radivojevic2023_recon_algo.reconstruct()` on the chosen unit's merged_template. Tune Stage 2 knobs if runtime requires (already empirically: upsample_factor=2 + pixel_um=10.0 ≈ 0.8s).
  4. Build an **ADAPTER** (not a new renderer) that converts `ReconstructionResult` → the on-disk artifact shape `plot_recons` (axon_recon's existing phase) expects. Audit what plot_recons reads first — it's in `pipeline/stages/reconstruct/phases/` somewhere. If the shape mismatch is fundamental (e.g. plot_recons expects gtr.pkl-shaped axon_velocity output), STOP AND ASK — do not invent a new renderer.
  5. Invoke `plot_recons` against BOTH outputs (axon_velocity_gtrs's existing reference output + radivojevic_recon's adapted output) on the SAME unit. Produces two PNGs from the SAME rendering code.
  6. Compose `comparison.png` side-by-side.
  7. File HARD-gate entry in `diagnostics_to_review.md` with all 3 artifacts (radivojevic PNG / axon_velocity PNG / composite). File real-data smoke entry in `smoke_log.md`.
  8. PAUSE for user review.

  **Promote when stable**: after 3 pre-gated diagnostics ship with no user complaints about deviation-from-explicit-instruction, promote (B1) — the "no improvising around explicit instructions" rule — into a guardrail and delete (B2)/(B3)/(B4) since the cohort of attempts will have proven the behavior is internalized.

- **[2026-05-21] Proactive plan audit — surface inconsistencies + inefficiencies as multiple-choice questions**: Loop must regularly audit `dev/notes/plans/active/*.md` (and `trackers/tech_debt.md`, `trackers/issues.md`) for logical problems and stale assumptions. When the user asks for status ("any blocks?", "any questions?", "where are we?"), these findings become first-class items surfaced alongside the active blockers.

  **(A1) What to look for**:
  - **Logical inconsistencies** — slice A says X, slice B says ¬X (e.g. one slice claims a phase will be deleted, another assumes it still exists)
  - **Internal contradictions** — plan says "do X then Y", but Y consumes something X doesn't produce (broken dependency chain)
  - **Stale assumptions** — plan written when guardrail Z was different, but Z has since changed and the plan didn't update (e.g. plan references `--force-replot` after it was renamed to `--replot`)
  - **Dead slices** — slice superseded by other work but not marked SHIPPED / SUPERSEDED (frequent after big refactors)
  - **Redundant slices** — two plans both schedule the same work; one should be the canonical owner
  - **Scope drift** — plan kept accreting work beyond its original goal; could be split, or some scope should move to a different plan
  - **Inefficient orderings** — plan does X then Y, but doing Y first would reveal whether X is even needed (cheap-info-first principle)
  - **Resource mis-matches** — slice plans for `n_workers=4` when the standard NERSC allocation is `n_workers=16`; or plan's expensive step could be moved out of the critical path

  **(A2) When to audit**:
  - **Continuously, opportunistically**: before starting a new slice FROM a plan, spend 1-2 minutes scanning the plan's adjacent slices + their dependencies for the patterns above. Cheap; catches most inconsistencies before they bite.
  - **At audit-pass mode** (queue empty per cadence ladder): do a deeper pass — read 1-2 active plans end-to-end + check against current_state.md + guardrails for stale assumptions.
  - **After 5+ slices of a plan ship**: drift check — does the plan's stated goal still match what the slices are actually building?
  - Do NOT audit every iteration; that's overhead. Audit-pass + opportunistic is the right cadence.

  **(A3) How to surface findings**:
  - Append to `dev/notes/brain/open_questions.md` under a new section `## 🔎 Plan-audit findings (loop-surfaced)` — one entry per finding, with the affected plan + slice + a 1-sentence statement of the problem.
  - Each finding MUST be phrased as a multiple-choice question per the B1/B2 format: 2-4 options (e.g. "(1) fix the inconsistency by editing slice X; (2) accept the inconsistency and document; (3) split the plan; (4) defer"). Loop picks a "recommended" option; user picks/adjusts when reviewing.
  - Findings are NOT urgent — they accumulate. When the user asks for status, the loop reads this section and surfaces a digest: "N plan-audit findings open; here are the top M by potential impact."
  - When a finding is resolved (user picks an option, loop executes), strike through the entry and link forward to the resolving commit.

  **(A4) Cap on volume**: at most 5 open audit findings at any time. If the loop finds a 6th, it must close (resolve OR defer-with-rationale) one of the existing 5 before filing the new one. Prevents accumulation of unactioned findings.

  **(A5) Tone**: audit findings should be NEUTRAL observations, not blame. The loop is auditing its own past work as often as anyone else's — most stale-assumption findings are about prior loop iterations. Keep findings factual ("slice 6 says X; slice 11 says ¬X; this is a contradiction"), not editorial ("slice 6 was poorly written").

  **Why**: per user 2026-05-21: "Include instructions to regularly search for logical inconsistencies or inefficiencies in plans. Turn those into questions for user when I ask." The user wants the loop to act as a continuous code/plan reviewer in addition to a slice executor — surfacing problems proactively rather than waiting for the user to discover them during review.

  **Promote when stable**: after 3 audit findings get filed AND resolved without user complaints about the format, promote (A1)/(A2)/(A3)/(A4)/(A5) into a guardrail (`brain/guardrails/plan_audit.md` would be a new doc) and delete this injection. The behavior stays — codified.

## 📝 User actions queued

Manual items the loop can't or shouldn't do — surfaced here so the user has one place to find them. Loop appends as needed; user prunes when done.

- **[2026-05-19] Delete the smoke-test repo `adamwea/__gh_auth_smoke_test`** on GitHub via web UI (`gh` token lacks `delete_repo` scope). Pure clutter; non-blocking.

- **[2026-05-21] Run queued salloc smokes** — `dev/notes/trackers/salloc_smokes_queued.md` lists kssynth slice 3b HEAVY waiting on an interactive allocation. NOTE: the radivojevic comparison this gates is currently inside the planning-pause; don't run this until the pause lifts.

## Shipped this week (2026-05-12 → 2026-05-18)

### axon_recon repo
- **Phase roster cleanup slice 1 — legacy reconstruct phases deleted** (commits `34bb353`, `2c9e1d3`, `17ca304` + commit_log companions):
  - `reports` mega-phase deleted (phase impl, config dataclass, CLI dispatch, YAML, tests).
  - `plot_templates` v1 deleted (phase impl, config field, CLI aliases, YAML, tests). `report_templates.consume` normalizer now canonicalizes on `plot_templates_v2` and rejects v1 strings.
  - `per_unit_processing` deleted (phase impl, the four nested sub-config dataclasses, monolithic-pipeline helpers `_report_scope_config` + `_disable_reports_config`, YAML, tests). `_run_reconstruct_templates_pipeline_monolithic` KEPT (still called by `run_reconstruct_templates_pipeline` when `phase_sequence is None`; ~70 template tests rely on it).
  - 0 new test failures across the slice; 15 pre-existing failures noted (see open_questions).
- **Reconstruct `--force-restart` wipes the whole stage output** (no more templates-cache stash-and-restore). Commit `5e2b883`.
- **Concat analyzer disabled at every recon production wrapper.** `legacy_include_concat` default flipped `True → False`; `_iter/_load_templates_phase_analyzers` hard-set `include_concat=False`; materialize call sites pass `include_concat=False`. Loader low-level path still honors `include_concat=True` for tests only. Tracker entry "Remove concat analyzer plumbing from recon stage" tracks full removal.
- **CLI scope flags landed**: `--profile`/`--task-profile`, `--target-wells` accepts ints, `--targets <ds>:<well>` per-pair filter. Process-wide overrides in `pipeline/config.py`. Tests pass.
- **summary.json per-pid tmp rename fix** (`logging/summary.py`). No more `FileNotFoundError` floods from concurrent srun writers.
- **`resolve_inner_worker_count` honors yaml/phase hints when slot is None.** No more silent `n_jobs=1` collapse in MPI workers. Test rewritten to lock in corrected fallback.

### SLAy repo
- **`accept_merge` assertion relaxed** (commit `426ba71`): `>=` instead of `==` for same-time intra-cluster collisions. Big wells (>700 KS units) no longer crash at the auto-merge step.
- **`accept_all_merges` aux-tsv sync**: cluster_KSLabel.tsv / cluster_Amplitude.tsv / cluster_ContamPct.tsv stay coherent with cluster_group.tsv after merges. Resolves the KS-extractor inner-join bug that was dropping new merged unit IDs in downstream consumers.

### Shifter image
- Latest digest in shifterimg: `62c8a06e6b` (built 2026-05-18 03:24:35, with concat-disabled wrappers).
- A later build `32638ea26b` (2026-05-18 04:57:38) includes the SLAy aux-tsv sync. **Both are present in shifterimg**; the 04:57 build is the most current.
- Docker.io path: `adammwea/axon-recon:pipeline-v2`.

### Validation runs
- **Smoke test for 260326/M08073/000208/well000** (DIV 36, 80k DMEM): recon stage with `--force-restart` produced 176 merged templates + 176 per-unit recon outputs, matching the post-SLAy good-label count from status replay. The concat-analyzer rip-out + aux-tsv sync together fixed the systematic post-merge undercount.
- **Job 53089489** (4-node interactive GPU, 16 wells) completed with ds4 spikesort_full results + 4 merge_SLAy retries. ds6/well002 host-OOM'd; others completed.

## In-flight

- **Phase roster cleanup plan** (`plans/active/phase_roster_cleanup_plan.md`): slices 1-9 + 11-13 + 14a-14b shipped (slice 10 plot_raster_threshold quality fix deferred per plan §3). Slice 14c still queued (blocker documented in open_questions: the monolithic per-target stage runner in preprocess/spikesort/reconstruct/analysis needs a refactoring decision before integration).
- **Parallelism post-migration cleanup plan** (`plans/active/parallelism_post_migration_cleanup_plan.md`): slices 3 + 4 + 5 + 6 + 7 + 8 + 9 shipped. Also `+` a production-code fix to `core/unit_plots.py` (invert_y_axis propagation) and a softening to `container_cli` on `--dry-run`. Pre-existing failures went from 21 → 0 in this module. **Remaining queued**: slices 1 (retire `resolve_stage_parallelism` + `StageParallelism.well_workers` — multi-file refactor across runner.py + execution/context.py + logging + 10+ test fixtures; smoke-test required), 2 (route remaining `inputs.n_jobs` through `resolve_inner_worker_count` — touches reconstruct/phases + templates/runner.py + preprocess; smoke-test required), 10 (lock guardrails doc to post-cleanup vocabulary; gated on slice 1 vocabulary changes landing).
- **kssynth + unitlink + unitmatch_phase plans** drafted in `plans/active/`.
  - `~/dev/pkgs/kssynth/` exists locally — **v1 feature-complete**. Slices 1-7 of the kssynth plan all landed (scaffold; cluster_tsv_sync; channel_grid; rasterize; partial_templates; merge_templates; io/ks_folder_writer + api.synthesize orchestrator + CLI). 58 tests green. Commits `7e236f3` → `d97036b` in the kssynth repo. Slice 8 (SLAy soft-imports kssynth) is the opt-in upstream-SLAy change; slice 9 is axon_recon recon-stage integration in THIS repo.
  - `~/dev/pkgs/unitlink/` exists locally — **v1 feature-complete**. Slices 1-7 of the unitlink plan all landed (scaffold; sorter_output_reader; union_grid; two_halves; output_writer; classical UMPy backend; api.match orchestrator + CLI). 53 tests green. Commits `81e4d3c` → `d92a6dc` in the unitlink repo. **Env dep chain confirmed hardened in `axon_recon` conda env (2026-05-19 loop run)**: `mat73`, `UnitMatchPy`, `kssynth`, `unitlink` all import cleanly; `tests/test_classical_backend.py` runs all 7 tests including `test_full_pipeline_with_real_umpy_minimal_input` + `test_params_override_overrides_match_threshold` with NO skips (no soft-skip fallback triggered). The originally-mocked behavior in slice 5's test design auto-promoted to real-UMPy on env presence. Container side will follow on the next shifter rebuild (`[full-cuda]` extras include `mat73`; Dockerfile separately installs `UnitMatchPy @ git+...`).
  - Both packages still need user-initiated `git push` to GitHub. axon_recon analysis-stage `unitmatch` phase: **slices 1-4 shipped this iteration** (commits `86672e2` slice 1 scaffold; `a18cb4b` slice 2 group discovery + path resolution in `core/unitmatch_groups.py`; `575e08b` slice 3 orchestrator invokes `unitlink.match` once per (chip, well) group with idempotent skip on subsequent group targets, output landing at `<output_root>/unitmatch/<chip>/<well>/`; `a7f8c51` slice 4 adds `--targets chip-well:<chip>:<well>` group form that expands against the data config). Phase is still `enabled: false` in both runtime YAMLs. Slice 5 (enable + login-node smoke) is the last v1 slice, gated on kssynth recon-stage integration (`ks_synthesizer_package_plan.md` slice 9) producing the inputs the orchestrator's `resolve_session_inputs` looks for; can advance once UMPy is wired into `unitlink` (per USER INJECTION #1, slice 5 of unitlink unblocks first).

## Locked decisions from 2026-05-18 pre-loop Q&A

Run-semantics:
- **Three invocation modes** for any stage: no-flag (auto-restart-from-first-broken), `--force-restart` (rmtree everything), `--replot` (run plot/report phases only, orthogonal to auto-restart).
- `--force-replot` is DELETED (`--replot` replaces it).
- **Checkpoint status enum**: `{missing, in_progress, ok, error, stale, skipped}`. `stale`/`error`/`in_progress`/`missing` all trigger force-restart of that phase + downstream in auto-restart mode. `skipped` stays skipped only if still configured that way; otherwise re-evaluated like `missing`.
- **`in_progress` marker**: phase writes a stub summary_json (`status: in_progress`, `started_at`, `pid`) BEFORE its main work begins, overwrites with `ok`/`error` on completion. Stranded `in_progress` = crashed process; auto-restart catches it.
- `dry_run` is its own status (`dry_run_ok`) and is treated like `missing` by auto-restart (dry-run doesn't actually run the phase).

Output locations:
- Iteration outputs: `/pscratch/sd/a/adammwea/dev_outputs/<plan_slice_name>/...` — one subdir per plan slice, granular.
- Reference data: `/pscratch/sd/a/adammwea/analyzed_data/...` — read-only. Never mutated during iteration.
- New `--output-root` CLI flag overrides `data_config.output_root` for iteration runs.

Working-data scope (re-confirmed):
- 80k DMEM well000 of M08073 chips, all DIVs, for v1 iteration.
- well001 OK as an additional sample if needed for parallelism / diversity smoke tests.
- Network scans: path pattern `/pscratch/sd/a/adammwea/raw_data/.../<date>/<chip>/Network/`. Two types per DIV (clustered groups of 4/9 channels, and fully sparse). v1 picks ONE type when network scans enter scope; the other is a future lever. Reserved for unitmatch v2 — out of v1 iteration scope.

Test policy:
- Delete tests that no longer make sense after a slice.
- Morph tests that should remain into the current mental model — done inline with the slice that requires it, not as a dedicated cleanup pass.

Process control:
- Model: default Opus 4.7; Sonnet 4.6 only for very concrete mechanical work.
- Mid-slice ambiguity: low-stakes → best-guess + note in `brain/notes.md`; high-stakes → ask user. If user is unresponsive: pause that slice, document blocker in `open_questions.md`, switch to another unblocked slice. Keep the loop moving.
- Login-node smokes: pass `--task-backend local_affinity` every time; cap 64 procs; use `--limit-*` flags to keep scope tiny. Bigger smokes → ask user to run on a real allocation.
- New repos (`kssynth`, `unitlink`) get local `git init` at slice 1; remote pushed by user when they create the GitHub repo.

## Queued / not started

Listed in tier-order from `dev/notes/plans/active/phase_roster_cleanup_plan.md` §"Execution order" (and the related response):

1. `phase_roster_cleanup_plan.md` (Tier 1)
2. `trackers/tech_debt.md` §"Remove `debug_mode` YAML blocks" + §"Remove concat analyzer plumbing from recon stage" (Tier 2, parallel)
3. `ks_synthesizer_package_plan.md` (Tier 3) — create the kssynth sibling repo + ship slices 1-7
4. `unitmatch_runner_package_plan.md` (Tier 3) — create the unitlink sibling repo + ship slices 1-7 (parallel with kssynth once APIs settle)
5. kssynth slice 9 (axon_recon recon-stage integration) + `unitmatch_phase_plan.md` slices 1-5
6. `trackers/tech_debt.md` §"Collapse `--force-restart` semantics" (Tier 4)
7. `trackers/tech_debt.md` §"Minimize / eliminate `resources.profiles`" (Tier 4)

Tier 5 (chip away anytime): parallelism_post_migration_cleanup_plan, spikesort/runner.py 10K-line decomposition, container preflight softening, container/MPI alignment notes.

- **`plans/completed/env_install_unification_plan.md`** (Tier 5, chip-away) — 8 slices to unify the install story onto pyproject.toml extras + `tools/setup_env.sh --editable-siblings`. **TIMING CONSTRAINT**: should land BEFORE the next shifter rebuild — otherwise the rebuild reapplies the old per-sibling-ARG mechanism and slice 6 of this plan throws it away two weeks later. If a near-term shifter rebuild is needed (e.g. to close the UMPy + mat73 gap), prefer holding the rebuild until plan slice 6 ships. Conversely, if Tier 1 work demands a rebuild urgently (kssynth slice 9 integration is the natural pressure point), prioritize at least plan slices 1-3 + 6 first.

- **`plans/active/analysis_propagation_video_plan.md`** (Tier 5, chip-away) — 9 slices to re-implement axon_velocity's branch-propagation video / GIF generation as a new analysis-stage phase (`propagation_video` or similar; name decided in slice 2). Expensive per-unit work, opt-in via YAML + targeted-on-demand via per-unit `--targets` triplet form. Slice 1 does git archeology to find the old recon-stage impl (may not exist; fine to wrap current `axon_velocity` API cleanly). Slice 8 produces the first real-data video and adds a HARD-gate diagnostic to `brain/diagnostics_to_review.md` for user approval — a natural end-of-day deliverable for some future loop run. NOT blocking anything else.

- **`plans/active/radivojevic_recon_algo_plan.md`** (Tier 4, gated, kickoff-after-integration) — 9 slices to reverse-engineer + clean-room re-implement Radivojevic 2023's reconstruction algorithm as a sibling package at `~/dev/pkgs/radivojevic2023_recon_algo/` (already scaffolded with user-provided literature/) plus a new `radivojevic_recon` phase in the recon stage. Alternative / comparison to `axon_velocity_gtrs`. **Kick-off trigger**: kssynth slice 9 AND unitmatch_phase slice 5 BOTH shipped (= first real end-to-end smoke through the new sibling packages). Slice 1 is research-only — read pre-populated PDFs (elife-86512 likely the target paper) + WebSearch/WebFetch for public code + produce algorithm-summary doc + user-gate. 5 USER GATEs total (slices 1, 3, 4, 6, 9) — high back-and-forth plan.

- **`plans/active/chip_layout_phase_split_plan.md`** (Tier 4, gated, kickoff-after-integration) — 8 slices to move `plot_full_chip_layout` from recon stage to analysis stage AND split into two phases: **Phase A** (white-bg timeline grid showing repeated sessions for a (chip, well) group) and **Phase B** (black-bg detailed per-session with electrode overlay + signal-strength-weighted RGB blending on electrode overlap). Cross-session unit colors via unitlink match table (preferred) or extremum-electrode fallback. **Kick-off trigger**: same as Radivojevic plan — kssynth slice 9 AND unitmatch_phase slice 5 both shipped (need unitlink match tables for color consistency). 2 USER GATEs (slices 5 + 7 visual diagnostics). Cross-session color helper (slice 3) is reused by future propagation_video / Radivojevic-comparison plotting.

- **`plans/completed/dashboard_ui_refinement_plan.md`** (Tier 4, no gating) — 9 slices to refactor and refine the Dash UI (`src/axon_recon/dashboard/`, ~2128-line app.py + 6 sibling modules). Addresses user-reported pain points: empty metrics rendering as broken UI; some plates missing from UI (hardcoded lists instead of filesystem-discovered); feature parity gap between plot types (box has features histogram/scatter don't); no box↔bar mode toggle; no tertiary grouping option; styling drift between plot types. Slice 1 = audit only (no code) — produces feature matrix + hardcoded-list inventory + empty-state gap list. Slice 4 introduces a shared `PlotConfig` abstraction that all plot types consume. Slices 6+7 add box↔bar toggle + tertiary grouping. Slice 8 unifies styling (possible palette-module synergy with chip_layout_phase_split_plan slice 3). **No dependency on Era 3 integration** — dashboard reads existing `analyzed_data/` outputs; can start any time the loop has bandwidth.

## Environment state

- Working on Perlmutter (NERSC). Logged in as `adammwea`.
- Shifter container: `adammwea/axon-recon:pipeline-v2`.
- Code lives at `/global/homes/a/adammwea/dev/pkgs/axon_recon/` (also accessible via `/global/u2/...` symlink — they're the same dir).
- Sibling packages at `/global/homes/a/adammwea/dev/pkgs/SLAy/`, `/global/homes/a/adammwea/dev/pkgs/UnitMatch/`.
- Pscratch for I/O: `/pscratch/sd/a/adammwea/{raw_data,analyzed_data,run_logs,smoke_logs}/`.

## Working-data scope (current iteration cycle)

**Locked to ONE cohort**: 80k DMEM well000 (M08073 chip family, well000 of each DIV's AxonTracking recording). Use this and only this for iteration development; ask the user before expanding scope.

- **Reference (read-only)**: `/pscratch/sd/a/adammwea/analyzed_data/Media_Density_T5_02182026_AR_axon_analysis_AW/Media_Density_T5_02182026_AR/<date>/M08073/AxonTracking/<rec>/well000/`. Existing per-stage outputs are the ground-truth reference for desired behavior. **Do not mutate these.** Compare against them.
- **Iteration output dir**: `/pscratch/sd/a/adammwea/dev_outputs/<feature>/...` — one subdir per feature or plan-slice. New run outputs go here; pscratch retention is fine since these are disposable.
- **Raw network scans**: available under the raw_data tree. Single-segment, non-concatenable. Need preprocess stage before any downstream work. Reserved for `unitmatch_phase_plan.md` slice 6 — DO NOT touch them during v1 iteration.
- **Smoke-test discipline**: targeted login-node only (`--task-backend local_affinity`, cap 64 procs, use `--limit-*` flags). Bigger than that → ask user to run on a real allocation.

## Known good baseline

For regression-check purposes, the following counts on `260326/M08073/000208/well000` (DIV 36, 80k DMEM) post-2026-05-18 fixes:
- post-SLAy unit IDs in `spike_clusters.npy`: 377
- post-SLAy cluster_KSLabel.tsv (after aux-tsv sync): 287 good + 312 mua = 599 rows
- recon merged templates: 176
- recon per-unit outputs: 176 (matches good count of post-merge units in spike_clusters; mua not reconstructed by default per `unit_label_filter`)
