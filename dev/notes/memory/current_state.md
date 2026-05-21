# Current state

Snapshot of what's shipped, in-flight, and queued. Updated as state changes; old facts get deleted, not commented out.

## ⚡ USER INJECTIONS

User-authored directives that override plan / tier order until satisfied. Read FIRST each iteration. Apply at the earliest applicable slice; when reliably internalized, promote the rule to a guardrail / CLAUDE.md slice protocol / plan and delete the entry here.

### Active

- **[2026-05-18] YAML hygiene as you go**: Every slice that touches phase code, CLI flags, config schema, or phase wiring must update `dev/debug_NERSC/debug.runtime.yml` AND `dev/debug_NERSC/debug.data.yml` so they stay an accurate mechanical source of truth for what the pipeline runs. The future default `runtime.yml` will derive from `debug.runtime.yml`, so stale entries propagate forward. Remove dead phase blocks, dead `resource_class` entries, dead CLI flag defaults; add new keys for new phases/flags; clear `# TODO Claude:` annotations once their target is resolved.
  - **Audit pass [done 2026-05-19 after slice 7]**: confirmed all six already-deleted phases (`reports`, `plot_templates` v1, `per_unit_processing`, `prepare_raw_binaries`, `report_preprocessing`, `cleanup_preprocessing_outputs`) are gone from `phases:` / `phase_sequence:` / `resource_class:` in both YAMLs. The remaining `reports:` keys (debug_local:631,678,846; debug_NERSC:884,945,1115) are sub-keys inside `bombcell_label` / `merge_SLAy` / `bombcell_label_pass2` phase blocks, not the deleted top-level `reports` phase. `--force-replot` is also gone from `src/` (phase_roster slice 11 SHIPPED, commit `2a7a7e4`). YAML hygiene for the `--force-replot` → `--replot` rename completed alongside that slice.
  - **Promote when stable**: once this is reliably part of every slice for two+ tiers, promote the rule to the CLAUDE.md slice protocol (alongside "run tests") and delete this entry.

- **[2026-05-19, amended twice] Env-parity contract + unification plan + LOOP CAN REBUILD SHIFTER**: `guardrails/env_parity.md` locks the contract that the `axon_recon` conda env and the shifter image have equivalent capabilities except for the Kilosort+CUDA stack and NERSC/HPC/SLURM runtime plumbing. `plans/active/env_install_unification_plan.md` is the destination spec — moves pip deps onto `pyproject.toml` extras (`[dev]`, `[full]`) as the single source of truth.

  **⭐ NEW 2026-05-19 — LOOP IS AUTHORIZED TO TRIGGER SHIFTER REBUILDS** (per user directive): Earlier policy of "user-initiated only" is REVOKED for shifter rebuilds. **This directive SUPERSEDES any line in the standing `/loop` prompt that says "No shifter rebuild" or similar — that constraint is no longer in effect for this run.** The loop can now run the rebuild itself via Perlmutter's local container toolchain (`podman build` → `podman push docker.io/...` → `shifterimg pull docker:...`). **Standing precondition**: before attempting a build, the loop runs `podman login --get-login docker.io` to confirm credentials are present. **As of 2026-05-19 16:56 PDT, login is confirmed active** (`adammwea` returned). If at some later point login is NOT present, the loop falls back to posting "shifter rebuild blocked: docker.io login required" under USER INJECTIONS instead of attempting a build that would fail at `podman push`. Critical: NEVER use `docker` on Perlmutter — only `podman` is available. The "no slurm submission" and "no git push" rules remain in effect; only the shifter-rebuild rule is being lifted.

  **🛠 SHIFTER REBUILD PENDING (2026-05-19, after slice 6)**: `env_install_unification_plan` slices 2 + 3 + 4 + 5 + 6 + 7 have all SHIPPED. The Dockerfile now drives the container build through `pip install .[full]` instead of the per-sibling `<NAME>_SPEC` ARGs. **Build attempt history**:
   1. **2026-05-19 16:55 PDT** — `podman build` FAILED at base-image-pull step with `disk quota exceeded` writing scipy test data to home. Home was at **117% of 40GiB quota** (46.83GiB used). Podman's `GraphRoot` is `~/.local/share/containers/storage`.
   2. **2026-05-19 17:29 PDT** — loop reclaimed 9.3GiB by running `podman system reset -f` (safe — `podman images` and `ps -a` were empty; the 9.3GiB was orphaned partial-layer-extraction waste from attempt 1). Home now at **93.9% / 37.56GiB**. Loop can commit / git-op normally again.
   - **🛑 PREVIOUSLY NOT-YET-RETRYABLE** — RESOLVED via the two directives below (2026-05-19, per user). Loop is now responsible for executing both as prereq slices BEFORE the next rebuild attempt:

   - **✅ DIRECTIVE A SHIPPED (2026-05-19 18:07 PDT)**: podman GraphRoot moved to `/pscratch/sd/a/adammwea/podman_storage`. `~/.config/containers/storage.conf` written with `[storage] driver = "overlay"` + the pscratch graphroot. `podman info` confirms graphroot resolves to pscratch (graphRootAllocated ≈ 44 PiB shows it's the pscratch lustre mount). Smoke-tested with `podman pull alpine` → ok + `podman rmi alpine` → cleaned. Home no longer gates podman builds. NOTE: standing "No pscratch overlay" rule remains in effect for ALL OTHER purposes — this override is scoped specifically to podman's image-build cache. Code stays in /global/homes; iteration outputs stay in /pscratch/.../dev_outputs/.

   - **✅ DIRECTIVE B SHIPPED (2026-05-19 18:50 PDT)**: split executed via the `--no-deps UnitMatchPy` approach (refined from the original "constraint file" alternative). Commits: `987053a` (pyproject `[full]` + `[full-cuda]` split), `338868e` (Dockerfile uses `[full-cuda]` + separate `pip install --no-deps "UnitMatchPy @ git+..."`), `6743847` (audit doc at `dev/notes/refs/kilosort4_base_audit.md`). Rebuild result:
     - `podman build` succeeded at 2026-05-19 18:50 PDT after ~35 minutes.
     - Final image size: **12.8 GiB** (down from the projected 17-20 GiB; matches the predicted 12-14 GiB target).
     - Tag: `docker.io/adammwea/axon-recon:pipeline-v2` (image ID `5f464e4e037d`).

   - **✅ SHIFTER ROUND COMPLETE — IMAGE READY WITH ONE KNOWN GAP (2026-05-19 21:56 PDT)**:
     1. ✅ User re-logged in to docker.io as `adammwea`.
     2. ✅ `podman push docker.io/adammwea/axon-recon:pipeline-v2` completed (12.8 GB; mostly delta against the previous push, so only ~13 blobs needed copying).
     3. ✅ `shifterimg pull docker:adammwea/axon-recon:pipeline-v2` completed; perlmutter shifter registry now `READY` at hash `9cdca44d9b` (2026-05-19T21:56:42) — replaces `32638ea26b` (2026-05-18T04:57:38).
     4. ✅ In-container smoke verified: `shifter --image=adammwea/axon-recon:pipeline-v2 python` — UnitMatchPy (+ bayes_functions, overlord, utils), torch 2.7.1+cu118 with CUDA True, spikeinterface 0.104.3, axon_velocity, numpy 1.26.4, scipy 1.16.0, sklearn 1.7.0, joblib 1.5.1, mat73 ALL import cleanly. axon_recon itself imports cleanly (note: the baked-in version is the snapshot AT BUILD TIME — `stage_aggregate_exit_code` shipped in `a8a87c4` AFTER the build, so the container's frozen copy doesn't have it yet; will be picked up on the NEXT shifter rebuild).
     5. ⚠️ **KNOWN GAP — SLAy missing from new image**: `import slay` raises `ModuleNotFoundError`. The old image at `32638ea26b` had SLAy installed via the now-removed `SLAY_SPEC` ARG. `env_install_unification_plan` slice 6 (Dockerfile collapse to `pip install .[full-cuda]`) dropped that ARG without adding SLAy to the `[full]`/`[full-cuda]` extras. `axon_recon` consumes SLAy at runtime via `importlib.import_module("slay.run")` in `stages/spikesort/runner.py:2008,2018`, so the `merge_SLAy` phase will fail with ImportError on this image. **RESOLVED via DIRECTIVE D below** — SLAy already has public remote `git@github.com:adamwea/SLAy.git`.
     6. kssynth + unitlink also remain ImportError per the original USER INJECTION #4 (GH-remotes hold). **RESOLVED via DIRECTIVE D below** — user lifted the hold; loop creates the remotes itself via `gh repo create` and pushes.

   - **✅ DIRECTIVE D SHIPPED (2026-05-20 00:25 PDT)**: all three in-image gaps closed. New shifter image `adammwea/axon-recon:pipeline-v2` (shifter hash `cfc82cc501`, docker hash `cbf32d30f09f`, 12.8 GB, READY at 2026-05-20T00:23:03) imports `slay` + `kssynth` + `unitlink` cleanly (with axon_recon's numpy-cupy fallback shim that's already wired into the runtime path). Full smoke verified UMPy + torch 2.7.1+cu118 CUDA True + spikeinterface 0.104.3 + numpy 1.26.4 all import.

     **Resolution journey** (preserved here for traceability; original directive body below for archeology):
     1. Build #1 — failed at private-repo clone (kssynth/unitlink created `--private`; container build context has no GH auth). User selected P1 (flip to public) — `gh repo edit … --visibility public` on both.
     2. Build #2 — failed at pip dep resolution: SLAy's `numpy>=2.2.6` conflicts with axon_recon's `numpy<2.0`. Restored the pre-slice-6 `SLAY_INSTALL_ARGS="--no-deps"` pattern: added `SLAY_GIT_URL` ARG to Dockerfile + second `pip install --no-deps "${SLAY_GIT_URL}"` step (mirrors UnitMatchPy); removed SLAy from pyproject `[full]`/`[full-cuda]` extras.
     3. Build #3 — SUCCESS. axon_recon, kssynth, unitlink wheels built cleanly; UMPy + SLAy installed via separate `--no-deps` steps. In-container smoke passes for all three with the numpy-cupy shim that mirrors `_install_numpy_cupy_fallback_module` in `stages/spikesort/runner.py:1649`.

     **Promoted rule** to `guardrails/env_parity.md` §"Sub-rules": "siblings with numpy upper-bound conflicts (currently SLAy and UnitMatchPy) install via separate `pip install --no-deps "${<NAME>_GIT_URL}"` steps in the Dockerfile, AFTER the main `pip install .[full-cuda]` step. The conda env installs them editably via `tools/install_dev_siblings.sh`." (Promotion deferred to a follow-up commit; that's mechanical.)

     **Era 3 integration UNBLOCKED**: kssynth slice 9 (axon_recon recon-stage integration) + unitmatch_phase slice 5 (enable + login-node smoke) now have all the runtime imports available inside the shifter image. The container's frozen axon_recon snapshot doesn't have the recent `stage_aggregate_exit_code` helper (a8a87c4 landed after the rebuild) — at runtime the wrapper bind-mounts host source so it picks up.

     **Original DIRECTIVE D body (2026-05-19, AUTHORIZED — loop owned this)**: close the three "in-image" gaps (SLAy, kssynth, unitlink) so the shifter image actually runs the full pipeline. **User authorizations granted with this directive**:
     - **(D1) Sibling-repo git pushes are AUTHORIZED**, scoped to: (a) pushing existing local commits to existing remotes, AND (b) creating new GitHub remotes via `gh repo create` for kssynth + unitlink. The standing "no git push" rule still applies to `axon_recon` itself — only sibling repos are being lifted.
     - **(D2) Push to WELL-NAMED BRANCHES**, not `main` (except for the inaugural publication of a brand-new repo where `main` IS the named branch). User will review and merge to main themselves.
     - **(D3) USER INJECTION #4 (GH-remotes hold for kssynth + unitlink) is REVOKED.** Replaced by this directive — loop creates the remotes when ready.

     **Environment audit (2026-05-19 22:37 PDT — pre-execution verification):**
     - `gh` CLI at `~/.local/bin/gh`, authenticated as `adamwea` with `repo` scope (can `gh repo create`).
     - SSH to `git@github.com` succeeds via `~/.ssh/id_ed25519_adamwea`.
     - `git@github.com:adamwea/SLAy.git` exists; local `main` has 2 unpushed commits (`f7c2173` aux-tsv sync, `426ba71` assertion relax) on top of upstream's `e77dab1`.
     - `~/dev/pkgs/kssynth` exists locally on `main`, NO upstream — needs `gh repo create`.
     - `~/dev/pkgs/unitlink` exists locally on `main`, NO upstream — needs `gh repo create`.

     **Loop execution sequence (slice-by-slice; each step is its own commit + commit_log entry):**
     1. **Push SLAy fixes to a named branch + open PR.** `cd ~/dev/pkgs/SLAy && git checkout -b claude/merge-fixes-2026-05 && git push -u origin claude/merge-fixes-2026-05`. Then `gh pr create --title "..." --body "..."` against `main`. **PR MERGE POLICY (per user 2026-05-19): USER-ONLY merge. Loop NEVER merges the PR — leaves it open for user review on GitHub web UI.** Record branch name + PR URL in the SLAy commit_log (separate from axon_recon's commit_log; SLAy keeps its own at `~/dev/pkgs/SLAy/dev/notes/commit_log.md` if one exists, otherwise commit body alone).
     2. **Create kssynth remote + push.** `cd ~/dev/pkgs/kssynth && gh repo create adamwea/kssynth --public --source=. --remote=origin --push`. **MUST be `--public`, NOT `--private`** — private repos break `pip install git+https://...` in the shifter build environment (no auth tokens baked into the container build context). SLAy / UnitMatch / axon_velocity are all public; kssynth + unitlink match. The `--push` flag publishes `main` (v1-complete content) to the new remote in one operation. Record the URL.
     3. **Create unitlink remote + push.** Same form: `cd ~/dev/pkgs/unitlink && gh repo create adamwea/unitlink --public --source=. --remote=origin --push`. **Public, not private** — same reason as kssynth.
     4. **Capture commit SHAs** for each sibling's `main` (`git rev-parse origin/main`) — these become the version pins in pyproject.toml.
     5. **Edit `pyproject.toml`**: add to `[full]` AND `[full-cuda]` extras (both — they must stay in sync):
        ```
        "SLAy @ git+https://github.com/adamwea/SLAy.git@<sha>",
        "kssynth @ git+https://github.com/adamwea/kssynth.git@<sha>",
        "unitlink @ git+https://github.com/adamwea/unitlink.git@<sha>",
        ```
        Use specific commit SHAs (not branch names) for reproducibility. SLAy's pin is the tip of `main` (NOT the unmerged feature branch — the runtime needs the published-and-stable code path; merge happens later when user reviews).
        - **Subtle**: SLAy's main does NOT yet contain the merge fixes (they're on the feature branch). For the immediate rebuild to also include those fixes, either (a) pin to the feature branch tip SHA (`f7c2173`) directly, OR (b) user merges the PR before rebuild, OR (c) accept that the shifter image doesn't have the merge fixes yet. Default choice: pin to feature-branch SHA `f7c2173` — the fixes are needed for the new image to be production-equivalent to the old one. Document this choice in the commit message so user can update later when the PR merges.
     6. **Sanity-check pyproject locally**: `pip install -e .[dev,full-cuda]` should succeed from a clean conda env (might require `--force-reinstall` of the siblings since their version pins changed). Run the existing test suite.
     7. **Rebuild shifter image**: `podman build` → `podman push` → `shifterimg pull` → in-container smoke verifying `import slay; import kssynth; import unitlink` all succeed.
     8. Update this USER INJECTIONS entry to `✅ DIRECTIVE D SHIPPED` once round 2 of the shifter rebuild completes and the import chain is verified in-container. Then promote the "siblings live in `[full]`/`[full-cuda]` as git URL pins" rule into the env_parity guardrail (it's already there as the target shape; just needs a "current shape" update to match).

     **✅ BLOCKER ON STEP 7 RESOLVED (2026-05-19 23:15 PDT) — USER CHOSE P1: REPOS FLIPPED TO PUBLIC**. User selected P1 ("make kssynth + unitlink PUBLIC") with stated reasoning that private was a mistake — research-tool sibling packages don't need privacy. Visibility flipped via `gh repo edit adamwea/kssynth --visibility public` + same for unitlink (no `--accept-visibility-change-consequences` flag in gh 2.49.0; bare `--visibility public` worked silently). Verified via REST API: both now `{"private": false, "visibility": "public"}`. DIRECTIVE D step 2 spec retroactively amended (commit incoming) to `--public` so any future re-execution doesn't repeat the bug. **Step 7 retry is unblocked** — loop reruns `podman build` → `podman push` → `shifterimg pull` → in-container smoke (now with `import slay; import kssynth; import unitlink` all expected to succeed) on its next iteration. Original BLOCKER entry preserved in commit `edb6f82` for archeology; P2 (BuildKit secrets) and P3 (SSH forwarding) discarded as unnecessary given P1's simplicity.

   - **✅ DIRECTIVE C PROMOTED TO GUARDRAIL (2026-05-19 22:33 PDT)**: docker.io authfile persistence is now codified in `guardrails/env_parity.md` §"Sub-rules" 1 under "Docker.io authfile MUST live at `$HOME/.config/containers/auth.json` on NERSC". The shifter round at 21:56 PDT confirmed no auth interruptions, satisfying the promotion criterion. Directive C's full body (debugging + verification log) is preserved in commit `eb3b290` for future archeology.

  **Slice-level discipline going forward**: any slice that adds/removes/upgrades a conda dep MUST update the appropriate artifact pair (pre-plan: env.yml + Dockerfile; post-plan-slice-2: pyproject.toml `[full]` extra; post-plan-slice-6: Dockerfile picks up automatically via `[full]`) AND post a "shifter rebuild needed: X" line under USER INJECTIONS. The loop now picks up that rebuild itself when credentials are in place. Read `guardrails/env_parity.md` for the full contract.

- **⭐ PRE-OVERNIGHT CLEARANCES (2026-05-19, set during pre-loop check)**: User answered 3 foreseeable gates ahead of the overnight loop run so they don't block.
  1. **Dashboard slice 7 — tertiary grouping UX: configurable per chart (YAML toggle).** Both `small-multiples` and `hierarchical-X-labels` render modes get implemented; user picks per chart via a per-phase YAML knob (e.g. `tertiary_render_mode: small_multiples | hierarchical_labels`) or a dashboard UI dropdown. ~1 extra commit vs picking one mode; most flexible long-term. Default mode TBD by the loop during slice 7 — pick `small_multiples` as default since it reads better for the typical few-value tertiary case; user can override per chart.
  2. **Radivojevic plan slice 1 — PRE-APPROVED to continue past USER GATE 1 into slice 2.** Loop ships slice 1 (literature read + code search + `dev/notes/refs/radivojevic2023_paper.md` + `dev/notes/refs/radivojevic2023_algorithm_summary.md` + input-vs-axon-velocity compat map), THEN immediately scaffolds the sibling package per slice 2 (`pyproject.toml`, package layout, README with citation, 2-3 sanity tests) WITHOUT pausing. **All slice-1 review questions get logged to `open_questions.md` under "Radivojevic slice 1 user-gate review" for AM user review.** Subsequent gates (slices 3, 4, 6, 9) still apply normally. Slice 3 (core algorithm implementation) DOES NOT START until the slice-1 questions are reviewed.
  3. **SLAy PR merge policy — USER-ONLY merge.** Codified in DIRECTIVE D step 1 above. Loop pushes the branch + opens the PR via `gh pr create` but never runs `gh pr merge`. User reviews and merges via GitHub web UI when convenient.

- **[2026-05-21] Phase-enable for tests (general rule)**: When a slice needs to smoke-test a phase that's currently YAML-disabled (`enabled: false`), the loop MUST enable the phase for the smoke — either temporarily via a YAML edit it reverts, or via a CLI override. Never test a phase while it's disabled and treat the no-op as validation. Per user 2026-05-21: "Feel free to enable phases as needed for tests. Don't test it disabled and think that validates it."
  - **✅ CLI FLAG SHIPPED 2026-05-21 (commit `5d28561`)**: `--force-enable PHASE[,PHASE...]` added to the recon stage CLI + shared `stages` subparser. Process-wide override pattern mirrors `--no-plot`/`--profile`/`--scratch-output`/`--output-root`. Helper `_apply_force_enable_phases` in `pipeline/runner.py`; setter/getter in `pipeline/config.py`. Applied inside `_run_reconstruct_substage_from_runtime` (execution path) AND `_build_reconstruct_allocation_preview` (`--alloc` preview path) AFTER YAML parsing but BEFORE phase-roster + target selection. Override covers both the recon stage_config tree and the templates substage tree. 15 new tests cover process-wide setter/getter invariants + helper edge cases + 1 real-YAML integration. Slice 3b smoke is now unblocked (`axon-recon stages reconstruct.kssynth --config ... --force-enable kssynth`).
  - **Use count toward promotion**: 1 (kssynth slice 3b — pending smoke execution). Promotion criterion = ≥2 slice uses; when slice 5 (full-stage smoke with `--force-enable kssynth,plot_recons,axon_velocity_gtrs`) or any other YAML-disabled phase smoke uses the flag, promote the rule and delete this injection entry. The CLI flag stays — it's general utility.

- **[2026-05-21] ✅ PROMOTED 2026-05-21 to `guardrails/loop_cadence.md`** (commit pending). Loop heartbeat reason format + shorter default cadence — the format ("Next iteration in {N}s — {sentence}") + ladder (90s active / 120s between-slices / 300s audit / 600-1200s blocked) is now locked into the guardrail doc. User-side follow-up: update the `/loop` prompt template to bake in the cadence ladder so future re-pastes inherit it (the loop can't edit the user's standing /loop prompt).

- **[2026-05-21] Real-data smoke log discipline**: New file `dev/notes/trackers/smoke_log.md` is the canonical log of smoke tests on REAL data + the bugs they reveal + how they get solved. The loop MUST append an entry whenever:
  - A login-node smoke on **real data** completes (pass OR fail) — even uneventful regression checks
  - A user-initiated `salloc`/`sbatch` run completes and the loop has access to the output dir
  - A HARD-gate visual diagnostic is reviewed and approved/rejected
  - **Do NOT log**: dry-run smokes, unit tests, synthetic-fixture smokes. Those don't count as data validation; they go in commit messages and the dry-run rollout plan.
  - **Format**: see the schema at the top of `smoke_log.md` — short headered fields (smoke command / cohort / commit / outcome / what ran / quantitative result / bugs revealed → fixed-in-commit / diagnostics link / baseline-established? / status).
  - **Why**: per user 2026-05-21 "let's start keeping a log of smoke tests using real data and the bugs they reveal and how they get solved." Passing unit tests is necessary but not sufficient — the user wants a separate, persistent record of what's actually been validated on data so future loop iterations don't treat "tests green" as equivalent to "feature works on M08073/well000."
  - **Immediate application**: kssynth slice 5's HARD-gate 176-template regression run (when it ships — currently gated on slice 3b being executable; 3b's dry-run short-path above is wiring-only, NOT a smoke_log entry) WILL be the first new real-data smoke entry. Any analyzers-cache-required smoke that user/loop runs to unblock slice 3b's REAL-data version (vs. dry-run) also counts.
  - **Backfilled at creation**: 2 historical entries seeded (2026-05-18 known-good baseline + 2026-05-18 Job 53089489 multi-well sweep) so the schema has examples and future regressions have a comparable.
  - **(S2 — added 2026-05-21) Worker-count validation discipline (per user)**: every smoke MUST scrutinize the logs for actual worker counts and verify they match what was requested. Concretely:
    - Before kicking off any smoke, the loop records the EXPECTED worker counts (cpus_per_task / n_jobs / well_workers / phase fanout) — derived from CLI args, the runtime YAML resource_class entries, and the slot.cpu_count budget.
    - DURING / AFTER the smoke, the loop scans the logs for `phase_parallelism event=...`, `Preprocess phase worker allocation`, `n_jobs_source=...`, `slot.cpu_count=...`, and equivalent lines. Records ACTUAL counts.
    - **Mismatch = smoke_log entry MUST flag it**: e.g. "REQUESTED cpus_per_task=10 but observed n_jobs=1 collapsing to serial in MPI worker (n_jobs_source=fallback)". A passing smoke that ran with the wrong worker count is NOT a green smoke — it's a silent failure that the smoke_log must capture.
    - **Env over-request = fail-fast**: if requested resources exceed what `os.sched_getaffinity(0)` / `SLURM_CPUS_PER_TASK` / `SLURM_GPUS_*` / cgroup limits actually report, the loop should NOT silently clamp and run anyway — it must surface as a smoke failure with the diagnostic "REQUESTED N workers; env reports M available; refusing to clamp silently." Tech-debt entry already exists; this aligns the smoke discipline with the long-term plan.
    - **Why**: per user 2026-05-21: "whenver running a smoke, pay careful attention to the logs and how many workers are being used. Is that the correct amount of workers we asked for? […] Check the environment, if we're asking for too many resoursces fail fast. I feel this plan hasnt been executed properly just based on a few smokes I've been trying to run. See recent logs. Profile was still clamping, and incorrectly."
  - **Promote when stable**: once the loop has appended ≥3 entries spontaneously across different slices (NOT counting the backfilled seeds), promote the rule into the CLAUDE.md slice protocol (alongside "visual diagnostics") and delete this injection.

- **[2026-05-21] Tightened visual-diagnostics trigger (after audit found 5 dashboard slices shipped without filing)**: The current CLAUDE.md rule ("when the claim depends on visual inspection") was too fuzzy and let the loop talk itself out of filing on dashboard slices 2 / 5 / 6 / 7 / 8 — all of which changed user-visible rendering. Tightened rule (effective immediately):
  - **(R1) Any slice that changes user-visible rendering MUST file a diagnostic entry.** This is now a hard requirement, not a judgment call. Examples that REQUIRE a filing:
    - Dashboard / UI changes (new plot type, new render mode, empty-state, style/palette/font change, new feature on an existing plot)
    - Any new phase whose output is a visual artifact (figure, plot, video, heatmap)
    - Any change that produces visually different output than before, even when tests pass
    - First real-data run of any new algorithm stage (Stage 1 → Stage 2 → Stage 3 transitions in Radivojevic each get their own filing)
  - **(R2) Stage transitions in multi-stage algorithms MUST file at EACH transition**, not just at the final end-to-end gate. For Radivojevic specifically: Stage 1 first real-output (peak lists overlaid on STA template = still visualizable as scatter on the channel grid), Stage 2 first electrical-image rendering, Stage 3 first skeleton + interconnect rendering, AND the final axon_velocity_gtrs-vs-radivojevic_recon comparison. Four separate filings minimum. Stage 1's diagnostic gets a soft gate (peak distribution sanity); the rest get HARD gates per existing GATE 3 spec.
  - **(R3) Diagnostic exemptions remain (don't file for)**: every routine pipeline plot during a normal run; pure-computation slices with numerical assertions covering validation; logs/text summaries (those go in commit messages, not diagnostics_to_review.md).
  - **(R5 — added 2026-05-21 after first SOFT-gate filing had only npy/tsv)**: ANY diagnostic with claimed visual content MUST include a rendered image format (PNG / SVG / PDF). npy / tsv / parquet alone is data, not a diagnostic. The visual file is the audit-trail artifact you can review at a glance. Data files can accompany it for re-rendering / downstream consumption. Per user feedback 2026-05-21: "I see the recon output but its npy and tsv files."
  - **(R4) Where artifacts live**: `/pscratch/sd/a/adammwea/dev_outputs/<plan_slice>/diagnostics/` per CLAUDE.md. If the slice's working dir is named after the slice (e.g. `radivojevic_first_run`), the diagnostics subdir lands inside it.
  - **Backfill scope (per user 2026-05-21)**: SKIPPED for dashboard slices 2/5/6/7/8 — user can review interactively any time. Going-forward only.
  - **Why**: per user 2026-05-21 audit: "Generally, was hoping to see more diagnostic plots... unless we truly haven't done any slices that warrant it." Audit found 5 dashboard slices shipped without diagnostics — loop has been under-filing. The diagnostics file is the audit trail that closes the gap between "tests green" and "the picture looks right."
  - **Promote when stable**: once 3+ slices (post-2026-05-21) have correctly filed diagnostics under R1/R2 without prompting, promote the rule into CLAUDE.md §"Visual diagnostics" (amend the existing section) and delete this injection.

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
  - Append to `dev/notes/memory/open_questions.md` under a new section `## 🔎 Plan-audit findings (loop-surfaced)` — one entry per finding, with the affected plan + slice + a 1-sentence statement of the problem.
  - Each finding MUST be phrased as a multiple-choice question per the B1/B2 format: 2-4 options (e.g. "(1) fix the inconsistency by editing slice X; (2) accept the inconsistency and document; (3) split the plan; (4) defer"). Loop picks a "recommended" option; user picks/adjusts when reviewing.
  - Findings are NOT urgent — they accumulate. When the user asks for status, the loop reads this section and surfaces a digest: "N plan-audit findings open; here are the top M by potential impact."
  - When a finding is resolved (user picks an option, loop executes), strike through the entry and link forward to the resolving commit.

  **(A4) Cap on volume**: at most 5 open audit findings at any time. If the loop finds a 6th, it must close (resolve OR defer-with-rationale) one of the existing 5 before filing the new one. Prevents accumulation of unactioned findings.

  **(A5) Tone**: audit findings should be NEUTRAL observations, not blame. The loop is auditing its own past work as often as anyone else's — most stale-assumption findings are about prior loop iterations. Keep findings factual ("slice 6 says X; slice 11 says ¬X; this is a contradiction"), not editorial ("slice 6 was poorly written").

  **Why**: per user 2026-05-21: "Include instructions to regularly search for logical inconsistencies or inefficiencies in plans. Turn those into questions for user when I ask." The user wants the loop to act as a continuous code/plan reviewer in addition to a slice executor — surfacing problems proactively rather than waiting for the user to discover them during review.

  **Promote when stable**: after 3 audit findings get filed AND resolved without user complaints about the format, promote (A1)/(A2)/(A3)/(A4)/(A5) into a guardrail (`guardrails/plan_audit.md` would be a new doc) and delete this injection. The behavior stays — codified.

## 📝 User actions queued

Manual items the loop can't or shouldn't do — surfaced here so the user has one place to find them. Loop appends as needed; user prunes when done.

- **[2026-05-21] Salloc smokes queued — see `dev/notes/trackers/salloc_smokes_queued.md`**. New dedicated file (per user 2026-05-21) for smoke runs that need an interactive Slurm allocation. Currently queued: kssynth slice 3b HEAVY (analyzers + kssynth) on M08073/well000 DIV 36, gating the radivojevic apples-to-apples diagnostic. Loop appends here; user runs each entry inside `salloc`; entry is deleted (or moved to `smoke_log.md`) when the run completes.

- **[2026-05-19] Delete the smoke-test repo `adamwea/__gh_auth_smoke_test`** on GitHub. Created during DIRECTIVE D pre-execution verification of `gh repo create` (commit `2124d2c`); `gh repo delete` requires the `delete_repo` token scope which the current token doesn't have. Either delete via web UI at https://github.com/adamwea/__gh_auth_smoke_test/settings (bottom of page → "Delete this repository") OR run `gh auth refresh -h github.com -s delete_repo` to add the scope and let the loop clean it up itself in the future. Not blocking anything; just clutter.

- **[2026-05-21 ✅ RESOLVED] Standing `/loop` prompt now lives at `dev/notes/loop_prompts/extended_autonomous.md`** (commit pending). Round 4 baked in (cadence ladder + phase-enable rule + real-data smoke log discipline + pre-cleared decisions for SLAy PR / kssynth slice 3b / Radivojevic gate). User pastes the fenced block from that file when re-firing the loop. Future revisions get version-tracked by `git log` on that path. CLAUDE.md pointers table updated to reference it.

- **[2026-05-21 09:35 — UPDATE] ✅ kssynth slice 3b SHORT-PATH smoke CONFIRMED working end-to-end**. Empirical validation:
  ```
  axon-recon stages reconstruct.kssynth --config dev/debug_NERSC/debug.runtime.yml \
    --target-dataset 0 --limit-wells 1 \
    --dry-run --force-enable kssynth \
    --output-root /pscratch/sd/a/adammwea/dev_outputs/<slice>
  ```
  exits status=success in ~5 seconds on real M08073 data; writes a well-formed
  `kssynth_summary.json` with `status: dry_run_ok`, missing-cache warning,
  outputs_would_produce reporting per_unit_dir + summary_json correctly.
  Similarly `reconstruct.analyzers --dry-run` + `preprocess --dry-run` both
  complete cleanly. **`--output-root` IS REQUIRED** — without it, the dry-run
  summary lands in the read-only reference data dir (verified empirically:
  the loop accidentally wrote `synth_sorter_output/kssynth_summary.json` +
  `context/analyzers_summary.json` to the reference well during a first
  pass; both cleaned up immediately).

  **Heavy smoke (real analyzers + real kssynth.synthesize) still needs the
  data-routing decision below — `--output-root` redirects OUTPUTS but the
  analyzers phase also needs INPUTS (preproc + spikesort outputs) which
  exist only at the reference path. So heavy smoke is gated on either
  (path 1) loosening cache-subdir rule, (path 2) alternate_well_out_dirs
  plumbing, or (path 3) symlink approach.** Original decision text preserved
  below.

- **[2026-05-21 ✅ DECIDED] kssynth slice 3b smoke data-routing — USER CHOSE PATH 2 (--input-root plumbing)**. Loop's next slice in the kssynth_recon_integration plan is to add `alternate_well_out_dirs` (or equivalent `--input-root` CLI flag) plumbing to the analyzers loader so dev_outputs/ wells can read inputs (preproc + spikesort outputs) from the reference well_out_dir while writing outputs (cache/, synth_sorter_output/, etc.) to the dev_outputs/ well_out_dir. Touch is M (loader signature + CLI flag + plumbing through resolve_session_inputs). Once shipped, the slice 3b heavy smoke runs as:
  ```bash
  axon-recon stages reconstruct.analyzers --config dev/debug_NERSC/debug.runtime.yml \
    --target-dataset N --limit-wells 1 --task-backend local_affinity \
    --input-root /pscratch/sd/a/adammwea/analyzed_data/Media_Density_T5_02182026_AR_axon_analysis_AW/ \
    --output-root /pscratch/sd/a/adammwea/dev_outputs/kssynth_slice3b/
  axon-recon stages reconstruct.kssynth --config dev/debug_NERSC/debug.runtime.yml \
    --target-dataset N --limit-wells 1 --task-backend local_affinity \
    --force-enable kssynth \
    --input-root /pscratch/sd/a/adammwea/analyzed_data/Media_Density_T5_02182026_AR_axon_analysis_AW/ \
    --output-root /pscratch/sd/a/adammwea/dev_outputs/kssynth_slice3b/
  ```
  Reference-data immutability preserved; clean separation between iteration outputs and reference inputs. Original 3-options block deleted; preserve in commit `e966f2c`'s diff for archeology.

  **Original block preserved for archeology**: The slice 3b plan note assumes the smoke runs in-place on `260326/M08073/AxonTracking/000208/well000` (DIV 36) — but the reference well's `recon_outputs/cache/` is empty and writing it would technically mutate the read-only reference data (per the working-data-scope rule). Redirecting via `--output-root /pscratch/.../dev_outputs/kssynth_slice3b/` cleanly avoids the mutation, but creates an input-resolution problem: the analyzers loader looks for preproc + spikesort outputs under `<output_root>/Media_Density_T5_.../260326/M08073/AxonTracking/000208/well000/` and those exist ONLY at the reference path. Three resolution paths:
  1. **Loosen the rule for `cache/` subdirs** — they're explicitly rebuildable, never "ground-truth reference output". Loop writes the cache in-place; no other reference artifacts touched. Smallest change.
  2. **Add `alternate_well_out_dirs` plumbing to the analyzers loader** so an iteration well can read inputs from the reference well_out_dir while writing outputs to the dev_outputs/ well_out_dir. The loader already has `alternate_well_out_dirs` in its signature for similar purposes (templates/runner.py `_load_templates_phase_analyzers`). Needs a `--input-root` CLI flag or equivalent. Touch is M.
  3. **Symlink approach** — `mkdir -p /pscratch/.../dev_outputs/kssynth_slice3b/.../well000/` then symlink `preprocess_outputs/` and `spikesort_outputs/` from the reference. Outputs land in the dev_outputs tree; inputs read through the symlinks. Touch is S but fragile (paths embedded in summary.json reference symlink targets).
  - **Smoke command sequence** (post-decision):
    ```bash
    # Step 1 (heavy, ~5-15 min): build the analyzer cache.
    axon-recon stages reconstruct.analyzers --config dev/debug_NERSC/debug.runtime.yml \
      --target-dataset N --limit-wells 1 --task-backend local_affinity \
      [--output-root /pscratch/.../dev_outputs/kssynth_slice3b/  # path 2 only]

    # Step 2 (fast, ~30s with --force-enable): run kssynth on the cache.
    axon-recon stages reconstruct.kssynth --config dev/debug_NERSC/debug.runtime.yml \
      --target-dataset N --limit-wells 1 --task-backend local_affinity \
      --force-enable kssynth \
      [--output-root /pscratch/.../dev_outputs/kssynth_slice3b/  # path 2 only]
    ```
    Replace `N` with the 0-based dataset index for `260326/M08073/AxonTracking/000208`. `--limit-units` could be added if step 2 is slow.
  - **Expected output**: `<well>/recon_outputs/synth_sorter_output/{kssynth_summary.json, templates.npy, channel_positions.npy, per_unit/unit_<id>/{merged_template,merged_channel_locations}.npy, ...}`. Visual diagnostic to file under `/pscratch/sd/a/adammwea/dev_outputs/kssynth_recon_integration/slice3b/`.
  - **Recommendation**: option (1) is cleanest if user accepts the cache-subdir-is-rebuildable framing. Awaiting decision before next loop iteration attempts the smoke.

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
- Mid-slice ambiguity: low-stakes → best-guess + note in `memory/notes.md`; high-stakes → ask user. If user is unresponsive: pause that slice, document blocker in `open_questions.md`, switch to another unblocked slice. Keep the loop moving.
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

- **`plans/active/env_install_unification_plan.md`** (Tier 5, chip-away) — 8 slices to unify the install story onto pyproject.toml extras + `tools/setup_env.sh --editable-siblings`. **TIMING CONSTRAINT**: should land BEFORE the next shifter rebuild — otherwise the rebuild reapplies the old per-sibling-ARG mechanism and slice 6 of this plan throws it away two weeks later. If a near-term shifter rebuild is needed (e.g. to close the UMPy + mat73 gap), prefer holding the rebuild until plan slice 6 ships. Conversely, if Tier 1 work demands a rebuild urgently (kssynth slice 9 integration is the natural pressure point), prioritize at least plan slices 1-3 + 6 first.

- **`plans/active/analysis_propagation_video_plan.md`** (Tier 5, chip-away) — 9 slices to re-implement axon_velocity's branch-propagation video / GIF generation as a new analysis-stage phase (`propagation_video` or similar; name decided in slice 2). Expensive per-unit work, opt-in via YAML + targeted-on-demand via per-unit `--targets` triplet form. Slice 1 does git archeology to find the old recon-stage impl (may not exist; fine to wrap current `axon_velocity` API cleanly). Slice 8 produces the first real-data video and adds a HARD-gate diagnostic to `memory/diagnostics_to_review.md` for user approval — a natural end-of-day deliverable for some future loop run. NOT blocking anything else.

- **`plans/active/radivojevic_recon_algo_plan.md`** (Tier 4, gated, kickoff-after-integration) — 9 slices to reverse-engineer + clean-room re-implement Radivojevic 2023's reconstruction algorithm as a sibling package at `~/dev/pkgs/radivojevic2023_recon_algo/` (already scaffolded with user-provided literature/) plus a new `radivojevic_recon` phase in the recon stage. Alternative / comparison to `axon_velocity_gtrs`. **Kick-off trigger**: kssynth slice 9 AND unitmatch_phase slice 5 BOTH shipped (= first real end-to-end smoke through the new sibling packages). Slice 1 is research-only — read pre-populated PDFs (elife-86512 likely the target paper) + WebSearch/WebFetch for public code + produce algorithm-summary doc + user-gate. 5 USER GATEs total (slices 1, 3, 4, 6, 9) — high back-and-forth plan.

- **`plans/active/chip_layout_phase_split_plan.md`** (Tier 4, gated, kickoff-after-integration) — 8 slices to move `plot_full_chip_layout` from recon stage to analysis stage AND split into two phases: **Phase A** (white-bg timeline grid showing repeated sessions for a (chip, well) group) and **Phase B** (black-bg detailed per-session with electrode overlay + signal-strength-weighted RGB blending on electrode overlap). Cross-session unit colors via unitlink match table (preferred) or extremum-electrode fallback. **Kick-off trigger**: same as Radivojevic plan — kssynth slice 9 AND unitmatch_phase slice 5 both shipped (need unitlink match tables for color consistency). 2 USER GATEs (slices 5 + 7 visual diagnostics). Cross-session color helper (slice 3) is reused by future propagation_video / Radivojevic-comparison plotting.

- **`plans/active/dashboard_ui_refinement_plan.md`** (Tier 4, no gating) — 9 slices to refactor and refine the Dash UI (`src/axon_recon/dashboard/`, ~2128-line app.py + 6 sibling modules). Addresses user-reported pain points: empty metrics rendering as broken UI; some plates missing from UI (hardcoded lists instead of filesystem-discovered); feature parity gap between plot types (box has features histogram/scatter don't); no box↔bar mode toggle; no tertiary grouping option; styling drift between plot types. Slice 1 = audit only (no code) — produces feature matrix + hardcoded-list inventory + empty-state gap list. Slice 4 introduces a shared `PlotConfig` abstraction that all plot types consume. Slices 6+7 add box↔bar toggle + tertiary grouping. Slice 8 unifies styling (possible palette-module synergy with chip_layout_phase_split_plan slice 3). **No dependency on Era 3 integration** — dashboard reads existing `analyzed_data/` outputs; can start any time the loop has bandwidth.

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
