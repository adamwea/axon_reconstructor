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

## 📝 User actions queued

Manual items the loop can't or shouldn't do — surfaced here so the user has one place to find them. Loop appends as needed; user prunes when done.

- **[2026-05-19] Delete the smoke-test repo `adamwea/__gh_auth_smoke_test`** on GitHub. Created during DIRECTIVE D pre-execution verification of `gh repo create` (commit `2124d2c`); `gh repo delete` requires the `delete_repo` token scope which the current token doesn't have. Either delete via web UI at https://github.com/adamwea/__gh_auth_smoke_test/settings (bottom of page → "Delete this repository") OR run `gh auth refresh -h github.com -s delete_repo` to add the scope and let the loop clean it up itself in the future. Not blocking anything; just clutter.

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
