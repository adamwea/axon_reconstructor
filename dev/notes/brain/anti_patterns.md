# Anti-patterns — lessons learned the hard way

Consolidated registry of "things that LOOK tempting in the moment but produce silent bugs / context drift / wasted work." Each entry is something this project has already paid for in lost iterations. Future loop instances read this to avoid the same mistakes.

**Reading rule for the loop**: when about to do something that matches a trap below, STOP and read the entry. The "right move" column is what to do instead. If the situation feels novel, propose adding a NEW anti-pattern via `open_questions.md` after surfacing it as a finding.

**Writing rule**: APPEND-only. User can prune (rarely). Format mirrors the auto-memory feedback files — many entries here originate as auto-memory and are surfaced here for in-repo visibility.

---

## AP-001 — Improvising around explicit user instructions

**The trap**: User said "use X" (e.g. "use existing plot_recons", "pick a high-branch unit", "produce a comparison"). Implementation hits friction (X has a shape mismatch, the target isn't available, etc.). Loop's instinct is to substitute (Y instead of X, or do half of X) to keep momentum.

**Why it's a trap**: The substitute produces an artifact that LOOKS like progress but answers a question the user didn't ask. Hours of work compound into the wrong deliverable. The user has to redo or reject everything.

**The right move**: STOP and write a focused multiple-choice question to `brain/open_questions.md`: "Spec says X, but Y is blocking; should I (1) do Z to enable X, (2) accept substitute W with caveats, (3) something else?" The 1-line question saves 80+ minutes of wrong work.

**Source**: 2026-05-21 — loop spent ~80 min on a Radivojevic SOFT-gate diagnostic that failed on 3 user-explicit requirements (use plot_recons, pick high-branch unit, produce comparison). Each improvisation was technically reasonable in isolation; cumulative output answered zero of the user's actual questions. Codified as USER INJECTION B1/B2 in `current_state.md` + auto-memory `feedback_no_improvise_around_explicit_instructions`.

---

## AP-002 — Validating a phase by running it disabled

**The trap**: A phase is YAML-disabled (`enabled: false`). Loop runs a smoke that hits the stage but skips the phase. Smoke "passes" — and loop reports the slice validated.

**Why it's a trap**: A passing no-op tells you the dispatcher works, not that the PHASE works. Silent confidence in untested code.

**The right move**: Enable the phase for the smoke. Use the `--force-enable PHASE` CLI flag (shipped commit `5d28561`) OR temporarily edit YAML + revert. The smoke MUST actually run the phase or it's not validation.

**Source**: 2026-05-21 user feedback during kssynth slice 3b design. Codified as USER INJECTION "Phase-enable for tests" + auto-memory `feedback_enable_phases_for_tests`.

---

## AP-003 — Self-graded code commits (actor-grading-itself)

**The trap**: After editing code, loop says "this looks right, committing." No external check. Tests pass → ship.

**Why it's a trap**: The actor reasoning that wrote the code is the same reasoning that grades it — same biases, same blind spots. Code that "looks right" to the actor often masks subtle errors that an unbiased reader would catch.

**The right move**: spawn a SEPARATE Explore subagent as critic per `brain/guardrails/critic_separation.md`. Narrow scope (diff + trusted-output fixture + invariants only). Pass/fail/concerns verdict. Block commit on fail. Push-back surfaces to user — never silent override.

**Source**: Brain-theory conversation 2026-05-21; structurally separate monitor module. Codified as `brain/guardrails/critic_separation.md`.

---

## AP-004 — Tests-green ≠ feature-works-on-real-data

**The trap**: Unit/integration test suite passes. Loop reports slice complete. Moves on.

**Why it's a trap**: Test fixtures are synthetic. Real data has corner cases (sparse templates, off-by-one channel maps, segment boundaries, NaN propagation) that fixtures don't capture. Code that passes tests can produce silently-wrong output on real cohorts.

**The right move**: maintain `dev/notes/trackers/smoke_log.md` separately from test results. Every real-data smoke gets an entry (cohort, command, outcome, quantitative results, bugs revealed). A passing test isn't equivalent to a green smoke_log entry.

**Source**: 2026-05-21 user direction. Codified as USER INJECTION + auto-memory `feedback_real_data_smoke_log`.

---

## AP-005 — Diagnostic without a rendered image

**The trap**: Slice produces output (numpy arrays, TSV tables, JSON summaries). Loop files a diagnostic entry pointing at those files + claims "diagnostic shipped."

**Why it's a trap**: Arrays + TSVs aren't VISUAL diagnostics; they're data dumps. The user can't review them at a glance — they have to load + render, which is what the loop should have done. The audit-trail artifact is missing.

**The right move**: every diagnostic with visual content MUST include a rendered image (PNG/SVG/PDF) as the user-visible artifact. Data files can ACCOMPANY for re-rendering / downstream consumption, but the image IS the diagnostic.

**Source**: 2026-05-21 user feedback "I see the recon output but its npy and tsv files." Codified as R5 in the tightened visual-diagnostics rule + auto-memory `feedback_visual_diagnostics_strict`.

---

## AP-006 — Bare halts instead of multiple-choice questions

**The trap**: Loop is blocked. Surfaces a description of the situation: "I am blocked because Y. Please advise." Waits.

**Why it's a trap**: User has to design the answer from scratch. High cognitive load. Loop's analysis (the options it considered, the tradeoffs, the recommendation) is hidden — so the user can't quickly pick.

**The right move**: every "ask" form MUST be 2-4 numbered options with `label / touch-size / tradeoff` per option + a Recommended option called out. Format mirrors `AskUserQuestion`. The user picks a number; loop executes.

**Source**: 2026-05-21 user direction "for now, instead of stopping ask it to just give me choices like you just did." Codified as B1/B2 in `current_state.md` + auto-memory `feedback_no_improvise_around_explicit_instructions` "How to apply" section.

---

## AP-007 — Bumping the /loop prompt mid-session without consolidation

**The trap**: User asks for a new rule. Loop adds it to current_state.md AND bumps the /loop prompt round (extending the standing prompt). Repeat. The prompt grows monotonically; rules accumulate without retirement.

**Why it's a trap**: Each round bump consumes user attention to re-paste; the prompt's coherence degrades; rules that should have been promoted to guardrails instead sit as injections forever. Bloat reduces signal.

**The right move**: bump the /loop prompt only when (a) authorizations change OR (b) major directives ship and can be removed as first-iteration priorities. Otherwise, new rules land in `current_state.md` USER INJECTIONS until promoted to guardrails. Don't bump every session.

**Source**: 2026-05-21 audit — observed the /loop prompt bumped 3 times in 90 minutes (rounds 5/6/7). Surfaced during the refinement pass that led to the 🛑 PAUSED state.

---

## AP-008 — Hardcoding sample rate or device parameters

**The trap**: A new analysis phase needs the sample rate. Loop sees `sampling_rate_hz: 10000` in the cohort's metadata and uses 10000 as a constant.

**Why it's a trap**: The pipeline runs on multiple devices (MaxTwo @10 kHz, MaxOne @20 kHz, future ThreeBrain/Sony at TBD rates). Hardcoded constants break silently when a new device joins. The same applies to electrode count (26,400), pitch (17.5 μm), and any other device-specific number.

**The right move**: read device params from the analyzer manifest at runtime (`preprocess.save_rec_metadata` is the authoritative producer). Parameterize anything that scales with sample rate as a RATIO not absolute Hz (e.g. `upsample_factor: 10` not `upsample_target_hz: 200000`). For threshold values, prefer STD-relative over absolute-μV.

**Source**: 2026-05-21 Radivojevic Q5 review. Codified as project auto-memory `project_axon_recon_device_diversity` + tech-debt entry "Device-agnostic source data" + comments in `brain/refs/radivojevic2023_algorithm_summary.md`.

---

## AP-009 — Paper-overs instead of principled fixes

**The trap**: A symptom appears (e.g. "profile clamps cpus_per_task at YAML default when srun gives more"). Loop ships a targeted patch that handles the symptom (e.g. "SLURM_CPUS_PER_TASK env wins over YAML at each phase"). Symptom goes away. Loop moves on.

**Why it's a trap**: The next plan that touches the same area finds the paper-over and has to revert it OR work around it. The cumulative effect is a code path with N patches on top of a fundamentally broken model. The principled fix (e.g. "env-only supply resolver — delete the profile concept entirely") would have been smaller in the long run.

**The right move**: when a symptom surfaces, ask "is this symptom of a deeper model problem?" If yes, surface the deeper problem as a finding/plan rather than papering. Specifically: if a fix would be reverted by a foreseeable other plan, DON'T ship the fix — ship the foreseeable other plan first.

**Source**: 2026-05-21 commits `1982a31` + `322e8fe` (SLURM env precedence + topology clamp). Both were paper-overs on top of profile-clamping; `resources_profiles_elimination_plan.md` slice 0 reverts them. Tracked in D-003 + the plan's See-also markers.

---

## AP-010 — Letting USER INJECTIONS accumulate without promotion

**The trap**: User authors an injection. Loop applies it. Next user msg adds another injection. Repeat. After 12+ active injections in current_state.md, the loop reads them all every iteration → context bloat → drift accelerator.

**Why it's a trap**: The injection-piling pattern is supposed to be temporary — each injection has a "Promote when stable" criterion. Without active promotion (to guardrails / CLAUDE.md slice protocol / plans), injections become permanent context overhead that shouldn't be permanent.

**The right move**: every iteration that ships, the loop AUDITS the open injections list. Anything that met its promotion criterion gets PROMOTED + DELETED from current_state.md. Resolved injections get retired immediately. The injection list should hover at 3-5 active, not grow to 12+.

**Source**: 2026-05-21 audit found 12 active injections, several with promotion criteria already met. Refinement pass collapsed to 4. The retire-and-promote pattern is now part of CLAUDE.md slice protocol step 12-14.

---

## AP-011 — Generating new rules instead of using existing controls

**The trap**: A new user concern surfaces. Loop's instinct is to write a new injection / new guardrail / new auto-memory to address it.

**Why it's a trap**: The repo already has controls for most concerns (CLAUDE.md entry protocol, slice protocol, guardrails, memory system). Adding NEW rules without checking whether existing ones cover the concern doubles the surface to maintain + dilutes attention to the original rules.

**The right move**: before writing a new rule, check whether existing controls cover the concern. If yes: NOTE that they do (so the user knows the existing rule applies). If no: write the minimum-viable rule + check whether it can replace or amend an existing rule rather than adding a parallel one.

**Source**: 2026-05-21 audit — observed the assistant generating ~6 injections + 4 /loop-prompt round bumps + 3 auto-memory files in 90 minutes, several of which overlapped with existing rules. The "STOP — what controls already exist?" pause is the corrective.

---

## AP-012 — Skipping the reference data tree as a verifier anchor

**The trap**: A new phase produces output. Loop checks "does this match the spec I wrote?" Spec is loop-authored → tautological pass.

**Why it's a trap**: The verifier needs an EXTERNAL ground truth (per critic separation, per trust hierarchy). Loop-authored specs aren't external. The reference data tree IS the external ground truth for shape/schema/structure (per D-002 blanket pin).

**The right move**: when authoring or shipping a new output, FIRST check `brain/trusted_outputs.md` TR-000's structural invariants + any TR-NNN specific invariants for the affected junction. The new output MUST match the analogous reference's shape OR be a justified differential (with the differential captured as a DC-NNN entry).

**Source**: 2026-05-21 Radivojevic SOFT-gate produced output that bore little structural resemblance to axon_velocity_gtrs's reference output for the same unit. User flagged: "New outputs should be identical or at least similar in shape to the current reference data." Codified as D-002 + TR-000.

---

## Promotion criterion

When an anti-pattern has appeared 3+ times AND its "right move" has been internalized (loop hasn't tripped on it for ≥5 sessions), promote the rule into a guardrail and link from here. Don't delete the entry; mark Status: PROMOTED so the lesson persists.
