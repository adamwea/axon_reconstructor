# Radivojevic-2023 reconstruction algorithm plan

## Motivation

Radivojevic et al. 2023 describe a reconstruction algorithm for axon
tracking on high-density MEAs. The user wants it implemented as an
**alternative and/or comparison to axon_recon's current `axon_velocity`
GTRS-based approach** in the recon stage. Sibling-package scaffold has
been started by the user at
`/global/homes/a/adammwea/dev/pkgs/radivojevic2023_recon_algo/` with
`literature/` and `notes/` subdirs already populated (see slice 1).

Reference code is reportedly NOT publicly available. Plan starts with
an exhaustive literature read + code-availability search (slice 1)
before committing to a clean-room re-implementation.

User direction (2026-05-19):
- "I want to implement [Radivojevic 2023's algorithm] in the recon stage
  as an alternative and/or comparison to axon_recon's algo."
- "Inputs are ideally comparable to what axon_velocity needs to generate
  gtrs but unsure."
- "I would eventually want to produce all the same plots and metrics that
  axon_velocity gtrs enable, if not more."
- "I'd later want to ship this with full attribution to the authors."
- "If you can find the code online, and avoid reinventing the wheel,
  obviously that'd be ideal."
- "This plan will certainly require some back and forth with me."

## Pre-populated literature (user-provided)

In `/global/homes/a/adammwea/dev/pkgs/radivojevic2023_recon_algo/literature/`:

- **`elife-86512-figures-v1.pdf`** — eLife article 86512 (almost certainly
  the primary "Radivojevic 2023" paper). Read in slice 1; confirm with user
  during slice 1 user-gate.
- **`Radivojevic and Rostedt Punga - Functional imaging of conduction
  dynamics in cortical...`** — possible second 2023 paper. May provide
  complementary algorithmic detail. Read in slice 1.
- `Radivojevic et al. - 2017 - Tracking individual action potentials...` —
  earlier methods paper; useful background for algorithm lineage.
- `Radivojevic et al. - 2016 - Electrical Identification and Selective
  Microstimulation...` — earlier methods; deeper background.
- `Franke et al. - 2015 - Bayes optimal template matching for spike sorting`
  — adjacent / not directly the target algorithm but useful context.

## Out of scope (v1)

- Replacing `axon_velocity` GTRS as the recon-stage default. New algo
  lives alongside, opt-in via YAML. Comparison work decides which (if
  either) becomes default later.
- Improving on or modifying the published method. Stay faithful to the
  paper for v1; deviations recorded in slice 9 as v2 questions.
- Publishing the sibling package to PyPI. Stays local + private per
  existing GH-remotes-hold policy.

## High-uncertainty plan structure

This plan has significantly more open questions than typical
sibling-package plans because:

- Source code may not be public — clean-room reimplementation carries
  inherent risk of misinterpretation.
- Input compatibility with `axon_velocity_gtrs` is assumed but not
  verified.
- "Same plots and metrics" may expand scope as we discover what
  Radivojevic's output supports.
- Implementing the method may imply new phases in OTHER stages
  (preprocess / spikesort / reconstruct) if input formats differ.

**Explicit user check-in gates** are built into the slice ladder. Loop
posts questions to `open_questions.md` at each gate and switches to
another plan until the user answers. Gates at slices 1, 3, 4, 6, 9.

## Slices

### Slice 1 — Literature + code search (RESEARCH ONLY, no code changes) — SHIPPED 2026-05-21

**Deliverables:**
- ✅ `dev/notes/refs/radivojevic2023_paper.md` — citation (DOI 10.7554/eLife.86512, published 2023-08-22) + data availability (Dryad doi:10.5061/dryad.gxd2547r1) + code availability finding (**NONE** — clean-room required) + lineage of related papers (Buccino 2022 → axon_velocity, Bullmann 2019 → hana, Radivojevic 2016/2017 → earlier methods).
- ✅ `dev/notes/refs/radivojevic2023_algorithm_summary.md` — algorithm spec (3 stages: adaptive thresholding + skeletonization + multi-step tracking), input/output spec, hardware assumptions (HD-MEA ~17.5 μm pitch, 20 kHz), input compat table vs `axon_velocity_gtrs`, list of slice-3 hyperparameter unknowns flagged for tuning.
- ✅ USER GATE 1 questions logged to `dev/notes/memory/open_questions.md` under "Radivojevic slice 1 user-gate review" (6 questions; question 3 — averaged-vs-per-spike STA — is the highest-impact one).

**Note on PDF reading**: the loop's environment lacks `pdftoppm` /
`pdftotext`; the pre-populated literature/ PDFs couldn't be read in
this iteration. All paper info gathered via WebFetch on the eLife
article page + WebSearch. The figures-only PDF in literature/ is
useful visual reference but doesn't carry the methods detail
(suffix `-figures-v1` = supplementary figures only). When PDF
extraction is available in a future env, slice 3's algorithm
implementation should re-read the full methods section to verify
hyperparameter defaults + answer question 3 above (raw per-spike STA
vs averaged template).

**Pre-overnight clearance**: per current_state.md "PRE-OVERNIGHT
CLEARANCES" item #2, loop proceeds INTO slice 2 (sibling-package
scaffold) without pausing. Slice 3 (core algorithm impl) DOES gate
on user review of the open_questions.md entries.

#### Original spec (preserved for slice 3 reference)


**Goal**: pin the exact paper; exhaust code-availability search;
produce a written algorithm summary; identify input requirements.

**Actions**:
- **Read the pre-populated PDFs** under
  `~/dev/pkgs/radivojevic2023_recon_algo/literature/`. Primary
  focus on `elife-86512-figures-v1.pdf` (likely the target paper);
  also read the Radivojevic + Rostedt Punga paper if it covers
  algorithm detail.
- **Identify the paper precisely** — record DOI + title + author list
  in this plan + in a new `dev/notes/refs/radivojevic2023_paper.md`.
- **Search for public code** via WebSearch + WebFetch tools:
  - GitHub: search "Radivojevic", "Hierlemann" (likely ETH Zurich lab
    affiliation), "MaxWell Biosystems".
  - Zenodo + FigShare: paper supplementary code archives.
  - The paper's supplementary materials / data availability section.
  - Lab pages: Hierlemann group at ETHZ.
  - Document findings in this plan (positive OR negative).
- **Write `dev/notes/refs/radivojevic2023_algorithm_summary.md`** — a
  structured summary: algorithm name, inputs, processing steps,
  outputs, hyperparameters, evaluation metrics, figure-level plot
  inventory. This is the source-of-truth doc the rest of the plan
  reads against.
- **Map inputs to `axon_velocity_gtrs`**: side-by-side table comparing
  what Radivojevic needs vs. what `axon_velocity_gtrs` currently
  consumes from the recon stage. Identify deltas.

**Output**:
- `dev/notes/refs/radivojevic2023_paper.md` (citation)
- `dev/notes/refs/radivojevic2023_algorithm_summary.md` (algorithm doc)
- Updated paper-id + code-search-result sections in this plan
- List of user check-in questions for slice 2 kickoff

**USER GATE 1**: pause + post questions to `open_questions.md`:
- Confirm paper identity (paper title + DOI).
- Review the algorithm summary doc.
- Confirm clean-room re-implementation path (OR, if code found,
  approve attribution + license strategy for direct use).
- Confirm or revise the input-delta map (which prerequisite data we
  need to compute that we don't already have).

### Slice 2 — Sibling-package scaffold

**Goal**: turn `~/dev/pkgs/radivojevic2023_recon_algo/` into a proper
sibling package (matches kssynth / unitlink shape).

**Actions**:
- `git init` (skip if user already did)
- `pyproject.toml` with core deps; specific deps TBD from slice 1's
  summary.
- Package layout:
  `radivojevic2023_recon_algo/{__init__,core,io,api}.py` (or whatever
  fits the algorithm's structure).
- Empty test scaffolding under `tests/`.
- README with paper citation + clean-room re-implementation
  disclaimer + attribution paragraph.
- `LICENSE` (decision in slice 9 user-gate).
- 2-3 sanity tests (import, version).

**Compliance**: per `guardrails/package_contracts.md` — SI-compliant
where it touches SpikeInterface conventions; no axon_recon-specific
deps.

**No remote**: hold per existing GH-remotes-hold policy.

### Slice 3 — Core algorithm implementation (most complex slice)

**Goal**: best-effort implementation of the algorithm's core processing
chain per the slice-1 summary doc.

**Actions**:
- Implement per the summary doc, step by step. One commit per logical
  step where possible.
- Heavy use of `open_questions.md` for ambiguities in the paper —
  every unresolved spec question gets logged with paper section
  reference + the two/three reasonable interpretations.
- Test each step with synthetic input where feasible.
- **USER GATE 2 every 2-3 sub-steps**: post sub-step results +
  ambiguity questions to `open_questions.md`; switch to another plan
  until user reviews.

**Risk note**: this slice may take many iterations and produce a v1
that diverges from the paper in subtle ways. Divergences documented
in slice 9.

### Slice 4 — Input compatibility check vs `axon_velocity_gtrs`

**Goal**: confirm whether the algorithm's inputs match what
`axon_velocity_gtrs` already consumes, or whether new prerequisite
phases are needed.

**Actions**:
- Diff input requirements (from slice 1 summary) against
  `axon_velocity_gtrs`'s actual inputs (templates, channel positions,
  sample rate, waveform extracts, …).
- **Compatible**: no new phases needed.
- **Incompatible**: document required prerequisite data. Could be new
  phases in preprocess / spikesort / reconstruct, OR new computations
  inside the sibling package, OR both. Surface for user decision.

**USER GATE 3**: surface any required new phases for user approval
BEFORE slice 5. New phases imply additional plan injections for those
stages.

### Slice 5 — Recon-stage phase wire-in

**Goal**: new `radivojevic_recon` phase (name TBD; could also be
`radivojevic_2023_recon` for explicit attribution) in
`pipeline/stages/reconstruct/` that calls into the sibling package.

**Actions**:
- Per the existing recon-stage phase template (compare to
  `axon_velocity_gtrs`).
- `enabled: false` default — opt-in per run.
- All scope flags + `--dry-run` + checkpoint markers per guardrails.
- YAML scaffolding in `debug.runtime.yml` + `debug.data.yml` per the
  YAML-hygiene injection.
- Per `env_parity` guardrail: editable install added to (the future)
  `tools/install_dev_siblings.{sh,py}` once env_install_unification
  plan ships slice 4; until then, document the editable install
  command under USER INJECTIONS.

### Slice 6 — First real-data smoke + HARD-gate correctness review

**Goal**: run on **one (dataset, well, unit)** of the 80k DMEM well000
cohort. Compare side-by-side against `axon_velocity_gtrs`.

**Actions**:
- Login-node smoke per standing constraints.
- Save outputs side-by-side under
  `/pscratch/sd/a/adammwea/dev_outputs/radivojevic_recon_smoke/`.
- **HARD-gate visual diagnostic**: produce comparison plots
  (Radivojevic vs `axon_velocity_gtrs`) — branch overlay,
  propagation timing comparison, every figure type slice 1 surfaced.
- Entry in `memory/diagnostics_to_review.md`.

**USER GATE 4**: user reviews + approves the first real-data output
before any downstream slices ship. **Most important user check-in of
the plan.** If outputs look obviously wrong vs. the paper figures,
slice 3 reopens.

### Slice 7 — Comparable plots + metrics

**Goal**: ship every plot + metric that `axon_velocity_gtrs` produces,
but sourced from the Radivojevic algorithm. Plus any additional
outputs the Radivojevic data supports (per slice 1 summary).

**Actions**:
- Inventory `axon_velocity_gtrs`'s plot + metric outputs.
- Implement equivalents using Radivojevic-output data.
- Tests for each output type.
- If the new algo can drive the analysis-stage propagation-video plan
  (separate plan), expose that path here too.

### Slice 8 — Comparison analysis phase

**Goal**: new analysis-stage phase that compares
`axon_velocity_gtrs` vs `radivojevic_recon` outputs for units where
both have run.

**Actions**:
- Per-unit comparison: agreement metrics, divergence metrics,
  computational-cost diff.
- Comparison plot suite (side-by-side figures, scatter, residuals).
- Lives in `pipeline/stages/analysis/comparison/` or similar.

### Slice 9 — Attribution + docs + divergences

**Goal**: ship the package + recon phase with full attribution and a
clear "what we did differently from the paper" doc.

**Actions**:
- Sibling package README: citation, author list, attribution
  paragraph, clean-room re-implementation disclaimer.
- `DIVERGENCES.md`: every place the implementation deviates from the
  paper (or where the paper was ambiguous and a judgment call was
  made).
- axon_recon main README: cite Radivojevic 2023 alongside
  axon_velocity.
- License compatibility check.

**USER GATE 5**: review attribution wording before declaring done.

## Smoke tests

- After slice 3: each algorithm step passes its synthetic unit test.
- After slice 5: dry-run on a small target list reports correct paths.
- After slice 6: real login-node smoke on one unit produces output;
  HARD-gate visual diagnostic flagged.
- After slice 7: every `axon_velocity_gtrs` plot/metric has a
  Radivojevic equivalent.

## Done criteria

- `axon-recon recon radivojevic_recon --targets <ds>:<well>` produces
  recon outputs.
- Phase wired into recon stage as `enabled: false`.
- Side-by-side comparison plots vs `axon_velocity_gtrs` available
  (analysis-stage comparison phase).
- Full attribution + DIVERGENCES doc shipped.
- All 5 USER GATEs approved.

## Tier + sequencing

**Tier 4 (gated)** — substantive work behind in-flight integration.
Per user direction (2026-05-19): "priority-wise, this can come after
implementing unitlink and kssynth."

Kick-off trigger: **kssynth slice 9** (axon_recon recon-stage
integration) AND **unitmatch_phase slice 5** (enable + login-node
smoke) BOTH shipped. Those are the natural "unitlink + kssynth
integration" markers — first real end-to-end smoke through axon_recon
using the new sibling packages. Until both land, this plan sits idle.

Once unblocked, **slice 1 (research-only)** is a good first pick — it's
bounded research that produces written docs + user-gate questions
without touching code. Slices 2-9 sequence behind slice 1 + user gates.

## Open questions (initial — will grow during execution)

- **Paper identity confirmation** (slice 1 user-gate): the literature
  folder strongly implies eLife 86512 is the target, but user should
  confirm.
- **Code availability** (slice 1): is the algorithm actually
  unavailable, or buried in supplementary materials / a GitHub repo we
  haven't found?
- **Input compatibility** (slice 4 user-gate): does the algorithm
  consume the same templates/footprints `axon_velocity` does? Or new
  preprocessing required?
- **New prerequisite phases**: if input compatibility fails, which
  stages need new phases? Implications for tier ordering of those
  ancillary plans.
- **License** (slice 9 user-gate): which license for the sibling
  package? Compatible with axon_recon's?
- **Author contact**: should we reach out to Radivojevic et al. for
  algorithm-detail questions, or attempt clean-room only? (User
  decision; loop doesn't email people.)
- **Plotting parity scope** (slice 7): does "all the same plots and
  metrics that axon_velocity gtrs enable" include the
  propagation-video work, or is that strictly future?
