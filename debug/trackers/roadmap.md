# Roadmap — future feature ideas

Living backlog of features and capabilities we want, but haven't yet scoped into a
plan. Entries point AT plans when they exist; plans don't duplicate entry content.

## Format

Each entry uses the structure below. Status one of: `idea` / `scoped` / `in-plan` /
`landed` / `dropped`. When status flips to `in-plan`, the body collapses to a
one-liner pointing at the plan doc.

```
### <Short title>
- **Status**: idea | scoped | in-plan | landed | dropped
- **Tags**: comma-separated (curation, algorithms, dashboard, analysis, infra, …)
- **Problem**: 1 paragraph — what's the unmet need / motivation
- **Approach sketch**: 1 paragraph — rough technical shape
- **Dependencies / blockers**: what has to be true before this can start
- **See also**: plan doc(s), prior commits, external refs
```

---

## Active entries

### Curation GUI features in the dashboard
- **Status**: idea
- **Tags**: curation, dashboard, spikesort, recon
- **Problem**: Bombcell and SpikeInterface ship curation GUIs, but neither sees the
  morphology data from the axon-recon reconstruct stage. We want a way to (a) launch
  the external GUIs against our spikesort outputs, and (b) do recon-aware merge
  candidate review (side-by-side circle_recon + branch overlay + similarity score),
  then re-run the merge + recon stages from the dashboard.
- **Approach sketch**: Three deliverables, ordered by unique value: (1) recon-aware
  merge-candidate tab inside the Dash app — click two units on a scatter, get a
  side-by-side comparison + accept/reject button that appends to
  `<well>/curation_inputs/recon_merge_proposals.tsv`; (2) wrap external GUIs as
  sibling top-level commands (`axon-recon curate sortingview …`,
  `axon-recon curate spikeinterface-gui …`) parallel to `axon-recon dashboard`, with
  dashboard buttons that print or `subprocess.Popen` the right command; (3) "Apply
  curation" button that spawns `axon-recon stages spikesort.merge_SLAy
  --use-curation-inputs` as a background subprocess with status polling. The dashboard
  stays read-only of pipeline outputs but writes only to `<well>/curation_inputs/`,
  preserving replayability.
- **Dependencies / blockers**: per-well lockfile + auto-reload-on-completion for the
  rerun flow; `--use-curation-inputs` flag on `spikesort.merge_SLAy` (new feature on
  the spikesort side); decide subprocess vs job-queue for rerun triggering.
- **See also**: discussed in conversation 2026-05-11; would earn its own plan
  (`curation-dashboard-extension-plan.md`) when scoped.

### Milos-style axon-tracking algorithm
- **Status**: idea
- **Tags**: algorithms, recon
- **Problem**: <TBD — user to elaborate. Working title only; refers to an
  alternative axon-tracking approach attributed to Milos that may improve on
  the current axon_velocity-based pipeline.>
- **Approach sketch**: <TBD>
- **Dependencies / blockers**: writeup of the algorithm; comparison criteria
  against the current pipeline; fixture dataset for benchmarking.
- **See also**: placeholder — populate when user provides details.

### Axon-vs-dendrite identification from signal characteristics
- **Status**: idea
- **Tags**: algorithms, analysis, recon
- **Problem**: <TBD — user to elaborate. The reconstruction stage currently
  treats every detected propagation as an axon. Some traced branches may be
  dendritic; classifying them by signal characteristics (waveform shape,
  velocity range, polarity, …) would let downstream analysis filter or stratify.>
- **Approach sketch**: <TBD>
- **Dependencies / blockers**: training/validation set of labeled
  axon-vs-dendrite branches (or a heuristic ruleset to start); decision on
  whether classification happens in the reconstruct stage as a per-branch label
  or in the analysis stage as a per-row column.
- **See also**: placeholder — populate when user provides details.
