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

### Post-templates bombcell + SLAy pass (label/merge on full template payloads)
- **Status**: idea
- **Tags**: spikesort, curation, recon
- **Problem**: bombcell labels and SLAy merges currently run on the raw kilosort
  artifacts (mean waveforms, cluster TSVs, automerge templates) BEFORE recon
  builds its per-segment + concat analyzer payloads. That means both algorithms
  reason about each unit using an incomplete picture of its template — only the
  channels and spike subset kilosort/SLAy chose to surface. Some units may be
  misclassified ("good" vs "mua" vs "noise") or mis-paired-for-merge because the
  available template is locally clean but globally noisy (or vice versa), and the
  feature inputs to bombcell/SLAy don't see that. Running bombcell + SLAy a
  second time, AFTER recon has materialized full per-segment + concat templates,
  could correct those calls.
- **Approach sketch**: After recon's `templates` stage finishes, expose a
  follow-on phase (or top-level command) that re-projects each unit's full
  per-segment + concat template payload into the feature shape bombcell expects
  (template waveforms per channel, amplitude/contam metrics derived from the
  larger spike set in the materialized payloads), then re-runs bombcell to emit a
  fresh labels JSON. Same pattern for SLAy: feed the materialized concat
  template into SLAy's similarity scorer (instead of the kilosort-mean-waveform
  template) and run another merge pass on the candidate pool. Outputs land at a
  distinct location (e.g. `bombcell_label_post_templates_outputs/`, parallel SLAy
  output root) so the original kilosort-derived labels/merges stay auditable.
  Decide downstream which label set the recon stage consumes via a config knob.
- **Dependencies / blockers**: requires the templates stage to emit per-unit
  template arrays in a shape bombcell/SLAy can ingest without re-running the
  kilosort post-processing pipeline. May need an adapter layer that mimics the
  kilosort folder layout from materialized payloads. The simpler version
  (`bombcell_label_pass2` on raw kilosort artifacts) was prototyped on
  2026-05-11 and shipped disabled-by-default: it hit a SpikeInterface
  `KiloSortSortingExtractor` inner-join bug — the extractor merges every
  `cluster_*.tsv` on `cluster_id`, and SLAy only updates `cluster_KSLabel.tsv`
  + `cluster_group.tsv` (not `cluster_Amplitude.tsv` / `cluster_ContamPct.tsv`),
  so the post-merge unit IDs get dropped by the intersection. To re-enable
  pass2 (or to do the post-templates version), either strip the stale per-
  metric TSVs before loading the sorting, force the loader through
  `_build_numpy_sorting_from_kilosort_raw` (already exists), or patch SLAy to
  write stub rows into the other cluster_*.tsv files for new unit IDs.
- **See also**: discussed 2026-05-11; recon-side label loader now prefers
  post-SLAy `cluster_KSLabel.tsv` over `bombcell_labels.json` for exactly this
  reason — `cluster_KSLabel.tsv` is the only artifact that has the complete
  post-merge label set (pass1 wrote bombcell labels into it for pre-merge IDs,
  SLAy appended inherited labels for new merge IDs).

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


### Editable / mount-based source for fast container iteration
- **Status**: idea
- **Tags**: infra, container, dx
- **Problem**: Every code edit to `axon_recon` or its sibling packages
  (UnitMatchPy, SLAy, axon_velocity, eventually mea_analysis) currently
  requires a full Shifter image rebuild + push + `shifterimg pull` cycle.
  On a NERSC head node this is ~15–30 min for a small Python change, which
  kills the inner-loop dev cadence when iterating on, e.g., a new CLI
  subcommand, a phase-runner tweak, or a SLAy parameter change. We want a
  way to point the running container at a host-mounted source tree so
  Python edits are picked up without rebuilding.
- **Approach sketch**: Three candidates discussed 2026-05-13:
  - **A (mount over site-packages)**: `--volume=<repo>/src/axon_recon:
    /home/miniconda3/lib/python3.11/site-packages/axon_recon:ro`. Direct
    shadow of the installed copy; works today with no Dockerfile changes
    but pins the site-packages path.
  - **B (`pip install -e` in the image + mount /opt/axon_recon at runtime)**:
    image has only an `.egg-link` / `.pth`; host source mounted at
    `/opt/axon_recon` is the canonical import path. Cleaner for new
    sub-package additions; slightly more dependent on pip-version behavior.
  - **C (PYTHONPATH prepend via env var)**: `--env=PYTHONPATH=/host_repo/src
    --volume=<repo>:/host_repo:ro`. Opt-in, portable across base images,
    needs no Dockerfile change. Works today as A's lazy cousin.
  Likely shape: A or B for axon_recon + UnitMatchPy + SLAy (frequently
  edited, stable entry points), C as escape hatch for one-off cases.
  Wrapping into `examples/perlmutter_*.sbatch` and the
  `axon-recon-container` lab-server wrapper would be the shipping form.
- **Dependencies / blockers**: decide which siblings actually need this
  (axon_velocity is a PyPI pin and would need to be added as a sibling
  checkout first; mea_analysis is currently blocked inside the container
  via `AXON_RECON_IN_CONTAINER=1` and would need that guard loosened
  intentionally per-run). Caveats to document: `.pyc` cache pollution
  between host Python and container Python (use `PYTHONDONTWRITEBYTECODE=1`
  or `:ro` mount), entry-point changes still need rebuild, C-extension
  rebuilds still need rebuild.
- **See also**: discussed 2026-05-13 after the first containerized
  `axon-recon status` run. `containers/axon-recon/rebuild_shifter.sh`
  remains the canonical rebuild path; this entry is the fast-path
  alternative for the inner dev loop.


### Bombcell tuning for young / sparse cultures
- **Status**: idea
- **Tags**: spikesort, bombcell, parameters
- **Problem**: At DIV 6 on the Media_Density_T5_02182026_AR dataset,
  bombcell labels every unit `noise` or `mua` even when there is real
  activity. Concrete observation 2026-05-13: dataset 1 wells —
  - well000 (1 good / 134 mua / 302 noise out of 438) — barely passes
  - well003 (0 good / 4 mua / 59 noise out of 63) — all-noise, SLAy skips
  - well005 (0 good / 5 mua / 530 noise out of 539) — all-noise with
    1.5M spikes and 539 sorted clusters present, still 0 pass.
  Mature cultures (DIV 26+) pass bombcell normally; the rejection is
  specific to early-DIV.
- **Approach sketch**: Either bombcell's default parameter set is too
  strict for the waveform amplitudes / spike counts / refractory
  violation rates seen in young cultures, or specific parameters
  (`minWvDuration`, `minSomaticAmplitude`, `maxRPVviolations`,
  `minSpatialDecaySlope`, `minPresenceRatio`, etc.) need to scale with
  DIV / SNR / firing rate. Two paths:
  1. **Manual sweep**: pick one DIV 6 well known to have real units
     (visually verifiable from waveform plots), sweep bombcell params
     to find a setting that passes a reasonable fraction. Capture
     the parameter set in `dev/debug_NERSC/debug.runtime.yml` (and
     `debug_local/`) under `stages.spikesort.phases.bombcell_label.parameters`.
  2. **DIV-conditional parameter sets**: declare presets that
     apply per-DIV bucket (`< 10 DIV`, `10-21 DIV`, `>= 21 DIV`).
     Requires runtime-config plumbing for conditional param maps.
- **Dependencies / blockers**: none for path (1); path (2) requires
  decision on conditional-config syntax for stage parameters.
- **See also**: discussed 2026-05-13 after the SLAy
  `n_samples=0` crash on dataset 1 wells 003 + 005. SLAy now
  short-circuits cleanly via the new `no_qualifying_units` skip path
  (sibling commit), so this is no longer a hard crash — but the
  wells still produce zero analysis-eligible units, which is data
  loss until bombcell is retuned.
