# Radivojevic 2023 — algorithm summary + input/output spec

> `radivojevic_recon_algo_plan` slice 1 deliverable #2.
> Source-of-truth doc for the clean-room re-implementation.
> Paper citation in [`radivojevic2023_paper.md`](./radivojevic2023_paper.md).

## TL;DR

**Algorithm name**: Functional axonal morphology reconstruction from
extracellular HD-MEA recordings (the paper does not give it a
pithy nickname — Buccino's "axon_velocity" name comes from the 2022
algorithm; this one is distinct).

**Three-stage pipeline**:
1. **Adaptive thresholding** — detect signal peaks across the
   spatiotemporal recording.
2. **Skeletonization** — infer conduction pathways between detected
   peaks.
3. **Multi-step tracking** — interconnect peaks across consecutive
   timeframes using direct matching, skeleton-assisted routing, and
   velocity-based indirect prediction.

**Output**: per-axon reconstruction comprising trajectory, branching
pattern, conduction timing (action-potential propagation across the
arbor), and signal amplitudes mapped across the entire arbor.

## Inputs

### Data the algorithm consumes

| Field | Source / shape | Notes |
|-------|---------------|-------|
| Spike-triggered average (STA) per unit | `electrode_template[n_channels, n_samples]` — the spike-aligned extracellular voltage trace at every electrode | Same shape as `axon_velocity_gtrs`'s `templates.npy` — should be drop-in compatible. |
| Electrode locations (x, y) | `electrode_locations[n_channels, 2]` (μm) | Same as `axon_velocity_gtrs`. |
| Sampling rate | scalar (Hz) | Paper uses 20 kHz (MEA1k). |
| (optional) raw STA per spike (not just average) | for the tracking step's velocity refinement | The 2023 paper's multi-step tracking may use the SPREAD of individual spike arrival times across electrodes to refine velocity estimates — this is what distinguishes it from the 2022 (Buccino) graph-based approach which only consumes the AVERAGED template. **TBD from full methods read.** |

### Hardware assumptions

- HD-MEA with ~17.5 μm electrode pitch (MaxWell HD-MEA1k spec).
- 26,400 simultaneous recording channels (or compatible subset).
- 20 kHz sampling rate.

Paper notes: algorithm performance "degrades" at 35+ μm pitch. Our
data (MaxWell MEA1k, ~17.5 μm in axon-tracking mode) is in spec.

### Preprocessing requirements (upstream of this algorithm)

- Spike sorting must have happened (the input is the
  spike-triggered average, so spike times must be available).
- Templates should be post-quality-control: only "clean" units fed
  in. Same QC bar as `axon_velocity` (the user already applies bombcell
  + SLAy QC in the spikesort stage).

## Algorithm steps (from the eLife abstract + algorithm overview)

### Stage 1: Adaptive thresholding

- Detects signal peaks (negative voltage excursions) in the STA across
  the spatiotemporal recording.
- "Adaptive" = progressively relaxed spatial/temporal constraints —
  i.e. the threshold is not a single static value; it's relaxed
  iteratively to capture lower-amplitude peaks while preserving spatial
  coherence.
- **Hyperparams (TBD from full methods)**: initial threshold (in σ
  units of the noise?), relaxation rate, spatial coherence radius
  (μm), temporal coherence window (ms).

### Stage 2: Skeletonization

- Builds "electrical images" — 2D maps of the recording at each
  timeframe.
- Applies image skeletonization (morphological thinning) to infer
  conduction pathways between detected peaks.
- **Hyperparams (TBD)**: image resolution, morphological structuring
  element, smoothing kernel.

### Stage 3: Multi-step tracking

- Iterates over consecutive timeframes (in order of action-potential
  propagation).
- Three sub-strategies for connecting peaks frame-to-frame:
  1. **Direct matching** — if peak in frame T+1 is within radius D of a
     peak in frame T, link them.
  2. **Skeleton-assisted routing** — if no direct match, follow the
     skeletonized pathway from stage 2 to find the most likely
     connection.
  3. **Indirect prediction** — if no skeleton path either, use the
     estimated local conduction velocity to predict where the peak
     should be in T+1 and search in a wider radius.
- **Hyperparams (TBD)**: direct-match radius (μm), skeleton-traversal
  cost, velocity-extrapolation window (ms), max-jump distance (μm).

## Outputs

### Per-unit (per-arbor) outputs

| Field | Shape / type | Description |
|-------|-------------|-------------|
| `trajectory` | list of (x, y, t) points (μm, ms) | Ordered axonal pathway from soma outward through branches. |
| `branches` | tree structure of trajectory segments | Each branch has a parent + bifurcation point. |
| `conduction_timing` | per-electrode peak time (ms) | When the AP arrived at each electrode along the arbor. |
| `signal_amplitudes` | per-electrode peak amplitude (μV) | Voltage trace amplitude along the arbor. |
| `propagation_velocity` | per-branch m/s | Local AP velocity for each branch segment. |
| `total_length_um` | scalar | Sum of all branch lengths. |

### Per-experiment aggregate outputs (paper highlights)

- Number of arbors reconstructed.
- Mean / median arbor length, branching factor.
- Velocity distribution across arbors.
- Cortical-vs-spinal comparison metrics (from the paper).

## Input compat with `axon_velocity_gtrs`

| `axon_velocity_gtrs` consumes (from recon stage) | Radivojevic 2023 needs | Compat status |
|---|---|---|
| `merged_template.npy` `(n_channels, n_samples)` | electrode_template_array, same shape | ✅ **drop-in** |
| `merged_channel_locations.npy` `(n_channels, 2)` | electrode_locations | ✅ **drop-in** |
| sampling_rate (from analyzer manifest) | sampling_rate | ✅ **drop-in** |
| (none — Buccino's algorithm uses just the averaged template) | per-spike STA SPREAD across electrodes (maybe — TBD) | ⚠️ **possible delta** — confirm during slice 3 (impl) |
| (none) | adaptive-threshold hyperparams + skeleton + tracking params | New phase config dataclass needed (slice 4) |

**Implication**: input contract is largely a superset of what
`axon_velocity_gtrs` consumes, so wiring this into the recon stage as
a sibling phase to `axon_velocity_gtrs` should be straightforward at
the input-resolution layer. The ALGORITHM is fundamentally different
(adaptive thresholding + skeletonization + multi-step tracking vs the
graph A* approach Buccino uses).

## Open questions for USER GATE 1

These get logged separately into
[`dev/notes/memory/open_questions.md`](../memory/open_questions.md)
under "Radivojevic slice 1 user-gate review". Resolved answers
update this doc.

1. **Paper identity confirmed?** Title + DOI per
   `radivojevic2023_paper.md` — please verify this is the intended
   target paper.
2. **Clean-room approach?** No public code found via methodical
   search; clean-room re-implementation is the only path. Please
   confirm OR identify a code source I missed.
3. **Does the algorithm need RAW spike-triggered-average per-spike,
   or just the averaged template?** This affects whether we can
   piggyback on existing `analyzers` phase output (which only stores
   the averaged template) or need a separate raw-STA cache phase.
   **HIGH-IMPACT** — please confirm by reading the methods section
   when you can.
4. **Input compat map look right?** Per the table above,
   `axon_velocity_gtrs`-compatible templates should drop in. Any
   missing input?
5. **Hyperparameter defaults**: paper figures will show concrete values
   — when we have access to the full methods text, those become the
   slice-3 defaults. Until then, I'll start slice 3 with reasonable
   guesses + flag them for tuning in slice 6.
6. **Algorithm-name preference?** Paper doesn't pick a nickname.
   Options for the sibling-package + phase name:
   - `radivojevic_recon` (current — matches the sibling-package dir
     name; descriptive of WHO not WHAT).
   - `electrical_imaging_recon` (descriptive of WHAT — matches the
     paper's framing of "electrical visualization").
   - `axon_skeleton_recon` (descriptive of the KEY STEP).
   Pick one to lock in for slice 4 onward.

## Slice 1 status: DONE (research-only)

- ✅ Paper identified (`radivojevic2023_paper.md`).
- ✅ Public code searched — none found for the 2023 algorithm; clean-room
  required.
- ✅ This algorithm-summary doc written (high-level; depth limited by
  inability to read the full methods PDF locally — `pdftotext` /
  `pdftoppm` not available in this env; full methods would need browser
  access to the eLife article).
- ✅ Input compat map vs `axon_velocity_gtrs` written.
- ⏭️ USER GATE 1 questions logged to `open_questions.md`.

Per current_state.md "PRE-OVERNIGHT CLEARANCES" item #2, this slice is
pre-approved to continue into slice 2 (sibling-package scaffold) WITHOUT
pausing — slice 3 (core algorithm impl) DOES wait for user to review
these slice-1 questions.
