# Radivojevic 2023 — paper identification + citation

> `radivojevic_recon_algo_plan` slice 1 deliverable #1.
> Pins the exact target paper for the clean-room re-implementation.

## Citation

**Radivojevic, M., & Rostedt Punga, A.** (2023). Functional imaging of
conduction dynamics in cortical and spinal axons. *eLife*, 12, e86512.

- **DOI**: https://doi.org/10.7554/eLife.86512
- **Published**: 2023-08-22
- **Affiliation (primary)**: MaxWell Biosystems AG, Basel, Switzerland
  (Milos Radivojevic); Uppsala University (Anna Rostedt Punga)
- **Preprint**: bioRxiv 2023.02.28.530461 (Feb 28, 2023)

## Abstract (verbatim)

> Mammalian axons are specialized for transmitting action potentials
> to targets within the central and peripheral nervous system. A
> growing body of evidence suggests that, besides signal conduction,
> axons play essential roles in neural information processing, and
> their malfunctions are common hallmarks of neurodegenerative
> diseases. The technologies available to study axonal function and
> structure integrally limit the comprehension of axon neurobiology.
> High-density microelectrode arrays (HD-MEAs) allow for accessing
> axonal action potentials at high spatiotemporal resolution, but
> provide no insights on axonal morphology. Here, we demonstrate a
> method for electrical visualization of axonal morphologies based on
> extracellular action potentials recorded from cortical and motor
> neurons using HD-MEAs. The method enabled us to reconstruct up to
> 5-cm-long axonal arbors and directly monitor axonal conduction
> across thousands of recording sites. We reconstructed 1.86 m of
> cortical and spinal axons in total and found specific features in
> their structure and function.

## Data + code availability

- **Data**: Dryad — `doi:10.5061/dryad.gxd2547r1`
  - CorticalNeurons.zip (4.11 GB), MotorNeurons.zip (4.48 GB).
  - Spike-sorted + spike-triggered-averaged signals across 26,400
    electrodes (MaxWell HD-MEA1k); `.mat` format.
  - **Posted 2023-08-16**.
- **Source code**: ❌ **NOT publicly available.** Dryad deposit
  contains only data; no GitHub/Zenodo/FigShare repository found in
  the paper, supplementary materials, or via direct web search for
  authors + title (2026-05-21 search session).
- **Adjacent code that IS public** (different algorithms, NOT the
  target):
  - **`axon_velocity`** (Buccino et al. 2022, J Neural Eng 19:026026,
    PMC7612575, DOI 10.1088/1741-2552/ac4dfb) —
    https://github.com/alejoe91/axon_velocity. This is the
    GTRS-based algorithm `axon_recon` ALREADY uses via
    `phases/axon_velocity_gtrs.py`. Different algorithmic approach
    (graph traversal vs Radivojevic's adaptive thresholding +
    skeletonization + multi-step tracking).
  - **`hana`** (Bullmann et al. 2019, Front Cellular Neurosci) —
    https://github.com/tbullmann/hdmea_axon. The 2019 hdmea axon
    pipeline (also Radivojevic is a co-author). Predecessor approach
    using spike-triggered averages — likely shares concepts with
    Radivojevic 2023 but is older.

## Algorithm one-paragraph summary (from paper)

Input: extracellular action potential recordings from HD-MEAs (Maxwell
HD-MEA1k, 26,400 electrodes, ~17.5 μm pitch). Processing has three
stages: (1) **adaptive thresholding** detects signal peaks at varying
amplitudes using progressively relaxed spatial/temporal constraints;
(2) **skeletonization** of electrical images infers conduction
pathways between detected peaks; (3) **multi-step tracking** algorithm
interconnects peaks across consecutive timeframes via direct matching,
skeleton-assisted routing, and indirect prediction based on estimated
conduction velocities. Output: "functional morphologies" — full
reconstruction of axonal trajectories, branching patterns, conduction
timing (action potential propagation across the arbor), and signal
amplitudes mapped across the entire arbor. Reported reconstruction
length: up to 5 cm per arbor; 1.86 m total across all reconstructed
cortical + spinal axons.

## Related papers (lineage)

- **Radivojevic et al. 2016**: "Electrical Identification and Selective
  Microstimulation of Neuronal Compartments Based on Features of
  Extracellular Action Potentials" (Nature Scientific Reports). Earlier
  methods paper; useful background for compartment identification.
- **Radivojevic et al. 2017**: "Tracking individual action potentials
  throughout mammalian axonal arbors" (eLife). Earlier methods paper;
  introduces multi-electrode AP tracking that the 2023 algorithm
  extends.
- **Bullmann et al. 2019** (above) — hdmea_axon pipeline.
- **Buccino et al. 2022** — `axon_velocity` (graph/A*-based).
- **Franke et al. 2015**: "Bayes optimal template matching for spike
  sorting" — adjacent / not directly the target algorithm but useful
  context for the spike-detection priors. Pre-populated by user in the
  literature/ dir.

## Note on `elife-86512-figures-v1.pdf`

The pre-populated PDF in
`~/dev/pkgs/radivojevic2023_recon_algo/literature/` is the
**FIGURES-only** supplement (per its filename suffix `-figures-v1`).
The main text + methods of the paper need to be read separately from
the eLife article page (https://elifesciences.org/articles/86512).
Slice 1's algorithm-summary doc is built from the abstract +
algorithm-overview section visible online; the figures-pdf is useful
for visual reference but doesn't carry the methods detail.

## Stage 1 — verbatim quotes (the ones that bit us)

> ⚠️ Use these quotes — not paraphrases — for any decision about thresholds,
> noise estimation, channel counts, or what a paper number represents.

### Noise estimation (page 6)

> "The thresholds were initially determined based on electrical noise
> observed from the HD-MEA chip. **Electrical noise was estimated from
> voltage traces sampled across an entire array during periods when the
> observed neuron was inactive, and the noise was estimated for each
> neuron separately.**"

Implications:
- Noise is sampled from the **raw voltage trace**, NOT the spike-triggered
  average template, NOT the template-derivative.
- Sampled during **inactive periods** of the neuron (no AP / axonal energy).
- **Per-neuron**, not per-channel.
- Templates are spike-triggered averages of **200 voltage traces per
  electrode** (also page 6) — so noise on the template is ~√200 ≈ 14× lower
  than raw-trace noise. Any MAD-on-template noise estimate is therefore
  ~14× too low compared to the paper's intended threshold scale.

### Adaptive thresholding (page 6)

> "In the first step, a simple planar threshold, set to **9 STD of the
> estimated noise**, was used to detect high-amplitude signal peaks…"
> "In the second step, confined thresholds, set to **2 STD of the estimated
> noise**, were applied locally to detect low-amplitude AP peaks. The
> confined thresholds were positioned on spatial and temporal coordinates
> of previously mapped peaks. The thresholds were confined spatially to
> **50 μm radii** and temporally to periods encompassing **three consecutive
> timeframes** (t_previous, t_current, t_next)…"
> "The third step…the threshold level was further lowered to **1 STD** of
> the estimated noise, and the spatial confinement was broadened to a
> **100 μm radius**."

### Time-derivative grid (page 6)

> "Axonal electrical images were obtained by **averaging 200 voltage traces
> per electrode**. **Time derivatives of averaged traces (μV/μs)** were
> computed for each of the electrodes, and the resulting data were divided
> into **400 consecutive timeframes (with 50 μs inter-frame intervals)**."

### Figure 7A representative-neuron numbers (page 10–12)

> "Representative examples of functional morphologies reconstructed for
> cortical and motor neurons are shown in Figure 7A…"
> "The reconstructed arbor of the cortical axon yielded a total length of
> **27.12 mm**, comprising 101 inter-branching segments and 53 axon
> terminals. **Axonal electrical activity was detected on 7295 electrodes,
> occupying an active area of 2.23 mm².**"
> "Axonal activity was detected on **6663 electrodes**, occupying an active
> area of 2.04 mm²." (motor neuron representative)

**⚠️ 7295 is wrong for tuning Stage-1 cumulative channel counts. Two
independent reasons:**

1. **It's a SINGLE representative cortical example.** The paper does not
   identify which of the 50 cortical neurons it is. Calibrating against it
   on an arbitrary other neuron (e.g. `myNeuron01`) is meaningless. The
   only aggregate stat available across all 50 is "average active area
   1.38±0.08 mm²" (page 12) — NOT a per-neuron channel count.

2. **7295 is the count of electrodes in the RECONSTRUCTED arbor**, not the
   count of channels that passed the three-step Stage-1 thresholding. The
   Stage-1 cumulative set `(9σ ∪ 2σ ∪ 1σ)` is a strict superset of the
   reconstructed arbor — many channels detected at Stage 1 never end up
   linked into a valid trajectory by Stage 2 (direct / skel-assisted /
   indirect). Matching our Stage-1 cumulative count to a post-Stage-2 arbor
   count tunes the wrong quantity entirely.

There is no Stage-1-cumulative channel count published in the paper.
Stop using 7295 as a tuning target.

### Algorithm performance numbers (page 10)

> "We were able to detect **45%, 74%, and 98%** of the actual peaks after
> the first, second, and third steps, respectively. We observed **no false
> peak detections** for thresholds set to 9, 2, and 1 STD of the noise in
> first, second, and third steps, respectively."
> "We could interconnect **70%, 85%, and 91%** of the mapped AP peaks after
> the first, second, and third steps, respectively."

### Aggregate population stats (page 12)

> "Average axonal lengths were **16.73±1.20** and **14.63±0.88 mm** for
> cortical and motor neurons, respectively (p=0.25). The average sizes of
> active areas were **1.38±0.08** and **1.61±0.08 mm²**…"
> "Average axial distances of axonal branching points were 0.27±0.01 and
> 0.43±0.01 mm for cortical and motor neurons, respectively (p<10⁻⁶)."
> "Average numbers of axon terminals were 29.92±1.64 and 17.82±1.03…"

## Errata in our reconstruction work

- **`tune_noise_mult.py --target 7295` was wrong on two axes.** First, 7295
  is a single arbitrary neuron from Fig 7A — not even known to be
  `myNeuron01`. Second, 7295 is a POST-RECONSTRUCTION arbor-channel count,
  not a Stage-1 cumulative pre-link channel count — we were matching our
  Stage-1 set to a quantity that the paper computed AFTER its Stage-2
  trajectory linking. There is no Stage-1-cumulative-channel-count
  published in the paper. NOISE_MULT=1.10312 has no principled basis.
- **Noise estimation method mismatch.** We use per-channel MAD on the
  template derivative; paper uses raw-trace inactive-period noise (page 6).
  Templates are spike-triggered averages of 200 trials so our noise floor
  is ~√200 (≈14×) lower in magnitude. We do not have raw-trace Dryad data
  (Dryad publishes only templates), so we cannot replicate paper-style
  noise on Dryad. NOISE_MULT is a knob used to BRIDGE this gap empirically
  — but it needs a real anchor, and the 7295 anchor was bogus.
- **What we don't have a target for yet.** No published paper number we
  can tune against gives us a defensible Stage-1 channel count on Dryad.
  Path forward: either drop empirical tuning (use NOISE_MULT=1.0 raw MAD),
  or replicate paper-style noise on our OWN unit_598 (where raw recording
  is accessible via spikeinterface) and verify the algorithm there.

## Sources

- [Paper at eLife](https://elifesciences.org/articles/86512)
- [DOI 10.7554/eLife.86512](https://doi.org/10.7554/eLife.86512)
- [Dryad data deposit (doi:10.5061/dryad.gxd2547r1)](https://datadryad.org/stash/dataset/doi:10.5061/dryad.gxd2547r1)
- [bioRxiv preprint (2023.02.28.530461)](https://www.biorxiv.org/content/10.1101/2023.02.28.530461v1)
- [ResearchGate PDF mirror](https://www.researchgate.net/publication/373298894_Functional_imaging_of_conduction_dynamics_in_cortical_and_spinal_axons)
