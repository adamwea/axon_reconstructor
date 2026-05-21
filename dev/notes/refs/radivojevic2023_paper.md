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

## Sources

- [Paper at eLife](https://elifesciences.org/articles/86512)
- [DOI 10.7554/eLife.86512](https://doi.org/10.7554/eLife.86512)
- [Dryad data deposit (doi:10.5061/dryad.gxd2547r1)](https://datadryad.org/stash/dataset/doi:10.5061/dryad.gxd2547r1)
- [bioRxiv preprint (2023.02.28.530461)](https://www.biorxiv.org/content/10.1101/2023.02.28.530461v1)
- [ResearchGate PDF mirror](https://www.researchgate.net/publication/373298894_Functional_imaging_of_conduction_dynamics_in_cortical_and_spinal_axons)
