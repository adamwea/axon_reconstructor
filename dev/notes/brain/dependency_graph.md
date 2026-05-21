# Dependency graph — phases, modules, plans, contracts

The persistent DAG. Lets the loop answer: "I touched X — what now needs re-verification?" Without this, contract drift goes silent and downstream code breaks invisibly.

**Reading rule for the loop**: when ANY slice ships that changes a node's contract (output shape, function signature, file layout), the loop MUST update the affected node's edges + push the dependent nodes onto a re-verify list.

**Writing rule**: loop UPDATES this file as a side-effect of shipping slices. Append edges + node entries; modify existing entries when a contract changes. Mark superseded edges struck-through but don't delete (history matters for audits).

---

## STATUS: SKELETON ONLY — POPULATED BY PHASE-ZERO MISSION (Z1)

The actual graph isn't built yet. Loop's first allowed-during-pause work is Z1 from `brain/objectives.md`: read existing code + plans, produce the DAG. This file is the destination for that output.

Until Z1 runs, the loop should treat the graph as "absent" — meaning ANY contract change requires a full conservative re-verification of all dependent code (i.e. don't ship anything that breaks the trusted-output baselines).

---

## Schema (for Z1's output)

When Z1 lands, each node entry looks like:

```
### <node-id> — <human label>
- **Kind**: phase | module | plan | sibling-package | YAML-key | CLI-flag | guardrail
- **Owns**: <list of file paths that implement this node>
- **Produces (contract)**: <interface signature OR output shape OR file layout>
- **Consumes (deps)**: [<list of node-ids this depends on>]
- **Depended-on-by**: [<list of node-ids whose contract assumes this one>]
- **Trusted outputs**: [<list of brain/trusted_outputs.md TR-xxx IDs that anchor this node>]
- **Last contract change**: <commit hash> on <date>
- **Re-verify when**: <one sentence on what kind of change triggers downstream re-verification>
```

Edges are derived from `Consumes` + `Depended-on-by` reciprocally (Z1 validates the bidirectional consistency).

## Junctions to look for (Z1 priorities)

These are the high-coverage nodes Z1 should identify first — certifying them transitively covers a lot of the pipeline:

1. **`merged_template.npy` per-unit layout** — consumed by every recon-stage downstream phase (plot_templates_v2, report_templates, axon_velocity_gtrs, eventually radivojevic_recon). Produced by `build_templates` historically, now also by `kssynth` slice 4 postprocess. SINGLE biggest junction in the recon stage.
2. **`gtr.pkl` / equivalent gtr-result shape** — axon_velocity_gtrs's output; consumed by plot_recons + dashboard. radivojevic_recon must produce something equivalent (per O3 DOD).
3. **Analyzer cache layout** — `recon_outputs/cache/analyzers/segments/*/` — consumed by every recon phase via `_load_templates_phase_analyzers`. Built by `reconstruct.analyzers` phase.
4. **`merged_channel_locations.npy` shape** — paired with `merged_template.npy`; same consumer set.
5. **Phase config dataclass shapes** — `ReconstructionPhasesConfig` aggregates per-phase config; changes here propagate to every phase's parser.
6. **Resource budget contract** — `current_phase_budget()` returns `_budget.cpus_per_task` etc.; changes here (e.g. resources_profiles slice 2 making it env-derived) propagate to every `inputs.n_jobs`-using call site.
7. **Per-stage CLI dispatch** — adding/renaming a phase requires CLI + runtime + YAML alignment.

## Cross-plan dependencies (already known — Z1 confirms + adds)

These are the cross-plan relationships already surfaced manually; Z1 should formalize + audit them:

- `kssynth_recon_integration_plan` slice 5 (enable+retire) → consumes `kssynth` phase + its slice 4 per-unit postprocess; produces the 176-template regression check (anchored in TR-001).
- `unitmatch_phase_plan` slice 5 → CONSUMES `kssynth_recon_integration_plan` slice 5 (needs the kssynth per-unit outputs as input).
- `radivojevic_recon_algo_plan` slice 3 sub-step 9 → CONSUMES `kssynth_recon_integration_plan` slice 3b heavy smoke (needs `merged_template.npy` for unit_0598).
- `chip_layout_phase_split_plan` slice 1 → CONSUMES unitmatch_phase slice 5 (needs match table for cross-session color consistency).
- `parallelism_post_migration_cleanup_plan` slice 2 → CONSUMES `resources_profiles_elimination_plan` slices 0-2 (locked sequencing per `open_questions.md` Finding #1).

## Until Z1 runs

When the pause lifts and the loop ships any slice before Z1 has completed, it must:
- Read this skeleton
- Recognize the graph is absent
- Add a "Z1 not yet run — conservative verification" note to the slice's commit body
- Re-verify TR-001 (the 176-template baseline) after the slice ships

Once Z1 lands, the graph drives the propagation list per-slice and conservative re-verification can relax.
