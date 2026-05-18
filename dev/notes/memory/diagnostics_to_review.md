# Diagnostics pending user review

Visual / tabular diagnostics that Claude produced during slices and that the user should look at before the slice is fully validated. Some things can only be confirmed visually (waveform shapes, template overlays, sort-quality plots, before/after comparison figures); a passing test suite is necessary but not sufficient.

## How this works

- When a slice's verification benefits from a visual or tabular artifact, Claude generates it during the smoke run, saves it under `/pscratch/sd/a/adammwea/dev_outputs/<plan_slice>/diagnostics/`, and adds an entry below before committing.
- The commit message references this file ("see `memory/diagnostics_to_review.md` entry <N>") so the user knows to check it.
- The user reviews when they can — no synchronous blocking unless Claude flagged the entry as a hard gate.
- Once reviewed, the user either marks the entry `✅ approved <date>` or replies with corrective feedback; Claude then updates the entry to `resolved` or addresses the feedback in a follow-up slice.

## Entry format

```
### <N>. <one-line title>
- **Date / commit**: 2026-05-NN / <commit-hash>
- **Plan / slice**: <plan>.<slice-N>
- **Artifact**: `<path under dev_outputs>`
- **What it shows**: <one sentence>
- **What to check**: <one sentence — what makes this PASS vs FAIL>
- **Gate level**: soft | hard
  - soft = nice to confirm, but downstream slices can proceed
  - hard = must be approved before downstream slices touch this surface
- **Status**: pending | ✅ approved <date> | ❌ rejected <date> (followup: <slice>)
```

## When to add an entry vs not

**Add an entry when:**
- The slice's claim depends on visual inspection (e.g. "templates look right", "raster looks clean", "merge plot shows correct candidate pairs").
- A regression check needs a before/after comparison figure.
- A new phase's first real-data run produces a plot that establishes the baseline.
- A bug fix produces visually different output than before — even when tests pass, the user should sanity-check the picture.

**Do NOT add an entry for:**
- Every routine plot the pipeline emits during a normal run. Those are part of the pipeline's regular outputs; the user can browse them anytime via the dashboard or `find`.
- Plots that have automated assertions (numerical counts, hash comparisons) that fully cover the validation. Tests are the validation; the plot is a bonus.
- Logs / text summaries — those go in commit messages, not here.

## Pruning

Keep the list short. When an entry is `approved`, leave it for ~one week so the audit trail survives, then prune. `rejected` entries stay until the followup slice resolves them, then prune.

---

## Entries

(initially empty — Claude appends entries as slices generate diagnostics)
