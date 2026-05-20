# Open questions

TBD decisions awaiting user input or empirical data. Each entry has a clear resolution criterion. When resolved, move the conclusion to `current_state.md`, `guardrails/`, or a plan; delete the entry from here.

## Awaiting empirical data (deferred until plan reaches the relevant slice)

- **Two-halves split granularity**: temporal midpoint is what UMPy expects. Could split finer for more same-neuron pairs per unit, but UMPy shape is hardcoded to `(..., 2)`. Decide after `unitlink` v1 results.
- **Per-chip match threshold tuning**: default `match_threshold: 0.5` from UMPy may be too permissive for HD-MEA. Add per-group calibration in `unitlink` v3? Wait for v1 + v2 empirical data.
- **Network-scan inclusion as default**: decide after `unitmatch_phase_plan.md` slice 7's measurement of marginal gain.
- **Both network-scan types in unitmatch v2**: v2 picks ONE type (lean clustered variant). The sparse variant may join in v3 if marginal gain measurement justifies it.
- **DeepUnitMatch HD-MEA training**: `unitlink` v2 wrapper supports it; training a HD-MEA model is its own project. Defer.
- **`init` / `cleanup` stage scope**: v1 = one phase each (`copy_src_to_scratch` / `wipe_src_scratch`); grow organically. Stages disabled by default for now but must work.
- **`concat_binary` resource class** after consolidation: keep spikesort-side budget. Plan §6 §3.
- **bombcell / SLAy code deletion timing**: never delete from spikesort code, just disable. User: "I think in the future we will only use the recon stage versions if we successfully implement them as I imagine, maybe then we delete them. but for now, just disable them."
- **`plot_raster_threshold` quality fix design**: needs design-doc-level thinking about colormap / per-segment channel toggling visualization. Defer.
- **Dashboard slice 7 tertiary-grouping UX**: third grouping dropdown for box/bar plots. Two reasonable UX paths:
  - (A) **Faceted small-multiples** — one plot per tertiary value. Scales visually with up to ~6-9 facet values; breaks down for high-cardinality tertiary (e.g. DIV ∈ {6,8,10,...,32}). Easy to read individual sub-plots; harder to compare values across sub-plots.
  - (B) **Nested hierarchical X-axis** — render primary × secondary × tertiary as compound x-tick labels (e.g. `wt | DIV10 | media_a`, `wt | DIV10 | media_b`, …). Scales to higher cardinality; keeps the comparison axis intact; visually busier. Plotly supports this via `category_orders` + multi-level group_col.
  - **Recommendation**: ship (B) first because it's the lower-risk extension of the existing box-plot rendering (slice 6 already renders nested groups for secondary; tertiary is just another dimension to fold into the category sort). Add (A) later as an opt-in `tertiary_mode: facet` knob if users want it. **User input wanted**: confirm (B)-first, or override to (A) if you want small-multiples as the default. **Resolution criterion**: user picks one; loop ships that slice 7.

## Test-suite latent failures (revealed by hygiene passes)

- **`test_progress.py::test_pipeline_progress_skips_tqdm_logging_redirect_for_rich_handler`**: pre-existing TabError fixed in commit `34ef166`, which unblocked module collection but exposed a latent assertion failure: `progress._bar.fp` is None after the with-block (line 111). Root cause: `__exit__` resets `self._bar = None` (`execution/progress.py:94`), so the assertion at test_progress.py:111 — `assert writes == [("", progress._bar.fp)]` — references a None attribute. **Resolution criterion**: capture the bar's `fp` inside the with-block (e.g. via a fixture or by reading `_FakeTqdm`'s captured bar), or rewrite the assertion to match the post-exit None state. Low priority — only one of 4 tests in the module; doesn't gate any plan.

## Per-slice empirical findings

- **resources.profiles elimination — per-srun-flag fallback when slot is missing**: when `--cpus-per-task` is passed on the command line, does the resolver use that directly, or compute from `os.sched_getaffinity(0)`? Both are reasonable; pick during the implementation slice. See `trackers/tech_debt.md` §"Minimize / eliminate `resources.profiles`".

- **Auto-restart with chip-well-group phase scope** (`unitmatch` phase): the auto-restart logic walks `phase_sequence` per target. For chip-well groups, the "target" is a group, not a (dataset, well) pair. Verify the logic generalizes when the unitmatch phase lands.
