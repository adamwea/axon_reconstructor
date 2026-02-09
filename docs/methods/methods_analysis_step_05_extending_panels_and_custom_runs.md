# Analysis Step — Part 5: Extending Panels and Custom Runs

Scope: this document covers where to modify the analysis stage if you want different panels, additional stages, or different unit selection.

Primary code path:
- `axon_reconstructor.pipeline.analysis.runner.analyze_units(...)`

---

## 1. Where panels are defined

Panels are currently defined directly inside `analyze_units(...)` as a list of dictionaries:

- each dict includes a `title` (informational) and a `path`

To add a new panel:

- add a new entry to that list
- update the renderer layout if you need more than 6 panels

---

## 2. Changing unit selection

Unit selection currently defaults to templates `merged_units` because that’s the most stable “canonical unit set”.

Alternatives you might implement:

- discover units from reconstruction output folders
- discover units from waveforms curated selections
- load a curated unit list file if you want analysis to reflect curation

---

## 3. Adding stage-aware placeholders

The analysis renderer already tolerates missing panels.

If you want clearer placeholders (e.g. include stage name or expected path), modify:

- `_try_load_image_rgba(...)` / `_try_load_image_pil(...)`
- placeholder text creation in `_render_unit_grid(...)` / `_render_unit_grid_pil(...)`

---

## 4. Batch execution

For batch runs, prefer passing `unit_limit` and using `force_restart=False` so reruns are incremental.

If you need “clean rebuild” behavior, set `force_restart=True`.
