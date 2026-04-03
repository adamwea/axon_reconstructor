# Analysis Step — Part 2: Unit Discovery and the Panel Contract

Scope: this document covers how analysis discovers unit ids and what “panels” it expects to be able to render.

Primary code path:
- `axon_reconstructor.pipeline.analysis.runner.analyze_units(...)`

---

## 1. Unit discovery

If `inputs.unit_ids` is provided:

- analysis renders exactly those unit ids.

Otherwise, analysis discovers units from a stable contract in templates outputs:

- `<well_out_dir>/stg4_templates_outputs/merged_units/`
- each unit directory is expected to be named `unit_<id>`

This is intentionally templates-driven because templates stage defines the “canonical unit set” handed to reconstruction.

If `inputs.unit_limit` is provided, analysis truncates the discovered unit list.

---

## 2. Panel contract (current grid layout)

For each unit id, analysis constructs a fixed list of panels (6 total) and attempts to load them as images.

Current mapping:

1. Templates topo footprint:
   - `<well>/stg4_templates_outputs/topo_unit_footprints/unit_<id>.png`

2. Templates zoomed merged contributing footprint:
   - `<well>/stg4_templates_outputs/footprints_zoomed/unit_<id>_merged_contributing_footprint_ptp_linear_zoom.png`

3. Templates propagation plot:
   - `<well>/stg4_templates_outputs/propagation_plots/unit_<id>.png`

4. Reconstruction branch velocities:
   - `<well>/stg5_reconstruction_outputs/by_unit/unit_<id>/branch_velocities.png`

5. Reconstruction raw branches (zoom):
   - `<well>/stg5_reconstruction_outputs/by_unit/unit_<id>/branches_raw_zoom.png`

6. Reconstruction graph heuristics:
   - `<well>/stg5_reconstruction_outputs/by_unit/unit_<id>/graph_heuristics.png`

---

## 3. Missing panels are non-fatal

If a panel file is missing or cannot be rendered, analysis inserts a placeholder in that grid slot.

This is deliberate:

- it’s better to produce a grid showing what’s available than to fail the entire analysis run because one plot is missing.
