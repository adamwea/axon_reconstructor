# Analysis Step — Part 3: Rendering Backend and Dependencies

Scope: this document explains how the summary grids are rendered and what optional dependencies improve output quality.

Primary code path:
- `axon_reconstructor.pipeline.analysis.runner._render_unit_grid(...)`

---

## 1. Matplotlib-first rendering

Analysis prefers matplotlib when available:

- it forces `Agg` backend
- it renders the grid to both PNG and PDF

This provides consistent layout and higher-quality PDF output.

---

## 2. Pillow fallback

If matplotlib is unavailable, analysis falls back to a Pillow-based montage renderer:

- creates a large RGBA canvas
- places each panel image into fixed slots
- writes PNG and a PDF (via Pillow conversion)

This fallback keeps the stage functional in minimal environments.

---

## 3. SVG support via `cairosvg`

Some upstream stages may emit SVG panels.

Analysis supports SVG panels *only if* `cairosvg` is installed:

- SVG is converted to PNG bytes via `cairosvg.svg2png(...)`
- then loaded with Pillow

If `cairosvg` is missing, SVG panels render as placeholders.

---

## 4. Minimal dependency summary

To render everything well:

- Required for analysis rendering:
  - `pillow`

- Recommended for better output:
  - `matplotlib`
  - `numpy`

- Optional for SVG:
  - `cairosvg`
