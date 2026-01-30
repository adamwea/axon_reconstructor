# Pipeline diagrams (Mermaid)

This folder contains slide-ready Mermaid diagrams as `.mmd` files, plus a helper script to render them to SVG/PNG.

## Files

- `pipeline_overview.mmd`
- `raw_preprocessing.mmd`
- `spikesorting.mmd`
- `waveforms.mmd`
- `templates.mmd`
- `reconstruction.mmd`

Rendered outputs are written to:

- `rendered/` (SVG + PNG)

## Render locally (recommended)

### 1) Install Mermaid CLI (`mmdc`)

If you don’t already have `mmdc`:

#### WSL / Conda-friendly install (recommended)

Install Node.js inside your Linux/conda environment so `node` exists in WSL:

- `conda install -c conda-forge nodejs`

Then install Mermaid CLI:

- `npm install -g @mermaid-js/mermaid-cli`

#### Generic install

- Install Node.js (via your preferred method)
- Then:

  `npm install -g @mermaid-js/mermaid-cli`

Verify:

- `mmdc --version`

### 2) Render diagrams

From repo root:

- `bash docs/diagrams/render.sh`

This creates:

- `docs/diagrams/rendered/*.svg`
- `docs/diagrams/rendered/*.png`

## View in VS Code

- Open any `.mmd` file
- Use a Mermaid preview extension, or paste into https://mermaid.live

## Notes

- The `.mmd` files are intentionally “presentation-ish”: minimal text, left-to-right flow, and stable filenames.
- If you want specific styling (colors, font sizes, dark mode), we can add a `mermaid-config.json` and wire it into the render script.

### WSL gotcha

If `npm` resolves to something like `/mnt/c/.../npm` and `mmdc` resolves to `/mnt/c/.../mmdc`, you’re using the Windows Node/NPM from inside WSL.
That often fails with `exec: node: not found` when rendering from WSL.

The fix is to install Node inside WSL (e.g. `conda install -c conda-forge nodejs`) and reinstall Mermaid CLI inside WSL.
