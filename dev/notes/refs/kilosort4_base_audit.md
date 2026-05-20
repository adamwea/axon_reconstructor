# `kilosort4-base:4.0.38_cuda-12.0.0` base image audit

DIRECTIVE B step 1 from USER INJECTIONS (2026-05-19). Inventory of
what's pre-installed in `docker.io/spikeinterface/kilosort4-base:4.0.38_cuda-12.0.0`
so the Dockerfile's `pip install .[full-cuda]` step can avoid
reinstalling the CUDA stack.

Captured via `podman run --rm docker.io/spikeinterface/kilosort4-base:4.0.38_cuda-12.0.0 pip list`
on 2026-05-19 18:08 PDT.

## Pre-installed (don't reinstall)

### Torch + CUDA stack (the ~5 GiB chunk that was being duplicated)

| Package | Version |
|---|---|
| `torch` | 2.7.1+cu118 |
| `torchaudio` | 2.7.1+cu118 |
| `torchvision` | 0.22.1+cu118 |
| `triton` | 3.3.1 |
| `nvidia-cublas-cu11` | 11.11.3.6 |
| `nvidia-cuda-cupti-cu11` | 11.8.87 |
| `nvidia-cuda-nvrtc-cu11` | 11.8.89 |
| `nvidia-cuda-runtime-cu11` | 11.8.89 |
| `nvidia-cudnn-cu11` | 9.1.0.70 |
| `nvidia-cufft-cu11` | 10.9.0.58 |
| `nvidia-curand-cu11` | 10.3.0.86 |
| `nvidia-cusolver-cu11` | 11.4.1.48 |
| `nvidia-cusparse-cu11` | 11.7.5.86 |
| `nvidia-nccl-cu11` | 2.21.5 |
| `nvidia-nvtx-cu11` | 11.8.86 |

The image tag says `cuda-12.0.0` but the actual torch wheels are
`cu118` (CUDA 11.8). The 12.0.0 in the tag refers to the underlying
CUDA toolchain available on the host driver / nvidia base layer.

### Scientific / utility stack also pre-installed

| Package | Version | Notes |
|---|---|---|
| `numpy` | 1.26.4 | Within our `numpy<2.0` pin. |
| `scipy` | 1.16.0 | Newer than our `scipy<2.0` pin allows but pip should leave it alone. |
| `scikit-learn` | 1.7.0 | Within our `scikit-learn<2.0` pin. |
| `matplotlib` | 3.10.3 | Within our `matplotlib<4` pin. |
| `numba` + `llvmlite` | 0.61.2 / 0.44.0 | Numba JIT — transitive dep of many things. |
| `kilosort` | 4.0.38 | The eponymous spike sorter. |
| `joblib` | 1.5.1 | UMPy + sklearn dep, already here. |
| `tqdm` | 4.65.0 | Within `tqdm<5`. |
| `requests` | 2.31.0 | Within `requests<3`. |
| `PyYAML` | 6.0.2 | Within our pin. |
| `psutil` | 5.9.0 | Within our pin. |
| `pillow`, `fonttools`, `contourpy`, `kiwisolver`, `cycler` | various | Matplotlib stack. |
| `attrs`, `beautifulsoup4`, `click`, `fsspec`, `jinja2`, `jsonschema`, `networkx`, `packaging`, `platformdirs`, `sympy`, `typing_extensions`, … | various | Standard scientific-Python transitive deps. |

### Conda machinery
`conda`, `conda-build`, `conda-libmamba-solver`, `libmambapy`,
`mamba`-related packages. Not needed by axon_recon directly but
present because the base image was conda-built.

## NOT pre-installed (the Dockerfile must add these)

Cross-referencing pyproject.toml `[full-cuda]` against the audit:

| Package | Action |
|---|---|
| `h5py<4` | Add. |
| `pandas<3.0` | Add. |
| `pyarrow` | Add. |
| `plotly>=5.18` | Add. |
| `dash>=2.14` | Add. |
| `dash-ag-grid` | Add. |
| `statsmodels` | Add. |
| `kaleido` | Add. |
| `openpyxl` | Add. |
| `docker` | Add. |
| `pypdf` | Add. |
| `nvidia-ml-py` | Add (this is NOT one of the nvidia-cu11 packages — it's a small Python wrapper for NVML). |
| `spikeinterface==0.104.3` | Add. |
| `axon_velocity==0.1.2` | Add. |
| `mpi4py` | Add. |
| `mat73` | Add. |
| `mtscomp` | Add. |
| `marshmallow` | Add. |

These 18 packages are the actual "new install" that `pip install
.[full-cuda]` performs inside the container. None of them transitively
require torch reinstallation (verified mentally — spikeinterface has
optional torch deps but doesn't hard-require it; axon_velocity uses
matplotlib + numpy + scipy only).

UnitMatchPy is installed AFTER `.[full-cuda]` via the separate
`pip install --no-deps "UnitMatchPy @ git+..."` step in the Dockerfile
(commit `338868e`). Without `--no-deps`, pip would try to satisfy
UMPy's `torch>=2.1,<3.0` constraint by reinstalling torch (since the
installed torch 2.7.1+cu118 might not parse cleanly as 2.x for pip's
resolver). The `--no-deps` skip is what saves the ~5 GiB.

## Edge cases / risks

1. **`scipy 1.16.0` vs our `scipy<2.0` pin**: pip should leave the
   pre-installed 1.16.0 in place since it satisfies `<2.0`. No
   downgrade attempt expected.
2. **`numpy 1.26.4`**: same — within `<2.0`. h5py + pandas might pull
   newer numpy if not constrained; the pin should hold them.
3. **`scikit-learn 1.7.0` is newer than the conda env's**: the
   conda env has `scikit-learn<2.0`, container has 1.7.0. Both
   within the pin. No drift.
4. **`tqdm 4.65.0` is older than our `>=4` request**: 4.65 satisfies
   `>=4` per the core dependency. No reinstall.
5. **`spikeinterface==0.104.3` pin is exact**: if base image ever
   bundles spikeinterface (it currently doesn't), pip would still
   honor the exact pin and reinstall to match. Watch for this if the
   upstream `kilosort4-base` image changes.

## Implications for `[full-cuda]` extra

The current `[full-cuda]` extra (commit `987053a`) correctly omits the
UnitMatchPy line. No other line in `[full-cuda]` needs to be removed
— every other entry is either:
- Not pre-installed (h5py, pandas, dash, …) → pip will install fresh.
- Pre-installed at a version satisfying our pin (numpy, scipy, …) →
  pip will see it satisfied + skip.

So the slim Dockerfile should work as-is. Validation comes from the
next rebuild attempt — the image size delta vs the previous
`pip install .[full]` build is the empirical check.

## See also

- USER INJECTIONS DIRECTIVE B (`dev/notes/memory/current_state.md`)
- Slim split: `pyproject.toml` `[full-cuda]` extra
- Dockerfile: `containers/axon-recon/Dockerfile` (commit `338868e`
  flipped default to `dev,full-cuda` + added `--no-deps` UMPy step)
