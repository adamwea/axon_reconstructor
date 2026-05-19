# syntax=docker/dockerfile:1
# Fully-qualified registry path so podman/podman-hpc (no default registry)
# resolves to Docker Hub. Docker accepts the qualified form unchanged.
FROM docker.io/spikeinterface/kilosort4-base:4.0.38_cuda-12.0.0

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# env_install_unification slice 6: the per-sibling SPEC ARGs
# (AXON_RECON_RUNTIME_SPEC, AXON_VELOCITY_SPEC, SPIKEINTERFACE_SPEC,
# UNITMATCH_SPEC, UNITMATCH_RUNTIME_SPEC, SLAY_SPEC, SLAY_RUNTIME_SPEC,
# MPI4PY_SPEC) were replaced by `pip install -e .[full]`. pyproject.toml's
# `[full]` extra is now the single source of truth for pip deps.
# Kilosort + CUDA inherit from the base image; NERSC/HPC plumbing
# (openmpi-bin, observability apt packages, Maxwell HDF5 plugin)
# stays here because conda/pyproject can't express it.
ARG AXON_RECON_OBSERVABILITY_APT_PACKAGES="time sysstat procps"
ARG AXON_RECON_MPI_APT_PACKAGES="openmpi-bin libopenmpi-dev"
ARG AXON_RECON_INSTALL_EXTRAS="dev,full"
ARG MAXWELL_HDF5_PLUGIN_URL="https://share.mxwbio.com/d/7f2d1e98a1724a1b8b35/files/?p=%2FLinux%2Flibcompression.so&dl=1"
ARG MAXWELL_HDF5_PLUGIN_DIR="/usr/local/lib/plugin"

ENV PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=1 \
  SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True \
  AXON_RECON_IN_CONTAINER=1 \
  HDF5_PLUGIN_PATH=${MAXWELL_HDF5_PLUGIN_DIR} \
  AXON_RECON_CACHE_ROOT=/tmp/axon-recon-cache \
  HOME=/tmp/axon-recon-cache/home \
  XDG_CACHE_HOME=/tmp/axon-recon-cache/xdg \
  MPLCONFIGDIR=/tmp/axon-recon-cache/matplotlib \
  NUMBA_CACHE_DIR=/tmp/axon-recon-cache/numba \
  PYTHONPYCACHEPREFIX=/tmp/axon-recon-cache/pycache

WORKDIR /opt/axon_recon

# Step 1: install non-pip prerequisites (apt + Maxwell HDF5 plugin
# binary). This layer caches as long as the apt + plugin URLs don't
# change.
COPY containers/axon-recon/install_maxwell_hdf5_plugin.py /opt/axon_recon/containers/axon-recon/install_maxwell_hdf5_plugin.py
RUN rm -rf /tmp/axon-recon-cache \
  && apt-get update \
  && if [ -n "${AXON_RECON_OBSERVABILITY_APT_PACKAGES}" ]; then DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ${AXON_RECON_OBSERVABILITY_APT_PACKAGES}; fi \
  && if [ -n "${AXON_RECON_MPI_APT_PACKAGES}" ]; then DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ${AXON_RECON_MPI_APT_PACKAGES}; fi \
  && rm -rf /var/lib/apt/lists/* \
  && python containers/axon-recon/install_maxwell_hdf5_plugin.py --url "${MAXWELL_HDF5_PLUGIN_URL}" --plugin-dir "${MAXWELL_HDF5_PLUGIN_DIR}" \
  && python -m pip install --upgrade pip setuptools wheel

# Step 2: copy the source tree.
COPY . /opt/axon_recon

# Step 3: install axon_recon plus the requested extras (default
# `dev,full`). pyproject.toml's `[full]` extra now pulls the full
# scientific stack, axon_velocity, spikeinterface, mpi4py, UnitMatchPy
# (git URL @ EnnyvanBeest/UnitMatch subdirectory=UnitMatchPy), and the
# sibling runtime deps (mat73, mtscomp, joblib, marshmallow).
#
# kssynth + unitlink remain Dockerfile-uninstalled (per USER INJECTION
# #4 in current_state.md). They'll be added to `[full]` once their GH
# remotes exist.
RUN if [ -n "${AXON_RECON_INSTALL_EXTRAS}" ]; then \
  python -m pip install ".[${AXON_RECON_INSTALL_EXTRAS}]"; \
  else \
  python -m pip install .; \
  fi \
  && chmod +x /opt/axon_recon/containers/axon-recon/entrypoint.sh \
  && chmod +x /opt/axon_recon/containers/axon-recon/smoke_cli.sh \
  && ln -sf /opt/axon_recon/containers/axon-recon/entrypoint.sh /usr/local/bin/axon-recon-entrypoint \
  && ln -sf /opt/axon_recon/containers/axon-recon/smoke_cli.sh /usr/local/bin/axon-recon-smoke-cli \
  && rm -rf /root/.cache/pip /tmp/axon-recon-cache /tmp/pip-* /tmp/build-*

ENTRYPOINT ["axon-recon-entrypoint"]
CMD ["--help"]
