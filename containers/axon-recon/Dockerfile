# syntax=docker/dockerfile:1
FROM spikeinterface/kilosort4-base:4.0.38_cuda-12.0.0

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG AXON_RECON_INSTALL_EXTRAS=""
ARG AXON_RECON_RUNTIME_SPEC="numpy<2.0 h5py<4 tqdm<5 requests<3 psutil matplotlib<4 pandas<3.0 openpyxl scipy<2.0 pyyaml docker pypdf scikit-learn<2.0 nvidia-ml-py"
ARG AXON_VELOCITY_SPEC="axon_velocity==0.1.2"
ARG SPIKEINTERFACE_SPEC="spikeinterface==0.103.2"
ARG UNITMATCH_SPEC=""
ARG UNITMATCH_RUNTIME_SPEC="joblib mat73 mtscomp"
ARG UNITMATCH_INSTALL_ARGS="--no-deps"
ARG SLAY_SPEC=""
ARG SLAY_RUNTIME_SPEC="marshmallow"
ARG SLAY_INSTALL_ARGS="--no-deps"
ARG MPI4PY_SPEC="mpi4py"
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

WORKDIR /opt/axon_reconstructor

COPY . /opt/axon_reconstructor

RUN rm -rf /tmp/axon-recon-cache \
  && python containers/axon-recon/install_maxwell_hdf5_plugin.py --url "${MAXWELL_HDF5_PLUGIN_URL}" --plugin-dir "${MAXWELL_HDF5_PLUGIN_DIR}" \
  && python -m pip install --upgrade pip setuptools wheel \
  && if [[ -n "${AXON_RECON_INSTALL_EXTRAS}" ]]; then \
  python -m pip install ".[${AXON_RECON_INSTALL_EXTRAS}]"; \
  else \
  python -m pip install .; \
  fi \
  && if [[ -n "${AXON_RECON_RUNTIME_SPEC}" ]]; then python -m pip install ${AXON_RECON_RUNTIME_SPEC}; fi \
  && if [[ -n "${AXON_VELOCITY_SPEC}" ]]; then python -m pip install "${AXON_VELOCITY_SPEC}"; fi \
  && if [[ -n "${SPIKEINTERFACE_SPEC}" ]]; then python -m pip install "${SPIKEINTERFACE_SPEC}"; fi \
  && if [[ -n "${UNITMATCH_SPEC}" ]]; then \
  if [[ -n "${UNITMATCH_RUNTIME_SPEC}" ]]; then python -m pip install ${UNITMATCH_RUNTIME_SPEC}; fi; \
  python -m pip install ${UNITMATCH_INSTALL_ARGS} "${UNITMATCH_SPEC}"; \
  fi \
  && if [[ -n "${SLAY_SPEC}" ]]; then \
  if [[ -n "${SLAY_RUNTIME_SPEC}" ]]; then python -m pip install ${SLAY_RUNTIME_SPEC}; fi; \
  python -m pip install ${SLAY_INSTALL_ARGS} "${SLAY_SPEC}"; \
  fi \
  && if [[ -n "${MPI4PY_SPEC}" ]]; then python -m pip install "${MPI4PY_SPEC}"; fi \
  && chmod +x /opt/axon_reconstructor/containers/axon-recon/entrypoint.sh \
  && chmod +x /opt/axon_reconstructor/containers/axon-recon/smoke_cli.sh \
  && ln -sf /opt/axon_reconstructor/containers/axon-recon/entrypoint.sh /usr/local/bin/axon-recon-entrypoint \
  && ln -sf /opt/axon_reconstructor/containers/axon-recon/smoke_cli.sh /usr/local/bin/axon-recon-smoke-cli \
  && rm -rf /root/.cache/pip /tmp/axon-recon-cache /tmp/pip-* /tmp/build-*

ENTRYPOINT ["axon-recon-entrypoint"]
CMD ["--help"]