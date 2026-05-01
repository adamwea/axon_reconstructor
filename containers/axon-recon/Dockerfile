# syntax=docker/dockerfile:1
FROM spikeinterface/kilosort4-base:4.0.6_cuda-12.0.0

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG AXON_RECON_INSTALL_EXTRAS=""
ARG SPIKEINTERFACE_SPEC=""
ARG UNITMATCH_SPEC=""
ARG SLAY_SPEC=""
ARG MPI4PY_SPEC="mpi4py"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XDG_CACHE_HOME=/tmp/axon-recon-cache/xdg \
    MPLCONFIGDIR=/tmp/axon-recon-cache/matplotlib \
    NUMBA_CACHE_DIR=/tmp/axon-recon-cache/numba \
    PYTHONPYCACHEPREFIX=/tmp/axon-recon-cache/pycache

WORKDIR /opt/axon_reconstructor

COPY . /opt/axon_reconstructor

RUN mkdir -p /tmp/axon-recon-cache \
    && python -m pip install --upgrade pip setuptools wheel \
    && if [[ -n "${AXON_RECON_INSTALL_EXTRAS}" ]]; then \
        python -m pip install ".[${AXON_RECON_INSTALL_EXTRAS}]"; \
    else \
        python -m pip install .; \
    fi \
    && if [[ -n "${SPIKEINTERFACE_SPEC}" ]]; then python -m pip install "${SPIKEINTERFACE_SPEC}"; fi \
    && if [[ -n "${UNITMATCH_SPEC}" ]]; then python -m pip install "${UNITMATCH_SPEC}"; fi \
    && if [[ -n "${SLAY_SPEC}" ]]; then python -m pip install "${SLAY_SPEC}"; fi \
    && python -m pip install pyyaml docker pypdf \
    && if [[ -n "${MPI4PY_SPEC}" ]]; then python -m pip install "${MPI4PY_SPEC}"; fi \
    && chmod +x /opt/axon_reconstructor/containers/axon-recon/entrypoint.sh \
    && chmod +x /opt/axon_reconstructor/containers/axon-recon/smoke_cli.sh \
    && ln -sf /opt/axon_reconstructor/containers/axon-recon/entrypoint.sh /usr/local/bin/axon-recon-entrypoint \
    && ln -sf /opt/axon_reconstructor/containers/axon-recon/smoke_cli.sh /usr/local/bin/axon-recon-smoke-cli \
    && rm -rf /root/.cache/pip /tmp/pip-* /tmp/build-*

ENTRYPOINT ["axon-recon-entrypoint"]
CMD ["--help"]