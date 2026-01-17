# Base image maintained by the lab for SpikeInterface + Kilosort4 runtime.
# Source: https://hub.docker.com/r/mandarmp/benshalomlab_spikesorter
FROM mandarmp/benshalomlab_spikesorter:latest

# Keep image changes minimal: we mainly add the Maxwell HDF5 plugin.
# Some base images run as root already; if not, this still works when root.

RUN python3 -m pip install --upgrade pip setuptools wheel

# Maxwell HDF5 plugin (needed to read some MaxWell-compressed .h5 recordings)
# Build with context set to `environments/` so this path exists.
COPY maxwell_hdf5_plugin /opt/maxwell_hdf5_plugin
ENV HDF5_PLUGIN_PATH=/opt/maxwell_hdf5_plugin/Linux

WORKDIR /workspace
CMD ["bash"]
