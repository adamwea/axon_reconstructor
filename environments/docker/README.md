# Kilosort4 / SpikeInterface Docker image (for NERSC Shifter)

This folder contains a minimal Dockerfile to build a Shifter-ready image for Perlmutter GPU interactive jobs.

- Base image: `mandarmp/benshalomlab_spikesorter:latest`
- Adds: Maxwell HDF5 plugin (`HDF5_PLUGIN_PATH=/opt/maxwell_hdf5_plugin/Linux`)

## Build (local PC)

From the repo root:

- Choose a new tag (don’t reuse old tags; Shifter caching is painful):
  - `TAG=v8`

- Build **Linux/amd64** (important if you’re on Apple Silicon):
  - `docker build --platform=linux/amd64 -t adammwea/axonkilo_docker:$TAG -f environments/docker/Dockerfile environments`

- Sanity check:
  - `docker run --rm adammwea/axonkilo_docker:$TAG python3 -c "import h5py; print('h5py', h5py.__version__)"`

## Push (local PC)

- `docker login`
- `docker push adammwea/axonkilo_docker:$TAG`

## Pull into Shifter (NERSC login node)

- `shifterimg pull docker:adammwea/axonkilo_docker:$TAG`
- `shifterimg images | grep axonkilo`

Then allocate an interactive GPU node:

- `salloc -A <YOUR_ALLOCATION> -q interactive -C gpu -t 04:00:00 --nodes=1 --gpus=1 --image=adammwea/axonkilo_docker:$TAG`

(You can also use the image directly via `--image` if it’s already in the cache.)
