#!/usr/bin/env bash
set -euo pipefail

axon-reconstructor --help >/dev/null
axon-reconstructor stages --help >/dev/null
python /opt/axon_reconstructor/containers/axon-recon/smoke_imports.py
echo "axon_recon container smoke passed"