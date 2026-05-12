#!/usr/bin/env bash
set -euo pipefail

axon-recon --help >/dev/null
axon-recon stages --help >/dev/null
python /opt/axon_recon/containers/axon-recon/smoke_imports.py
echo "axon_recon container smoke passed"