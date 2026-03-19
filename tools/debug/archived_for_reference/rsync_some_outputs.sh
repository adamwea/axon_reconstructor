# Resolve symlinked paths to real filesystem paths
UNIT_DIR="$(readlink -f "/home/adamm/dev/symlinks/local_RBS_data/outputs/Media_Density_T3_07012025_AR/250728/M07137/AxonTracking/000225/well001/reconstruction_outputs/by_unit/unit_112")"
WELL_DIR="${UNIT_DIR%/reconstruction_outputs/by_unit/*}"

SRC1="$(readlink -f "$WELL_DIR/preprocess_outputs")"
SRC2="$(readlink -f "$WELL_DIR/spikesorting_outputs")"
DST1="$(readlink -m "$WELL_DIR/stg1_preprocess_outputs")"
DST2="$(readlink -m "$WELL_DIR/stg2_spikesorting_outputs")"

echo "SRC1=$SRC1"; echo "SRC2=$SRC2"; echo "DST1=$DST1"; echo "DST2=$DST2"

# Safety: sources must exist; only delete destinations
test -d "$SRC1" && test -d "$SRC2"
rm -rf -- "$DST1" "$DST2"
mkdir -p "$DST1" "$DST2"

# Loud rsync output (so you can see activity)
rsync -a --checksum --delete --human-readable --info=progress2,stats2 "$SRC1"/ "$DST1"/
rsync -a --checksum --delete --human-readable --info=progress2,stats2 "$SRC2"/ "$DST2"/

# Verify (0 means identical)
echo "stg1_diffs=$(rsync -a --checksum --delete --dry-run --itemize-changes "$SRC1"/ "$DST1"/ | wc -l)"
echo "stg2_diffs=$(rsync -a --checksum --delete --dry-run --itemize-changes "$SRC2"/ "$DST2"/ | wc -l)"