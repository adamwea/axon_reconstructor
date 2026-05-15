#!/usr/bin/env bash
# Delete bombcell_label_outputs/ and merge_SLAy/ from every well in the
# analysis tree, so the next pipeline run regenerates them from scratch.
#
# By default this is a DRY RUN — pass --commit to actually delete.
#
# Usage:
#   dev/debug_NERSC/cleanup_bombcell_slay_outputs.sh           # preview
#   dev/debug_NERSC/cleanup_bombcell_slay_outputs.sh --commit  # delete

set -euo pipefail

ANALYSIS_ROOT="${ANALYSIS_ROOT:-/pscratch/sd/a/adammwea/analyzed_data/Media_Density_T5_02182026_AR_axon_analysis_AW/Media_Density_T5_02182026_AR}"

COMMIT=0
for arg in "$@"; do
	case "$arg" in
		--commit) COMMIT=1 ;;
		--help|-h)
			grep '^#' "$0" | sed 's/^# \{0,1\}//'
			exit 0
			;;
		*) echo "Unknown arg: $arg" >&2; exit 2 ;;
	esac
done

echo "analysis root: $ANALYSIS_ROOT"
echo "mode:          $([ "$COMMIT" -eq 1 ] && echo COMMIT || echo "dry-run (use --commit to delete)")"
echo

bombcell_dirs=()
slay_dirs=()
while IFS= read -r -d '' p; do bombcell_dirs+=("$p"); done < <(
	find "$ANALYSIS_ROOT" -maxdepth 7 -type d -name bombcell_label_outputs -print0 2>/dev/null
)
while IFS= read -r -d '' p; do slay_dirs+=("$p"); done < <(
	find "$ANALYSIS_ROOT" -maxdepth 7 -type d -name merge_SLAy -print0 2>/dev/null
)

echo "bombcell_label_outputs dirs found: ${#bombcell_dirs[@]}"
echo "merge_SLAy dirs found:             ${#slay_dirs[@]}"
echo

for dir in "${bombcell_dirs[@]}" "${slay_dirs[@]}"; do
	if [ "$COMMIT" -eq 1 ]; then
		echo "rm -rf $dir"
		rm -rf "$dir"
	else
		echo "would rm -rf $dir"
	fi
done

if [ "$COMMIT" -eq 1 ]; then
	echo
	echo "Done. Now rebuild concat_analyzer + re-run merge_SLAy:"
	echo "  shifter --image=adammwea/axon-recon:pipeline-v2 \\"
	echo "    axon-recon stages spikesort.concat_analyzer \\"
	echo "      --config dev/debug_NERSC/debug.runtime.yml \\"
	echo "      --task-profile perlmutter_cpu --task-backend mpi --force-restart"
	echo "  # then:"
	echo "  shifter --image=adammwea/axon-recon:pipeline-v2 \\"
	echo "    axon-recon stages spikesort.merge_SLAy \\"
	echo "      --config dev/debug_NERSC/debug.runtime.yml \\"
	echo "      --task-profile perlmutter_cpu --task-backend mpi --force-restart"
fi
