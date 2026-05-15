#!/usr/bin/env bash
# Rename every well's analysis_outputs/ and recon_outputs/ to
# *_with_bombcell so they're preserved for comparison before we
# force-restart recon (and re-run analysis) without bombcell labels.
#
# DRY-RUN by default. Pass --commit to actually rename.
#
# Behavior:
#   - For each well dir under $ANALYSIS_ROOT (max depth 7) that contains
#     a directory named analysis_outputs/ or recon_outputs/, rename to
#     <name>_with_bombcell.
#   - If the target name already exists, the operation is SKIPPED for
#     that well and a warning is printed (no clobber).
#   - find_outputs_to_skip honors a pre-existing _with_bombcell folder
#     so re-runs of this script are idempotent.
#
# Usage:
#   dev/debug_NERSC/jobs/sans_bombcell_rerun/rename_outputs.sh
#   dev/debug_NERSC/jobs/sans_bombcell_rerun/rename_outputs.sh --commit

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
echo "mode:          $([ "$COMMIT" -eq 1 ] && echo COMMIT || echo "dry-run (use --commit to actually rename)")"
echo

renamed=0
skipped_collision=0
total=0
for name in analysis_outputs recon_outputs; do
	while IFS= read -r -d '' src; do
		total=$((total + 1))
		dst="${src}_with_bombcell"
		if [ -e "$dst" ]; then
			echo "SKIP collision: $dst already exists -- $src untouched"
			skipped_collision=$((skipped_collision + 1))
			continue
		fi
		if [ "$COMMIT" -eq 1 ]; then
			echo "mv $src -> $dst"
			mv "$src" "$dst"
			renamed=$((renamed + 1))
		else
			echo "would mv $src -> $dst"
		fi
	done < <(find "$ANALYSIS_ROOT" -maxdepth 7 -type d -name "$name" -print0 2>/dev/null)
done

echo
echo "summary:"
echo "  candidates inspected: $total"
if [ "$COMMIT" -eq 1 ]; then
	echo "  renamed:              $renamed"
fi
echo "  collisions skipped:   $skipped_collision"
