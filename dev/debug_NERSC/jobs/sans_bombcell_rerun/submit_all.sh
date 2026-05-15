#!/usr/bin/env bash
# Submit the sans-bombcell rerun jobs. Three flat lists of "<idx>:<well>"
# entries below; submission iterates each and chains afterok deps as needed.
#
# Stage templates live next to this script under ./sbatches/:
#   preproc.sbatch        (CPU shared,  4 h)
#   spikesort_full.sbatch (GPU shared,  6 h, 1 GPU)
#   merge_SLAy.sbatch     (GPU shared,  6 h, 1 GPU)
#   recon.sbatch          (CPU shared,  6 h, --force-restart)
#
# Modes:
#   --dry-run   print what would be submitted (default)
#   --submit    actually submit
#
# NERSC policy notes:
#   * GPU jobs use `-q shared -C gpu -G 1` (NOT `-q gpu_shared`) and the _g
#     account. The named `gpu_shared` qos exists in `sacctmgr` but is rejected
#     by NERSC's server-side policy filter; the shared qos auto-routes to
#     shared_gpu_ss11 when -C gpu + -G N + -A *_g is set.
#   * cpus_per_task=32 is the minimum NERSC allows for 1 GPU on shared (= 16
#     physical cores via SMT2). The srun line inside each GPU sbatch uses
#     --threads-per-core=1 so axon-recon still sees a 16-physical envelope,
#     matching the perlmutter_gpu YAML profile.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
SBATCH_DIR="$HERE/sbatches"
LOG_DIR="$HERE/logs"
mkdir -p "$LOG_DIR"

export REPO_ROOT

# === plan: well buckets ====================================================
# full  = preproc (CPU)  -> spikesort_full (GPU) -> recon (CPU, force)
# slay  = merge_SLAy (GPU) -> recon (CPU, force)
# recon = recon (CPU, force)  [no dep]

FULL_WELLS=(
	"4:well000" "4:well001" "4:well002" "4:well003" "4:well004" "4:well005"
)

SLAY_WELLS=(
	"1:well002"
	"3:well000"
	"5:well003"
	"6:well000" "6:well002" "6:well004" "6:well005"
	"7:well002"
	"8:well000" "8:well001" "8:well002" "8:well005"
	"10:well000" "10:well001" "10:well002" "10:well003" "10:well005"
	"11:well001" "11:well002" "11:well003" "11:well005"
	"12:well000" "12:well001" "12:well002"
	"13:well000" "13:well001" "13:well002" "13:well003" "13:well004" "13:well005"
	"14:well000" "14:well001" "14:well002" "14:well003" "14:well004" "14:well005"
)

RECON_ONLY_WELLS=(
	"0:well000" "0:well001" "0:well002" "0:well003" "0:well004" "0:well005"
	"1:well000" "1:well001" "1:well003" "1:well004" "1:well005"
	"2:well000" "2:well001" "2:well002" "2:well003" "2:well004" "2:well005"
	"3:well001" "3:well002" "3:well003" "3:well004" "3:well005"
	"5:well000" "5:well001" "5:well002" "5:well004" "5:well005"
	"6:well001" "6:well003"
	"7:well000" "7:well001" "7:well003" "7:well004" "7:well005"
	"8:well003" "8:well004"
	"11:well000" "11:well004"
)

MODE="dry-run"
for arg in "$@"; do
	case "$arg" in
		--dry-run) MODE="dry-run" ;;
		--submit)  MODE="submit" ;;
		--help|-h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
		*) echo "Unknown arg: $arg" >&2; exit 2 ;;
	esac
done

submit_one() {
	# submit_one <sbatch_template> <idx> <well> <kind> [<dep_jobid>]
	local sbf="$1" idx="$2" well="$3" kind="$4" dep="${5:-}"
	local out="$LOG_DIR/${kind}_ds${idx}_${well}_%j.out"
	local jobname="${kind}_ds${idx}_${well}"
	local dep_flag=()
	[ -n "$dep" ] && dep_flag=(--dependency="afterok:${dep}")
	if [ "$MODE" = "submit" ]; then
		sbatch --parsable \
			--job-name="$jobname" \
			--output="$out" \
			--export="ALL,IDX=${idx},WELL=${well}" \
			"${dep_flag[@]}" \
			"$sbf"
	else
		echo "DRY: sbatch --parsable --job-name=$jobname --output=$out --export=ALL,IDX=${idx},WELL=${well} ${dep_flag[*]} $sbf" >&2
		echo "DRY-${jobname}"
	fi
}

echo "=== mode: $MODE ==="
echo

# 1) full-chain wells: preproc -> spikesort -> recon
for entry in "${FULL_WELLS[@]}"; do
	idx="${entry%%:*}"; well="${entry##*:}"
	j1=$(submit_one "$SBATCH_DIR/preproc.sbatch"        "$idx" "$well" preproc)
	j2=$(submit_one "$SBATCH_DIR/spikesort_full.sbatch" "$idx" "$well" spikesort "$j1")
	j3=$(submit_one "$SBATCH_DIR/recon.sbatch"          "$idx" "$well" recon     "$j2")
	echo "ds${idx}/${well} full        preproc=$j1 spikesort=$j2 recon=$j3"
done

# 2) slay wells: merge_SLAy -> recon
for entry in "${SLAY_WELLS[@]}"; do
	idx="${entry%%:*}"; well="${entry##*:}"
	j1=$(submit_one "$SBATCH_DIR/merge_SLAy.sbatch" "$idx" "$well" merge_SLAy)
	j2=$(submit_one "$SBATCH_DIR/recon.sbatch"      "$idx" "$well" recon "$j1")
	echo "ds${idx}/${well} slay        merge_SLAy=$j1 recon=$j2"
done

# 3) recon-only wells: just recon (no dep)
for entry in "${RECON_ONLY_WELLS[@]}"; do
	idx="${entry%%:*}"; well="${entry##*:}"
	j1=$(submit_one "$SBATCH_DIR/recon.sbatch" "$idx" "$well" recon)
	echo "ds${idx}/${well} recon-only  recon=$j1"
done

echo
total=$(( ${#FULL_WELLS[@]} * 3 + ${#SLAY_WELLS[@]} * 2 + ${#RECON_ONLY_WELLS[@]} ))
echo "total submissions ($MODE): $total"
[ "$MODE" = "dry-run" ] && echo "re-run with --submit to actually submit"
