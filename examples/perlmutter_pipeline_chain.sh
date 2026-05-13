#!/bin/bash
# Submit a dependent spikesort -> reconstruct sbatch chain on Perlmutter.
#
# Flow:
#   1. Scan dev/debug_NERSC/debug.data.yml for datasets that have at least
#      one included well missing the spikesort completion marker
#      (<well>/spikesort_outputs/merge_SLAy/merge_stage_summary.json).
#   2. If any incomplete datasets found: submit perlmutter_spikesort.sbatch
#      targeting just those indices, on 4 GPU nodes (regular QoS, 4h).
#   3. Submit perlmutter_reconstruct.sbatch on 4 CPU nodes (regular QoS, 4h)
#      targeting the reconstruct dataset list (default: 0-8, "all except the
#      last 4 datasets").
#   4. Reconstruct depends on spikesort completing with exit 0 (--dependency=
#      afterok). If no incomplete spikesort, reconstruct goes in with no
#      dependency and runs as soon as the CPU queue picks it up.
#
# Env overrides:
#   DRY_RUN=1          — print what would happen (alloc details, datasets,
#                        srun commands) without calling sbatch.
#   RUNTIME_CFG=...    — path to the runtime yml (default dev/debug_NERSC/debug.runtime.yml).
#   RECONSTRUCT_TARGETS=...  — comma-separated dataset list (default
#                        "0,1,2,3,4,5,6,7,8" = all except last 4).
#
# Examples:
#   DRY_RUN=1 examples/perlmutter_pipeline_chain.sh
#   RECONSTRUCT_TARGETS="0-12" examples/perlmutter_pipeline_chain.sh
#   examples/perlmutter_pipeline_chain.sh                  # real submission

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_CFG="${RUNTIME_CFG:-dev/debug_NERSC/debug.runtime.yml}"
RECONSTRUCT_TARGETS="${RECONSTRUCT_TARGETS:-0,1,2,3,4,5,6,7,8}"
DRY_RUN="${DRY_RUN:-0}"

cd "$REPO_ROOT"

print_section() {
    echo
    echo "=== $1 ==="
}

print_sbatch_directives() {
    local sbatch_file=$1
    grep '^#SBATCH' "$sbatch_file" | sed 's/^/    /'
}

print_srun_block() {
    local sbatch_file=$1
    awk '/^srun /,/^$/{print "    " $0}' "$sbatch_file"
}

# ---------- Step 1: scan for incomplete spikesort ----------
print_section "Spikesort completeness scan"
echo "runtime cfg:           ${RUNTIME_CFG}"
echo "reconstruct targets:   ${RECONSTRUCT_TARGETS}"
echo "dry-run:               ${DRY_RUN}"
echo

# --verbose sends a per-dataset status table to stderr; stdout captures the
# incomplete-index CSV.
INCOMPLETE="$(python3 examples/detect_incomplete_spikesort.py --verbose "$RUNTIME_CFG")"
echo

if [[ -n "${INCOMPLETE}" ]]; then
    echo "Incomplete datasets (CSV): ${INCOMPLETE}"
else
    echo "All included datasets are fully spikesorted; no spikesort sbatch needed."
fi

# ---------- Dry-run output ----------
if [[ "${DRY_RUN}" == "1" ]]; then
    if [[ -n "${INCOMPLETE}" ]]; then
        print_section "SPIKESORT — would submit"
        echo "    sbatch examples/perlmutter_spikesort.sbatch \"${INCOMPLETE}\""
        echo
        echo "  Allocation directives:"
        print_sbatch_directives examples/perlmutter_spikesort.sbatch
        echo
        echo "  Resulting srun (inside the alloc):"
        print_srun_block examples/perlmutter_spikesort.sbatch
        echo
        echo "  Target datasets: ${INCOMPLETE}"
        echo "  (spikesort stage processes ALL included wells of these datasets,"
        echo "   not just the missing ones. Already-complete wells will be re-processed"
        echo "   unless the runner detects existing per-phase artifacts and skips.)"
    fi

    print_section "RECONSTRUCT — would submit"
    if [[ -n "${INCOMPLETE}" ]]; then
        echo "    sbatch --dependency=afterok:<SPIKESORT_JOBID> examples/perlmutter_reconstruct.sbatch \"${RECONSTRUCT_TARGETS}\""
        echo "    (reconstruct waits in (Dependency) state until spikesort exits 0;"
        echo "     auto-cancelled if spikesort fails)"
    else
        echo "    sbatch examples/perlmutter_reconstruct.sbatch \"${RECONSTRUCT_TARGETS}\""
        echo "    (no dependency — spikesort skipped)"
    fi
    echo
    echo "  Allocation directives:"
    print_sbatch_directives examples/perlmutter_reconstruct.sbatch
    echo
    echo "  Resulting srun (inside the alloc):"
    print_srun_block examples/perlmutter_reconstruct.sbatch
    echo
    echo "  Target datasets: ${RECONSTRUCT_TARGETS}"

    print_section "DRY RUN COMPLETE — no jobs submitted"
    echo "Re-run without DRY_RUN=1 to actually submit:"
    echo "    examples/perlmutter_pipeline_chain.sh"
    exit 0
fi

# ---------- Step 2: real submission ----------
SPIKESORT_JOB=""
JOB_DEPS=()
if [[ -n "${INCOMPLETE}" ]]; then
    SPIKESORT_JOB="$(sbatch --parsable examples/perlmutter_spikesort.sbatch "${INCOMPLETE}")"
    echo "submitted spikesort job ${SPIKESORT_JOB} (datasets ${INCOMPLETE})"
    JOB_DEPS=(--dependency="afterok:${SPIKESORT_JOB}")
fi

# ---------- Step 3: submit reconstruct ----------
RECONSTRUCT_JOB="$(sbatch --parsable "${JOB_DEPS[@]}" examples/perlmutter_reconstruct.sbatch "${RECONSTRUCT_TARGETS}")"
echo "submitted reconstruct job ${RECONSTRUCT_JOB} (datasets ${RECONSTRUCT_TARGETS})"

# ---------- Step 4: summary ----------
print_section "Pipeline chain submitted"
if [[ -n "${SPIKESORT_JOB}" ]]; then
    echo "  spikesort:    ${SPIKESORT_JOB}   (datasets ${INCOMPLETE})"
    echo "  reconstruct:  ${RECONSTRUCT_JOB}   (datasets ${RECONSTRUCT_TARGETS}, depends on ${SPIKESORT_JOB})"
else
    echo "  reconstruct:  ${RECONSTRUCT_JOB}   (datasets ${RECONSTRUCT_TARGETS}, no dependencies)"
fi
echo
echo "Monitor with: squeue -u \$USER -o '%.10i %.9P %.2t %.10M %.10L %.20R'"
