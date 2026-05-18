#!/bin/bash
# Rerun ds4 full-spikesort (6 wells) + merge_SLAy retry on the 4 OOM'd wells.
# Each srun is --exclusive so slurm queues steps when GPUs are saturated; no
# oversubscription risk. `wait` at the bottom blocks until all 10 finish.
#
# Logs go to a fresh, timestamp-stamped folder per script invocation so each
# rerun is easy to tell apart from previous attempts. The parent script also
# prints a one-line STARTED/FINISHED marker for each srun to stdout, with
# elapsed time + rc on completion, so you can tail the allocation prompt to
# see progress without grepping individual log files.

set -uo pipefail
cd /global/u2/a/adammwea/dev/pkgs/axon_recon

export MPICH_GPU_SUPPORT_ENABLED=1
# Mitigate the CUDA allocator fragmentation seen in ds12_well002's merge_SLAy
# (156 MiB alloc failed with 38.80 GiB pinned by PyTorch). Doesn't help wells
# whose single allocation > GPU capacity — those need slay_params tuning.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fresh log folder per run so attempts don't overlap on disk.
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOGDIR="dev/debug_NERSC/jobs/sans_bombcell_rerun/logs_${RUN_TAG}"
mkdir -p "$LOGDIR"
echo "logs -> $LOGDIR"
echo "slurm job id: ${SLURM_JOB_ID:-<not-in-allocation>}"

# Parent-script lifecycle log of each well srun, mirrored to the terminal.
# Each backgrounded srun runs inside a subshell that prints STARTED before the
# srun call and FINISHED (with rc + elapsed) after; the printf ensures each
# line is emitted atomically so concurrent subshells don't tear each other.
launch_well() {
  local stage="$1"; shift   # e.g. "spikesort" or "merge_SLAy"
  local tag="$1"; shift     # e.g. "ds4:well000"
  local log="$1"; shift     # absolute or repo-relative path to per-srun log
  # remaining args: the full srun command + args
  (
    local start=$(date +%s)
    printf '[%s] STARTED  %-12s %-18s -> %s\n' \
      "$(date '+%H:%M:%S')" "$stage" "$tag" "$log"
    "$@" > "$log" 2>&1
    local rc=$?
    local elapsed=$(( $(date +%s) - start ))
    printf '[%s] FINISHED %-12s %-18s rc=%-3d elapsed=%ds\n' \
      "$(date '+%H:%M:%S')" "$stage" "$tag" "$rc" "$elapsed"
  ) &
}

# -----------------------------------------------------------------------------
# ds4 full spikesort: 260302/M06804/000077 wells 000–005 (6 wells)
# -----------------------------------------------------------------------------
for WELL in well000 well001 well002 well003 well004 well005; do
  LOG="${LOGDIR}/spikesort_ds4_${WELL}_${SLURM_JOB_ID}.out"
  launch_well spikesort "ds4:${WELL}" "$LOG" \
    srun --exclusive -N 1 -n 1 -c 16 --cpu-bind=cores --threads-per-core=1 --gpus=1 --mem=56G \
      shifter axon-recon stages spikesort \
        --config dev/debug_NERSC/debug.runtime.yml \
        --profile perlmutter_gpu \
        --task-backend mpi \
        --cpus-per-task 16 \
        --targets "4:${WELL}" \
        --force-restart
done

# -----------------------------------------------------------------------------
# merge_SLAy retries on the 4 wells that OOM'd previously:
#   ds1:well002  = 260224/M06804/000032/well002 (591 KS units)
#   ds6:well000  = 260305/M06804/000099/well000 (720 KS units)
#   ds6:well002  = 260305/M06804/000099/well002 (849 KS units, host RAM OOM)
#   ds12:well002 = 260319/M06804/000174/well002 (642 KS units)
# ae_chan was halved 300 -> 150 in the yaml; combined with expandable_segments
# above this should clear the smaller cases. ds6:well002 may still hit host
# RAM ceiling — flag if it does.
# -----------------------------------------------------------------------------
for PAIR in 1:well002 6:well000 6:well002 12:well002; do
  IDX="${PAIR%%:*}"
  WELL="${PAIR##*:}"
  LOG="${LOGDIR}/merge_SLAy_ds${IDX}_${WELL}_${SLURM_JOB_ID}.out"
  launch_well merge_SLAy "ds${IDX}:${WELL}" "$LOG" \
    srun --exclusive -N 1 -n 1 -c 16 --cpu-bind=cores --threads-per-core=1 --gpus=1 --mem=56G \
      shifter axon-recon stages spikesort.merge_SLAy \
        --config dev/debug_NERSC/debug.runtime.yml \
        --profile perlmutter_gpu \
        --task-backend mpi \
        --cpus-per-task 16 \
        --targets "$PAIR" \
        --force-restart
done

wait
echo "[$(date '+%H:%M:%S')] all 10 sruns finished. logs in $LOGDIR"
