#!/usr/bin/env bash
# Continuously monitor CPU utilization across every node in the current
# Slurm allocation. Prints one line per node per sample so you can see
# whether SI parallelism / n_jobs is actually fanning out across cores.
#
# Run from a login node or your salloc shell. Ctrl-C to stop.
#
# Usage:
#   dev/debug_NERSC/watch_cpu_usage.sh                          # busy-core count per node
#   dev/debug_NERSC/watch_cpu_usage.sh aggregate                # %usr per node (one row per sample)
#   SLURM_JOB_ID=12345 dev/debug_NERSC/watch_cpu_usage.sh       # target a specific job
#   INTERVAL=5 dev/debug_NERSC/watch_cpu_usage.sh               # 5s sample period
#
# Output (default "busy-core" mode):
#   0: 10:35:14 126/128 cores busy
#   1: 10:35:14 128/128 cores busy
#   2: 10:35:14 124/128 cores busy
#   3: 10:35:14 127/128 cores busy
#
# Sum across the ranks ≈ total busy cores in the allocation. ~all cores
# busy → SI parallelism is fanning out. Most ranks at low counts means
# only one worker per node is doing real work (n_jobs=1 leak).

set -euo pipefail

MODE="${1:-busy}"
INTERVAL="${INTERVAL:-2}"
JOBID="${SLURM_JOB_ID:-${SLURM_JOBID:-}}"

if [[ -z "${JOBID}" ]]; then
	# Try to auto-discover the user's most recent running job.
	JOBID="$(squeue -u "$USER" -h -o '%i' -t RUNNING 2>/dev/null | head -1 || true)"
	if [[ -z "${JOBID}" ]]; then
		echo "ERROR: no running Slurm job found for $USER. Pass SLURM_JOB_ID=<jobid> explicitly." >&2
		exit 2
	fi
	echo "auto-detected SLURM_JOB_ID=${JOBID}" >&2
fi

# Determine node count from the job spec so the srun fan-out matches.
NODE_COUNT="$(scontrol show job "${JOBID}" -o 2>/dev/null \
	| tr ' ' '\n' | awk -F= '/^NumNodes=/ {print $2; exit}')"
if [[ -z "${NODE_COUNT}" || "${NODE_COUNT}" == "0" ]]; then
	NODE_COUNT=1
fi

echo "watching jobid=${JOBID} nodes=${NODE_COUNT} mode=${MODE} interval=${INTERVAL}s" >&2
echo "press Ctrl-C to stop." >&2
echo >&2

case "${MODE}" in
	busy)
		# Per-node count of "cores with >50% non-idle". Forces 24h time
		# format so awk can parse, and filters on Average: rows which are
		# always emitted by `mpstat -P ALL <interval> 1`.
		srun --jobid="${JOBID}" --overlap -N "${NODE_COUNT}" --ntasks-per-node=1 --label \
			bash -c 'while :; do
				mpstat -P ALL '"${INTERVAL}"' 1 2>/dev/null | awk "
					/^Average:/ && \$2 != \"CPU\" && \$2 != \"all\" {
						tot++
						if ((100 - \$NF) > 50) busy++
					}
					END { printf \"%s %d/%d cores busy\n\", strftime(\"%T\"), busy, tot }
				"
			done'
		;;
	aggregate|agg)
		# %usr/%sys/%idle averaged across all cores per node. Filters
		# mpstat output to just the "all" rows for compactness.
		srun --jobid="${JOBID}" --overlap -N "${NODE_COUNT}" --ntasks-per-node=1 --label \
			bash -c 'export S_TIME_FORMAT=ISO; mpstat '"${INTERVAL}"' | awk "/all/ && !/CPU/ {print}"'
		;;
	*)
		echo "ERROR: unknown mode '${MODE}'. Use 'busy' (default) or 'aggregate'." >&2
		exit 2
		;;
esac
