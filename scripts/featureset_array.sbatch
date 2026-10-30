#!/bin/bash
#SBATCH --job-name=megan_featureset
#SBATCH --partition=dev
#SBATCH --output=/tmp/featureset_runs/slurm_%A_%a.out
#SBATCH --error=/tmp/featureset_runs/slurm_%A_%a.out
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#
# The AqSolDB feature-set comparison, as a Slurm array.
#
# Submitted with a "%2" concurrency limit so only two tasks run at once: each needs
# ~5.8 GB of host RAM against ~20 GB free, which keeps the requested 5 GB margin. The
# GPU is not the constraint - a run uses only ~0.4 GB of VRAM - and this node has no
# GPU gres configured, so every task simply shares the one card.
#
# Slurm rather than a backgrounded shell script on purpose: a nohup'd process is still
# in the launching session's process group, so when that session ends the whole tree is
# killed. That is exactly what happened on the first attempt - both runs were killed
# mid-plot at the same instant despite having completed successfully. A batch job is
# owned by slurmd and outlives the session.
#
# Submit with:
#   sbatch --array=0-6%2 scripts/featureset_array.sbatch

set -u
cd /media/ssd/Programming/graph_attention_student
source .venv/bin/activate

# One (arm, seed) pair per array index. Seed 0 of "full" and "portable" already
# completed in the first attempt and are not repeated here.
COMBOS=(
  "portable_strict 0"
  "full            1"
  "portable        1"
  "portable_strict 1"
  "full            2"
  "portable        2"
  "portable_strict 2"
)

read -r ARM SEED <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"
TAG="${ARM}_seed${SEED}"

echo "[$(date +%H:%M:%S)] task ${SLURM_ARRAY_TASK_ID}: ${TAG} on $(hostname)"

export MEGAN_MARGIN_SCALE=12.0
export MEGAN_EPOCHS=150
export MEGAN_SPLIT_SEED=$SEED
export OMP_NUM_THREADS=4

python "graph_attention_student/experiments/vgd_torch__megan__aqsoldb__featureset__${ARM}.py" \
  > "/tmp/featureset_runs/${TAG}.log" 2>&1
STATUS=$?

echo "[$(date +%H:%M:%S)] task ${SLURM_ARRAY_TASK_ID}: ${TAG} exited ${STATUS}"
exit $STATUS
