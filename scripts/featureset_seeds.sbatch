#!/bin/bash
#SBATCH --job-name=megan_featureset_seeds
#SBATCH --partition=dev
#SBATCH --output=/tmp/featureset_runs/slurm_%A_%a.out
#SBATCH --error=/tmp/featureset_runs/slurm_%A_%a.out
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#
# Additional seeds for the AqSolDB feature-set comparison.
#
# The array index is decoded into (arm, seed) over the cross product of ARMS and the
# seeds given in MEGAN_SEEDS, so the same script serves any number of extra seeds:
#
#   MEGAN_SEEDS="3 4" sbatch --array=0-5%2 --dependency=afterany:7 \
#       scripts/featureset_seeds.sbatch
#
# The "%2" concurrency limit keeps host RAM within the 5 GB margin (each run peaks
# around 8 GB against ~20 GB free). The dependency matters just as much: two arrays each
# limited to %2 would otherwise run four tasks at once and blow that budget.
#
# Note that tasks routinely exit 137. That is the post-run visualisation being
# OOM-killed, not a training failure: vgd_torch__megan.py tracks one matplotlib figure
# per epoch for the training animation, so memory grows with EPOCHS and peaks after the
# results are already on disk. Every metric and the diagnostic report are written before
# it, so a 137 run is still a complete, usable result.

set -u
cd /media/ssd/Programming/graph_attention_student
source .venv/bin/activate

ARMS=(full portable portable_strict)
read -r -a SEEDS <<< "${MEGAN_SEEDS:-3 4}"

ARM=${ARMS[$((SLURM_ARRAY_TASK_ID % ${#ARMS[@]}))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / ${#ARMS[@]}))]}
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
