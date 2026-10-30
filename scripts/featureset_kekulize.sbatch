#!/bin/bash
#SBATCH --job-name=megan_kekulize
#SBATCH --partition=dev
#SBATCH --output=/tmp/featureset_runs/slurm_%A_%a.out
#SBATCH --error=/tmp/featureset_runs/slurm_%A_%a.out
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#
# Does aromatic perception carry signal for the GNN?
#
# The "portable" arm is re-run with every molecule kekulized and its aromatic flags
# cleared, so aromatic rings become alternating single/double bonds and nothing is marked
# aromatic. Everything else - the 38 node columns, the model, the seeded splits - is
# unchanged, and the non-kekulized runs from array 7/14 serve as the paired baseline.
#
# Submit with:
#   sbatch --array=0-4%2 scripts/featureset_kekulize.sbatch
#
# Tasks routinely exit 137: the post-run visualisation is OOM-killed after the results
# are already written, so a 137 run is still complete. See featureset_seeds.sbatch.

set -u
cd /media/ssd/Programming/graph_attention_student
source .venv/bin/activate

SEED=$SLURM_ARRAY_TASK_ID
TAG="portable_kekulized_seed${SEED}"

echo "[$(date +%H:%M:%S)] task ${SLURM_ARRAY_TASK_ID}: ${TAG} on $(hostname)"

export MEGAN_MARGIN_SCALE=12.0
export MEGAN_EPOCHS=150
export MEGAN_SPLIT_SEED=$SEED
export OMP_NUM_THREADS=4

python graph_attention_student/experiments/vgd_torch__megan__aqsoldb__featureset__portable_kekulized.py \
  > "/tmp/featureset_runs/${TAG}.log" 2>&1
STATUS=$?

echo "[$(date +%H:%M:%S)] task ${SLURM_ARRAY_TASK_ID}: ${TAG} exited ${STATUS}"
exit $STATUS
