#!/bin/bash
# Same content as train_plainnets_a100.sbatch but with .sh extension for sites
# that prefer it. This is still a Slurm sbatch script and should be submitted
# with `sbatch` (extension doesn't affect sbatch), e.g.
#   sbatch --array=0-4 slurm/train_plainnets_a100.sh

#!/bin/bash
#SBATCH --job-name=plainnet_train
#SBATCH --partition=gpu-a100-q
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/plainnet_%A_%a.out
#SBATCH --error=slurm_logs/plainnet_%A_%a.err
# NOTE: this script is intended to be submitted as an array job, e.g.
#   sbatch --array=0-4 slurm/train_plainnets_a100.sh
# or use the provided helper `slurm/submit_plainnets.sh` which computes the
# correct array range and calls sbatch for you.

set -euo pipefail

# ensure logs directory exists
mkdir -p slurm_logs

# If your cluster uses module system or conda, uncomment and customize below.
# source /etc/profile.d/modules.sh
# module load cuda/11.7
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate my_torch_env

module load pytorch-extra-py39-cuda11.8-gcc11

# -----------------------------------------------------------------------------
# Configuration list: each entry is the CLI arguments to pass to `python3 train.py`.
# Edit/add entries to run more/other experiments. Keep each entry quoted.
# The order here defines SLURM_ARRAY_TASK_ID mapping (0..N-1).
# -----------------------------------------------------------------------------
CONFIGS=(
"--arch plainnet20 --lr 0.1  --batch-size 128 --epochs 200 --save-dir results/plainnet20"
"--arch plainnet32 --lr 0.1  --batch-size 128 --epochs 200 --save-dir results/plainnet32"
"--arch plainnet44 --lr 0.1  --batch-size 128 --epochs 200 --save-dir results/plainnet44"
"--arch plainnet56 --lr 0.1  --batch-size 128 --epochs 200 --save-dir results/plainnet56"
"--arch plainnet110 --lr 0.01 --batch-size 64  --epochs 200 --save-dir results/plainnet110"
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
NUM_CONFIGS=${#CONFIGS[@]}

if [ "$TASK_ID" -ge "$NUM_CONFIGS" ]; then
  echo "SLURM_ARRAY_TASK_ID ($TASK_ID) out of range (0..$((NUM_CONFIGS-1)))"
  exit 1
fi

ARGS=${CONFIGS[$TASK_ID]}

# Print environment info (useful for debugging)
echo "Job: $SLURM_JOB_ID  ArrayTask: $TASK_ID  Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Running: python3 train.py $ARGS"

# Run the training command
python3 train.py $ARGS

# End of script
