#!/bin/bash
#SBATCH --job-name=robust_train
#SBATCH --partition=gpu-a100-q
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/robust_%A_%a.out
#SBATCH --error=slurm_logs/robust_%A_%a.err
# NOTE: this script is intended to be submitted as an array job, e.g.
#   sbatch --array=0-4 slurm/robust_train_a100.sh

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
# Configuration list: each entry is the CLI arguments to pass to `python3 robust_train.py`.
# Edit/add entries to run more/other experiments. Keep each entry quoted.
# The order here defines SLURM_ARRAY_TASK_ID mapping (0..N-1).
# -----------------------------------------------------------------------------
CONFIGS=(
"--arch plainnet20 --lr 0.1  --batch-size 128 --epochs 200 --save-dir results/robust_plainnet20"
"--arch plainnet32 --lr 0.1  --batch-size 128 --epochs 200 --save-dir results/robust_plainnet32"
"--arch plainnet44 --lr 0.1  --batch-size 128 --epochs 200 --save-dir results/robust_plainnet44"
"--arch plainnet56 --lr 0.1  --batch-size 128 --epochs 200 --save-dir results/robust_plainnet56"
"--arch plainnet110 --lr 0.1 --batch-size 64  --epochs 200 --save-dir results/robust_plainnet110"
"--arch resnet20 --lr 0.1  --batch-size 128 --epochs 200 --save-dir results/robust_resnet20"
"--arch resnet32 --lr 0.1  --batch-size 128 --epochs 200 --save-dir results/robust_resnet32"
"--arch resnet44 --lr 0.1  --batch-size 128 --epochs 200 --save-dir results/robust_resnet44"
"--arch resnet56 --lr 0.1  --batch-size 128 --epochs 200 --save-dir results/robust_resnet56"
"--arch resnet110 --lr 0.1 --batch-size 64  --epochs 200 --save-dir results/robust_resnet110"
"--arch wideresnet28_10 --lr 0.1 --batch-size 128 --epochs 200 --save-dir results/robust_wideresnet28_10"
"--arch wideresnet28_2 --lr 0.1  --batch-size 128 --epochs 200 --save-dir results/robust_wideresnet28_2"
"--arch wideresnet40_2 --lr 0.1  --batch-size 128 --epochs 200 --save-dir results/robust_wideresnet40_2"
"--arch wideresnet16_8 --lr 0.1  --batch-size 128 --epochs 200 --save-dir results/robust_wideresnet16_8"
"--arch vit32 --lr 0.1  --batch-size 128 --epochs 200 --save-dir results/robust_vit32"
"--arch vit56 --lr 0.1  --batch-size 128 --epochs 200 --save-dir results/robust_vit56"
"--arch vit110 --lr 0.1  --batch-size 128 --epochs 200 --save-dir results/robust_vit110"
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
echo "Running: python3 robust_train.py $ARGS"

# Run the training command
python3 robust_train.py $ARGS
