#!/bin/bash
# Helper to submit the robust train array job with the correct --array range
# Usage: ./slurm/submit_robust.sh

set -euo pipefail

# Count the number of configurations in the sbatch file
# We look for lines containing "--arch" which are part of the CONFIGS array
# This assumes the format in slurm/robust_train_a100.sh
N=$(grep -c '"--arch' slurm/robust_train_a100.sh)

if [ "$N" -eq 0 ]; then
  echo "No configs found in slurm/robust_train_a100.sh"
  exit 1
fi

ARRAY=0-$((N-1))

echo "Found $N configurations."
echo "Submitting array job with range $ARRAY"

# Ensure logs dir exists
mkdir -p slurm_logs

sbatch --array=$ARRAY slurm/robust_train_a100.sh

echo "Submitted. Use 'squeue -u $(whoami)' to check status."
