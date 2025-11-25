#!/bin/bash
# Helper to submit the plainnet array job with the correct --array range
# Usage: ./slurm/submit_plainnets.sh
# This script computes the number of configured experiments in the sbatch file
# and submits a job array that runs each configuration on one GPU.

set -euo pipefail

# Mirror of the CONFIGS defined in slurm/train_plainnets_a100.sbatch.
# Keep this list synchronized with the CONFIGS array in that file.
CONFIGS=(
  plainnet20
  plainnet32
  plainnet44
  plainnet56
  plainnet110
)

N=${#CONFIGS[@]}
if [ $N -eq 0 ]; then
  echo "No configs to submit. Edit this script or slurm/train_plainnets_a100.sbatch to add experiments."
  exit 1
fi

ARRAY=0-$((N-1))

echo "Submitting array job with range $ARRAY (total $N tasks)"
# Ensure the local logs directory exists so Slurm can write stdout/err files
# into the repository directory (some sites do not create parent directories).
mkdir -p slurm_logs

# submit the .sh variant (some sites prefer .sh extension). The script is
# still an sbatch script and should be submitted with `sbatch`.
sbatch --array=$ARRAY slurm/train_plainnets_a100.sh

echo "Submitted. Use 'squeue -u $(whoami)' to check status."
