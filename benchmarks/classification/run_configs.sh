#!/bin/bash

# Script to launch multiple configs with sbatch
# Usage: ./run_configs.sh [-p PARTITION] [-g NUM_GPUS] config1.py config2.py ...
#   -p PARTITION   Specify partition (default: uses sbatch script default)
#   -g NUM_GPUS    Number of GPUs per node: 1 or 4 (default: 1)

PARTITION=""
NUM_GPUS=1

while getopts "p:g:" opt; do
  case $opt in
  p)
    PARTITION="$OPTARG"
    ;;
  g)
    NUM_GPUS="$OPTARG"
    ;;
  \?)
    echo "Invalid option: -$OPTARG" >&2
    exit 1
    ;;
  esac
done
shift $((OPTIND - 1))

if [ $# -eq 0 ]; then
  echo "Usage: $0 [-p PARTITION] [-g NUM_GPUS] <config1> <config2> ..."
  echo "Example: $0 -p A100-40GB configs/classification/cifar100/my_runs/resnet/r50_mixup.py"
  echo "Example: $0 -g 4 configs/classification/cifar100/my_runs/resnet/r50_mixup.py"
  echo ""
  echo "Partitions available: A100-40GB, RTXA6000, L40S, A100-80GB, H100-PCI"
  echo "GPU options: 1 (single GPU), 4 (4 GPUs with sbatch_dist_train_4gpu.sh)"
  exit 1
fi

if [ "$NUM_GPUS" != "1" ] && [ "$NUM_GPUS" != "4" ]; then
  echo "Error: NUM_GPUS must be 1 or 4, got: $NUM_GPUS"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$NUM_GPUS" = "4" ]; then
  SBATCH_SCRIPT="${SCRIPT_DIR}/sbatch_dist_train_4gpu.sh"
else
  SBATCH_SCRIPT="${SCRIPT_DIR}/sbatch_train_single_gpu.sh"
fi

if [ ! -f "$SBATCH_SCRIPT" ]; then
  echo "Error: sbatch script not found at $SBATCH_SCRIPT"
  exit 1
fi

for CFG in "$@"; do
  if [ ! -f "$CFG" ]; then
    echo "Warning: Config file not found: $CFG, skipping..."
    continue
  fi

  CONFIG_NAME=$(basename "$CFG" .py)

  # Extract dataset name from path (e.g., configs/classification/cifar100/... -> cifar100)
  DATASET=$(echo "$CFG" | cut -d'/' -f3)
  JOB_NAME="openmixup ${DATASET} ${CONFIG_NAME}"

  echo "Launching: $CFG"
  echo "Job name: $JOB_NAME"

  if [ -n "$PARTITION" ]; then
    sbatch -J "$JOB_NAME" --partition="$PARTITION" "$SBATCH_SCRIPT" "$CFG"
  else
    sbatch -J "$JOB_NAME" "$SBATCH_SCRIPT" "$CFG"
  fi
done

echo "All jobs submitted!"
