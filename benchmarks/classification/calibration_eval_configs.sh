#!/bin/bash

# Script to launch multiple configs with sbatch
# Usage: ./calibration_eval_configs.sh [-p PARTITION] [-k KEYS] checkpoint1.pth checkpoint2.pth ...
#   -p PARTITION   Specify partition (default: uses sbatch script default)
#   -k KEYS        Evaluation mode: calibration, fgsm, or pgd (default: calibration)
#
# Checkpoint paths can be:
#   - With timestamped subfolder: work_dirs/classification/cifar100/resnet/2026_03_06_15/checkpoint.pth
#     -> config_name = resnet (parent folder), dataset = cifar100 (2 levels up)
#   - Without timestamped subfolder: work_dirs/classification/cifar100/resnet/checkpoint.pth
#     -> config_name = checkpoint (filename stem), dataset = cifar100 (2 levels up)

PARTITION=""
KEYS="calibration"

while getopts "p:k:" opt; do
  case $opt in
  p)
    PARTITION="$OPTARG"
    ;;
  k)
    KEYS="$OPTARG"
    if [[ "$KEYS" != "calibration" && "$KEYS" != "fgsm" && "$KEYS" != "pgd" ]]; then
      echo "Error: -k must be one of: calibration, fgsm, pgd" >&2
      exit 1
    fi
    ;;
  \?)
    echo "Invalid option: -$OPTARG" >&2
    exit 1
    ;;
  esac
done
shift $((OPTIND - 1))

if [ $# -eq 0 ]; then
  echo "Usage: $0 [-p PARTITION] [-k calibration|fgsm|pgd] <checkpoint1> <checkpoint2> ..."
  echo "Example: $0 -p A100-40GB -k fgsm work_dirs/classification/cifar100/my_runs/resnet/r18_mixups_CE/latest.pth"
  echo ""
  echo "Partitions available: A100-40GB, RTXA6000, L40S, A100-80GB, H100-PCI"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/sbatch_eval_calibration_single_gpu.sh"

if [ ! -f "$SBATCH_SCRIPT" ]; then
  echo "Error: sbatch script not found at $SBATCH_SCRIPT"
  exit 1
fi

# Regex pattern for timestamped folder (e.g., 2026_03_06_15, 20260224_183416)
# Matches folders starting with date-like pattern: digits_..._digits
TIMESTAMP_PATTERN='^\d+_\d+_\d+$'

for CHECKPOINT in "$@"; do
  if [ ! -f "$CHECKPOINT" ]; then
    echo "Warning: Checkpoint file not found: $CHECKPOINT, skipping..."
    continue
  fi

  # Get the directory containing the checkpoint
  CHECKPOINT_DIR=$(dirname "$CHECKPOINT")
  CHECKPOINT_BASENAME=$(basename "$CHECKPOINT")

  # Get parent folder name
  PARENT_FOLDER=$(basename "$CHECKPOINT_DIR")

  # Check if parent folder matches timestamp pattern
  if [[ "$PARENT_FOLDER" =~ $TIMESTAMP_PATTERN ]]; then
    # Timestamp subfolder: config name is the grandparent folder
    CONFIG_NAME=$(basename "$(dirname "$CHECKPOINT_DIR")")
    # Go up from timestamp dir until we hit the folder before "classification"
    CURRENT="$CHECKPOINT_DIR"
    PREV=""
    while [ "$(basename "$CURRENT")" != "classification" ]; do
      PREV="$CURRENT"
      CURRENT=$(dirname "$CURRENT")
    done
    DATASET=$(basename "$PREV")
  else
    # No timestamp: config name is the parent folder name
    CONFIG_NAME="$PARENT_FOLDER"
    # Go up from checkpoint dir until we hit the folder before "classification"
    CURRENT="$CHECKPOINT_DIR"
    PREV=""
    while [ "$(basename "$CURRENT")" != "classification" ]; do
      PREV="$CURRENT"
      CURRENT=$(dirname "$CURRENT")
    done
    DATASET=$(basename "$PREV")
  fi

  JOB_NAME="eval ${KEYS} ${DATASET} ${CONFIG_NAME}"

  echo "Launching: $CHECKPOINT"
  echo "  Config name: $CONFIG_NAME"
  echo "  Dataset: $DATASET"
  echo "  Keys: $KEYS"
  echo "  Job name: $JOB_NAME"

  if [ -n "$PARTITION" ]; then
    sbatch -J "$JOB_NAME" --partition="$PARTITION" "$SBATCH_SCRIPT" "$CHECKPOINT" --keys "$KEYS"
  else
    sbatch -J "$JOB_NAME" "$SBATCH_SCRIPT" "$CHECKPOINT" --keys "$KEYS"
  fi
done

echo "All jobs submitted!"
