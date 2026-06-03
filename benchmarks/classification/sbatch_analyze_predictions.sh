#!/bin/bash
# Usage:
#   sbatch benchmarks/classification/sbatch_analyze_predictions.sh \
#       <config> <checkpoint> [extra args]
#
# Example:
#   sbatch benchmarks/classification/sbatch_analyze_predictions.sh \
#       configs/classification/cifar10/r18_cutmix.py \
#       work_dirs/r18_cutmix/latest.pth \
#       --output work_dirs/r18_cutmix/predictions.npz

#SBATCH --partition=H100,H100-RP,H100-PCI,A100-80GB,A100-40GB,A100-PCI,RTXA6000,L40S,RTX3090
#SBATCH --job-name="analyze_pred"
#SBATCH --output=/netscratch/nauen/slurm/%x-%j-%N.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=01:00:00

CONTAINER_IMAGE="/netscratch/nauen/images/openmixup_v2.sqsh"
CFG=$1
CHECKPOINT=$2
PY_ARGS=${@:3}

GLOG_vmodule=MemcachedClient=-1 \
  srun \
  --kill-on-bad-exit=1 \
  --container-workdir="$(pwd)" \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/fscratch/$USER:/fscratch/$USER,/ds-sds:/ds-sds:ro,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
  --container-image=${CONTAINER_IMAGE} \
  python -u tools/analyze_predictions.py \
  ${CFG} \
  ${CHECKPOINT} \
  --device cpu \
  ${PY_ARGS}
