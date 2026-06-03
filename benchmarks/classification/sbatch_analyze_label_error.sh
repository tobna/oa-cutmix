#!/bin/bash
# Usage:
#   sbatch benchmarks/classification/sbatch_analyze_label_error.sh \
#       <config> [extra args]
#
# Example:
#   sbatch benchmarks/classification/sbatch_analyze_label_error.sh \
#       configs/classification/tiny_imagenet/r18/cutmix_fga/r18_cutmix_fga_abs.py \
#       --output work_dirs/r18_cutmix_fga_abs/label_error.npz \
#       --n-samples 100

#SBATCH --partition=H100,H100-RP,H100-PCI,A100-80GB,A100-40GB,A100-PCI,RTXA6000,L40S,RTX3090
#SBATCH --job-name="label_error"
#SBATCH --output=/netscratch/nauen/slurm/%x-%j-%N.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=02:00:00

CONTAINER_IMAGE="/netscratch/nauen/images/openmixup_v2.sqsh"
CFG=$1
PY_ARGS=${@:2}

GLOG_vmodule=MemcachedClient=-1 \
  srun \
  --kill-on-bad-exit=1 \
  --container-workdir="$(pwd)" \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/fscratch/$USER:/fscratch/$USER,/ds-sds:/ds-sds:ro,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
  --container-image=${CONTAINER_IMAGE} \
  python -u tools/analyze_label_error.py \
  ${CFG} \
  --num-workers 8 \
  ${PY_ARGS}
