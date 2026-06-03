#!/bin/bash

#SBATCH --partition="A100-40GB,A100-PCI,A100-80GB"
#SBATCH --job-name="mix_speed"
#SBATCH --output=/netscratch/nauen/slurm/%x-%j-%N.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G

CONTAINER_IMAGE="/netscratch/nauen/images/openmixup_v2.sqsh"
PY_ARGS="$@"

srun \
  --kill-on-bad-exit=1 \
  --container-workdir="$(pwd)" \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/fscratch/$USER:/fscratch/$USER,/ds-sds:/ds-sds:ro,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
  --container-image=${CONTAINER_IMAGE} \
  python -u tools/benchmark_mix_speed.py \
  ${PY_ARGS}
