#!/bin/bash

#SBATCH --partition="A100-40GB,RTXA6000,L40S,A100-80GB,H100-PCI,A100-PCI,H100,H100-RP"
#SBATCH --job-name="openmixup"
#SBATCH --output=/netscratch/nauen/slurm/%x-%j-%N.out
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=512G

CONTAINER_IMAGE="/netscratch/nauen/images/openmixup_v2.sqsh"
CFG=$1
PY_ARGS=${@:2}
GPUS_PER_NODE=1

# train
GLOG_vmodule=MemcachedClient=-1 \
  srun \
  --kill-on-bad-exit=1 \
  --container-workdir="$(pwd)" \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/fscratch/$USER:/fscratch/$USER,/ds-sds:/ds-sds:ro,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
  --container-image=${CONTAINER_IMAGE} \
  python -u tools/train.py \
  $CFG \
  --seed 0 --launcher="slurm" ${PY_ARGS}
