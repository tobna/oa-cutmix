#!/usr/bin/env bash

set -e
set -x

PARTITION="A100-40GB,RTXA6000,L40S,A100-80GB,H100-PCI"
CONTAINER_IMAGE="/netscratch/nauen/images/openmixup_v2.sqsh"
CFG=$1
PY_ARGS=${@:2}
GPUS=1 # When changing GPUS, please also change imgs_per_gpu in the config file accordingly to ensure the total batch size is 256.
GPUS_PER_NODE=1
CPUS_PER_TASK=${CPUS_PER_TASK:-24}
SRUN_ARGS=${SRUN_ARGS:-""}

JOB_NAME="$(echo ${CFG%.*} | sed -e 's\.*/\openmixup \g')"

# train
GLOG_vmodule=MemcachedClient=-1 \
  srun -p ${PARTITION} \
  --job-name "${JOB_NAME}" \
  --gres=gpu:${GPUS_PER_NODE} \
  --ntasks=${GPUS} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} \
  --kill-on-bad-exit=1 \
  --mem=64G \
  --container-workdir="$(pwd)" \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/fscratch/$USER:/fscratch/$USER,/ds-sds:/ds-sds:ro,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
  --container-image=${CONTAINER_IMAGE} \
  ${SRUN_ARGS} \
  python -u tools/train.py \
  $CFG \
  --seed 0 --launcher="slurm" ${PY_ARGS}
