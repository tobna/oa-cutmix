#!/bin/bash

#SBATCH --partition="H100,H200,B200,H200-PCI,H100-PCI"
#SBATCH --job-name="sam3 detect"
#SBATCH --output=/netscratch/nauen/slurm/%x-%j-%N.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

CONTAINER_IMAGE="/netscratch/nauen/images/image_captioning_v8.sqsh"

srun \
  --export="ALL,TQDM_DISABLE=1" \
  --kill-on-bad-exit=1 \
  --container-workdir="$PWD" \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/fscratch/$USER:/fscratch/$USER,/ds-sds:/ds-sds:ro,/ds:/ds:ro,"$PWD":"$PWD" \
  --container-image=${CONTAINER_IMAGE} \
  python -u tools/prepare_data/sam3_detect_imagenet.py "$@"
