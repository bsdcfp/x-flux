#!/bin/bash
# Run the docker container with the specified parameters
# WORKSPACE_PATH=$(pwd)
WORKSPACE_PATH="/home/fuping.chu"
C_NAME="fuping-flux-cu124"
FLAGS="-itd --privileged "
# IMAGE_URL="harbor.shopeemobile.com/aip/aip-image-hub/aip-prod/projects/0/pytorch2.0-cu12.1-py3.10-trt8.6:py3.10-cu12.1-pt2.0-trt8.6-vscode1.82.2-d540733753"
IMAGE_URL="harbor.shopeemobile.com/aip/aip-image-hub/aip-prod/projects/0/pytorch2.3-cu12.4-ngc24.04:py3.10-cu12.4-pt2.3-trt8.6-nccl2.21.5-2e5b376c"
# IMAGE_URL="harbor.shopeemobile.com/aip/aip-image-hub/aip-prod/projects/123/pytorch2.5-cu12.6-py3.10-trt10.3:py3.10-cu12.6-pt2.5-trt10.3-vscode1.82.2-f6754f38f4"
DATASETS="/home/fuping.chu/datasets"
MODEL_ZOO="/home/fuping.chu/model_zoo"

cmd="docker run -u root $FLAGS --name ${C_NAME} \
  --gpus all \
  --shm-size=16g \
  --net=host \
  -w /workspace \
  -v ${WORKSPACE_PATH}:/workspace \
  -v ${DATASETS}:/datasets \
  -v ${MODEL_ZOO}:/model_zoo \
  ${IMAGE_URL} bash"
echo $cmd
$cmd
