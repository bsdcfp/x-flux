#!/bin/bash
# Run the docker container with the specified parameters
WORKSPACE_PATH=$(pwd)
C_NAME="fuping-flux-worker2"
FLAGS="-itd --privileged "
# IMAGE_URL="harbor.shopeemobile.com/aip/aip-image-hub/aip-prod/projects/0/pytorch2.0-cu12.1-py3.10-trt8.6:py3.10-cu12.1-pt2.0-trt8.6-vscode1.82.2-d540733753"
IMAGE_URL="pytorch2.0-cu12.1-py3.10-trt8.6:py3.10-cu12.1-pt2.0-trt8.6-vscode1.82.2-d540733753-fuping-flux"
DATASETS="/data1/luwen.miao/data/flux/flux_shortsentence"
MODEL_ZOO="/data1/luwen.miao/ckpt/flux/"

cmd="docker run $FLAGS --name ${C_NAME} \
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
