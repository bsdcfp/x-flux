#!/bin/bash 
set +x
ENV_PATH=/workspace/flux_env/master_env.sh
source $ENV_PATH
DEVICES_ID=4
WORLD_SIZE=2
export CUDA_VISIBLE_DEVICES=$DEVICES_ID
export RANK=1
echo "Running On GPU_$DEVICES_ID, RANK_$RANK"

# export NCCL_SOCKET_IFNAME=eth0 
# export NCCL_IB_DISABLE=1 
export NCCL_DEBUG=INFO
#export NCCL_HOST_ID=0

accelerate launch \
  --num_processes $WORLD_SIZE \
  --num_machines $WORLD_SIZE \
  --machine_rank $RANK \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --deepspeed_multinode_launcher standard \
  --use_deepspeed \
  --mixed_precision bf16 \
  train_flux_deepspeed.py --config 'train_configs/test_finetune.yaml' &
wait
