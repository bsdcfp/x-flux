#!/bin/bash 
set +x
ENV_PATH=/workspace/flux_env/master_env.sh
source $ENV_PATH
# DEVICES_ID=(0 1 4 5)
DEVICES_ID=0
# WORLD_SIZE=${#DEVICES_ID[@]}
WORLD_SIZE=2
export CUDA_VISIBLE_DEVICES=0
export RANK=0
echo "Running On GPU_$CUDA_VISIBLE_DEVICES, RANK_$RANK"

nsys profile \
	-y 600 \
	-d 20 \
	--stats=true \
        -t cuda,osrt,nvtx \
	-o flux_zero2_timeline_worker1_$RANK \
        --gpu-metrics-devices=$DEVICES_ID \
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
#         accelerate launch \
#           --num_processes $WORLD_SIZE \
#           --num_machines $WORLD_SIZE \
#           --machine_rank $RANK \
#           --main_process_ip $MASTER_ADDR \
#           --main_process_port $MASTER_PORT \
#           --deepspeed_multinode_launcher standard \
#           --use_deepspeed \
#           --mixed_precision bf16 \
#           train_flux_deepspeed.py --config 'train_configs/test_finetune.yaml' &
wait
