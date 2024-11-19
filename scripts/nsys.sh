#!/bin/bash 

ENV_PATH=/workspace/flux_env/master_env.sh
source $ENV_PATH
i=$1
echo "device id: $i"
export CUDA_VISIBLE_DEVICES=$i
export RANK=$rank_id
nsys profile \
            -y 600 \
            -d 20 \
            --stats=true \
            -t cuda,osrt,nvtx \
            -o flux_zero2_timeline_$i \
    	--gpu-metrics-device=$i \
      accelerate launch \
     --num_processes $WORLD_SIZE \
     --num_machines $WORLD_SIZE \
     --machine_rank $RANK \
     --main_process_ip $MASTER_ADDR \
     --main_process_port $MASTER_PORT \
     --deepspeed_multinode_launcher standard \
     --use_deepspeed \
     --mixed_precision bf16 \
     train_flux_deepspeed.py --config 'train_configs/test_finetune.yaml'
