#!/bin/bash 

ENV_PATH=/workspace/flux_env/master_env.sh
source $ENV_PATH
DEVICES_ID=(0 1 4 5)
WORLD_SIZE=${#DEVICES_ID[@]}

rank_id=0
for i in ${DEVICES_ID[@]}; do
    export CUDA_VISIBLE_DEVICES=$i
    export RANK=$rank_id
    if [ $RANK -eq 0 ];then
        echo "Running On GPU_$i"
	echo "source  $ENV_PATH"
        echo "export CUDA_VISIBLE_DEVICES=$i"
        echo "export RANK=$rank_id"
        echo "nsys profile \
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
         train_flux_deepspeed.py --config 'train_configs/test_finetune.yaml' &"
    else
        echo "Running On GPU_$i"
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
    fi
    let rank_id+=1
done
wait
