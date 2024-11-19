#!/bin/bash 

ENV_PATH=/workspace/flux_env/master_env.sh
source $ENV_PATH
# DEVICES_ID=(0 1 4 5)
DEVICES_ID=(0 1)
WORLD_SIZE=${#DEVICES_ID[@]}
BATCH=1
ZERO_MODE=zero2
		#-o flux_ncu_${ZERO_MODE}_worker${}_batch${batch}

rank_id=0
for i in ${DEVICES_ID[@]}; do
    export CUDA_VISIBLE_DEVICES=$i
    export RANK=$rank_id
    if [ $RANK -eq 0 ];then
        echo "Running On GPU_$i, RANK_$RANK"
	#bash nsys.sh $i &
	ncu \
		--kernel-name elementwise_kernel \
		--launch-skip 10428 \
		-o flux_ncu.ncu-rep \
		--launch-count 1 \
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
    else
            echo "Running On GPU_$i, RANK_$RANK"
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
