#!/bin/bash 
set +x
# is_dry_run=true
is_dry_run=false

ENV_PATH=/workspace/flux_env/master_env.sh
source $ENV_PATH
# DEVICES_ID=(0 4 5 7)
DEVICES_ID=(0 1)
WORLD_SIZE=${#DEVICES_ID[@]}
# ACCELERATE_CONFIG=train_configs/ds_config.yaml
ACCELERATE_CONFIG=train_configs/ds_config_zero3.yaml

rank_id=0
for i in ${DEVICES_ID[@]}; do
    echo "Running On GPU_$i"
    export CUDA_VISIBLE_DEVICES=$i
    export RANK=$rank_id
    cmd="accelerate launch \
	    --config_file ${ACCELERATE_CONFIG} \
      --num_processes $WORLD_SIZE \
      --num_machines $WORLD_SIZE \
      --machine_rank $RANK \
      --main_process_ip $MASTER_ADDR \
      --main_process_port $MASTER_PORT \
      --deepspeed_multinode_launcher standard \
      --use_deepspeed \
      --mixed_precision bf16 \
      train_flux_deepspeed.py --config 'train_configs/test_finetune.yaml' "
    if [ "$is_dry_run" = false ]; then
	    eval $cmd &
    else
            echo "export CUDA_VISIBLE_DEVICES=$i"
            echo "export RANK=$rank_id"
	    echo $cmd
    fi

    let rank_id+=1
done
wait
