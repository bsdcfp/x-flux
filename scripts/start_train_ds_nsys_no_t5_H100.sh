#!/bin/bash 
set +x
#is_dry_run=true
is_dry_run=false

EXP_ID=$(date "+%Y%m%d-%H%M%S")
echo "Experiment ID:  ${EXP_ID}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(dirname "$SCRIPT_DIR")"
cd $WORKDIR

ENV_PATH=$WORKDIR/scripts/flux_env/master_env_H100.sh
source $ENV_PATH
# DEVICES_ID=(0 4 5 7)
# DEVICES_ID=(6)
DEVICES_ID=(0 1)
WORLD_SIZE=${#DEVICES_ID[@]}
DEEPSPEED_CONFIG=train_configs/ds_config_zero2_offload.json
ACCELERATE_CONFIG=train_configs/ds_config.yaml

ZERO_MODE=zero2
SINGLE_DEVICE_BATCH=4

START_TIME=$(date +%s)
echo "Start Time: $(date)"

rank_id=0
for i in ${DEVICES_ID[@]}; do
    export CUDA_VISIBLE_DEVICES=$i
    export RANK=$rank_id
    if [ $RANK -eq 0 ];then
        echo "Running On GPU_$i, RANK_$RANK"
        cmd="nsys profile \
	  -y 120 \
          -d 30 \
          --stats=true \
          -t cuda,osrt,nvtx \
          --cuda-memory-usage=true \
          --gpu-metrics-device=$i \
          -o flux_timeline_${ZERO_MODE}_workers${WORLD_SIZE}_batch${SINGLE_DEVICE_BATCH}_gpu$i \
          -f true \
          accelerate launch \
            --num_processes $WORLD_SIZE \
            --num_machines $WORLD_SIZE \
            --machine_rank $RANK \
            --main_process_ip $MASTER_ADDR \
            --main_process_port $MASTER_PORT \
            --deepspeed_multinode_launcher standard \
            --use_deepspeed \
            --mixed_precision bf16 \
            train_flux_deepspeed_no_t5_nvtx.py --config 'train_configs/test_finetune_H100.yaml' "
    else
            echo "Running On GPU_$i, RANK_$RANK"
          cmd="accelerate launch \
            --num_processes $WORLD_SIZE \
            --num_machines $WORLD_SIZE \
            --machine_rank $RANK \
            --main_process_ip $MASTER_ADDR \
            --main_process_port $MASTER_PORT \
            --deepspeed_multinode_launcher standard \
            --use_deepspeed \
            --mixed_precision bf16 \
            train_flux_deepspeed.py --config 'train_configs/test_finetune_H100.yaml' "
    fi

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
END_TIME=$(date +%s)
echo "End Time: $(date)"

# 计算并输出总耗时
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "Total Elapsed Time: $ELAPSED_TIME seconds"
