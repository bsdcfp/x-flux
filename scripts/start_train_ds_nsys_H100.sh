#!/bin/bash 
set +x
is_dry_run=false

EXP_ID=$(date "+%Y%m%d-%H%M%S")
echo "Experiment ID:  ${EXP_ID}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(dirname "$SCRIPT_DIR")"
cd $WORKDIR

ENV_PATH=$WORKDIR/scripts/flux_env/master_env_H100.sh
source $ENV_PATH
DEVICES_ID=(1 2 3 6)
DEVICES_ID=(2 3 4 5)
DEVICES_ID=(0 1 2 3 4 5 6 7)
DEVICES_ID=(4 5 6 7)
DEVICES_ID=(0 1 2 3)
# DEVICES_ID=(4 5)
# DEVICES_ID=(0 4 5 7)

SINGLE_DEVICE_BATCH=6

WORLD_SIZE=${#DEVICES_ID[@]}
TRAINING_CONFIG=train_configs/test_finetune_H100.yaml
TRAINING_CONFIG=train_configs/test_finetune_b8.yaml
TRAINING_CONFIG=train_configs/test_finetune_b1.yaml
TRAINING_CONFIG=train_configs/test_finetune_b4.yaml
TRAINING_CONFIG=train_configs/test_finetune_b6.yaml
DEEPSPEED_CONFIG=train_configs/ds_config_zero2_offload.json
DEEPSPEED_CONFIG=train_configs/ds_config_zero3_allreduce.json
DEEPSPEED_CONFIG=train_configs/ds_config_zero3_opt.json
DEEPSPEED_CONFIG=train_configs/ds_config_zero3_default.json
DEEPSPEED_CONFIG=train_configs/ds_config_zero3_offload.json
DEEPSPEED_CONFIG=train_configs/ds_config_zero0_default.json
DEEPSPEED_CONFIG=train_configs/ds_config_zero2_default.json
DEEPSPEED_CONFIG=train_configs/ds_config_zero2_opt.json

ZERO_MODE=zero2-opt

# MAIN_SCRIPT=train_flux_deepspeed.py
# NVTX_MAIN_SCRIPT=train_flux_deepspeed_nvtx.py
# MAIN_SCRIPT=train_flux_deepspeed_no_t5.py
# NVTX_MAIN_SCRIPT="train_flux_deepspeed_total_params.py --nvtx"
NVTX_MAIN_SCRIPT=train_flux_deepspeed_no_t5.py
NVTX_MAIN_SCRIPT="train_flux_deepspeed_no_t5_nvtx.py --nvtx"

MAIN_SCRIPT=train_flux_deepspeed_total_params_no_t5.py
MAIN_SCRIPT=train_flux_deepspeed_total_params.py
MAIN_SCRIPT=train_flux_deepspeed_no_t5.py

START_TIME=$(date +%s)
echo "Start Time: $(date)"

rank_id=0
for i in ${DEVICES_ID[@]}; do
    export CUDA_VISIBLE_DEVICES=$i
    export RANK=$rank_id
    echo "Running On GPU_$i, RANK_$RANK"
    cmd="nsys profile \
      -y 120 \
      -d 30 \
      --capture-range=cudaProfilerApi \
      -t cuda,osrt,nvtx,cudnn,cublas \
      --cuda-memory-usage=true \
      --gpu-metrics-device=$i \
      -o logs/nsys/flux_timeline_${ZERO_MODE}_workers${WORLD_SIZE}_batch${SINGLE_DEVICE_BATCH}_gpu$i \
      -f true \
      accelerate launch \
        --num_processes $WORLD_SIZE \
        --num_machines $WORLD_SIZE \
        --machine_rank $RANK \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        --deepspeed_multinode_launcher standard \
        --use_deepspeed \
        --deepspeed_config_file ${DEEPSPEED_CONFIG} \
        --mixed_precision bf16 \
        ${NVTX_MAIN_SCRIPT} --config ${TRAINING_CONFIG}"

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
