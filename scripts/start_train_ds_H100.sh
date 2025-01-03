#!/bin/bash 
set +x
# is_dry_run=true
EXP_ID=$(date "+%Y%m%d-%H%M%S")
echo "Experiment ID:  ${EXP_ID}"

is_dry_run=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(dirname "$SCRIPT_DIR")"
cd $WORKDIR

ENV_PATH=$WORKDIR/scripts/flux_env/master_env_H100.sh
source $ENV_PATH

# DEVICES_ID=(0 4 5 7)
# DEVICES_ID=(6)
DEVICES_ID=(4 5 6 7)
DEVICES_ID=(0 1 2 3)
DEVICES_ID=(0 1)

MIXED_PRECISION=bf16
MIXED_PRECISION=fp8

WORLD_SIZE=${#DEVICES_ID[@]}

TRAINING_CONFIG=train_configs/test_finetune_b8.yaml
TRAINING_CONFIG=train_configs/test_finetune_b6.yaml
TRAINING_CONFIG=train_configs/test_finetune_b1.yaml
TRAINING_CONFIG=train_configs/test_finetune_b4.yaml
TRAINING_CONFIG=train_configs/test_finetune_H100.yaml

DEEPSPEED_CONFIG=train_configs/ds_config_zero2_offload.json
DEEPSPEED_CONFIG=train_configs/ds_config_zero3_allreduce.json
DEEPSPEED_CONFIG=train_configs/ds_config_zero3_opt.json
DEEPSPEED_CONFIG=train_configs/ds_config_zero3_default.json
DEEPSPEED_CONFIG=train_configs/ds_config_zero3_offload.json
DEEPSPEED_CONFIG=train_configs/ds_config_zero0_default.json
DEEPSPEED_CONFIG=train_configs/ds_config_zero1_default.json
DEEPSPEED_CONFIG=train_configs/ds_config_zero2_default.json
DEEPSPEED_CONFIG=train_configs/ds_config_zero2_opt.json
DEEPSPEED_CONFIG=train_configs/ds_config_zero2_opt_msamp.json

MAIN_SCRIPT="train_flux_deepspeed_gradient_ckpt_no_t5.py --gradient_checkpointing"
MAIN_SCRIPT="train_flux_deepspeed_gradient_ckpt.py --gradient_checkpointing"
MAIN_SCRIPT=train_flux_deepspeed.py
MAIN_SCRIPT=train_flux_deepspeed_no_t5_fp8.py
MAIN_SCRIPT=train_flux_deepspeed_no_t5.py

START_TIME=$(date +%s)
echo "Start Time: $(date)"

# cmd="python3 scripts/prepare_dataset.py \
# 	--config ${TRAINING_CONFIG} \
# 	--overwrite \
# 	--max_workers 2 \
# 	--num_workers 8"
# if [ "$is_dry_run" = false ]; then
#     export CUDA_VISIBLE_DEVICES=$DEVICES_ID
#     eval $cmd 
# else
#     echo $cmd
# fi
# 
# END_PREPARE_DATA_TIME=$(date +%s)
# ELAPSED_TIME=$((END_PREPARE_DATA_TIME - START_TIME))
# echo "Total prepare datasets Time: $ELAPSED_TIME seconds"

rank_id=0
for i in ${DEVICES_ID[@]}; do
    echo "Running On GPU_$i"
    export CUDA_VISIBLE_DEVICES=$i
    export RANK=$rank_id
    cmd="accelerate launch \
      --num_processes $WORLD_SIZE \
      --num_machines $WORLD_SIZE \
      --machine_rank $RANK \
      --main_process_ip $MASTER_ADDR \
      --main_process_port $MASTER_PORT \
      --deepspeed_multinode_launcher standard \
      --use_deepspeed \
      --deepspeed_config_file ${DEEPSPEED_CONFIG} \
      --mixed_precision $MIXED_PRECISION \
      ${MAIN_SCRIPT}  --config ${TRAINING_CONFIG} "
    if [ "$is_dry_run" = false ]; then
	    eval $cmd | tee -a $WORKDIR/logs/train_log_${EXP_ID}.txt &
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
