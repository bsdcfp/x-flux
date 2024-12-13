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
# DEVICES_ID=(0)
# DEVICES_ID=(0 2 3 4)
DEVICES_ID=(0 1 2 3 4 5 6 7)
WORLD_SIZE=${#DEVICES_ID[@]}
ACCELERATE_CONFIG=train_configs/accelerate_config.yaml
DEEPSPEED_CONFIG=train_configs/ds_config_zero2_offload.json
# --deepspeed_config_file ${DEEPSPEED_CONFIG} \
# --config_file ${ACCELERATE_CONFIG} \
      # --use_deepspeed \
      # --mixed_precision bf16 \
rank_id=0

# 记录开始时间
START_TIME=$(date +%s)
echo "Start Time: $(date)"

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
      --mixed_precision bf16 \
      train_flux_deepspeed_no_t5.py --config 'train_configs/test_finetune_H100.yaml' "
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

# 记录结束时间
END_TIME=$(date +%s)
echo "End Time: $(date)"

# 计算并输出总耗时
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "Total Elapsed Time: $ELAPSED_TIME seconds"
