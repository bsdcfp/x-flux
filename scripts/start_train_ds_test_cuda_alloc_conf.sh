#!/bin/bash 
set +x
# is_dry_run=true
is_dry_run=false

ENV_PATH=/workspace/project_flux/x-flux/flux_env/master_env.sh
source $ENV_PATH
# DEVICES_ID=(0 4 5 7)
DEVICES_ID=(0)
WORLD_SIZE=${#DEVICES_ID[@]}
ACCELERATE_CONFIG=train_configs/ds_config.yaml
# ACCELERATE_CONFIG=train_configs/ds_config_zero3.yaml

WORKDIR=/workspace/project_flux/x-flux
cd $WORKDIR

rank_id=0
for i in ${DEVICES_ID[@]}; do
    echo "Running On GPU_$i"
    export CUDA_VISIBLE_DEVICES=$i
    export RANK=$rank_id
    # export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:512"
    # export PYTORCH_CUDA_ALLOC_CONF="pinned_use_cuda_host_register:True,pinned_num_register_threads:8"
    # export PYTORCH_CUDA_ALLOC_CONF="pinned_use_cuda_host_register:True,pinned_num_register_threads:8,roundup_power2_divisions:[256:1,512:2,1024:4,>:8]"
    export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync"
    # export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync,pinned_use_cuda_host_register:True,pinned_num_register_threads:8"
    # export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    # export TORCH_LOGS="recompiles"
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