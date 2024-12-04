#!/bin/bash 
set +x
# is_dry_run=true
is_dry_run=false

ENV_PATH=/workspace/project_flux/x-flux/flux_env/master_env.sh
source $ENV_PATH
# DEVICES_ID=(0 4 5 7)
DEVICES_ID=0
WORKDIR=/workspace/project_flux/x-flux

echo "Running On GPU_$DEVICES_ID"
export CUDA_VISIBLE_DEVICES=$DEVICES_ID
cd $WORKDIR
cmd="python3 train_flux_deepspeed.py --config 'train_configs/test_finetune.yaml' "
eval $cmd &

wait
