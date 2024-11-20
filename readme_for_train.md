# 启动容器

- pull image

```sh
docker pull harbor.shopeemobile.com/aip/aip-image-hub/aip-prod/projects/123/pytorch2.5-cu12.6-py3.10-trt10.3:py3.10-cu12.6-pt2.5-trt10.3-vscode1.82.2-f6754f38f4
```

- start container

```sh
#!/bin/bash
# Run the docker container with the specified parameters

WORKSPACE_PATH="/home/fuping.chu"
C_NAME="fuping-flux"
FLAGS="-itd --privileged "
IMAGE_URL="harbor.shopeemobile.com/aip/aip-image-hub/aip-prod/projects/123/pytorch2.5-cu12.6-py3.10-trt10.3"
DATASETS="/home/fuping.chu/datasets"
MODEL_ZOO="/home/fuping.chu/model_zoo"

cmd="docker run -u root $FLAGS --name ${C_NAME} \
  --gpus all \
  --shm-size=16g \
  --net=host \
  -w /workspace \
  -v ${WORKSPACE_PATH}:/workspace \
  -v ${DATASETS}:/datasets \
  -v ${MODEL_ZOO}:/model_zoo \
  ${IMAGE_URL} bash"
echo $cmd
$cmd
```

- 安装依赖

```sh
cd /workspace/project_flux/x_flux
pip install -r requirements.txt
```

# 显存情况分析

利用[pytorch.org/memory_viz](https://pytorch.org/memory_viz)分析内存使用情况，详细说明请见[链接](https://pytorch.org/docs/stable/torch_cuda_memory.html)

- 增加代码

```c
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if step == 0:
                torch.cuda.memory._record_memory_history()
            if step == 1:
                torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
```

- 文件解析
    - 文件有点大，一个step有1.6g，解析会有点慢

# 耗时分析-nsys

利用nsystem分析模型在CUDA上耗时情况。

- dump脚本

```sh
#!/bin/bash 

ENV_PATH=/workspace/flux_env/master_env.sh
source $ENV_PATH
# DEVICES_ID=(0 1 4 5)
DEVICES_ID=(4 5)
WORLD_SIZE=${#DEVICES_ID[@]}
BATCH=1
ZERO_MODE=zero2

rank_id=0
for i in ${DEVICES_ID[@]}; do
    export CUDA_VISIBLE_DEVICES=$i
    export RANK=$rank_id
    if [ $RANK -eq 0 ];then
        echo "Running On GPU_$i, RANK_$RANK"
	#bash nsys.sh $i &
	nsys profile \
		-y 600 \
		-d 20 \
		--stats=true \
                -t cuda,osrt,nvtx \
		-o flux_timeline_${ZERO_MODE}_worker${WORLD_SIZE}_batch${BATCH}_$i \
    	        --gpu-metrics-devices=$i \
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

```

- 文件解析
    - 在笔记本上安装NVIDIA Nsight Systems UI客户端，打开.nsys-rep文件