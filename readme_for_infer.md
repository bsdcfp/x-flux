# 启动容器
* pull image
```sh
docker pull harbor.shopeemobile.com/aip/aip-image-hub/aip-prod/projects/123/pytorch2.5-cu12.6-py3.10-trt10.3:py3.10-cu12.6-pt2.5-trt10.3-vscode1.82.2-f6754f38f4
```

* start container
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

* 安装依赖
```sh
cd /workspace/project_flux/x_flux
pip install -r requirements.txt
```

# FLUX推理
* 推理脚本
```py
import torch
from diffusers import FluxPipeline

model_path="/model_zoo/flux.1_dev/"
pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

prompt = "a tiny astronaut hatching from an egg on the moon"
out = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    height=768,
    width=1360,
    num_inference_steps=50,
).images[0]
out.save("image.png")
```

* 运行脚本
```sh
python3 test_flux_diffusers.py 
```

# SD3推理