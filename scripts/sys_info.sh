#!/bin/bash
# 获取 CPU 信息
CPU_INFO=$(lscpu | grep 'Model name:' | awk -F ': ' '{print $2}'| awk '{$1=$1};1')
CPU_CORES=$(lscpu | grep 'CPU(s):' | awk -F ': ' '{print $2}' | head -n 1 | awk '{$1=$1};1')
CPU_THREADS=$(lscpu | grep 'Thread(s) per core:' | awk -F ': ' '{print $2}' | awk '{$1=$1};1')
CPU_SOCKETS=$(lscpu | grep 'Socket(s):' | awk -F ': ' '{print $2}' | awk '{$1=$1};1')
CPU_MODEL=$(lscpu | grep 'Model name:' | awk -F ': ' '{print $2}' | awk '{$1=$1};1')
CPU_FREQUENCY=$(lscpu | grep 'CPU MHz:' | awk -F ': ' '{print $2}' | head -n 1 | awk '{$1=$1};1')

# 获取操作系统信息
OS=$(cat /etc/os-release | grep PRETTY_NAME | cut -d '"' -f 2)

# 获取 Docker 镜像信息
DOCKER_IMAGE=$(docker ps | grep fuping | awk '{print $2}')

# 获取 CUDA 版本
CUDA_VERSION=$(nvcc --version | grep release | awk '{print $5}' | cut -d ',' -f 1)

# 获取 GPU 信息
GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader)

# 获取 Python 版本
PYTHON_VERSION=$(python3 --version)

# 获取 PyTorch 版本
PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")

# 获取 vLLM 版本
VLLM_VERSION=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "vLLM not installed")

# 获取 Flash Attention 版本
FLASH_ATTENTION_VERSION=$(python3 -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null || echo "Flash Attention not installed")

# 获取 Transformers 版本
TRANSFORMERS_VERSION=$(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "Transformers not installed")


# 输出信息
echo "CPU Model: $CPU_MODEL"
echo "CPU Cores: $CPU_CORES"
echo "CPU Threads: $CPU_THREADS"
echo "CPU Sockets: $CPU_SOCKETS"
echo "CPU Frequency: $CPU_FREQUENCY"
echo "OS: $OS"
echo "Docker Image: $DOCKER_IMAGE"
echo "GPU: $GPU_INFO"
echo "CUDA: $CUDA_VERSION"
echo "Python: $PYTHON_VERSION"
echo "PyTorch: $PYTORCH_VERSION"
echo "vLLM: $VLLM_VERSION"
echo "Flash Attention: $FLASH_ATTENTION_VERSION"
echo "Transformers: $TRANSFORMERS_VERSION"
