'''
File: monitor_gpu.py
Project: scripts
Created Date: Monday November 25th 2024
Author: Miaoluwen
-----
Last Modified: Monday November 25th 2024 4:13:37 pm
Modified By: the developer formerly known as Miaoluwen at <miaoluwen@sinodata.net.cn>
-----
Copyright (c) 2024 Sinodata
-----
'''
import torch
import time
import fire

# 定义 GPU 使用监控函数

def monitor_gpu_usage(interval=1, duration=60, gpu_indices="0"):
    """
    监控 GPU 使用率和显存占用
    Args:
        interval (int): 采样间隔（秒）
        duration (int): 监控总时长（秒）
    Returns:
        metrics (dict): 包含平均和最大利用率的信息
    """
    import pynvml  # 需要安装 pynvml：pip install nvidia-ml-py
    if isinstance(gpu_indices, int):
        gpu_indices = str([gpu_indices])
    #gpu_indices = [int(idx) for idx in gpu_indices.split(",")]

    # 初始化 NVIDIA 管理库
    pynvml.nvmlInit()
    gpu_metrics = {
      idx: {
        "gpu_utilizations": [],
        "memory_utilizations": []
      } for idx in gpu_indices
    }

    for _ in range(int(duration / interval)):
        for gpu_id in gpu_indices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)  # 选择监控的 GPU
            
            # 获取 GPU 利用率
            util_rate = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            # 获取显存使用
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = mem_info.used / 1024**2  # 转为 MB
            mem_total = mem_info.total / 1024**2  # 转为 MB
            mem_usage_rate = (mem_used / mem_total) * 100

            # 记录当前值
            gpu_metrics[gpu_id]["gpu_utilizations"].append(util_rate)
            gpu_metrics[gpu_id]["memory_utilizations"].append(mem_usage_rate)
            
            # print(f"GPU {gpu_id} gpu_metrics: {gpu_metrics[gpu_id]}")
            print(f"GPU {gpu_id} gpu_metrics collected. util_rate: {util_rate}%, mem_usage_rate: {mem_usage_rate:.2f}%")

        time.sleep(interval)
    
    result = {}

    for gpu_id in gpu_indices:
        # 计算平均值和最大值
        gpu_utilizations = gpu_metrics[gpu_id]["gpu_utilizations"]
        memory_utilizations = gpu_metrics[gpu_id]["memory_utilizations"]
        metrics = {
            "gpu_util_avg": sum(gpu_utilizations) / len(gpu_utilizations),
            "gpu_util_max": max(gpu_utilizations),
            "mem_util_avg": sum(memory_utilizations) / len(memory_utilizations),
            "mem_util_max": max(memory_utilizations),
        }
        result[gpu_id] = metrics

    # 清理资源
    pynvml.nvmlShutdown()
    return result


# 示例：监控 GPU 使用 10 秒，每 1 秒采样一次
if __name__ == "__main__":
    metrics = fire.Fire(monitor_gpu_usage)
    for gpu_id, data in metrics.items():
        print(f"GPU {gpu_id}:")
        print(f"  平均利用率: {data['gpu_util_avg']:.2f}%")
        print(f"  最大利用率: {data['gpu_util_max']:.2f}%")
        print(f"  平均显存占用率: {data['mem_util_avg']:.2f}%")
        print(f"  最大显存占用率: {data['mem_util_max']:.2f}%")
