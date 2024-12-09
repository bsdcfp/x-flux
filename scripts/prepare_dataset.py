import os
import sys
import torch
import argparse
import random
from torch.utils.data import DataLoader
from einops import rearrange, repeat
from omegaconf import OmegaConf
from concurrent.futures import ThreadPoolExecutor,as_completed

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.flux.modules.conditioner import HFEmbedder
from src.flux.modules.autoencoder import AutoEncoder
from src.flux.sampling import prepare
from src.flux.util import (load_ae, load_clip,load_t5)
from image_datasets.dataset import loader

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
def get_models(name: str, device, is_schnell: bool):
    t5 = load_t5(device, max_length=256 if is_schnell else 512)
    clip = load_clip(device)
    vae = load_ae(name, device)
    return vae, t5, clip

def process_one(one_data, t5, clip, vae, output_dir, index, device):
    img, prompts = one_data
    x_1 = vae.encode(img.to(device).to(torch.float32))

    # 准备数据
    prepared_data = prepare(t5=t5, clip=clip, img=x_1, prompt=prompts)
    x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    prepared_data["img"] = x_1
    for key, value in prepared_data.items():
        if value.dim() > 1:
            prepared_data[key] = value.squeeze(0).to("cpu")

    # 保存预处理后的数据
    output_path = os.path.join(output_dir, f"processed_{index}.pt")
    torch.save(prepared_data, output_path)
    print(f"Saved one_data {index} to {output_path}")

def process_and_save_data(
    t5: HFEmbedder,
    clip: HFEmbedder,
    vae: AutoEncoder,
    input_dir: str,
    output_dir: str,
    img_size: int = 512,
    device: str = "cuda:0",
    **kwargs
):
    train_dataloader = loader(train_batch_size=1,
                             num_workers=8,
                             img_dir=input_dir,
                             img_size=img_size)
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        for index, batch_data in enumerate(train_dataloader):
            future = executor.submit(process_one, batch_data, t5, clip, vae, output_dir, index, device)
            futures.append(future)
        
        for future in as_completed(futures):
            try:
                future.result()  # This will raise an exception if the task failed
            except Exception as e:
                print(f"An error occurred: {e}")

# def process_and_save_data(
#     t5: HFEmbedder,
#     clip: HFEmbedder,
#     vae: AutoEncoder,
#     input_dir: str,
#     output_dir: str,
#     img_size: int = 512,
#     device: str = "cuda:0",
#     **kwargs
# ):
#     train_dataloader = loader(train_batch_size=1,
#                              num_workers = 8,
#                              img_dir = input_dir,
#                              img_size = img_size)
#     # 遍历数据集
#     for index, batch_data in enumerate(train_dataloader):
#         img, prompts = batch_data
#         x_1 = vae.encode(img.to(device).to(torch.float32))

#         # 准备数据
#         prepared_data = prepare(t5=t5, clip=clip, img=x_1, prompt=prompts)
#         x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

#         # 保存预处理后的数据
#         output_path = os.path.join(output_dir, f"processed_{index}.pt")
#         torch.save(prepared_data, output_path)
#         print(f"Saved batch_data {index} to {output_path}")

def load_and_print_file(dir_path):
    if not os.path.exists(dir_path):
        print(f"Error: File {dir_path} does not exist.")
        return

    pt_files = [f for f in os.listdir(dir_path) if f.endswith('.pt')]
    if not pt_files:
        print(f"Error: No .pt files found in directory {dir_path}.")
        return

    random_file = random.choice(pt_files)
    file_path = os.path.join(dir_path, random_file)

    data = torch.load(file_path)
    print(f"Randomly selected file: {random_file}")
    print(f"Data in {file_path}:")
    # print(data)
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite output dir",
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config)

    # 设置输出目录
    output_dir = f"{config.data_config.img_dir}_processed"
    load_and_print_flag = False

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    else:
        load_and_print_flag = True

    if args.overwrite:
        load_and_print_flag = False

    if load_and_print_flag:
        load_and_print_file(output_dir)
    else:
        print(load_and_print_flag)
        is_schnell = config.model_name == "flux-schnell"

        vae, t5, clip = get_models(name=config.model_name, 
                                device=get_device(), 
                                is_schnell=is_schnell)

        vae.requires_grad_(False)
        t5.requires_grad_(False)
        clip.requires_grad_(False)

        # 调用函数进行处理和保存
        process_and_save_data(t5, clip, vae, config.data_config.img_dir, output_dir)
