import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from omegaconf import OmegaConf
from copy import deepcopy
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from src.flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from src.flux.util import (configs, load_ae, load_clip,
                       load_flow_model2, load_t5)

from src.flux.aip_profiler import memory_profiler
from image_datasets.dataset import loader

if is_wandb_available():
    import wandb
logger = get_logger(__name__, log_level="INFO")
from torchinfo import summary  
from torch.profiler import profile, schedule, tensorboard_trace_handler

tracing_schedule = schedule(skip_first=3, wait=2, warmup=2, active=3, repeat=2)
trace_handler = tensorboard_trace_handler(dir_name="logs/flux_torch_profile", use_gzip=True)

def get_models(name: str, device, offload: bool, is_schnell: bool):
    t5 = load_t5(device, max_length=256 if is_schnell else 512)
    clip = load_clip(device)
    clip.requires_grad_(False)
    model = load_flow_model2(name, device="cpu")
    vae = load_ae(name, device="cpu" if offload else device)
    return model, vae, t5, clip

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
        "--gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    args = parser.parse_args()


    return args

# @memory_profiler("./snapshot_mem_timeline_flux.1_train.pickle")
def main():

    args = parse_args()
    configs = OmegaConf.load(args.config)
    is_schnell = configs.model_name == "flux-schnell"
    logging_dir = os.path.join(configs.output_dir, configs.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=configs.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=configs.gradient_accumulation_steps,
        mixed_precision=configs.mixed_precision,
        log_with=configs.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


    if accelerator.is_main_process:
        if configs.output_dir is not None:
            os.makedirs(configs.output_dir, exist_ok=True)

    dit, vae, t5, clip = get_models(name=configs.model_name, device=accelerator.device, offload=False, is_schnell=is_schnell)

    vae.requires_grad_(False)
    t5.requires_grad_(False)
    clip.requires_grad_(False)
    dit = dit.to(torch.float32)
    dit.train()
    if args.gradient_checkpointing:
        dit.enable_gradient_checkpointing()

    optimizer_cls = torch.optim.AdamW
    #you can train your own layers
    for n, param in dit.named_parameters():
        if 'txt_attn' not in n:
            param.requires_grad = False

    print(sum([p.numel() for p in dit.parameters() if p.requires_grad]) / 1000000, 'parameters')
    optimizer = optimizer_cls(
        [p for p in dit.parameters() if p.requires_grad],
        lr=configs.learning_rate,
        betas=(configs.adam_beta1, configs.adam_beta2),
        weight_decay=configs.adam_weight_decay,
        eps=configs.adam_epsilon,
    )

    train_dataloader = loader(**configs.data_config)
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / configs.gradient_accumulation_steps)
    if configs.max_train_steps is None:
        configs.max_train_steps = configs.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        configs.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=configs.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=configs.max_train_steps * accelerator.num_processes,
    )
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if configs.resume_from_checkpoint:
        if configs.resume_from_checkpoint != "latest":
            path = os.path.basename(configs.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(configs.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{configs.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            configs.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            dit_state = torch.load(os.path.join(configs.output_dir, path, 'dit.bin'), map_location='cpu')
            dit_state2 = {}
            for k in dit_state.keys():
                dit_state2[k[len('module.'):]] = dit_state[k]
            dit.load_state_dict(dit_state2)
            optimizer_state = torch.load(os.path.join(configs.output_dir, path, 'optimizer.bin'), map_location='cpu')['base_optimizer_state']
            optimizer.load_state_dict(optimizer_state)

            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0
    dit, optimizer, _, lr_scheduler = accelerator.prepare(
        dit, optimizer, deepcopy(train_dataloader), lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        configs.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        configs.mixed_precision = accelerator.mixed_precision


    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / configs.gradient_accumulation_steps)
    if overrode_max_train_steps:
        configs.max_train_steps = configs.num_train_epochs * num_update_steps_per_epoch
    configs.num_train_epochs = math.ceil(configs.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(configs.tracker_project_name, {"test": None})

    timesteps = list(torch.linspace(1, 0, 1000).numpy())
    total_batch_size = configs.train_batch_size * accelerator.num_processes * configs.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {configs.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {configs.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {configs.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {configs.max_train_steps}")

    progress_bar = tqdm(
        range(0, configs.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, configs.num_train_epochs):
        train_loss = 0.0
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=tracing_schedule,
            on_trace_ready=trace_handler,
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
            with_flops=True,
            with_modules=True
        # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
        # used when outputting for tensorboard
        ) as prof:
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(dit):
                    img, prompts = batch
                    with torch.no_grad():
                        x_1 = vae.encode(img.to(accelerator.device).to(torch.float32))
                        inp = prepare(t5=t5, clip=clip, img=x_1, prompt=prompts)
                        x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

                    bs = img.shape[0]
                    t = torch.sigmoid(torch.randn((bs,), device=accelerator.device))
                    t_expand = t[:, None, None]
                    x_0 = torch.randn_like(x_1).to(accelerator.device)
                    # x_t = (1 - t) * x_1 + t * x_0
                    x_t = (1 - t_expand) * x_1 + t_expand * x_0
                    bsz = x_1.shape[0]
                    guidance_vec = torch.full((x_t.shape[0],), 4, device=x_t.device, dtype=x_t.dtype)

                    # Predict the noise residual and compute loss
                    model_pred = dit(img=x_t.to(weight_dtype),
                                    img_ids=inp['img_ids'].to(weight_dtype),
                                    txt=inp['txt'].to(weight_dtype),
                                    txt_ids=inp['txt_ids'].to(weight_dtype),
                                    y=inp['vec'].to(weight_dtype),
                                    timesteps=t.to(weight_dtype),
                                    guidance=guidance_vec.to(weight_dtype),)
                    
                    #loss = F.mse_loss(model_pred.float(), (x_1 - x_0).float(), reduction="mean")
                    loss = F.mse_loss(model_pred.float(), (x_0 - x_1).float(), reduction="mean")

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(configs.train_batch_size)).mean()
                    train_loss += avg_loss.item() / configs.gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(dit.parameters(), configs.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                prof.step()
                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0

                    if global_step % configs.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if configs.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(configs.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= configs.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - configs.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(configs.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(configs.output_dir, f"checkpoint-{global_step}")
                            os.makedirs(save_path, exist_ok=True)
                            torch.save(dit.state_dict(), os.path.join(save_path, 'dit.bin'))
                            torch.save(optimizer.state_dict(), os.path.join(save_path, 'optimizer.bin'))
                            #accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step >= configs.max_train_steps:
                    break

        print(prof.key_averages().table(
            sort_by="cuda_time_total", 
            row_limit=10
        ))

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
