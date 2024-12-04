import os
import sys
import torch
torch.cuda.memory._record_memory_history(
                max_entries=10_000_000)
from diffusers import FluxPipeline
# torch.cuda.memory._record_memory_history(True)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.flux.aip_profiler import memory_profiler

# @memory_profiler("./snapshot_mem_timeline_flux.1_infer.pickle")
def infer(pipe):
    prompt = "a tiny astronaut hatching from an egg on the moon"
    # warmup
    pipe(
            prompt=prompt,
            guidance_scale=3.5,
            height=768,
            width=1360,
            num_inference_steps=1,
        )
    # Non-default profiler schedule allows user to turn profiler on and off
    # on different iterations of the training loop;
    out = pipe(
            prompt=prompt,
            guidance_scale=3.5,
            height=768,
            width=1360,
            num_inference_steps=1,
        ).images[0]
    if isinstance(out, dict):
        output_tensor = next(iter(out.values()))
    else:
        output_tensor = out
    make_dot(output_tensor).render("pipe_graph", format="png")

    out.save("flux_astronaut.png")

if __name__ == "__main__":
    model_path="/model_zoo/flux.1_dev"
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    # pipe.enable_model_cpu_offload()

    infer(pipe)
    # torch.cuda.memory._dump_snapshot("./mem_prof_flux.1_2.pickle")
    # 在最后保存快照
    torch.cuda.memory._dump_snapshot("./snapshot_mem_timeline_flux.1_infer.pickle")
    # 停止记录
    torch.cuda.memory._record_memory_history(enabled=None) 