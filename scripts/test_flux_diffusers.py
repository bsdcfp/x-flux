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
out.save("flux_astronaut.png")