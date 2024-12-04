import torch
from diffusers import StableDiffusion3Pipeline
model_path="/model_zoo/stable-diffusion-3-medium-diffuser/"
pipe = StableDiffusion3Pipeline.from_pretrained(model_path)
pipe.to("cuda")
prompt = "a tiny astronaut hatching from an egg on the moon"
# prompt="a photo of a cat holding a sign that says hello world",
image = pipe(
    prompt=prompt,
    negative_prompt="",
    num_inference_steps=1,
    height=768,
    width=1360,
    guidance_scale=3.5,
).images[0]

trace_file = "trace_prof_sd3.json"
with torch.profiler.profile(
    # Currently only supports events on CUDA/GPU side, so do not add ProfilerActivity.CPU
    # this restriction is being removed
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    ) as prof:
        image = pipe(
            prompt=prompt,
            negative_prompt="",
            num_inference_steps=1,
            height=768,
            width=1360,
            guidance_scale=3.5,
        ).images[0]
    
prof.export_chrome_trace(trace_file)
# image.save("sd3_astronaut.png")