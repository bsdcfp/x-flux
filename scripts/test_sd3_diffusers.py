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
    num_inference_steps=50,
    height=768,
    width=1360,
    guidance_scale=3.5,
).images[0]

image.save("sd3_astronaut.png")