#main.py
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# model = "CompVis/stable-diffusion-v1-4"   # default stable diffusion model
model = "OFA-Sys/small-stable-diffusion-v0" # small stable diffusion model
 
pipe = StableDiffusionPipeline.from_pretrained(model, revision="fp16", torch_dtype=torch.float16)
 
pipe.to("cuda")
 
prompts = [
	"a girl walking in the parisian streets at night surounded by lights",
	"a group of boys playing basketball in the streets of new orleans",
	"a jazz choir singing at an old night casino",
	"a guy playing videogames in hell"
]
 
def generateImage(prompt, filename):
	with autocast("cuda"):
		output = pipe(prompt)
	for image in output.images:
		image.save(filename)
 
for i, p in enumerate(prompts):
	generateImage(p, f"{i}.png")
