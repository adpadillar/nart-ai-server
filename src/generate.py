# main.py
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from io import BytesIO
import base64


MODEL = "CompVis/stable-diffusion-v1-4"
warm = False
pipe = None


def prepPipeline():
    global warm
    global pipe

    if not warm:
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL, revision="fp16", torch_dtype=torch.float16)

        pipe.to("cuda")
        pipe.safety_checker = None
        pipe.requires_safety_checker = False

        warm = True

    return pipe


def generate(prompt: str, height: int, width: int, steps: int):
    pipe = prepPipeline()

    with autocast("cuda"):
        output = pipe(prompt, height=height, width=width,
                      num_inference_steps=steps)
    for image in output.images:
        # base64 encode image
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        base64_img_str = img_str.decode("utf-8")

        return base64_img_str
