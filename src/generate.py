# main.py
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from io import BytesIO
import base64

output = BytesIO()

MODEL = "CompVis/stable-diffusion-v1-4"


def generate(prompt: str):
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL, revision="fp16", torch_dtype=torch.float16)

    pipe.to("cuda")

    with autocast("cuda"):
        output = pipe(prompt)
    for image in output.images:
        # base64 encode image
        image.save(output, format="JPEG")
        image_data = output.getvalue()
        base_64_image = base64.b64encode(image_data)
        if not isinstance(image_data, str):
            # Python 3, decode from bytes to string
            image_data = image_data.decode()
        data_url = 'data:image/jpg;base64,' + image_data

        return data_url
