# main.py
import torch
import cv2
import numpy as np
from torch import autocast
from diffusers import StableDiffusionPipeline
from io import BytesIO
import base64
from super_image import EdsrModel, ImageLoader
from PIL import Image


MODEL = "CompVis/stable-diffusion-v1-4"
warm = False
pipe = None


def prepPipeline():
    global warm
    global pipe
    global u_model

    if not warm:
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL, revision="fp16", torch_dtype=torch.float16)

        pipe.to("cuda")
        pipe.safety_checker = None
        pipe.requires_safety_checker = False

        u_model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4)

        warm = True

    return pipe


def generate(prompt: str, height: int, width: int, steps: int):
    pipe = prepPipeline()

    with autocast("cuda"):
        output = pipe(prompt, height=height, width=width,
                      num_inference_steps=steps)
    for image in output.images:
        input_1 = ImageLoader.load_image(image)
        preds_1 = u_model(input_1)
        
        ImageLoader.save_image(preds_1, "./out.png")
        
        im = cv2.imread("./out.png")
        dst = cv2.fastNlMeansDenoisingColored(im, None, 11, 6, 7, 21)
        cv2.imwrite("./out.png", dst)

        im = Image.open("./out.png")

        buffered = BytesIO()
        im.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        base64_img_str = img_str.decode("utf-8")

        return base64_img_str
