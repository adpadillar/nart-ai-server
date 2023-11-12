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
        print("preparing pipeline")
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL, revision="fp16", torch_dtype=torch.float16)

        pipe.to("cuda")
        pipe.safety_checker = None
        pipe.requires_safety_checker = False

        u_model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4)

        warm = True
        print("pipeline ready!")

    return pipe


def generate(prompt: str, height: int, width: int, steps: int):
    pipe = prepPipeline()

    with autocast("cuda"):
        print("generating prompt: ", prompt)
        output = pipe(prompt, height=height, width=width,
                      num_inference_steps=steps)

    for image in output.images:
        print("upscaling...")
        input_1 = ImageLoader.load_image(image)
        preds_1 = u_model(input_1)
        
        ImageLoader.save_image(preds_1, "./out.png")
        
        #print("denoising...")
        #im = cv2.imread("./out.png")
        #dst = cv2.fastNlMeansDenoisingColored(im)
        #cv2.imwrite("./out.png", dst)

        im = Image.open("./out.png")

        buffered = BytesIO()
        print("saving image in buffer")
        im.save(buffered, format="PNG")
        print("encoding image to str")
        img_str = base64.b64encode(buffered.getvalue())
        print("decoding str to utf-8")
        base64_img_str = img_str.decode("utf-8")
        print("done! sending response")

        return base64_img_str
