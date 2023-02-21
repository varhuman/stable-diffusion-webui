import json
import requests
import io
import base64
from PIL import Image, PngImagePlugin

url = "http://127.0.0.1:7860"

option_payload = {
    "sd_model_checkpoint": "basil mix.ckpt",
    "CLIP_stop_at_last_layers": 2
}

response = requests.post(url=f'{url}/sdapi/v1/options', json=option_payload)

payload = {
    "prompt": "pink eyes, pink panties, white bra, pink hair, masterpiece",
    "steps": 110,
    "Sampler": "Euler a",
    "CFG scale": 7,
    "Seed": 1589504574,
    "Face restoration": "GFPGAN",
    "Size": "512x512",
    "Model hash": "bbf07e3a1c",
    "AddNet Enabled": True, 
    "AddNet Module 1": "LoRA",                                                           
    "AddNet Model 1": "日服v1(9b32c25cb009)", 
    "AddNet Weight A 1": 0.5, 
    "AddNet Weight B 1": 0.5, 
    "AddNet Module 2": "LoRA", 
    "AddNet Model 2": "koreanDollLikeness_v10(e2e472c06607)", 
    "AddNet Weight A 2": 0.7, 
    "AddNet Weight B 2": 0.7
}

response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)

r = response.json()

for i in r['images']:
    image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))

    png_payload = {
        "image": "data:image/png;base64," + i
    }
    response2 = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)

    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("parameters", response2.json().get("info"))
    image.save('./fortest/output/output.png', pnginfo=pnginfo)