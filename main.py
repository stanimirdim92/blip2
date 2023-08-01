import asyncio
import aiohttp
from asgiref import sync
from fastapi import FastAPI
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration , BitsAndBytesConfig
import torch
#from torch import bfloat16, float16
import time
#import uvicorn
#import requests
import os
from pydantic import BaseModel
from io import BytesIO

#threads_num = 8
#torch.set_num_threads(threads_num)
#os.environ["OM_NUM_THREADS"] = str(threads_num)
torch.set_grad_enabled(False)

#cache
use_cache=False
save_model = False


dtype=torch.bfloat16
model_name = "Salesforce/instructblip-vicuna-7b"
prompt_textbox="Can you tell me about this image in detail?"
save_dir = "./vit"

#nf8_config = BitsAndBytesConfig(
#     load_in_8bit=True,
#     llm_int8_enable_fp32_cpu_offload=True,
#)
nf4_config =  BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=dtype,
)

if use_cache == True and len(os.listdir(save_dir)) != 0:
    processor = InstructBlipProcessor.from_pretrained(save_dir)
    model = InstructBlipForConditionalGeneration.from_pretrained(save_dir,
        quantization_config=nf4_config,
    )
else:
    processor = InstructBlipProcessor.from_pretrained(model_name)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=nf4_config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
#    model.eval()
#    model.tie_weights()
    if save_model == True:
        processor.save_pretrained(save_dir)
        model.save_pretrained(save_dir)


# min_length=1,
# repetition_penalty=1.5,
# length_penalty=1.0,
generate_kwargs = dict(
    max_new_tokens=150,
    do_sample=False,
    min_new_tokens=10,
    temperature=0.75,
    num_beams=1,
    top_p=0.85,
    top_k=0,
)

def async_aiohttp_get_all(urls):
    async def get_all(urls):
        async with aiohttp.ClientSession() as session:
            async def fetch(url):
                async with session.get(url) as response:
                    return Image.open(BytesIO(await response.content.read())).convert("RGB")
            return await asyncio.gather(*[
                fetch(url) for url in urls
            ])
    # call get_all as a sync function to be used in a sync context
    return sync.async_to_sync(get_all)(urls)


def infer(image_path, prompt):
    t_start = time.time()

    texts = {}

    for idx, image in enumerate(async_aiohttp_get_all(image_path)):
        input_tokens = processor(images=image, text=prompt, return_tensors="pt").to(model.device, dtype)

        print("*** Running generate")

        outputs = model.generate(**input_tokens,**generate_kwargs)
        texts[image_path[idx]] = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        print(f"Start to finish: {time.time() - t_start:.3f} secs")

    return texts

############ FAST API ################

class ImageData(BaseModel):
     images: list[str]

app = FastAPI()

@app.post("/process_images/")
def process_images(image_data: ImageData):
    return {"output": infer(image_data.images, prompt_textbox)}
