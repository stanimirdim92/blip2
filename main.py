from fastapi import FastAPI
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration , BitsAndBytesConfig
import torch
import time
import requests
import os
#from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch, dispatch_model
#from huggingface_hub import snapshot_download


threads_num = 8
torch.set_num_threads(threads_num)
os.environ["OM_NUM_THREADS"] = str(threads_num)
torch.set_grad_enabled(False)

#os.environ['TRANSFORMERS_CACHE'] = './cache/'
#time
t_start = time.time()

#cache
use_cache=False
save_model = False


dtype=torch.bfloat16
model_name = "Salesforce/instructblip-vicuna-7b"
prompt_textbox="Can you tell me about this image in detail?"
save_dir = "./vit"
#weights_path = snapshot_download(repo_id="Salesforce/instructblip-vicuna-7b")
images = [
"https://stanimir.dev.smart-ads.eu/storage/images/2023/06/05/Hckj0opmngxNGTDDoRZ840VswS2HuN4n.png",
"https://petyo.processing.smart-ads.eu/storage/images/website/d91ac1aee382c197a71d355186a93dbe/BavarietCA_quer.jpg",
"https://petyo.processing.smart-ads.eu/storage/images/website/d91ac1aee382c197a71d355186a93dbe/GXNiRW_.jpeg",
"https://petyo.processing.smart-ads.eu/storage/images/website/d91ac1aee382c197a71d355186a93dbe/Artbeat_quer.jpg",
"https://petyo.processing.smart-ads.eu/storage/images/website/d91ac1aee382c197a71d355186a93dbe/logo_.png",
"https://petyo.processing.smart-ads.eu/storage/images/website/d91ac1aee382c197a71d355186a93dbe/SaschaKlaarXXL_.jpg",
"https://petyo.processing.smart-ads.eu/storage/images/website/d91ac1aee382c197a71d355186a93dbe/bg_saal.jpg",
"https://petyo.processing.smart-ads.eu/storage/images/website/d91ac1aee382c197a71d355186a93dbe/sddefault.jpg",
]

#nf8_config = BitsAndBytesConfig(
#     load_in_8bit=True,
#     llm_int8_enable_fp32_cpu_offload=True,
#     llm_int8_has_fp16_weight=True,
#)
nf4_config =  BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=dtype,
#    llm_int8_enable_fp32_cpu_offload=True,
)


def infer(image_path, prompt):
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
    #        local_files_only=True
        )
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
        temperature=0.8,
        num_beams=1,
        top_p=0.85,
        top_k=0,
    )

    texts = {}
    with torch.no_grad():
        for image_num in  image_path:
            with Image.open(requests.get(image_num, stream=True).raw).convert("RGB") as image:
                input_tokens = processor(images=image, text=prompt,return_tensors="pt").to(model.device, dtype)

                print("*** Running generate")

                outputs = model.generate(**input_tokens,**generate_kwargs)
                texts[image_num] = processor.batch_decode(outputs, skip_special_tokens=True)[0]
#                print(image_num,processor.batch_decode(outputs, skip_special_tokens=True)[0])
                print(f"Start to finish: {time.time() - t_start:.3f} secs")

    return texts



############ FAST API ################
app = FastAPI()

@app.get("/")
def read_root():
    return {"asd": "test4e"}
    return {"output": infer(images, prompt_textbox)}

