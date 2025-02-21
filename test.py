from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
model = InstructBlipForConditionalGeneration.from_pretrained("/data/dtt/pretrain_model_or_weight/instruct_blip2",device_map = 'auto',quantization_config=quantization_config)
processor = InstructBlipProcessor.from_pretrained("/data/dtt/pretrain_model_or_weight/instruct_blip2")

device = "cuda" if torch.cuda.is_available() else "cpu"
#model.to(device)
url = "/data/dtt/projects/distil-cd-main/COCO_val2014_000000000133.jpg"
image = Image.open(url).convert("RGB")
prompt = "What is unusual about this image?"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
#import ipdb;ipdb.set_trace()
outputs = model.generate(
    **inputs,
    do_sample=False,
    num_beams=5,
    max_length=256,
    min_length=1,
    top_p=0.9,
    repetition_penalty=1.5,
    length_penalty=1.0,
    temperature=1,
    num_return_sequences=1,
)
import ipdb;ipdb.set_trace()
generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
print(generated_text)