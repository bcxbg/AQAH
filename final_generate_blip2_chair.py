
import argparse
import torch
import os
import json
import sys
import os
from torchvision import transforms
from PIL import Image
import math
import matplotlib.pyplot as plt
from lavis.models import load_model_and_preprocess
import tqdm
from transformers import GenerationConfig
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer,AdamW


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--max_new_tokens",type = int,default=128)
parser.add_argument('--type_dataset',type = str,default='coco')
parser.add_argument('--type_prompt',type = str,default='I2')
parser.add_argument('--type_decoding',type = str,default='beam')
parser.add_argument('--type_method',type=str,default = 'AQAH')
args = parser.parse_args()

#disable_torch_init()
#from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
disable_torch_init()
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)

mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
norm = transforms.Normalize(mean, std)

total_image_path = "/data/dtt/projects/distil-cd-main/coco_dataset/image"
new_jsonl = []

# T5
model_name = "/data/dtt/projects/hallucination/hallucination/code_coco/code/pretrained/t5_batch_size64_lr0.00005_num_epochs120"
tokenizer = T5Tokenizer.from_pretrained(model_name)
t5model = T5ForConditionalGeneration.from_pretrained(model_name)
t5model = t5model.to(device)
root_image_path = "/data/dtt/dataset/MSCOCO/val2014/" if args.type_dataset != 'gqa' else "/data/dtt/dataset/gqa/"
#inference
for single_image_path in tqdm.tqdm(os.listdir(total_image_path)[:200]):
    
    # image
    image_id = int((single_image_path.split('/')[-1])[-10:-4])
    image_path = total_image_path + '/' + single_image_path
    raw_image = Image.open(image_path)
    raw_image = raw_image.convert("RGB")
    image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(device)


    # generate 6 question 
    qu = "Describe the details in the picture." if args.type_prompt == "I2" else "Provide a brief description of the given image."

    
    with torch.inference_mode():
        with torch.no_grad():
            output_texts = model.generate(
                samples = {'prompt':qu,
                            'image':norm(image_tensor)},
                num_beams=10,
                max_length=64,
                min_length=1,
                top_p=1,
                repetition_penalty=1.5,
                length_penalty=1,
                num_captions=6,
                temperature=1,
                )
    #import ipdb;ipdb.set_trace()
    #print(output_texts)

    # T5
    input_T5 = " "
    for i in range(len(output_texts)):
        input_T5 = input_T5 + '{}'.format(i+1) + output_texts[i]+'\n'
    T5_inputs = tokenizer(text = [input_T5],return_tensors = "pt",padding=True)
    T5_inputs_ids = T5_inputs['input_ids'].to(device)
    res = t5model.generate(T5_inputs_ids)
    T5_output = tokenizer.batch_decode(res, skip_special_tokens=True)
    #import ipdb;ipdb.set_trace()
    #print(T5_output)
    # Re-generate
    re_prompt = "{}".format(T5_output[0])
    with torch.inference_mode():
        with torch.no_grad():
            re_output_text = model.generate(
                samples = {'prompt':re_prompt,
                            'image':norm(image_tensor)},
                num_beams=1,
                max_length=64,
                min_length=1,
                top_p=1,
                repetition_penalty=1.5,
                length_penalty=1,
                num_captions=1,
                temperature=1,
                )[0]
    #import ipdb;ipdb.set_trace()
    #print(re_output_text)
    re_output_text = re_output_text.replace('<s>','')

    # Final Generation
    final_prompt = "<image>\nNote the question and answer pair:'{}.{}'\nPlease {}".format(T5_output[0],re_output_text,qu) if args.type_method == 'AQAH' else "<image>\nUSER:{}\nASSISTANT:".format(qu)
    print(final_prompt)
    num_beams = 5 if args.type_decoding == 'beam' else 1
   
    with torch.inference_mode():
        with torch.no_grad():
            output_pope = model.generate(
                samples = {'prompt':final_prompt,
                            'image':norm(image_tensor)},
                num_beams=num_beams,
                max_length=192,
                min_length=1,
                top_p=1,
                repetition_penalty=1.5,
                length_penalty=1,
                num_captions=1,
                temperature=1,
                
                )[0]
    
    print(output_pope)
    new_jsonl.append({"image_id":image_id,"caption":output_pope})
    print('\n')
with open('test_outputs/{}_blip2_{}_{}.jsonl'.format(args.type_method,args.type_prompt,args.type_decoding),'w') as file:
    for inst in new_jsonl:
        json.dump(inst,file)
        file.write('\n')


#CUDA_VISIBLE_DEVICES=5 python final_generate_blip2_chair.py
