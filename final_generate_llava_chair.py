
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
# from lavis.models import load_model_and_preprocess
import tqdm
from transformers import GenerationConfig
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer,AdamW
from transformers import LlavaForConditionalGeneration,  GenerationConfig,AutoProcessor

#from llava.utils import disable_torch_init

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--max_new_tokens",type = int,default=128)
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

model = LlavaForConditionalGeneration.from_pretrained("/data/dtt/pretrain_model_or_weight/llava-1.5-7b-hf",device_map = "auto",quantization_config=quantization_config)
processor = AutoProcessor.from_pretrained("/data/dtt/pretrain_model_or_weight/llava-1.5-7b-hf")
generation_config = GenerationConfig(
        pad_token_id = processor.tokenizer.pad_token_id,
        eos_token_id = processor.tokenizer.eos_token_id,
        bos_token_id = processor.tokenizer.bos_token_id,
        output_attentions = True,
        output_hidden_states = True,
        output_scores = True,
        output_logits = True,
        return_dict_in_generate = False,
    )

# mean = (0.48145466, 0.4578275, 0.40821073)
# std = (0.26862954, 0.26130258, 0.27577711)
# norm = transforms.Normalize(mean, std)
total_image_path = "/data/dtt/projects/distil-cd-main/coco_dataset/image"
new_jsonl = []
# T5
#import ipdb;ipdb.set_trace()
model_name = "/data/dtt/projects/hallucination/hallucination/code_coco/code/pretrained/t5_batch_size64_lr0.00005_num_epochs120"
tokenizer = T5Tokenizer.from_pretrained(model_name)
t5model = T5ForConditionalGeneration.from_pretrained(model_name)
t5model = t5model.to(device)
#inference
for single_image_path in tqdm.tqdm(os.listdir(total_image_path)[:250]):
    
    # image
    image_id = int((single_image_path.split('/')[-1])[-10:-4])
    image_path = total_image_path + '/' + single_image_path
    raw_image = Image.open(image_path)
    raw_image = raw_image.convert("RGB")
   
    # generate 6 question 
    original_qu = "Describe the details in the picture." if args.type_prompt == "I2" else "Provide a brief description of the given image."
    qu = "<image>\nUSER:{}\nASSISTANT:".format(original_qu) 
    
    first_caption_inputs = processor(images =  raw_image, text=qu, return_tensors="pt").to(device,torch.float16)

    with torch.inference_mode():
        with torch.no_grad():
            first_outputs = model.generate(
                input_ids=first_caption_inputs['input_ids'],  
                pixel_values=first_caption_inputs['pixel_values'],
                attention_mask=first_caption_inputs['attention_mask'],
                num_beams=6,
                top_p = 1,
                repetition_penalty=1,
                do_sample = True,
                num_return_sequences = 6,
                max_new_tokens = 64,
                generation_config = generation_config
                )
    #import ipdb;ipdb.set_trace()
    first_output_texts = processor.batch_decode(first_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    torch.cuda.empty_cache()
    #import ipdb;ipdb.set_trace()
    #print(output_texts)

    # T5
    input_T5 = " "
    for i in range(len(first_output_texts)):
        temp_prompt = first_output_texts[i].split("ASSISTANT:")[-1]
        input_T5 = input_T5 + '{}.'.format(i+1) + temp_prompt +'\n'
    T5_inputs = tokenizer(text = [input_T5],return_tensors = "pt",padding=True)
    T5_inputs_ids = T5_inputs['input_ids'].to(device)
    #import ipdb;ipdb.set_trace()
    res = t5model.generate(T5_inputs_ids)
    T5_output = tokenizer.batch_decode(res, skip_special_tokens=True)
    #import ipdb;ipdb.set_trace()
    #print(T5_output)
    # Re-generate
    re_prompt = "<image>\nUSER:{}\nASSISTANT:".format(T5_output[0])
    regenerate_inputs = processor(images =  raw_image, text=re_prompt , return_tensors="pt").to(device,torch.float16)
    torch.cuda.empty_cache()
    with torch.inference_mode():
        with torch.no_grad():
            re_output = model.generate(
                input_ids=regenerate_inputs['input_ids'],  
                pixel_values=regenerate_inputs['pixel_values'],
                attention_mask=regenerate_inputs['attention_mask'],
                num_beams=1,
                top_p = 1,
                repetition_penalty=1,
                do_sample = True,
                num_return_sequences = 1,
                max_new_tokens = 64,
                generation_config = generation_config
                )
    torch.cuda.empty_cache()   
    #import ipdb;ipdb.set_trace()
    #print(re_output_text)
    #re_output_text = re_output_text.replace('<s>','')
    re_output_text = processor.batch_decode(re_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    re_output_text = re_output_text.split("ASSISTANT:")[-1]
    
    
    # Final Generation
    final_prompt = "<image>\nUSER: Note the question and answer pair:'{}.{}'\nPlease {}\nASSISTANT:".format(T5_output[0],re_output_text,original_qu) if args.type_method == 'AQAH' else "<image>\nUSER:{}\nASSISTANT:".format(original_qu)
    print(final_prompt)
    final_inputs = processor(images = raw_image, text=final_prompt , return_tensors="pt").to(device,torch.float16)
    num_beams = 5 if args.type_decoding == 'beam' else 1
    top_k = 15 if args.type_decoding != 'beam' else None
   
   
    with torch.inference_mode():
        with torch.no_grad():
            output_pope = model.generate(
                input_ids=final_inputs ['input_ids'],  
                pixel_values=final_inputs ['pixel_values'],
                attention_mask=final_inputs ['attention_mask'],
                num_beams=num_beams,
                top_p = 1,
                repetition_penalty=1,
                do_sample = True,
                num_return_sequences = 1,
                max_new_tokens = 128,
                generation_config = generation_config,
                top_k = top_k
                )
    torch.cuda.empty_cache()
    output_pope_text = processor.batch_decode(output_pope, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    output_pope_text = output_pope_text.split("ASSISTANT:")[-1]
    print(output_pope_text)
    new_jsonl.append({"image_id":image_id,"caption":output_pope_text})
    #pred_list = recorder([output_pope_text], pred_list)
    #print(pred_list)
    #import ipdb;ipdb.set_trace()
    #print(output_pope_text)
    print('\n')

with open('test_outputs/{}_llava_{}_{}.jsonl'.format(args.type_method,args.type_prompt,args.type_decoding),'w') as file:
    for inst in new_jsonl:
        json.dump(inst,file)
        file.write('\n')
#print_acc(pred_list,label_list,args)


#CUDA_VISIBLE_DEVICES=5 python final_generate_llava_chair.py 
