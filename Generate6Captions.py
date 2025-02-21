import os
import torch
from PIL import Image
#from transformers import Blip2Processor, Blip2ForConditionalGeneration
import json
import PIL
import requests
import argparse
import tqdm
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import numpy as np

def decode_out(out,processor):
    output_ls = []
    for idx,sen in enumerate(out):
        output_text = processor.decode(sen, skip_special_tokens=True)
        output_ls.append(output_text)
    return output_ls
def get_image(VQG_path):
    image_name_ls = os.listdir(VQG_path)[:200]
    rs_image_dict = {image_name:VQG_path+'/'+image_name for image_name in image_name_ls}
    return rs_image_dict

# def processInBLIP2(model,image_path,processor,qtext,device,generate_config):
    
#     try:
#         image = Image.open(image_path).convert('RGB')
#     except PIL.UnidentifiedImageError:
#         print(f"Cannot open the image: {image_path}")
#         return None,None    
    
#     inputs = processor(image, qtext, return_tensors="pt").to(device, torch.float16)
#     topKtopPout = model.generate(**inputs,
#                               do_sample=True,
#                               top_k=generate_config['top_k'],
#                               top_p = generate_config['top_p'],
#                               temperature=generate_config['temperature'],
#                               num_return_sequences=generate_config['num_return_sequences_topK'],
#                               early_stopping=generate_config['early_stopping'],
#                               max_length=generate_config['max_length'])
#     top_k_text_ls = decode_out(out=topKtopPout,processor=processor)
#     beamsearchout = model.generate(**inputs,
#                                    num_beams = generate_config['num_beams'],
#                                    length_penalty= generate_config['length_penalty'],
#                                    early_stopping=generate_config['early_stopping'],
#                                    num_return_sequences=generate_config['num_return_sequences_beam'],
#                                    max_length=generate_config['max_length'])
#     beam_text_ls = decode_out(out = beamsearchout,processor=processor)
#     return top_k_text_ls,beam_text_ls

if __name__=='__main__': 
    parser = argparse.ArgumentParser(description='example')
    parser.add_argument("--pre_trained_model_path",type = str,default="/data/dtt/pretrained_model/llava-v15-7b",help='pretrained model')
    parser.add_argument('--dataset_path',type = str,default="/data/dtt/project/LLaVA-main/hallucination/code/new_VG_100K",help = 'path of dataset')
    parser.add_argument('--output_path',type = str,default= "/data/dtt/project/LLaVA-main/hallucination/code",help = 'output file path')
    args = parser.parse_args()

    pre_trained_model_path = args.pre_trained_model_path
    VQG_path = args.dataset_path
    image_files = get_image(VQG_path) # image_file = "view.jpg,view.jpg,view.jpg"
    #import ipdb;ipdb.set_trace()
    #qtext = "Describe the details in the picture, as detailed as possible"
    #qtext = "Generate a brief description for the image."
    #prompt = "Generate a brief description for the image."
    prompt = "Describe the details in the picture, as detailed as possible"
    # generate_config = {
    #     'top_k':15,
    #     'top_p':1,
    #     'temperature':1,
    #     'num_return_sequences_topK':4,
    #     'early_stopping':False,
    #     'max_length':50,
    #     'num_beams':20,
    #     'length_penalty':1,
    #     'num_return_sequences_beam':2,
    # }
    num_top_k = 4
    num_beam = 2
    none_image_name = []
    model_path = pre_trained_model_path
    model_base = None
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name
    )
    rs_dict = {}
    for image_name,image_path in tqdm.tqdm(image_files.items(),colour = 'red'):
        temp_rs_dict = {}
        args = type('Args', (), {
            "model":model,
            "model_name":model_name,
            "tokenizer":tokenizer,
            "image_processor":image_processor,
            "context_len":context_len,
            "query": prompt,
            "conv_mode": None,
            "image_file": image_path,
            "sep": ",",
            "temperature": 1,
            "top_k":15,
            "top_p": 1,
            'num_beams':1,
            'early_stopping':False,
            'max_length':50,
            'length_penalty':1,
            'max_new_tokens':50
        })()
        top_k_ls = []
        for i in range(num_top_k):
            top_kp_i = eval_model(args)
            top_k_ls.append(top_kp_i)
        
        temp_rs_dict['top_k_text'] = top_k_ls
        #temp_rs_dict['generate_config'] = args
        beam_ls = []
        args = type('Args', (), {
            "model":model,
            "model_name":model_name,
            "tokenizer":tokenizer,
            "image_processor":image_processor,
            "context_len":context_len,
            "query": prompt,
            "conv_mode": None,
            "image_file": image_path,
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            'num_beams':20,
            'max_new_tokens':50
        })()
        for j in range(num_beam):
            beam_str_j = eval_model(args)
            beam_ls.append(beam_str_j)
        temp_rs_dict['beam_text:'] = beam_ls
        rs_dict[image_name] = temp_rs_dict
        if beam_ls == None or top_k_ls == None:
            none_image_name.append(image_name)
        
    with open("/data/dtt/project/LLaVA-main/hallucination/code/original_details/6_captions_I2.json", "w") as file:
        json.dump(rs_dict, file)
    # rs_dict = {}
    # none_image_name = []
    # for image_name,image_path in tqdm.tqdm(rs_image_dict.items(),colour = 'red'):
    #     top_k_text_ls,beam_text_ls = processInBLIP2(model,image_path,processor,qtext,device,generate_config)
    #     temp_rs = {
    #         'top_k_text':top_k_text_ls,
    #         'beam_text:':beam_text_ls,
    #         'generate_config':generate_config,
    #     }
    #     rs_dict[image_name] = temp_rs
    #     if top_k_text_ls == None:
    #         none_image_name.append(image_name)
    # with open("VG_BLIP2_6_captions.json", "w") as file:
    #     json.dump(rs_dict, file)