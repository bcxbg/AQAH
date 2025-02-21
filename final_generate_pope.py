# import os
# import torch
# from PIL import Image
# #from transformers import Blip2Processor, Blip2ForConditionalGeneration
# import json
# import PIL
# import requests
# import argparse
# import tqdm
# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import get_model_name_from_path
# from llava.eval.run_llava import eval_model
# import numpy as np

# POPE_PATH_coco = {
#     "random": "pope_dataset/coco_pope_random.json",
#     "popular": "pope_dataset/coco_pope_popular.json",
#     "adversarial": "pope_dataset/coco_pope_adversarial.json",
# }
# POPE_PATH_aokvqa = {
#     "random": "pope_dataset/POPE/aokvqa/aokvqa_pope_random.json",
#     "popular": "pope_dataset/POPE/aokvqa/aokvqa_pope_popular.json",
#     "adversarial": "pope_dataset/POPE/aokvqa/aokvqa_pope_adversarial.json",
# }

# POPE_PATH_gqa = {
#     "random": "pope_dataset/POPE/gqa/gqa_pope_random.json",
#     "popular": "pope_dataset/POPE/gqa/gqa_pope_popular.json",
#     "adversarial": "pope_dataset/POPE/gqa/gqa_pope_adversarial.json",
# }

# POPE_PATH_ls = {
#     'coco':POPE_PATH_coco,
#     'aokvqa':POPE_PATH_aokvqa,
#     'gqa':POPE_PATH_gqa
# }

# def print_acc(pred_list, label_list,args):#,question_list,logits_list,attention_list,hidden_states_list):
#     pos = 1
#     neg = 0
#     yes_ratio = pred_list.count(1) / len(pred_list)
#     # unknown_ratio = pred_list.count(2) / len(pred_list)
#     count = 20
#     TP, TN, FP, FN = 0, 0, 0, 0
#     rs_list = []
    
#     for pred, label in zip(pred_list, label_list):
#         if pred == pos and label == pos:
#             TP += 1
#         elif pred == pos and label == neg:
#             FP += 1
#         elif pred == neg and label == neg:
#             TN += 1
#         elif pred == neg and label == pos:
#             FN += 1

#     print('TP\tFP\tTN\tFN\t')
#     print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

#     precision = float(TP) / float(TP + FP)
#     recall = float(TP) / float(TP + FN)
#     f1 = 2*precision*recall / (precision + recall)
#     acc = (TP + TN) / (TP + TN + FP + FN)
#     acc_txt = 'Accuracy: {}'.format(acc)
#     precision_txt = 'Precision: {}'.format(precision)
#     recall_txt = 'Recall: {}'.format(recall)
#     F1_txt = 'F1 score: {}'.format(f1)
#     yes_ratio = 'Yes ratio: {}'.format(yes_ratio)
#     print('Accuracy: {}'.format(acc))
#     print('Precision: {}'.format(precision))
#     print('Recall: {}'.format(recall))
#     print('F1 score: {}'.format(f1))
#     print('Yes ratio: {}'.format(yes_ratio))
#     output_path = "pope_output/original/original_VCD_blip2_{}_{}.txt".format(args.type_dataset,args.type_question)
#     with open(output_path,'w') as file:
#         file.write(acc_txt + '\n')
#         file.write(precision_txt+'\n')
#         file.write(recall_txt+'\n')
#         file.write(F1_txt+'\n')
#         file.write(yes_ratio+'\n')


# def recorder(out, pred_list):
#     NEG_WORDS = ["No", "not", "no", "NO"]
#     for line in out:

#         line = line.replace('.', '')
#         line = line.replace(',', '')
#         words = line.split(' ')
#         if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
#             pred_list.append(0)
#         else:
#             pred_list.append(1)
    
#     return pred_list

# def decode_out(out,processor):
#     output_ls = []
#     for idx,sen in enumerate(out):
#         output_text = processor.decode(sen, skip_special_tokens=True)
#         output_ls.append(output_text)
#     return output_ls
# def get_image(VQG_path):
#     image_name_ls = os.listdir(VQG_path)
#     rs_image_dict = {image_name:VQG_path+'/'+image_name for image_name in image_name_ls}
#     return rs_image_dict



# def get_processed_prompt(pre_prompt,image_value,question_value):
#     top_k_list = image_value["top_k_text"]
#     beam_list = image_value['beam_text:']
#     temp_idx = -1
#     temp_top_k_len = 128
#     for i in range(len(top_k_list)):
#         if top_k_list[i] == None:
#             continue
#         if len(top_k_list[i]) < temp_top_k_len:
#             temp_idx = i
#             temp_top_k_len = len(top_k_list[i])
#     rs_top_k = top_k_list[temp_idx] if temp_idx != -1 else None

#     temp_idx = -1
#     temp_beam_len = 128
#     for i in range(len(beam_list)):
#         if beam_list[i] == None:
#             continue
#         if len(beam_list[i]) < temp_beam_len:
#             temp_idx = i
#             temp_beam_len = len(beam_list[i])
#     rs_beam = beam_list[temp_idx] if temp_idx != -1 else None
#     rs_prompt = None
    
#     if (question_value[:8] == "Question") or (question_value[:8] == "question"):
#         pre_prompt = pre_prompt + "{}".format(question_value) + " answer:"
#     else:
#         pre_prompt = pre_prompt + "Question:{}".format(question_value) + " answer:"
#     rs_prompt = pre_prompt
#     if rs_top_k!= None:
#         rs_prompt = pre_prompt+rs_top_k.strip()
#     if rs_beam != None:
#         rs_prompt = rs_prompt + " and "+ rs_beam.strip() + "\'."
#         print(rs_prompt)
#     return rs_prompt
# if __name__=='__main__': 
#     parser = argparse.ArgumentParser(description='example')
#     parser.add_argument("--pre_trained_model_path",type = str,default="/data/dtt/pretrained_model/llava-v15-7b",help='pretrained model')
#     parser.add_argument('--dataset_path',type = str,default="/data/dtt/project/LLaVA-main/hallucination/code/new_VG_100K",help = 'path of dataset')
#     parser.add_argument('--output_path',type = str,default= "/data/dtt/project/LLaVA-main/hallucination/code",help = 'output file path')
#     args = parser.parse_args()

#     pre_trained_model_path = args.pre_trained_model_path
#     VQG_path = args.dataset_path
#     image_files = get_image(VQG_path) # image_file = "view.jpg,view.jpg,view.jpg"
#     #import ipdb;ipdb.set_trace()
#     #qtext = "Describe the details in the picture, as detailed as possible"
#     #qtext = "Generate a brief description for the image."
#     #prompt = "Generate a brief description for the image."

#     # generate_config = {
#     #     'top_k':15,
#     #     'top_p':1,
#     #     'temperature':1,
#     #     'num_return_sequences_topK':4,
#     #     'early_stopping':False,
#     #     'max_length':50,
#     #     'num_beams':20,
#     #     'length_penalty':1,
#     #     'num_return_sequences_beam':2,
#     # }
#     num_top_k = 4
#     num_beam = 2
#     none_image_name = []
#     model_path = pre_trained_model_path
#     model_base = None
#     model_name = get_model_name_from_path(model_path)
#     tokenizer, model, image_processor, context_len = load_pretrained_model(
#         model_path, model_base, model_name
#     )
#     total_question_dict = {}
#     answer_path = "/data/dtt/project/LLaVA-main/hallucination/code/question_answer/total_captions.json"
#     with open(file = answer_path,mode = 'r',encoding='utf-8') as file:
#         answer_dict = json.load(file)
#     rs_dict = {}
#     question_path = "/data/dtt/project/LLaVA-main/hallucination/code/generate_question/generate_question_6.json"
#     with open(file = question_path,mode = 'r',encoding= 'utf-8') as file:
#         total_question_dict = json.load(file)
#     #import ipdb;ipdb.set_trace()
#     #pre_prompt = "Generate a brief description for the image. Note that the existing question and answer: \'"
#     pre_prompt = "Describe the details in the picture, as detailed as possible Note that the existing question and answer: \'"
    
    
#     for image_name,image_path in tqdm.tqdm(image_files.items(),colour = 'red'):
#         if image_name not in total_question_dict:
#             continue
#         temp_rs_dict = {}
#         image_value = answer_dict[image_name]
#         question_value = total_question_dict[image_name] 
#         prompt = get_processed_prompt(pre_prompt,image_value,question_value)
#         args = type('Args', (), {
#             "model":model,
#             "model_name":model_name,
#             "tokenizer":tokenizer,
#             "image_processor":image_processor,
#             "context_len":context_len,
#             "query": prompt,
#             "conv_mode": None,
#             "image_file": image_path,
#             "sep": ",",
#             "temperature": 1,
#             "top_k":15,
#             "top_p": 1,
#             'num_beams':1,
#             'early_stopping':False,
#             'max_length':50,
#             'length_penalty':1,
#             'max_new_tokens':50
#         })()
#         #eval_model(args)
#         top_k_ls = []
#         for i in range(num_top_k):
#             top_kp_i = eval_model(args)
#             top_k_ls.append(top_kp_i)
        
#         temp_rs_dict['top_k_text'] = top_k_ls
#         #temp_rs_dict['generate_config'] = args
#         beam_ls = []
#         args = type('Args', (), {
#             "model":model,
#             "model_name":model_name,
#             "tokenizer":tokenizer,
#             "image_processor":image_processor,
#             "context_len":context_len,
#             "query": prompt,
#             "conv_mode": None,
#             "image_file": image_path,
#             "sep": ",",
#             "temperature": 0,
#             "top_p": None,
#             'num_beams':20,
#             'max_new_tokens':50
#         })()
#         for j in range(num_beam):
#             beam_str_j = eval_model(args)
#             beam_ls.append(beam_str_j)
#         temp_rs_dict['beam_text:'] = beam_ls
#         rs_dict[image_name] = temp_rs_dict
#         if beam_ls == None or top_k_ls == None:
#             none_image_name.append(image_name)
    
#     with open("/data/dtt/project/LLaVA-main/hallucination/code/final_generate/total_captions_500_2700_I2.json", "w") as file:
#         json.dump(rs_dict, file)
#     # rs_dict = {}
#     # none_image_name = []
#     # for image_name,image_path in tqdm.tqdm(rs_image_dict.items(),colour = 'red'):
#     #     top_k_text_ls,beam_text_ls = processInBLIP2(model,image_path,processor,qtext,device,generate_config)
#     #     temp_rs = {
#     #         'top_k_text':top_k_text_ls,
#     #         'beam_text:':beam_text_ls,
#     #         'generate_config':generate_config,
#     #     }
#     #     rs_dict[image_name] = temp_rs
#     #     if top_k_text_ls == None:
#     #         none_image_name.append(image_name)
#     # with open("VG_BLIP2_6_captions.json", "w") as file:
#     #     json.dump(rs_dict, file)




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
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

#from llava.utils import disable_torch_init
POPE_PATH_coco = {
    "random": "pope_dataset/coco_pope_random.json",
    "popular": "pope_dataset/coco_pope_popular.json",
    "adversarial": "pope_dataset/coco_pope_adversarial.json",
}
POPE_PATH_aokvqa = {
    "random": "pope_dataset/POPE/aokvqa/aokvqa_pope_random.json",
    "popular": "pope_dataset/POPE/aokvqa/aokvqa_pope_popular.json",
    "adversarial": "pope_dataset/POPE/aokvqa/aokvqa_pope_adversarial.json",
}

POPE_PATH_gqa = {
    "random": "pope_dataset/POPE/gqa/gqa_pope_random.json",
    "popular": "pope_dataset/POPE/gqa/gqa_pope_popular.json",
    "adversarial": "pope_dataset/POPE/gqa/gqa_pope_adversarial.json",
}

POPE_PATH_ls = {
    'coco':POPE_PATH_coco,
    'aokvqa':POPE_PATH_aokvqa,
    'gqa':POPE_PATH_gqa
}

def print_acc(pred_list, label_list,args):#,question_list,logits_list,attention_list,hidden_states_list):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)
    # unknown_ratio = pred_list.count(2) / len(pred_list)
    count = 20
    TP, TN, FP, FN = 0, 0, 0, 0
    rs_list = []
    
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    acc_txt = 'Accuracy: {}'.format(acc)
    precision_txt = 'Precision: {}'.format(precision)
    recall_txt = 'Recall: {}'.format(recall)
    F1_txt = 'F1 score: {}'.format(f1)
    yes_ratio = 'Yes ratio: {}'.format(yes_ratio)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    output_path = "pope_output/{}_blip2_{}_{}.txt".format(args.type_method,args.type_dataset,args.type_question)
    with open(output_path,'w') as file:
        file.write(acc_txt + '\n')
        file.write(precision_txt+'\n')
        file.write(recall_txt+'\n')
        file.write(F1_txt+'\n')
        file.write(yes_ratio+'\n')
        file.write('TP\tFP\tTN\tFN\t')
        file.write('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

def recorder(out, pred_list):
    NEG_WORDS = ["No", "not", "no", "NO"]
    for line in out:

        line = line.replace('.', '')
        line = line.replace(',', '')
        words = line.split(' ')
        if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
            pred_list.append(0)
        else:
            pred_list.append(1)
    
    return pred_list

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--max_new_tokens",type = int,default=128)
parser.add_argument("--type_question",type=str,default='popular')
parser.add_argument('--type_dataset',type = str,default='coco')
parser.add_argument('--type_method',type = str,default='AQAH')

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
#model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)
model = InstructBlipForConditionalGeneration.from_pretrained("/data/dtt/pretrain_model_or_weight/instruct_blip2",device_map = 'auto',quantization_config=quantization_config)
processor = InstructBlipProcessor.from_pretrained("/data/dtt/pretrain_model_or_weight/instruct_blip2")

# mean = (0.48145466, 0.4578275, 0.40821073)
# std = (0.26862954, 0.26130258, 0.27577711)
# norm = transforms.Normalize(mean, std)

question_ls = []
POPE_path = POPE_PATH_ls[args.type_dataset]
pope_path = POPE_path[args.type_question]
with open(pope_path,'r') as file:
    for line in file.readlines():
        question_ls.append(json.loads(line))
pred_list,label_list = [],[]


# T5
model_name = "/data/dtt/projects/hallucination/hallucination/code_coco/code/pretrained/t5_batch_size64_lr0.00005_num_epochs120"
tokenizer = T5Tokenizer.from_pretrained(model_name)
t5model = T5ForConditionalGeneration.from_pretrained(model_name)
t5model = t5model.to(device)
root_image_path = "/data/dtt/dataset/MSCOCO/val2014/" if args.type_dataset != 'gqa' else "/data/dtt/dataset/gqa/"
#inference
for inst in tqdm.tqdm(question_ls):
    
    # image
    image_path = root_image_path + inst['image']
    raw_image = Image.open(image_path)
    raw_image = raw_image.convert("RGB")
    #image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    if args.type_method != 'original':
        # generate 6 question 
        qu = "Describe the details in the picture."
        sample_inputs = processor(images=raw_image, text=qu, return_tensors="pt").to(device)

        with torch.inference_mode():
            with torch.no_grad():
                output_texts = model.generate(
                    **sample_inputs,
                    num_beams=10,
                    max_new_tokens=64,
                    min_length=1,
                    top_p=1,
                    repetition_penalty=1.5,
                    length_penalty=1,
                    num_return_sequences=6,
                    temperature=1,
                    )

        output_texts = processor.batch_decode(output_texts, skip_special_tokens=True)          
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
        re_generate_inputs = processor(images =  raw_image, text=re_prompt , return_tensors="pt").to(device)
        with torch.inference_mode():
            with torch.no_grad():
                re_output_text = model.generate(
                    **re_generate_inputs,
                    num_beams=1,
                    max_new_tokens=64,
                    min_length=1,
                    top_p=1,
                    repetition_penalty=1.5,
                    length_penalty=1,
                    num_return_sequences=1,
                    temperature=1,
                    )
        #import ipdb;ipdb.set_trace()
        #print(re_output_text)

        re_output_text = processor.batch_decode(re_output_text, skip_special_tokens=True)[0].strip()
        re_output_text = re_output_text.replace('<s>','')
        #print(re_output_text)
        # Final Generation
        pope_input = inst['text']
        
        final_prompt = "Note that '{}'. {}.".format(re_output_text,pope_input) #Please answer with yes or no.
        #print(final_prompt)
    else:
        final_prompt = inst['text']
    #final_prompt = '{}'.format(pope_input)
    print(final_prompt)
    final_inputs = processor(images =  raw_image, text=final_prompt , return_tensors="pt").to(device)
    # label
    label = inst['label']
    if label=='yes':
        label = 1
    else:
        label = 0
    label_list = label_list + [label]
    with torch.inference_mode():
        with torch.no_grad():
            output_pope = model.generate(
                **final_inputs,
                num_beams=5,
                max_new_tokens=128,
                min_length=1,
                top_p=1,
                repetition_penalty=1.5,
                length_penalty=1,
                num_return_sequences=1,
                temperature=1,
                )
    #print(output_pope)
    output_pope = processor.batch_decode(output_pope, skip_special_tokens=True)[0].strip()
    pred_list = recorder([output_pope], pred_list)
    
    #import ipdb;ipdb.set_trace()
    print(output_pope)
    print(pred_list)
    print('\n')

print_acc(pred_list,label_list,args)


#CUDA_VISIBLE_DEVICES=7 python final_generate_pope.py --type_question adversarial --type_dataset gqa
