
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
    output_path = "pope_output/{}_llava_{}_{}.txt".format(args.type_method,args.type_dataset,args.type_question)
    with open(output_path,'w') as file:
        file.write(acc_txt + '\n')
        file.write(precision_txt+'\n')
        file.write(recall_txt+'\n')
        file.write(F1_txt+'\n')
        file.write(yes_ratio+'\n')


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

question_ls = []
POPE_path = POPE_PATH_ls[args.type_dataset]
pope_path = POPE_path[args.type_question]
with open(pope_path,'r') as file:
    for line in file.readlines():
        question_ls.append(json.loads(line))
pred_list,label_list = [],[]


# T5
#import ipdb;ipdb.set_trace()
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
    if args.type_method != 'original':
        #generate 6 question 
        qu = "<image>\nUSER:Describe the details in the picture, as detailed as possible\nASSISTANT:"
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
        pope_input = inst['text']
        
        final_prompt = "<image>\nUSER: Note that '{}' And you should answer the question:{}\nASSISTANT:".format(re_output_text,pope_input) #, please answer with yes or no.
        #final_prompt = final_prompt if args.type_method == 'AQAH' else "<image>\nUSER:{}\nASSISTANT:".format(pope_input)
    else:
        final_prompt = "<image>\nUSER:{}\nASSISTANT:".format(inst['text'])
    print(final_prompt)
    final_inputs = processor(images = raw_image, text=final_prompt , return_tensors="pt").to(device,torch.float16)

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
                input_ids=final_inputs ['input_ids'],  
                pixel_values=final_inputs ['pixel_values'],
                attention_mask=final_inputs ['attention_mask'],
                num_beams=1,
                top_p = 1,
                repetition_penalty=1,
                do_sample = True,
                num_return_sequences = 1,
                max_new_tokens = 32,
                generation_config = generation_config
                )
    torch.cuda.empty_cache()
    output_pope_text = processor.batch_decode(output_pope, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    output_pope_text = output_pope_text.split("ASSISTANT:")[-1]
    pred_list = recorder([output_pope_text], pred_list)
    #print(pred_list)
    #import ipdb;ipdb.set_trace()
    print(output_pope_text)
    print('\n')

print_acc(pred_list,label_list,args)


#CUDA_VISIBLE_DEVICES=1 python final_generate_pope_llava.py --type_question adversarial --type_dataset gqa
