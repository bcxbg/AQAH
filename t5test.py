from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer,AdamW
import torch
from transformers import get_scheduler
import re
from easydict import EasyDict as EDict
import time
import random
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import json
import copy
import tqdm
import argparse

def get_BLUE(original_text,predict_text):
    mean_bleu_score = 0
    for i in range(len(predict_text)):
        reference = [original_text[i].split()]
        prediction = predict_text[i].split()
        print('\n')
        print("reference:{}".format(reference))
        print("prediction:{}".format(prediction))
        weights = [0.5, 0.5, 0, 0]
        smoothing_function = SmoothingFunction().method3
        bleu_score = sentence_bleu(reference, prediction,weights = weights,smoothing_function=smoothing_function)
        mean_bleu_score+=bleu_score
        print("BLEU score for prediction", i+1, ":", bleu_score)
        print("\n")
        time.sleep(1)
    mean_bleu_score/=len(predict_text)
    print("mean:{}".format(mean_bleu_score))
    return mean_bleu_score
def get_inputs(tokenizer,input_strings_ls):
    inputs = tokenizer(text = input_strings_ls,return_tensors = "pt",padding=True)
    '''
    {'input_ids': tensor([[ 100,   19,    8,  166, 7142,    5,    1],
        [ 100,   19,    8,  511,   80,    5,    1]]),
    'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1]])}
    '''
    return inputs
def eval(model,tokenizer,val_loader,config):
    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
    rs_dict = {}
    model.to(device)
    with torch.no_grad():
        for step in tqdm.tqdm(range(len(val_loader)),colour = 'red'):
            batch = val_loader[step]
            image_id_list = []
            image_value_list = []
            
            for image_dict in batch:
                for image_id,image_value in image_dict.items():
                    image_id_list.append(image_id)
                    image_value_list.append(image_value)
            inputs = get_inputs(tokenizer=tokenizer,input_strings_ls=image_value_list)
            inputs_ids = inputs['input_ids'].to(device)
            res = model.generate(inputs_ids)
            output = tokenizer.batch_decode(res, skip_special_tokens=True)
            for idx,image_id in enumerate(image_id_list):
                rs_dict[image_id] = output[idx]
                
    with open(config.output_dir,encoding = 'latin-1',mode = 'a+') as file:
        json.dump(rs_dict,file)
    return rs_dict

def process_dataset_blip2(dataset_dicts):
    #total_file = json.load(dataset_path)
    new_dict = {}
    for image_id,image_value in dataset_dicts.items():
        if image_id == "none_image_name":
            continue
        question1 = image_value["top_k_text"]
        question2 = image_value["beam_text:"]
        if question1 != None and question2 != None:
            question = question1+question2
        else:
            if question1 != None:
                question = question1
            else:
                if question2 != None:
                    question = question2
                else:
                    continue
        image_value["total_question"] = question
        new_dict[image_id] = image_value
    
    return list(new_dict.items())


def get_process_question(image_value):
    details_list = image_value['total_question']
    total_detail = ""
    for idx,single_detail in enumerate(details_list):
        if single_detail == None:
            #import ipdb;ipdb.set_trace()
            continue
        total_detail = total_detail+"{}. ".format(idx+1)+single_detail+"\n "
    return total_detail
def get_dataloader_blip2(dataset_ls,batch_size):
    total_single_batch = []
    single_batch = []
    for idx,(image_id,image_value) in enumerate(dataset_ls):
        if len(single_batch) < batch_size:
            #import ipdb;ipdb.set_trace()
            single_batch.append({image_id:get_process_question(image_value)})
        else:
            total_single_batch.append(copy.deepcopy(single_batch))
            single_batch = []
            single_batch.append({image_id:get_process_question(image_value)})
    if len(single_batch) != 0:
        total_single_batch.append(single_batch)
    print("total_single_batch:{}".format(len(total_single_batch)))
    return total_single_batch
    
def combine_dict(dict_paths):
    dicts_ls = []
    for single_path in dict_paths:
        #import ipdb;ipdb.set_trace()
        with open(single_path,encoding = 'latin-1',mode = 'r') as file:
            single_dict = json.load(file)
        dicts_ls.append(single_dict)
    rs_dict = {}
    for single_dict in dicts_ls:
        rs_dict.update(single_dict)
    return rs_dict

    
def plot_acc(acc_ls):
    epochs = range(1, len(acc_ls) + 1)
    plt.plot(epochs, acc_ls, marker='o')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.grid(True)
    plt.savefig("accuracy.pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count_flag", type = int, default=5)
    parser.add_argument("--batch_size", type = int, default=16)
    args = parser.parse_args()  
    model_name = "/data/dtt/project/blip2/video_caption_decoder/t5_batch_size64_lr0.00005_num_epochs120"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    count_flag = args.count_flag
    dataset_path = "/data/dtt/project/LLaVA-main/hallucination/code/original_details/6_captions_I2.json"
    dataset_path_ls = [dataset_path]
    total_dicts = combine_dict(dataset_path_ls)
    dict_question = process_dataset_blip2(total_dicts)
    #import ipdb;ipdb.set_trace()
    random.shuffle(dict_question)
    total_length = len(dict_question)
    print("total_lenth:{}".format(total_length))
    config = EDict(
        output_dir ="/data/dtt/project/LLaVA-main/hallucination/code/generate_question/generate_question_6_I2.json"
    )
    total_loader = get_dataloader_blip2(dict_question,args.batch_size)
    rs_dict = eval(model,tokenizer,total_loader,config)
    