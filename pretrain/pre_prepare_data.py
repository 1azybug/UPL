from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn
import os
import random
from tqdm import tqdm
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, required=True, help='Directory including the configuration file')
    return parser.parse_args()


def get_long_text_list(dataset_repo, output_dir, min_len, max_len):
    # cache long text for preventing full dataset traversal on each preparation. 
    if os.path.exists(f'{output_dir}/long_text.json'):
        with open(f'{output_dir}/long_text.json', 'r', encoding='utf-8') as f:
            long_text_list =  json.load(f)
        return long_text_list

    dataset = load_dataset(dataset_repo, split="train", streaming=True)

    long_text_list = []
    for example in tqdm(dataset, desc="Processing examples"):
        if min_len*2 <= len(example["text"]) <= max_len*6:  # one token \approx 2~6 char, here filter very long and very short text
            long_text_list.append(example["text"])
        
    with open(f'{output_dir}/long_text.json', 'w', encoding='utf-8') as f:
        json.dump(long_text_list, f, ensure_ascii=False)

    return long_text_list

    

def get_examples(model_id, dataset_repo, samples_num, min_len, max_len, instruction_dataset_repo, output_dir):
    model_name = model_id.split('/')[-1]
    train_data_name = f"{output_dir}/train_"+model_name+"_"+str(samples_num)+f"samples_{min_len}-{max_len}len_sorted.pt"
    eval_data_name = f"{output_dir}/eval_"+model_name+"_"+str(samples_num)+f"samples_{min_len}-{max_len}len.pt"

    if os.path.exists(train_data_name):
        print("loading data...")
        return torch.load(train_data_name), torch.load(eval_data_name)
    print(f"preparing data :train_data_name:{train_data_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    long_text_list = get_long_text_list(dataset_repo, output_dir, min_len, max_len)

    examples = []
    for text in tqdm(long_text_list, desc="Processing examples"):
        
        ids = tokenizer(text)["input_ids"]
        
        if len(ids)<min_len:
            continue
        if len(ids)>max_len:
            continue
        # half for prefix, half for LM
        last_start = len(ids) // 2
        
        inputs = torch.LongTensor(ids[:last_start])
        lm_target = torch.LongTensor(ids[last_start:])
        examples.append({"inputs":inputs,"lm_target":lm_target})

        if len(examples) == samples_num+1000:
            break

    # 分割验证集和训练集，并对训练集排序
    eval_examples = examples[:1000]
    train_examples = examples[1000:]
    
    print(train_examples[0]["inputs"].shape)
    # 按输入长度从短到长排序训练集
    train_examples_sorted = sorted(train_examples, key=lambda x: x["inputs"].size(0))
    print(train_examples_sorted[0]["inputs"].shape)
    
    # 保存处理后的数据
    torch.save(train_examples_sorted, train_data_name)
    torch.save(eval_examples, eval_data_name)


    print(len(train_examples_sorted))
    print(len(eval_examples))
    return train_examples_sorted, eval_examples

    
if __name__ == "__main__":

    args = parse_args()
    with open(args.work_dir+"/config.json") as f:
        config=json.load(f)
    
    training_config = config["pretrain_training_config"]
    config["data_config"]["model_id"] = training_config["model_id"]

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config["data_config"]["output_dir"] = output_dir

    train_examples, eval_examples = get_examples(**config["data_config"])

"""
cd pretrain
python pre_prepare_data.py --work_dir '../experiment/debug/quick'

"""