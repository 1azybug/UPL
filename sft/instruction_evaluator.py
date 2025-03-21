import json
import os
import sys

import matplotlib.pyplot as plt
import torch
from peft import PeftModel
from rouge import Rouge
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import numpy as np
import argparse
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import logging
from nltk.translate.bleu_score import sentence_bleu

from torch.nn import DataParallel
import torch.multiprocessing as mp

from instruction_prepare_data import get_examples
from model.modeling import get_model, save_adapter, load_adapter
from instruction_dataloader import get_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, default='instruction_rank-128_cl-lm_mrqa', required=False, help='Directory including the configuration file')
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='total batch size')
    return parser.parse_args()

class Evaluator:

    def __init__(self, config, work_dir, batch_size, tokenizer):
        self.config = config
        self.work_dir = work_dir
        self.batch_size = batch_size
        self.device_count = torch.cuda.device_count()
        self.tokenizer = tokenizer

    # def draw_loss(self):
    #     with open(os.path.join(self.work_dir,"instruction_info.json")) as f:
    #         info_list=json.load(f)

    #     lm_loss_values = [-1 if 'lm_loss' not in entry['training_loss'] else entry['training_loss']["lm_loss"] for entry in info_list]
    #     compress_loss_values = [-1 if 'compress_loss' not in entry['training_loss'] else entry['training_loss']["compress_loss"] for entry in info_list]
    #     step_values = [entry['steps'] for entry in info_list]
    #     lr_values = [entry['learning_rate'] for entry in info_list]
        
    #     plt.figure(figsize=(10, 5))
    #     if lm_loss_values[0] != -1:
    #         plt.plot(step_values, lm_loss_values, label="lm_loss")
    #     if compress_loss_values[0] != -1:
    #         plt.plot(step_values, compress_loss_values, label='compress_loss')


    #     plt.xlabel("step")
    #     plt.ylabel("loss")
    #     plt.title(self.work_dir)
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig(os.path.join(self.work_dir, 'instruction_training_loss.png'))
    #     plt.show()
    #     plt.close()


    def draw_ema_loss(self, alpha):

        def exponential_moving_average(values,alpha):
            ema = [values[0]]  # 初始化EMA的第一个值为原始值的第一个值
            for value in values[1:]:
                ema.append(alpha * value + (1 - alpha) * ema[-1])
            return ema

        with open(os.path.join(self.work_dir,"instruction_info.json")) as f:
            info_list=json.load(f)

        lm_loss_values = [-1 if 'lm_loss' not in entry['training_loss'] else entry['training_loss']["lm_loss"] for entry in info_list]

        step_values = [entry['steps'] for entry in info_list]
        lr_values = [entry['learning_rate'] for entry in info_list]

        
        plt.figure(figsize=(10, 5))
        if lm_loss_values[0] != -1:
            plt.plot(step_values, exponential_moving_average(lm_loss_values,alpha=alpha), label="lm_loss")

        plt.xlabel("step")
        plt.ylabel(f"loss(ema_alpha={alpha})")
        plt.title(f"{self.work_dir}_loss(ema_alpha={alpha})")
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(os.path.join(self.work_dir, 'instruction_ema_training_loss.png'))
        plt.close()

    def evaluate(self, rank=0):
        training_config = self.config["sft_training_config"]
        task_config = self.config["sft_task_config"]
        train_examples, eval_examples = get_examples(**self.config["data_config"])
        example_num_per_gpu = len(eval_examples)//training_config["device_count"]

        if rank <= self.device_count-2:
            eval_examples = eval_examples[rank*example_num_per_gpu:(rank+1)*example_num_per_gpu]
        else:
            eval_examples = eval_examples[rank*example_num_per_gpu:]

        print(f"[INFO] GPU{rank}: eval_examples[{rank*example_num_per_gpu}:{rank*example_num_per_gpu+len(eval_examples)}], nums:{len(eval_examples)}")

        dataset = get_dataset(task_config["task_type"], eval_examples, batch_size=self.batch_size)
        loader = DataLoader(dataset, batch_size=None)
        
        model = get_model(training_config["model_id"], task_config, rank)
        model = load_adapter(model, save_path_and_name=self.work_dir+'/instruction_adapter.pt', log=True)
        model.eval()

        info_list=[]
        with torch.no_grad():
            for inputs in tqdm(loader,total=len(eval_examples)//self.batch_size):
                inputs = {key:(value.to(rank) if value is not None else None) for key,value in inputs.items()}
                # output = model(inputs=inputs)
                generate_text = model.lm_inference(inputs)
                info_list.append({"generate_text": generate_text})

        with open(self.work_dir+f'/instruction_eval_info_list_{rank}.json', 'w', encoding='utf-8') as f:
            json.dump(info_list, f, ensure_ascii=False)
        
        
        
                
    def run(self, rank):
        # draw training loss
        if rank==0:
            # self.draw_loss()
            self.draw_ema_loss(alpha=0.1)
        self.evaluate(rank)


def evaluate(rank, args, world_size, tokenizer):

    with open(args.work_dir+"/output/config.json") as f:
        config=json.load(f)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(args.work_dir+f'/output/instruction_evaluate_info_rank{rank}.txt', mode='w'),
            logging.StreamHandler()
        ]
    )
    
    evaluator = Evaluator(config, args.work_dir+"/output", args.batch_size, tokenizer)
    evaluator.run(rank)

# def cal_avg_loss(args, config):
#     lm_loss = []
#     compress_loss = []
#     use_compress_loss = False
#     with open(args.work_dir+f'/instruction_info.json', 'r') as f:
#         data = json.load(f)
#         for run in data:
#             lm_loss.append(run['training_loss']['lm_loss'])
#             if 'compress_loss' in run['training_loss']:
#                 use_compress_loss = True
#                 compress_loss.append(run['training_loss']['compress_loss'])
#     avg_lm_loss = np.mean(lm_loss)
#     if use_compress_loss:
#         avg_compress_loss = np.mean(compress_loss)
#         return avg_lm_loss,avg_compress_loss
#     else:
#         return avg_lm_loss, -1

# def cal_cl_token_acc(cl_generate_text, examples_list, tokenizer):
#     correct_tokens = 0
#     total_tokens = 0
#     acc = []
#     info_list = []
#     for cl_gen_text, examples in zip(cl_generate_text, examples_list):
#         input_text = examples["context"]
#         cl_gen_text = tokenizer.decode(cl_gen_text, skip_special_tokens=True)
#         cl_gen_text = cl_gen_text.split("Context:\n ", 1)[-1]
#         cl_gen_ids = tokenizer(cl_gen_text, add_special_tokens=False)["input_ids"]
#         input_ids = tokenizer(input_text, add_special_tokens=False)["input_ids"]

#         total_tokens += len(cl_gen_ids)
#         correct_tokens += sum(1 for o,d in zip(cl_gen_ids, input_ids) if o == d)
#         acc.append(0)
#         # acc.append(correct_tokens / total_tokens)
#         correct_tokens = 0
#         total_tokens = 0
#         info_list.append({"input_text": input_text,
#                           "cl_generate_text": cl_gen_text})
#     with open(args.work_dir+f'/instruction_cl_generate_text.json', 'w', encoding='utf-8') as f:
#         json.dump(info_list, f, ensure_ascii=False, indent=4)
#     return np.mean(acc)


# Launch multi-process eval
if __name__ == "__main__":
    args = parse_args()
    world_size = torch.cuda.device_count()

    with open(args.work_dir + "/output/config.json") as f:
        config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(config["data_config"]["model_id"])

    sys.setrecursionlimit(10000)
    if not os.path.exists(args.work_dir+f'output/instruction_eval_info_list_0.json'):
        mp.spawn(evaluate,
                args=(args, world_size, tokenizer),
                nprocs=world_size,
                join=True)


    info_list = []
    for i in range(world_size):
        with open(args.work_dir+f'/output/instruction_eval_info_list_{i}.json', 'r', encoding='utf-8') as f:
            list_i =  json.load(f)
        info_list += list_i

    generate_text = [entry["generate_text"] for entry in info_list]

    print("calculate BLEU4...")
    instruction_dataset_name = config["data_config"]["instruction_dataset_repo"].split('/')[-1]
    if os.path.exists(f'{instruction_dataset_name}_test_instruction_dataset.json'):
        with open(f'{instruction_dataset_name}_test_instruction_dataset.json', 'r', encoding='utf-8') as f:
            examples_list =  json.load(f)

    instruction_inference_results = []
    bleu4_list = []
    rouge1_scores = []
    rouge = Rouge()
    for gen_text, example in zip(generate_text, examples_list):
        ##########################取其中一个答案用于训练即可##############################
        # answer = ""
        # for i in range(len(example["answers"])):
        #     if i == len(example["answers"])-1:
        #         answer += example["answers"][i]
        #     else:
        #         if example[("answers")][i] == example[("answers")][i+1]:
        #             answer += example["answers"][i]
        #             break
        #         answer += example[("answers")][i] + " "
        ################################################################################
        answer = example["answers"][0]
        ans_text = answer
        gen_text = tokenizer.decode(gen_text, skip_special_tokens=True)
        print("answer: " , answer)
        print("gen_text: " , gen_text)
        # if gen_text == "." or gen_text == "":
        #     gen_text = "test"

        gen_ids = tokenizer(gen_text, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(ans_text, add_special_tokens=False)["input_ids"]
        bleu4 = sentence_bleu([answer_ids], gen_ids, weights=(0.25, 0.25, 0.25, 0.25))
        example["generate"] = gen_text
        example["bleu4"] = bleu4

        if gen_text == "." or gen_text == "":
            example["rouge-f1"] = 0.0
            rouge1_scores.append(0.0)
        else:

            scores = rouge.get_scores(gen_text, ans_text)
            example["rouge-f1"] = scores[0]["rouge-1"]["f"]
            rouge1_scores.append(scores[0]["rouge-1"]["f"])

        instruction_inference_results.append(example)
        bleu4_list.append(bleu4)

    avg_bleu4 = np.mean(bleu4_list)
    print(f"avg_bleu4:{avg_bleu4}")
    # avg_lm_loss, avg_compress_loss = cal_avg_loss(args, config)
    # print(f"avg_lm_loss:{avg_lm_loss}")
    # print(f"avg_compress_loss:{avg_compress_loss}")
    rouge1_f1 = np.mean(rouge1_scores)
    print(f"rouge1_f1:{rouge1_f1}")
    with open(args.work_dir+f'/output/instruction_brief_eval_info.json', 'w', encoding='utf-8') as f:
        json.dump(f"avg_bleu4:{avg_bleu4}, rouge1_f1:{rouge1_f1}", f, ensure_ascii=False)

    with open(args.work_dir+f'/output/instruction_inference_results.json', 'w', encoding='utf-8') as f:
        json.dump(instruction_inference_results, f, ensure_ascii=False, indent=4)


"""
python ./pre_evaluator.py --work_dir CompressLLM --batch_size 1

CUDA_VISIBLE_DEVICES=0,1 HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=0 python ./pre_evaluator.py --work_dir CompressLLM --batch_size 1

Todo: multi gpu parallel inference
"""