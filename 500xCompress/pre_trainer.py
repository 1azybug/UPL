import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LinearLR, ConstantLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch.multiprocessing as mp
import os
import time
import json
from tqdm import tqdm
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

from pre_prepare_data import get_examples
from pre_modeling import get_model, save_adapter, load_adapter
from pre_dataloader import get_dataset

import logging
import wandb

# 配置日志，同时输出到屏幕和文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('log.txt', mode='w'),
        logging.StreamHandler()
    ]
)
def get_wsd_scheduler(optimizer, training_steps):

    W = 300
    S = training_steps - W
    D = 0

    warmup_scheduler = LinearLR(optimizer, start_factor=1/W, total_iters=W)
    stable_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=S)
    final_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=0)

    milestones = [W, W+S]

    wsd_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, stable_scheduler, final_scheduler], milestones=milestones)

    return wsd_scheduler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, default='project',required=False, help='Directory including the configuration file, for saving model')
    parser.add_argument('--port', type=str, default='14527', required=False, help='port for ddp training')
    return parser.parse_args()


# Initialize process group
def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # Initialize the distributed environment
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def training_step(ddp_model, inputs, rank, accumulation_steps):
    inputs = {key:value.to(rank) for key,value in inputs.items()}
    output = ddp_model(inputs=inputs)
    loss = output["loss"]
    loss /= accumulation_steps
    loss.backward()
    # 计算当前的梯度范数
    grad_norm = calculate_gradient_norm(ddp_model)
    output["loss_info"]["grad_norm"] = grad_norm
    return output["loss_info"]

def calculate_gradient_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:  # 检查梯度是否存在
            param_norm = param.grad.data.norm(2)  # 计算每个参数梯度的L2范数
            total_norm += param_norm.item() ** 2  # 累加每个梯度范数的平方
    total_norm = total_norm ** 0.5  # 求平方根得到整体L2范数
    return total_norm

def count_parameters(model, config):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embedding_params = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and ('lm_head' in name or 'emb' in name))
    non_embedding_params = trainable_params - embedding_params
    
    config["Total_parameters"] = total_params
    config["Trainable_parameters"] = trainable_params
    config["Embedding_parameters"] = embedding_params
    config["non-Embedding_parameters"] = non_embedding_params
    
    # print(f"Total parameters: {total_params}")
    # print(f"Trainable parameters: {trainable_params}")
    # print(f"Embedding parameters: {embedding_params}")
    # print(f"non-Embedding parameters: {non_embedding_params}")
    
    # embedding_percentage = (embedding_params / total_params) * 100
    # print(f"Embedding parameters percentage: {embedding_percentage:.2f}%")
    
    # trainable_percentage = (trainable_params / total_params) * 100
    # print(f"Trainable parameters percentage: {trainable_percentage:.2f}%")
    logging.info(f"Total parameters: {total_params}")
    logging.info(f"Trainable parameters: {trainable_params}")
    logging.info(f"Embedding parameters: {embedding_params}")
    logging.info(f"non-Embedding parameters: {non_embedding_params}")

    embedding_percentage = (embedding_params / total_params) * 100
    logging.info(f"Embedding parameters percentage: {embedding_percentage:.2f}%")

    trainable_percentage = (trainable_params / total_params) * 100
    logging.info(f"Trainable parameters percentage: {trainable_percentage:.2f}%")

    # logging.info(f"below parameters will be trained")
    # for name, p in model.named_parameters():
    #     if p.requires_grad:
    #         logging.info(f"{name}")


# Training process
def train(rank, args, world_size):

    if rank==0:
        wandb.init(project="local-experiment", entity="1762204162-", mode="disabled")
        # wandb.init(project="local-experiment", entity="1762204162-", mode="disabled")

    
    with open(args.work_dir+"/config.json") as f:
        config=json.load(f)

    setup(rank, world_size, args.port)
    torch.cuda.set_device(rank)

    training_config = config["training_config"]
    task_config = config['task_config']

    # assert world_size == training_config["device_count"], "device_count wrong"
    # assert training_config["total_batch_size"] == training_config['batch_size_per_device']*training_config["device_count"]*training_config["gradient_accumulation_steps"]
    # assert training_config["segment_len"] == task_config["segment_size"]
    # assert task_config["mem_size"]*task_config["head_num"] == task_config["segment_size"]

    config["data_config"]["model_id"] = training_config["model_id"]
    print(config["data_config"])
    train_examples, eval_examples = get_examples(**config["data_config"])

    # cal the total step
    training_steps = len(train_examples)//training_config["total_batch_size"]

    # drop last examples
    train_examples = train_examples[:training_steps*training_config["total_batch_size"]]
    if rank==0:
        # print(f"[INFO] total_examples:{len(train_examples)} | training_steps:{training_steps}")
        logging.info(f"[INFO] total_examples:{len(train_examples)} | training_steps:{training_steps}")
        
    # assigning data to each GPU individually
    example_num_per_gpu = len(train_examples) // training_config["device_count"]
    start_index = rank * example_num_per_gpu
    end_index = start_index + example_num_per_gpu
    train_examples = train_examples[start_index:end_index]


    # print(f"[INFO] rank{rank} training examples[{start_index}:{end_index}] | example_nums:{len(train_examples)} | training_steps:{training_steps}")
    logging.info(f"[INFO] rank{rank} training examples[{start_index}:{end_index}] | example_nums:{len(train_examples)} | training_steps:{training_steps}")

    # Instantiate the model and move it to the corresponding GPU
    model = get_model(training_config["model_id"], task_config, rank)

    # model = load_adapter(model, save_path_and_name=args.work_dir+'/adapter.pt', log=False)
    if rank == 0:
        count_parameters(model, config)
    
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    if rank == 0:
        for name, param in model.named_parameters():
            print(f"{name} - Shape: {param.shape} - requires_grad = {param.requires_grad}")
    # ddp_model = model

    # Instantiate the data loader
    dataset = get_dataset(task_config["task_type"], train_examples, training_config['batch_size_per_device'])    

    loader = DataLoader(dataset, batch_size=None)
    # print(len(loader))

    # Instantiate  optimizer
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=training_config["learning_rate"], betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = get_wsd_scheduler(optimizer, training_steps)
    accumulation_steps = training_config["gradient_accumulation_steps"]
    step_num = 0
    
    optimizer.zero_grad()
    ddp_model.train()
    
    info_list = []
    start_time = time.time()

    
    # for i, group in enumerate(optimizer.param_groups):
    #     logging.info(f"Parameter Group {i}:")
    #     for param in group['params']:
    #         for name, param_in_model in ddp_model.named_parameters():
    #             if param is param_in_model:
    #                 logging.info(f"{name}: {param.size()}")
                    
    for epoch in range(1):

        def save():
            if rank!=0:
                return
            with open(os.path.join(args.work_dir,"info.json"),'w') as f:
                json.dump(info_list,f,indent=4)

            with open(os.path.join(args.work_dir,"config.json"),'w') as f:
                json.dump(config,f,indent=4)
                
            save_adapter(ddp_model.module,save_path_and_name=os.path.join(args.work_dir,"adapter.pt"))
            # torch.save(optimizer.state_dict(),os.path.join(training_config["save_dir"],"optimizer.pt"))

        if rank == 0:
            progress_bar = tqdm(total=training_steps*accumulation_steps)

        for inputs in loader:
            step_num += 1

            # print(f"\n{'-'*80}\nstep{step_num}\ndevice:[{rank}]\ninputs:{inputs}\n{'-'*80}")  # for check
            # logging.info(f"\n{'-'*80}\nstep{step_num}\ndevice:[{rank}]\ninputs:{inputs}\n{'-'*80}")

            if (task_config["task_type"]=="Compress") and ("addition" in task_config) and (task_config["addition"]=="without_compress_loss"):
                del inputs["compress_targets"]

            if step_num % accumulation_steps == 0:
                loss = training_step(ddp_model,inputs,rank,accumulation_steps)
            else:
                with ddp_model.no_sync():
                    loss = training_step(ddp_model,inputs,rank,accumulation_steps)

            # logging.info(f"\n{'-'*80}\nstep{step_num}\ndevice:[{rank}]\nmodel_head_parameter:{torch.mean(ddp_model.module.compress_head.weight)}\n{'-'*80}")
            # logging.info(f"\n{'-'*80}\nstep{step_num}\ndevice:[{rank}]\nmodel_head_parameter:{ddp_model.module.compress_head[0].weight[:3,:3]}\n{'-'*80}")
            # logging.info(f"\n{'-'*80}\nstep{step_num}\ndevice:[{rank}]\nmodel_head_parameter_grad:{model.compress_head[0].weight.grad[:3,:3]}\n{'-'*80}")
            
            info_list.append({
                "run_time(hours)":(time.time()- start_time)/3600,
                "total_steps":training_steps,
                "steps":step_num/accumulation_steps, 
                "training_loss":loss, 
                "learning_rate":optimizer.param_groups[0]['lr']})
            
            if rank==0:
                wandb.log({
                    "run_time(hours)":(time.time()- start_time)/3600,
                    "total_steps":training_steps,
                    "steps":step_num/accumulation_steps, 
                    "training_loss":loss, 
                    "learning_rate":optimizer.param_groups[0]['lr']})

            if step_num % accumulation_steps == 0:


                # for i, group in enumerate(optimizer.param_groups):
                #     logging.info(f"Parameter Group {i}:")
                #     for param in group['params']:
                #         for name, param_in_model in ddp_model.named_parameters():
                #             if param is param_in_model:
                #                 logging.info(f"{name}: {param.size()}")
                #                 logging.info(f"requires_grad:{param.requires_grad}, grad is not None:{param.grad is not None}")
                
                
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), training_config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            


            if step_num % (training_config["log_step"]*accumulation_steps) == 0:
                if rank == 0:
                    # print(info_list[-1])
                    logging.info(info_list[-1])
    
            if step_num % (training_config["save_step"]*accumulation_steps) == 0:
                save()

            if rank == 0:
                progress_bar.update(1)
        if rank == 0:
            progress_bar.close()
        save()



# Launch multi-process training
if __name__ == "__main__":
    args = parse_args()
    world_size = torch.cuda.device_count()
    # world_size = 4

    # if os.path.exists(args.work_dir+f'/adapter.pt'):
    #     print("adapter is already exist")
    #     exit(0)
    mp.spawn(train,
             args=(args,world_size),
             nprocs=world_size,
             join=True)

"""
# 用 > train.log 无法实时查看输出
CUDA_VISIBLE_DEVICES=4,5,6,7 python ./pre_trainer.py --config_path ./models/Memorization/config.json --port 12353
"""




