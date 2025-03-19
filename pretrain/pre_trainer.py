import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from path_config import BASE_PATH
sys.path.append(BASE_PATH)
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch.multiprocessing as mp
import os
import time
import json
from tqdm import tqdm
from transformers.models.llama.configuration_llama import LlamaConfig
import argparse

from pre_prepare_data import get_examples
from model.modeling import get_model, save_adapter, load_adapter
from pre_dataloader import get_dataset

import logging
import wandb

from util.utils import get_wsd_scheduler, training_step, setup, count_parameters

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, default='project',required=False, help='Directory including the configuration file, for saving model')
    parser.add_argument('--port', type=str, default='14527', required=False, help='port for ddp training')
    return parser.parse_args()



# Training process
def train(rank, args, world_size):

    if rank==0:
        wandb.init(project="local-experiment", entity="1762204162-", mode="disabled")

    with open(args.work_dir+"/config.json") as f:
        config=json.load(f)

    setup(rank, world_size, args.port)
    torch.cuda.set_device(rank)

    training_config = config["pretrain_training_config"]
    task_config = config['pretrain_task_config']

    assert world_size == training_config["device_count"], "device_count wrong"
    assert training_config["total_batch_size"] == training_config['batch_size_per_device']*training_config["device_count"]*training_config["gradient_accumulation_steps"]
    assert training_config["chunk_size"] == task_config["chunk_size"]
    assert task_config["mem_size"]*task_config["compress_ratio"] == task_config["chunk_size"]

    config["data_config"]["model_id"] = training_config["model_id"]
    output_dir = "output"
    config["data_config"]["output_dir"] = output_dir
    # print(config)
    train_examples, eval_examples = get_examples(**config["data_config"])

    # cal the total step
    training_steps = len(train_examples)//training_config["total_batch_size"]

    # drop last examples
    # train_examples = train_examples[training_steps*training_config["total_batch_size"]-4800:training_steps*training_config["total_batch_size"]]  # for checking pids of longer context 
    train_examples = train_examples[:training_steps*training_config["total_batch_size"]]
    if rank==0:
        logging.info(f"[INFO] total_examples:{len(train_examples)} | training_steps:{training_steps}")
        
    indices = [i for i in range(len(train_examples))]

    device_count = training_config["device_count"]
    # The data is interleaved to obtain a batch containing samples of similar length
    train_examples = train_examples[rank::device_count]    # train_examples = train_examples[start_index:end_index]

    
    logging.info(f"[INFO] rank{rank} training examples: {indices[rank::device_count][:4]} ... {indices[rank::device_count][-4:]}| example_nums:{len(train_examples)} | training_steps:{training_steps}")
    
    model = get_model(training_config["model_id"], task_config, rank)
    
    # check non-frozen parameters
    if rank == 0:
        count_parameters(model, config)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    # Instantiate the data loader
    dataset = get_dataset(task_config["task_type"], train_examples, training_config['batch_size_per_device'])
    loader = DataLoader(dataset, batch_size=None)
    # Instantiate  optimizer
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=training_config["learning_rate"], betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = get_wsd_scheduler(optimizer, training_steps)
    accumulation_steps = training_config["gradient_accumulation_steps"]
    step_num = 0
    
    optimizer.zero_grad()
    ddp_model.train()
    
    info_list = []
    start_time = time.time()
                    
    for epoch in range(1):

        def save():
            if rank!=0:
                return
            with open(os.path.join(args.work_dir,f"{output_dir}/info.json"),'w') as f:
                json.dump(info_list,f,indent=4)

            with open(os.path.join(args.work_dir,f"{output_dir}/config.json"),'w') as f:
                json.dump(config,f,indent=4)
                
            save_adapter(ddp_model.module,save_path_and_name=os.path.join(args.work_dir,f"{output_dir}/adapter.pt"))

        if rank == 0:
            progress_bar = tqdm(total=training_steps*accumulation_steps)

        for inputs in loader:
            step_num += 1

            if step_num % accumulation_steps == 0:
                loss = training_step(ddp_model,inputs,rank,accumulation_steps)
            else:
                with ddp_model.no_sync():
                    loss = training_step(ddp_model,inputs,rank,accumulation_steps)

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
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), training_config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if step_num % (training_config["log_step"]*accumulation_steps) == 0:
                if rank == 0:
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

    output_dir = "output"
    if not os.path.exists(os.path.join(args.work_dir,output_dir)):
        os.makedirs(os.path.join(args.work_dir,output_dir))

    mp.spawn(train,
             args=(args,world_size),
             nprocs=world_size,
             join=True)

"""
CUDA_VISIBLE_DEVICES=5,7 python ./pre_trainer.py --work_dir '../experiment/debug/quick' --port 14529
"""




