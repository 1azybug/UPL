import logging
import wandb
from path_config import BASE_PATH
import sys
sys.path.append(BASE_PATH)
import random
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import optimizer
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch.multiprocessing as mp
import os
import time
import json
from tqdm import tqdm
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import argparse
import torch.distributed as dist
from model.modeling import CompressLLM
from model.lora import LinearLoraLayer, EmbeddingLoraLayer
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import ConstantLR
from torch.optim.lr_scheduler import SequentialLR

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


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # Initialize the distributed environment
    dist.init_process_group("nccl", rank=rank, world_size=world_size)




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
    embedding_params = sum(
        p.numel() for name, p in model.named_parameters() if p.requires_grad and ('lm_head' in name or 'emb' in name))
    non_embedding_params = trainable_params - embedding_params

    config["Total_parameters"] = total_params
    config["Trainable_parameters"] = trainable_params
    config["Embedding_parameters"] = embedding_params
    config["non-Embedding_parameters"] = non_embedding_params

    logging.info(f"Total parameters: {total_params}")
    logging.info(f"Trainable parameters: {trainable_params}")
    logging.info(f"Embedding parameters: {embedding_params}")
    logging.info(f"non-Embedding parameters: {non_embedding_params}")

    embedding_percentage = (embedding_params / total_params) * 100
    logging.info(f"Embedding parameters percentage: {embedding_percentage:.2f}%")

    trainable_percentage = (trainable_params / total_params) * 100
    logging.info(f"Trainable parameters percentage: {trainable_percentage:.2f}%")



def training_step(ddp_model, inputs, rank, accumulation_steps):
    # inputs = {key:value.to(rank) for key,value in inputs.items()}
    inputs = {key:value.to(rank) if value is not None else None for key,value in inputs.items()}
    output = ddp_model(inputs=inputs)
    loss = output["loss"]
    loss /= accumulation_steps
    loss.backward()
    # 计算当前的梯度范数
    grad_norm = calculate_gradient_norm(ddp_model)
    output["loss_info"]["grad_norm"] = grad_norm
    return output["loss_info"]







