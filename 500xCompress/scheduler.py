import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, SequentialLR, LinearLR, ConstantLR, ReduceLROnPlateau
import torch.optim as optim


def get_wsd_scheduler(optimizer, training_steps):

    W = 300
    S = int(0.00005*training_steps)
    D = training_steps-W-S

    warmup_scheduler = LinearLR(optimizer, start_factor=1/W, total_iters=W)
    stable_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=S)
    decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1/D,total_iters=D)

    milestones = [W, W+S]

    wsd_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, stable_scheduler, decay_scheduler], milestones=milestones)

    return wsd_scheduler


def get_stage_scheduler(scheduler_config, optimizer, training_steps):
    return ReduceLROnPlateau(optimizer, 'min', factor=scheduler_config["factor"], patience=scheduler_config["patience"])

def get_cosine_scheduler(scheduler_config, optimizer, training_steps):
    W = scheduler_config["warmup_steps"]
    D = training_steps-W

    warmup_scheduler = LinearLR(optimizer, start_factor=1/W, total_iters=W)
    decay_scheduler = CosineAnnealingLR(optimizer, T_max=D, eta_min=scheduler_config["min_lr"])

    milestones = [W]

    return SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=milestones)


def get_scheduler(scheduler_config, optimizer, training_steps):
    if scheduler_config["type"] == "wsd":
        print("[INFO] Use WSD Scheduler")
        return get_wsd_scheduler(scheduler_config, optimizer, training_steps)

    elif scheduler_config["type"] == "stage":
        print("[INFO] Use ReduceLROnPlateau Scheduler")
        return get_stage_scheduler(scheduler_config, optimizer, training_steps)
    elif scheduler_config["type"] == "cosine":
        return get_cosine_scheduler(scheduler_config, optimizer, training_steps)
    else:
        print("[INFO] Don't use scheduler")
        return None