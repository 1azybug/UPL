
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


def freeze_encoder(model):
    for name, param in model.named_parameters():
        print(name)
        if name == "compress_head.weight" or name == "mem_tokens" or name == "special_tokens":
            continue
        param.requires_grad = False

def freeze_decoder(model):
    for name, param in model.named_parameters():
        param.requires_grad = False


def load_adapter(model, save_path_and_name='adapter.pt', log=False):
    adapter_state_dict = torch.load(save_path_and_name, map_location='cpu')  # 先加载到CPU
    if log:
        print("Loading adapter parameters:")
        for name, _ in adapter_state_dict.items():
            print(f"[Load Adapter] {name}")

    # 将adapter的权重转移到模型的设备上
    adapter_state_dict = {k: v.to(model.device) for k, v in adapter_state_dict.items()}

    model.load_state_dict(adapter_state_dict, strict=False)
    return model

def load_model_with_adapter(model_id, task_config, rank, save_path_and_name='adapter.pt', log=False):
    model = get_model(model_id, task_config, rank)
    load_adapter(model, save_path_and_name, log)
    return model

def get_model_for_compress(model_id, task_config, rank):
    def add_compress_lora(model, task_config):
        for name, module in model.named_children():

            if name == "compress_head":
                continue
            if isinstance(module, nn.Linear) and ((name == "q_proj") or (name == "v_proj")):
                setattr(model, name, LinearLoraLayer(module.in_features, module.out_features, r=128,
                                                     weight=module.weight.data.clone()))
            # elif isinstance(module, nn.Embedding):
            #     setattr(model, name,
            #             EmbeddingLoraLayer(module.num_embeddings, module.embedding_dim, module.padding_idx, r=128,
            #                                weight=module.weight.data.clone()))
            else:
                # Recursively apply this function to submodules
                add_compress_lora(module, task_config)

    def add_multi_lora(model, task_config):
        for name, module in model.named_children():
            if name == "compress_head":
                continue
            if isinstance(module, nn.Linear):
                setattr(model, name,
                        TripleLinearLoraLayer(module.in_features, module.out_features, r_cl=16, r_lm=16, r_cl_prime=16,
                                              weight=module.weight.data.clone()))
            elif isinstance(module, nn.Embedding):
                setattr(model, name,
                        TripleEmbeddingLoraLayer(module.num_embeddings, module.embedding_dim, module.padding_idx,
                                                 r_cl=128, r_lm=128, r_cl_prime=128, weight=module.weight.data.clone()))
            else:
                # Recursively apply this function to submodules
                add_multi_lora(module, task_config)

    if task_config["use_multi_lora"]:
        model = CompressLLM(
            model_id,
            mem_size=task_config["mem_size"],
            head_num=task_config["head_num"],
            device_rank=rank,
            task_config=task_config
        )
        add_multi_lora(model, task_config)
    else:
        model = CompressLLM(
            model_id,
            mem_size=task_config["mem_size"],
            head_num=task_config["head_num"],
            device_rank=rank,
            task_config=task_config
        )
        freeze_encoder(model)
        add_compress_lora(model, task_config)
    return model

def get_model(model_id, task_config, rank):
    if task_config["task_type"] == "Compress":
        return get_model_for_compress(model_id, task_config, rank)
    raise Exception("Don't exist [{task_type}] task.")

def save_adapter(model, save_path_and_name='adapter.pt', log=False):
    adapter_name = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            if log:
                print("[Save Adapter]", name)
            adapter_name.add(name)

    state_dict = model.state_dict()
    adapter_state_dict = {name: param for name, param in state_dict.items() if name in adapter_name}
    torch.save(adapter_state_dict, save_path_and_name)


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







