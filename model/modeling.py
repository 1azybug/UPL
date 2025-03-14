import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from path_config import BASE_PATH
sys.path.append(BASE_PATH)
import logging
import pdb
import queue
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F
import math
import transformers
from model.lora import LinearLoraLayer, EmbeddingLoraLayer
class CompressLLM(torch.nn.Module):
    def __init__(self, model_id, mem_size, head_num, device_rank, task_config):
        super().__init__()
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=f"cuda:{device_rank}",
        )
        self.decoder = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=f"cuda:{device_rank}",
        )
        freeze_decoder(self.decoder)
        self.device = f"cuda:{device_rank}"
        self.task_config = task_config
        config = self.model.config
        self.vocab_size = config.vocab_size
        self.segment_len = task_config["segment_size"]
        self.mem_tokens = nn.Parameter(self.model.model.embed_tokens.weight.new_zeros((mem_size, config.hidden_size)), requires_grad=True)
        self.special_tokens = nn.Parameter(self.model.model.embed_tokens.weight.new_zeros((2, config.hidden_size)), requires_grad=True)
        self.head_num = head_num
        self.compress_head = nn.Linear(config.hidden_size, head_num*config.vocab_size, bias=False, device=f"cuda:{device_rank}",
                                            dtype=self.model.model.embed_tokens.weight.dtype)

        mean = torch.mean(self.model.model.embed_tokens.weight).item()
        std = torch.std(self.model.model.embed_tokens.weight).item()
        nn.init.normal_(self.mem_tokens, mean=mean, std=std)
        nn.init.normal_(self.special_tokens, mean=mean, std=std)

    def forward(self,inputs):
        tot_loss = 0
        tot_task = 0
        loss_info = {}
        compress_token_ids, compress_token, end_idx = self.compress(inputs)

        # compress loss
        if self.task_config["use_compress_loss"]:
            # [B,mem_size,emb_size] -> [B,mem_size,head_num*vocab_size]
            logits =  self.compress_head(compress_token)
            # [B,mem_size,head_num*vocab_size] -> [B,tot_Seg_len,V] -> [B,seq_len,V]
            logits = logits.float()
            logits = logits.contiguous().view(-1, self.vocab_size)
            compress_targets = inputs["input_ids"].contiguous().view(-1).to(logits.device)
            compress_loss = self.loss_fct(logits, compress_targets)
            loss_info["compress_loss"] = compress_loss.item()
            tot_loss += compress_loss
            tot_task += 1

        if self.task_config["use_ae_loss"]:
            inputs_embeds = self.model.model.embed_tokens(inputs['input_ids'])
            bsz, seq_len, emb_size = inputs_embeds.size()
            # [1,E] -> [1,1,E] -> [B,1,E]
            expand_ae_token = self.special_tokens[0:1].unsqueeze(0).expand(bsz, 1, emb_size)
            # [B,mem_size,E];     [B,1,E];      [B,seq_len-1,E]
            ae_emb = torch.cat([compress_token, expand_ae_token, inputs_embeds[:, :-1, :]], dim=1)
            position_ids = torch.arange(1, inputs_embeds.size(1)+1, device=inputs_embeds.device).unsqueeze(0)
            ae_position_ids = torch.cat([compress_token_ids, position_ids - 1], dim=1)
            if "wo_pe" in self.task_config:
                outputs = self.decoder(inputs_embeds=ae_emb)
            else:
                outputs = self.decoder(position_ids=ae_position_ids, inputs_embeds=ae_emb)
            # [B,mem_size+S,V] -> [B,S,V]
            logits = outputs.logits[:, compress_token.size(1):]
            inputs['ae_targets'] = inputs['input_ids'].contiguous().view(-1).to(logits.device)
            ae_loss = self.loss_fct(logits.contiguous().view(-1, self.vocab_size), inputs['ae_targets'])
            loss_info["ae_loss"] = ae_loss.item()
            tot_loss += ae_loss
            tot_task += 1

        # LM loss
        if self.task_config["use_lm_loss"]:
            lm_target_emb = self.model.model.embed_tokens(inputs['lm_targets'][:, :-1])
            bsz, seq_len, emb_size = lm_target_emb.size()
            # [1,E] -> [1,1,E] -> [B,1,E]
            expand_lm_token = self.special_tokens[1:2].unsqueeze(0).expand(bsz, 1, emb_size)
            lm_emb = torch.cat([compress_token, expand_lm_token,lm_target_emb],dim=1)
            latter_position_ids = torch.arange(end_idx+1,end_idx+seq_len+2,device=lm_target_emb.device).unsqueeze(0)
            lm_position_ids = torch.cat([compress_token_ids,latter_position_ids-1],dim=1)
            if "wo_pe" in self.task_config:
                outputs = self.decoder(inputs_embeds=lm_emb)
            else:
                outputs = self.decoder(inputs_embeds=lm_emb, position_ids=lm_position_ids)
            # [B,mem_size+S,V] -> [B,S,V]
            logits = outputs.logits[:,compress_token.size(1):]
            logits = logits[:, 1:]
            logits = logits.contiguous().view(-1, self.vocab_size)
            inputs["instruction_target"] = inputs["instruction_target"].contiguous().view(-1).to(logits.device)
            lm_loss = self.loss_fct(logits, inputs["instruction_target"])
            loss_info["lm_loss"] = lm_loss.item()
            tot_loss += lm_loss
            tot_task += 1
        loss = tot_loss/tot_task
        return {"loss":loss, "loss_info":loss_info}

    def compute_num_segments(self, total_length):
        assert total_length > 0
        num_segments = math.ceil(total_length / self.segment_len)
        return num_segments

    def compress(self, inputs):
        bsz, total_length = inputs['input_ids'].size()
        inputs['input_ids'] = inputs['input_ids'][:, :total_length - (total_length % self.head_num)]
        bsz, total_length = inputs['input_ids'].size()
        # num_segments: 片段个数 | segment_mem_size: 片段中mem的大小 | segment_length：片段长度
        num_segments = self.compute_num_segments(total_length)
        segment_length = self.segment_len
        # 收集compress_token
        compress_token = None
        compress_token_ids = None
        for segment_idx in range(num_segments):
            # 片段token：0-510 -> 510-1020 -> 1020-1530 -> ...
            start_idx = segment_idx * segment_length
            end_idx = min((segment_idx + 1) * segment_length, total_length)
            segment_input_ids = inputs['input_ids'][:, start_idx:end_idx]
            # ->LlamaForCausalLM->LlamaModel->embed_tokens
            inputs_embeds = self.model.model.embed_tokens(segment_input_ids)

            bsz, seq_len, emb_size = inputs_embeds.size()
            # 为了适配最后一个片段不足510
            mem_size = round((end_idx - start_idx) // self.head_num)
            mem_tokens = self.mem_tokens[:mem_size, :]
            expand_mem = mem_tokens.unsqueeze(0).expand(bsz, mem_size, emb_size)
            encode_inputs_embeds = torch.cat([inputs_embeds, expand_mem], dim=1)

            # [1,seq_len]
            position_ids = torch.arange(start_idx + 1, end_idx + 1, device=inputs_embeds.device).unsqueeze(0)
            # [1,mem_size]：compress token position information, the step is compression ratio
            mem_position_ids = torch.arange((start_idx + (self.head_num + 1) // 2), end_idx + 1, step=self.head_num,
                                            device=inputs_embeds.device).unsqueeze(0)
            # [1,seq_len+mem_size]
            encode_position_ids = torch.cat([position_ids, mem_position_ids], dim=1)

            if compress_token_ids is None:
                compress_token_ids = mem_position_ids
            else:
                # 将新的 mem_hidden 拼接到 compress_token
                compress_token_ids = torch.cat((compress_token_ids, mem_position_ids), dim=1)

            # make three masks：cl_mask、lm_mask、cl_prime_mask
            if "wo_pe" in self.task_config:
                outputs = self.model(inputs_embeds=encode_inputs_embeds, output_hidden_states=True)
            else:
                outputs = self.model(position_ids=encode_position_ids, inputs_embeds=encode_inputs_embeds,
                                     output_hidden_states=True)

            hidden_states = outputs.hidden_states[-1]
            # [B,mem_size,emb_size]
            mem_hidden = hidden_states[:, -mem_size:]
            # 在第一次循环时初始化 compress_token
            if compress_token is None:
                compress_token = mem_hidden
            else:
                # 将新的 mem_hidden 拼接到 compress_token
                compress_token = torch.cat((compress_token, mem_hidden), dim=1)
        return compress_token_ids, compress_token, end_idx
    def lm_inference(self,inputs):
        compress_token_ids, compress_token, end_idx = self.compress(inputs)
        lm_target_emb = self.model.model.embed_tokens(inputs['lm_targets'])
        bsz, seq_len, emb_size = lm_target_emb.size()
        expand_lm_token = self.special_tokens[1:2].unsqueeze(0).expand(bsz, 1, emb_size)

        lm_emb = torch.cat([compress_token, expand_lm_token, lm_target_emb], dim=1)
        latter_position_ids = torch.arange(end_idx, end_idx + seq_len + 1, device=lm_target_emb.device).unsqueeze(0)
        lm_position_ids = torch.cat([compress_token_ids, latter_position_ids], dim=1)

        generate_text = []
        past_key_values = None
        next_inputs_embeds = lm_emb.clone()
        next_position_ids = lm_position_ids.clone()
        for i in range(1024):
            if "wo_pe" in self.task_config:
                out = self.decoder(inputs_embeds=next_inputs_embeds,
                                 past_key_values=past_key_values,
                                 use_cache=True)
            else:
                out = self.decoder(position_ids=next_position_ids,
                                 inputs_embeds=next_inputs_embeds,
                                 past_key_values=past_key_values,
                                 use_cache=True)
            # [B,S,V] -> [B,V]
            logit = out.logits[:, -1]
            past_key_values = out.past_key_values
            # [B,V]->[B]
            next_token_id = torch.argmax(logit, dim=-1)
            # [B]->[B,E]->[B,1,E]
            next_inputs_embeds = self.model.model.embed_tokens(next_token_id).unsqueeze(1).to(lm_target_emb.device)
            # [1, seq_len]/[1,1] -> [1,1]
            next_position_ids = next_position_ids[:,-1:]+1
            generate_text.append(next_token_id.item())
            if next_token_id.item() == 2:
                return generate_text
        return generate_text

    def cl_inference(self, inputs):
        compress_token_ids, compress_token, end_idx = self.compress(inputs)
        logits = self.compress_head(compress_token).float()
        logits = logits.contiguous().view(-1, self.vocab_size)
        generate_text = torch.argmax(logits, dim=-1).tolist()
        return generate_text

    def ae_inference(self, inputs):
        compress_token_ids, compress_token, end_idx = self.compress(inputs)
        bsz, total_length, emb_size = inputs['input_ids'].size()
        inputs_embeds = self.model.model.embed_tokens(inputs['input_ids'])
        # [1,E] -> [1,1,E] -> [B,1,E]
        expand_ae_token = self.special_tokens[0:1].unsqueeze(0).expand(bsz, 1, emb_size)

        #                  [B,mem_size,E];   [B,1,E]
        ae_emb = torch.cat([compress_token, expand_ae_token], dim=1)
        position_ids = torch.arange(1, inputs_embeds.size(1)+1, device=inputs_embeds.device).unsqueeze(0)
        ae_position_ids = torch.cat([compress_token_ids, position_ids[:, :1] - 1], dim=1)

        generate_text = []
        past_key_values = None
        next_inputs_embeds = ae_emb.clone()
        next_position_ids = ae_position_ids.clone()

        for i in range(1024):
            if "wo_pe" in self.task_config:
                out = self.decoder(inputs_embeds=next_inputs_embeds, past_key_values=past_key_values, use_cache=True)
            else:
                out = self.decoder(position_ids=next_position_ids, inputs_embeds=next_inputs_embeds, past_key_values=past_key_values, use_cache=True)
            # [B,S,V] -> [B,V]
            logit = out.logits[:, -1]
            past_key_values = out.past_key_values
            # [B,V]->[B]
            next_token_id = torch.argmax(logit, dim=-1)
            # [B]->[B,E]->[B,1,E]
            next_inputs_embeds = self.model.model.embed_tokens(next_token_id).unsqueeze(1).to(inputs_embeds.device)
            next_position_ids = position_ids[:, i:i + 1]
            generate_text.append(next_token_id.item())
        return generate_text



    # def forward(self, inputs):
    #     loss_info = {}
    #     inputs_embeds = self.model.model.embed_tokens(inputs["input_ids"])
    #     lm_target_emb = self.model.model.embed_tokens(inputs['lm_targets'][:, :-1])
    #     encode_inputs_embeds = torch.cat([inputs_embeds, lm_target_emb], dim=1)
    #     outputs = self.model(
    #         inputs_embeds=encode_inputs_embeds,
    #     )
    #     # [B,mem_size+S,V] -> [B,S,V]
    #     logits = outputs.logits[:,inputs_embeds.size(1):]
    #     logits = logits.contiguous().view(-1, self.vocab_size)
    #     inputs["instruction_target"] = inputs["instruction_target"].contiguous().view(-1).to(logits.device)
    #     lm_loss = self.loss_fct(logits, inputs["instruction_target"])
    #     loss_info["lm_loss"] = lm_loss.item()
    #     loss = lm_loss
    #     return {"loss": loss, "loss_info": loss_info}



    def vanilla_llama_inference(self, inputs):
        inputs_embeds = self.model.model.embed_tokens(inputs["input_ids"])
        lm_target_emb = self.model.model.embed_tokens(inputs['lm_targets'])
        encode_inputs_embeds = torch.cat([inputs_embeds, lm_target_emb], dim=1)

        generate_text = []
        past_key_values = None
        next_inputs_embeds = encode_inputs_embeds.clone()
        for i in range(1024):
            # print(f"next_position_ids:{next_position_ids}")
            out = self.model(inputs_embeds=next_inputs_embeds,
                                     past_key_values=past_key_values,
                                     use_cache=True)
            # [B,S,V] -> [B,V]
            logit = out.logits[:, -1]
            past_key_values = out.past_key_values
            # [B,V]->[B]
            next_token_id = torch.argmax(logit, dim=-1)
            next_inputs_embeds = self.model.model.embed_tokens(next_token_id).unsqueeze(1).to(inputs_embeds.device)
            generate_text.append(next_token_id.item())
            if next_token_id.item() == 2:
                return generate_text
        return generate_text


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
