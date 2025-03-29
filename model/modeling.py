import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from path_config import BASE_PATH
sys.path.append(BASE_PATH)
import logging
import pdb
import queue
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F
import math
import transformers
from model.lora import LinearLoraLayer
import matplotlib.pyplot as plt
import seaborn as sns

class CompressLLM(torch.nn.Module):
    def __init__(self, model_id, mem_size, compress_ratio, device_rank, task_config):
        super().__init__()
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
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
        self.chunk_size = task_config["chunk_size"]
        self.mem_tokens = nn.Parameter(self.model.model.embed_tokens.weight.new_zeros((mem_size, config.hidden_size)), requires_grad=True)
        self.special_tokens = nn.Parameter(self.model.model.embed_tokens.weight.new_zeros((2, config.hidden_size)), requires_grad=True)
        self.compress_ratio = compress_ratio
        self.mem_size = mem_size

        mean = torch.mean(self.model.model.embed_tokens.weight).item()
        std = torch.std(self.model.model.embed_tokens.weight).item()
        nn.init.normal_(self.mem_tokens, mean=mean, std=std)
        nn.init.normal_(self.special_tokens, mean=mean, std=std)

    def forward(self,inputs):
        tot_loss = 0
        tot_task = 0
        loss_info = {}

        # context position ids:[1,......,end_idx]
        compress_token_ids, compress_token, end_idx = self.compress(inputs)

##########################################################AE Task########################################################################

        if self.task_config["is_pretrain"] and self.task_config["use_ae_loss"]:
            # print("AE Task")
            inputs_embeds = self.decoder.model.embed_tokens(inputs['ae_targets'])  
            bsz, seq_len, emb_size = inputs_embeds.size()
            # [1,E] -> [1,1,E] -> [B,1,E]
            expand_ae_token = self.special_tokens[0:1].unsqueeze(0).expand(bsz, 1, emb_size)
            # [B,mem_size,E];     [B,1,E];      [B,seq_len-1,E]
            ae_emb = torch.cat([compress_token, expand_ae_token, inputs_embeds[:, :-1, :]], dim=1)

            # ae_pids:[0], ae_target[:, :-1] pids:[1,......,end_idx] because drop the last token to predict [1,...,end_idx+1] the last one is <eos>.
            position_ids = torch.arange(0, inputs_embeds.size(1), device=inputs_embeds.device).unsqueeze(0)
            ae_position_ids = torch.cat([compress_token_ids, position_ids], dim=1)
            # print(f"ae_position_ids:{ae_position_ids}")
            if self.task_config["use_pe"]:
                outputs = self.decoder(position_ids=ae_position_ids, inputs_embeds=ae_emb)
            else:
                outputs = self.decoder(inputs_embeds=ae_emb)
            # [B,mem_size+S,V] -> [B,S,V]
            logits = outputs.logits[:, compress_token.size(1):]
            inputs['ae_targets'] = inputs['ae_targets'].contiguous().view(-1).to(logits.device)
            ae_loss = self.loss_fct(logits.contiguous().view(-1, self.vocab_size), inputs['ae_targets'])  # [ae]+context[:-1] -> context[:]
            loss_info["ae_loss"] = ae_loss.item()

            if "ae_weight" not in self.task_config:
                tot_loss += ae_loss
                tot_task += 1
            else:
                # print(f"ae_weight:{self.task_config['ae_weight']}; tot_task:{tot_task}")
                tot_loss += ae_loss*self.task_config["ae_weight"]
                tot_task += self.task_config["ae_weight"]
                # print(f"ae_weight:{self.task_config['ae_weight']}; tot_task:{tot_task}")
        
#######################################################LM Task################################################################################

        if self.task_config["is_pretrain"] and self.task_config["use_lm_loss"]:
            # print("LM Task")
            lm_target_emb = self.decoder.model.embed_tokens(inputs['lm_targets'][:, :-1])
            bsz, seq_len, emb_size = lm_target_emb.size()
            # [1,E] -> [1,1,E] -> [B,1,E]
            expand_lm_token = self.special_tokens[1:2].unsqueeze(0).expand(bsz, 1, emb_size)
            lm_emb = torch.cat([compress_token, expand_lm_token, lm_target_emb],dim=1)

            # context pids:[1,......,end_idx] 
            # lm_pids:[end_idx], lm_target_pids:[end_idx+1,......]
            latter_position_ids = torch.arange(end_idx,end_idx+seq_len+1,device=lm_target_emb.device).unsqueeze(0)
            lm_position_ids = torch.cat([compress_token_ids,latter_position_ids],dim=1)
            # print(f"lm_position_ids:{lm_position_ids}")
            if self.task_config["use_pe"]:
                outputs = self.decoder(inputs_embeds=lm_emb, position_ids=lm_position_ids)
            else:
                outputs = self.decoder(inputs_embeds=lm_emb)
            # [B,mem_size+S,V] -> [B,S,V]
            logits = outputs.logits[:, compress_token.size(1):]
            logits = logits.contiguous().view(-1, self.vocab_size)
            inputs['lm_targets'] = inputs['lm_targets'].contiguous().view(-1).to(logits.device)
            lm_loss = self.loss_fct(logits, inputs['lm_targets'])
            loss_info["lm_loss"] = lm_loss.item()

            if "lm_weight" not in self.task_config:
                tot_loss += lm_loss
                tot_task += 1
            else:
                # print(f"lm_weight:{self.task_config['lm_weight']}; tot_task:{tot_task}")
                tot_loss += lm_loss*self.task_config["lm_weight"]
                tot_task += self.task_config["lm_weight"]           
                # print(f"lm_weight:{self.task_config['lm_weight']}; tot_task:{tot_task}")


######################################################QA Task####################################################################
        # LM loss
        if self.task_config["is_sft"] and self.task_config["use_lm_loss"]:
            # print("QA Task")
            lm_target_emb = self.decoder.model.embed_tokens(inputs['lm_targets'][:, :-1])
            bsz, seq_len, emb_size = lm_target_emb.size()
            # [1,E] -> [1,1,E] -> [B,1,E]
            expand_lm_token = self.special_tokens[1:2].unsqueeze(0).expand(bsz, 1, emb_size)
            lm_emb = torch.cat([compress_token, expand_lm_token,lm_target_emb],dim=1)
            # context position ids:[1,......,end_idx];
            #                                         [LM] position ids:[end_idx];  QA position ids:[end_idx+1,.......]
            latter_position_ids = torch.arange(end_idx,end_idx+seq_len+1,device=lm_target_emb.device).unsqueeze(0)
            lm_position_ids = torch.cat([compress_token_ids,latter_position_ids],dim=1)
            # print(f"lm_position_ids:{lm_position_ids}")
            if self.task_config["use_pe"]:
                outputs = self.decoder(inputs_embeds=lm_emb, position_ids=lm_position_ids)
            else:
                outputs = self.decoder(inputs_embeds=lm_emb)
            # [B,mem_size+S,V] -> [B,S,V]
            logits = outputs.logits[:,compress_token.size(1):]

            #  in prepare_data.py, we drop the fisrt -100, so here we drop the [LM]'s logits which is used to predict the fisrt -100.
            #  but it's no influence because -100 are not used to calculate the loss.
            logits = logits[:, 1:]    
            logits = logits.contiguous().view(-1, self.vocab_size)
            inputs["instruction_target"] = inputs["instruction_target"].contiguous().view(-1).to(logits.device)
            lm_loss = self.loss_fct(logits, inputs["instruction_target"])
            loss_info["lm_loss"] = lm_loss.item()
            tot_loss += lm_loss
            tot_task += 1


        loss = tot_loss/tot_task
        return {"loss":loss, "loss_info":loss_info}

    def compute_num_chunks(self, total_length):
        assert total_length > 0
        num_chunks = math.ceil(total_length / self.chunk_size)
        return num_chunks

    def get_uniform_position_ids(self, x_1, x_n, ratio):
        return torch.arange((x_1 + (ratio - 1) // 2), x_n, step=ratio, device=self.device).unsqueeze(0)

    def compress(self, inputs):
        bsz, total_length = inputs['input_ids'].size()
        ######################################应该不需要截断context##########################################
        # inputs['input_ids'] = inputs['input_ids'][:, :total_length - (total_length % self.compress_ratio)]
        # bsz, total_length = inputs['input_ids'].size()
        ##########################################################################################################
        # num_chunks: 片段个数 | chunk_mem_size: 片段中mem的大小 | chunk_sizegth：片段长度
        num_chunks = self.compute_num_chunks(total_length)
        chunk_size = self.chunk_size
        # 收集compress_token
        compress_token = None
        compress_token_ids = None
        for chunk_idx in range(num_chunks):
            # 片段token：0-509 -> 510-1019 -> 1020-1529 -> ...
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, total_length)
            chunk_input_ids = inputs['input_ids'][:, start_idx:end_idx]
            # ->LlamaForCausalLM->LlamaModel->embed_tokens
            inputs_embeds = self.model.model.embed_tokens(chunk_input_ids)

            bsz, seq_len, emb_size = inputs_embeds.size()

            #################################不需要截断##############################################
            # 为了适配最后一个片段不足510
            # mem_size = round((end_idx - start_idx) // self.compress_ratio)
            # mem_tokens = self.mem_tokens[:mem_size, :]
            # expand_mem = mem_tokens.unsqueeze(0).expand(bsz, mem_size, emb_size)
            ########################################################################################
            expand_mem = self.mem_tokens.unsqueeze(0).expand(bsz, self.mem_size, emb_size)

            encode_inputs_embeds = torch.cat([inputs_embeds, expand_mem], dim=1)

            # [1,seq_len]
            position_ids = torch.arange(start_idx + 1, end_idx + 1, device=inputs_embeds.device).unsqueeze(0)
            # [1,mem_size]：compress token position information, the step is compression ratio
            mem_position_ids = self.get_uniform_position_ids(x_1=start_idx + 1, x_n=start_idx+chunk_size, ratio=self.compress_ratio)
            # [1,seq_len+mem_size]
            encode_position_ids = torch.cat([position_ids, mem_position_ids], dim=1)
            # print(f"encode_position_ids:{encode_position_ids}")

            if compress_token_ids is None:
                compress_token_ids = mem_position_ids
            else:
                # 将新的 mem_hidden 拼接到 compress_token
                compress_token_ids = torch.cat((compress_token_ids, mem_position_ids), dim=1)

            if self.task_config["use_pe"]:
                outputs = self.model(position_ids=encode_position_ids, inputs_embeds=encode_inputs_embeds,
                                     output_hidden_states=True,output_attentions=True)
            else:
                outputs = self.model(inputs_embeds=encode_inputs_embeds, output_hidden_states=True,output_attentions=True)

            self.attn_analysis(total_length, outputs, chunk_input_ids)
            hidden_states = outputs.hidden_states[-1]
            # [B,mem_size,emb_size]
            mem_hidden = hidden_states[:, -self.mem_size:]
            # 在第一次循环时初始化 compress_token
            if compress_token is None:
                compress_token = mem_hidden
            else:
                # 将新的 mem_hidden 拼接到 compress_token
                compress_token = torch.cat((compress_token, mem_hidden), dim=1)
        return compress_token_ids, compress_token, end_idx

    def lm_inference(self,inputs,generate_num=1024):
        compress_token_ids, compress_token, end_idx = self.compress(inputs)
        lm_target_emb = self.decoder.model.embed_tokens(inputs['lm_targets'])
        bsz, seq_len, emb_size = lm_target_emb.size()
        expand_lm_token = self.special_tokens[1:2].unsqueeze(0).expand(bsz, 1, emb_size)

        lm_emb = torch.cat([compress_token, expand_lm_token, lm_target_emb], dim=1)
        # context position ids:[1,......,end_idx]
        #                                         [LM] position ids:[end_idx];  QA position ids:[end_idx+1,.......] 
        latter_position_ids = torch.arange(end_idx, end_idx + seq_len + 1, device=lm_target_emb.device).unsqueeze(0)
        lm_position_ids = torch.cat([compress_token_ids, latter_position_ids], dim=1)

        generate_text = []
        past_key_values = None
        next_inputs_embeds = lm_emb.clone()
        next_position_ids = lm_position_ids.clone()
        for i in range(generate_num):
            if self.task_config["use_pe"]:
                out = self.decoder(position_ids=next_position_ids,
                                   inputs_embeds=next_inputs_embeds,
                                   past_key_values=past_key_values,
                                   use_cache=True)
            else:
                out = self.decoder(inputs_embeds=next_inputs_embeds,
                                   past_key_values=past_key_values,
                                   use_cache=True)
            # [B,S,V] -> [B,V]
            logit = out.logits[:, -1]
            past_key_values = out.past_key_values
            # [B,V]->[B]
            next_token_id = torch.argmax(logit, dim=-1)
            # [B]->[B,E]->[B,1,E]
            next_inputs_embeds = self.decoder.model.embed_tokens(next_token_id).unsqueeze(1).to(lm_target_emb.device)  
            # [1, seq_len]/[1,1] -> [1,1]
            next_position_ids = next_position_ids[:,-1:]+1
            generate_text.append(next_token_id.item())
            if next_token_id.item() == self.tokenizer.eos_token_id:
                return generate_text
        return generate_text


    def ae_inference(self, inputs):
        compress_token_ids, compress_token, end_idx = self.compress(inputs)
        bsz, tot_mem_size, emb_size = compress_token.size()
        # [1,E] -> [1,1,E] -> [B,1,E]
        expand_ae_token = self.special_tokens[0:1].unsqueeze(0).expand(bsz, 1, emb_size)

        #                  [B,tot_mem_size,E];   [B,1,E]
        ae_emb = torch.cat([compress_token, expand_ae_token], dim=1)

        # ae_pid:[0]  shape:[1]->[1,1]->[B,1]
        position_ids = torch.arange(0, 1, device=compress_token.device).unsqueeze(0).expand(bsz, 1)
        ae_position_ids = torch.cat([compress_token_ids, position_ids], dim=1)

        generate_text = []
        past_key_values = None
        next_inputs_embeds = ae_emb.clone()
        next_position_ids = ae_position_ids.clone()

        for i in range(inputs['input_ids'].size(-1)+20):
            # print(f"next_pids:{next_position_ids}")
            if self.task_config["use_pe"]:
                out = self.decoder(position_ids=next_position_ids, inputs_embeds=next_inputs_embeds, past_key_values=past_key_values, use_cache=True)
            else:
                out = self.decoder(inputs_embeds=next_inputs_embeds, past_key_values=past_key_values, use_cache=True)
            # [B,S,V] -> [B,V]
            logit = out.logits[:, -1]
            past_key_values = out.past_key_values
            # [B,V]->[B]
            next_token_id = torch.argmax(logit, dim=-1)
            # [B]->[B,E]->[B,1,E]
            next_inputs_embeds = self.decoder.model.embed_tokens(next_token_id).unsqueeze(1).to(compress_token.device)
            # next_position_ids:[B,S] -> [B,1]
            next_position_ids = next_position_ids[:,-1:]+1
            generate_text.append(next_token_id.item())
            if next_token_id.item() == self.tokenizer.eos_token_id:
                return generate_text
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
            if next_token_id.item() == self.tokenizer.eos_token_id:
                return generate_text
        return generate_text

    def attn_analysis(self, total_length, outputs, chunk_input_ids):
        save_dir = "your experiment config path"
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        if total_length>=510:
            attentions = outputs.attentions
            mem_tokens = [f"[MEM{i}]" for i in range(self.mem_size)]
            input_text = self.tokenizer.convert_ids_to_tokens(chunk_input_ids.tolist()[0])
            input_text.extend(mem_tokens)
            for layer_index in range(len(attentions)):
                # 选取当前层的注意力权重
                attention = attentions[layer_index].squeeze(0)  # (num_heads, seq_len, seq_len)
                # 计算所有注意力头的加和
                total_attention = attention.sum(dim=0).to(torch.float32).cpu().numpy()  # (seq_len, seq_len)
                # 绘制综合注意力热力图
                plt.figure(figsize=(150, 150))
                sns.heatmap(
                    total_attention, 
                    xticklabels=input_text, 
                    yticklabels=input_text, 
                    cmap="Reds",  # 由浅粉色到深红色
                    square=True
                )
                # 旋转标签以避免重叠
                plt.title(f"Summed Attention Map - Layer {layer_index + 1}")
                # 保存图像到本地
                file_name = f"attention_layer{layer_index+1}_summed.png"
                file_path = os.path.join(save_dir, file_name)
                plt.savefig(file_path, format="png")
                print(f"Summed Attention map saved at: {file_path}")
                plt.close()  # 关闭当前图像，释放内存
            exit()
def freeze_encoder(model):
    for name, param in model.named_parameters():
        if name == "mem_tokens" or name == "special_tokens":
            continue
        param.requires_grad = False

def freeze_decoder(model):
    for name, param in model.named_parameters():
        param.requires_grad = False


def load_adapter(model, save_path_and_name='adapter.pt', log=False):
    adapter_state_dict = torch.load(save_path_and_name, map_location='cpu')  # 先加载到CPU

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
            if isinstance(module, nn.Linear) and ((name == "q_proj") or (name == "v_proj")):
                setattr(model, name, LinearLoraLayer(module.in_features, module.out_features, r=128,
                                                     weight=module.weight.data.clone()))
            else:
                # Recursively apply this function to submodules
                add_compress_lora(module, task_config)

    model = CompressLLM(
        model_id,
        mem_size=task_config["mem_size"],
        compress_ratio=task_config["compress_ratio"],
        device_rank=rank,
        task_config=task_config
    )

    # freeze all the model except mem tokens and special tokens
    freeze_encoder(model)
    # only add lora to encoder, don't add lora to model.decoder
    add_compress_lora(model.model, task_config)
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
