import logging
import pdb
import queue
import sys

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F
import math
import transformers
sys.path.append('/mnt/zhaorunsong/lx/compress/')
# noinspection PyUnresolvedReferences
from modify_code import modify_llama


class TripleLinearLoraLayer(nn.Module):
    def __init__(self, in_features, out_features, r_cl=16, r_lm=16, r_cl_prime=16, scale=1.0, weight=None):
        super(TripleLinearLoraLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = 2  # Scaling factor

        # 原始权重矩阵 W
        self.weight = nn.Parameter(weight, requires_grad=False)

        # 压缩 token 的 LoRA 模块参数 (A_cl, B_cl)
        self.lora_A_cl = nn.Parameter(torch.zeros((in_features, r_cl), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        self.lora_B_cl = nn.Parameter(torch.zeros((r_cl, out_features), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)

        # LLM token 的 LoRA 模块参数 (A_lm, B_lm)
        self.lora_A_lm = nn.Parameter(torch.zeros((in_features, r_lm), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        self.lora_B_lm = nn.Parameter(torch.zeros((r_lm, out_features), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)

        # 额外的压缩 token 的 LoRA 模块参数 (A_cl', B_cl')
        self.lora_A_cl_prime = nn.Parameter(torch.zeros((in_features, r_cl_prime), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        self.lora_B_cl_prime = nn.Parameter(torch.zeros((r_cl_prime, out_features), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)

        # 初始化 LoRA 参数
        nn.init.kaiming_uniform_(self.lora_A_cl, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_cl)
        nn.init.kaiming_uniform_(self.lora_A_lm, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_lm)
        nn.init.kaiming_uniform_(self.lora_A_cl_prime, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_cl_prime)

    def forward(self, x, mask):
        # 原始权重的计算结果，只计算一次
        result = F.linear(x, self.weight)

        # 检查并应用每种 mask
        if "cl_mask" in mask:
            x_cl = x * mask["cl_mask"]
            result_cl = self.scale * (x_cl @ self.lora_A_cl @ self.lora_B_cl)
            result += result_cl

        if "lm_mask" in mask:
            x_lm = x * mask["lm_mask"]
            result_lm = self.scale * (x_lm @ self.lora_A_lm @ self.lora_B_lm)
            result += result_lm

        if "cl_prime_mask" in mask:
            x_cl_prime = x * mask["cl_prime_mask"]
            result_cl_prime = self.scale * (x_cl_prime @ self.lora_A_cl_prime @ self.lora_B_cl_prime)
            result += result_cl_prime

        return result

class TripleEmbeddingLoraLayer(nn.Module):
    def __init__(self, in_features, out_features, padding_idx, r_cl=128, r_lm=128, r_cl_prime=128, scale=1.0, weight=None):
        super(TripleEmbeddingLoraLayer, self).__init__()
        self.num_embeddings = in_features
        self.embedding_dim = out_features
        self.padding_idx = padding_idx
        self.scale = 2  # Scaling factor

        # 原始权重矩阵 W
        self.weight = nn.Parameter(weight, requires_grad=False)

        # 压缩 token 的 LoRA 模块参数 (A_cl, B_cl)
        self.lora_A_cl = nn.Parameter(torch.zeros((in_features, r_cl), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        self.lora_B_cl = nn.Parameter(torch.zeros((r_cl, out_features), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)

        # LLM token 的 LoRA 模块参数 (A_lm, B_lm)
        self.lora_A_lm = nn.Parameter(torch.zeros((in_features, r_lm), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        self.lora_B_lm = nn.Parameter(torch.zeros((r_lm, out_features), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)

        # 额外的压缩 token 的 LoRA 模块参数 (A_cl', B_cl')
        self.lora_A_cl_prime = nn.Parameter(torch.zeros((in_features, r_cl_prime), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        self.lora_B_cl_prime = nn.Parameter(torch.zeros((r_cl_prime, out_features), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)

        # 初始化 LoRA 参数
        nn.init.zeros_(self.lora_A_cl)
        nn.init.normal_(self.lora_B_cl)
        nn.init.zeros_(self.lora_A_lm)
        nn.init.normal_(self.lora_B_lm)
        nn.init.zeros_(self.lora_A_cl_prime)
        nn.init.normal_(self.lora_B_cl_prime)

    def forward(self, x, mask):
        # 计算一次嵌入的基准结果
        result = F.embedding(x, self.weight, self.padding_idx)  # 初始化结果

        # 检查每个 mask 并应用相应的 LoRA 层
        if "cl_mask" in mask:
            x_cl = x * mask["cl_mask"]
            after_A_cl = F.embedding(x_cl, self.lora_A_cl, self.padding_idx)
            result_cl = self.scale * (after_A_cl @ self.lora_B_cl)
            result += result_cl

        if "lm_mask" in mask:
            x_lm = x * mask["lm_mask"]
            after_A_lm = F.embedding(x_lm, self.lora_A_lm, self.padding_idx)
            result_lm = self.scale * (after_A_lm @ self.lora_B_lm)
            result += result_lm

        if "cl_prime_mask" in mask:
            x_cl_prime = x * mask["cl_prime_mask"]
            after_A_cl_prime = F.embedding(x_cl_prime, self.lora_A_cl_prime, self.padding_idx)
            result_cl_prime = self.scale * (after_A_cl_prime @ self.lora_B_cl_prime)
            result += result_cl_prime

        return result
# from peft import prepare_model_for_kbit_training

class LinearLoraLayer(nn.Module):
    # No bias in LLama3 LinearLayer
    def __init__(self, in_features, out_features, r=16, weight=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.dropout = nn.Dropout(0.05)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.scale = 2  # The alpha value is usually twice the rank
        self.lora_A = nn.Parameter(torch.zeros((in_features, r), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros((r, out_features), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = F.linear(x, self.weight)
        result += self.scale*(x@self.lora_A@self.lora_B)
        return result
    

class EmbeddingLoraLayer(nn.Module):
    # No bias in LLama3 LinearLayer
    def __init__(self, in_features, out_features, padding_idx, r=128, weight=None):
        super().__init__()
        self.num_embeddings = in_features
        self.embedding_dim = out_features
        self.padding_idx = padding_idx
        self.dropout = nn.Dropout(0.05)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.scale = 2  # The alpha value is usually twice the rank
        self.lora_A = nn.Parameter(torch.zeros((in_features, r), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros((r, out_features), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        nn.init.zeros_(self.lora_A)
        nn.init.normal_(self.lora_B)
        
        
    def forward(self, x):
        result = F.embedding(x, self.weight, self.padding_idx)
        after_A = F.embedding(x, self.lora_A, self.padding_idx)
        result += self.scale*(after_A@self.lora_B)
        return result
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
        bsz, total_length = inputs['input_ids'].size()
        inputs['input_ids'] = inputs['input_ids'][:,:total_length - (total_length % self.head_num)]
        bsz, total_length = inputs['input_ids'].size()
        mem_size = self.mem_tokens.size(0)
        # num_segments: 片段个数 | segment_mem_size: 片段中mem的大小 | segment_length：片段长度
        num_segments = self.compute_num_segments(total_length)
        segment_length = self.segment_len
        # 收集compress_token
        compress_token = None
        compress_token_ids = None
        all_trimmed_past_key_values = []
        for segment_idx in range(num_segments):
            # 片段token：0-510 -> 510-1020 -> 1020-1530 -> ...
            start_idx = segment_idx * segment_length
            end_idx = min((segment_idx + 1) * segment_length, total_length)
            segment_input_ids = inputs['input_ids'][:, start_idx:end_idx]
            # ->LlamaForCausalLM->LlamaModel->embed_tokens
            # todo:1.
            if self.task_config["use_multi_lora"]:
                mask = {"lm_mask": torch.ones_like(segment_input_ids)}
                inputs_embeds = self.model.model.embed_tokens(segment_input_ids, mask)
            else:
                inputs_embeds = self.model.model.embed_tokens(segment_input_ids)

            bsz, seq_len, emb_size = inputs_embeds.size()
            # 为了适配最后一个片段不足510
            mem_size = round((end_idx - start_idx) // self.head_num)
            mem_tokens = self.mem_tokens[:mem_size,:]
            expand_mem = mem_tokens.unsqueeze(0).expand(bsz, mem_size, emb_size)
            encode_inputs_embeds = torch.cat([inputs_embeds, expand_mem], dim=1)

            # [1,seq_len]
            position_ids = torch.arange(start_idx + 1, end_idx + 1, device=inputs_embeds.device).unsqueeze(0)
            # [1,mem_size]：compress token position information, the step is compression ratio
            mem_position_ids = torch.arange((start_idx + (self.head_num + 1) // 2), end_idx + 1, step=self.head_num,device=inputs_embeds.device).unsqueeze(0)
            # [1,seq_len+mem_size]
            encode_position_ids = torch.cat([position_ids, mem_position_ids], dim=1)

            if compress_token_ids is None:
                compress_token_ids = mem_position_ids
            else:
                # 将新的 mem_hidden 拼接到 compress_token
                compress_token_ids = torch.cat((compress_token_ids, mem_position_ids), dim=1)

            # make three masks：cl_mask、lm_mask、cl_prime_mask
            # todo:2.
            if self.task_config["use_multi_lora"]:
                mask = make_masks(inputs_embeds, expand_mem)
                # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
                if "wo_pe" in self.task_config:
                    # print("no pe in here")
                    outputs = self.model(inputs_embeds=encode_inputs_embeds, output_hidden_states=True, mask=mask)
                else:
                    outputs = self.model(position_ids=encode_position_ids, inputs_embeds=encode_inputs_embeds,
                                         output_hidden_states=True, mask=mask)
            else:
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

            past_key_values = outputs.past_key_values
            # print(past_key_values.shape)
            trimmed_past_key_values = tuple(
                (layer_key[:, :, -mem_size:, :], layer_value[:, :, -mem_size:, :])
                for layer_key, layer_value in past_key_values
            )
            all_trimmed_past_key_values.append(trimmed_past_key_values)
        # compress_token = self.compress_project(compress_token)
        tot_loss = 0
        tot_task = 0
        loss_info = {}
        # 首先，获取层数（假设所有片段的层数都是一样的）
        num_layers = len(all_trimmed_past_key_values[0])
        merged_past_key_values = []

        # 对于每一层，遍历所有片段，将该层的 key 和 value 分别拼接在一起
        for layer_idx in range(num_layers):
            keys = []
            values = []
            for trimmed in all_trimmed_past_key_values:
                # trimmed 是一个 tuple，其中 trimmed[layer_idx] = (layer_key, layer_value)
                layer_key, layer_value = trimmed[layer_idx]
                keys.append(layer_key)
                values.append(layer_value)
            # 假设 mem_size 的维度在 dim=2 上，将所有片段的 key 和 value 拼接起来
            merged_key = torch.cat(keys, dim=2)
            merged_value = torch.cat(values, dim=2)
            merged_past_key_values.append((merged_key, merged_value))

        # 转换为 tuple 形式
        concatenated_past_key_values = tuple(merged_past_key_values)
        if self.task_config["use_compress_loss"]:
            # print("compress_targets will be used")
            # [B,mem_size,emb_size] -> [B,mem_size,head_num*vocab_size]
            logits =  self.compress_head(compress_token)
            # extract original logits
            # [B,mem_size,head_num*vocab_size] -> [B,tot_Seg_len,V] -> [B,seq_len,V]
            logits = logits.float()
            logits = logits.contiguous().view(-1, self.vocab_size)
            compress_targets = inputs["input_ids"].contiguous().view(-1).to(logits.device)

            compress_loss = self.loss_fct(logits, compress_targets)
            loss_info["compress_loss"] = compress_loss.item()
            tot_loss += compress_loss
            tot_task += 1


        # LM loss
        if 'lm_targets' in inputs and self.task_config["use_lm_loss"]:
            if self.task_config["use_multi_lora"]:
                mask = {"lm_mask": torch.ones_like(inputs['lm_targets'][:,:-1])}
                # [B,seq_len-1] -> [B,seq_len-1,E]
                lm_target_emb = self.model.model.embed_tokens(inputs['lm_targets'][:,:-1], mask)
            else:
                lm_target_emb = self.model.model.embed_tokens(inputs['lm_targets'][:, :-1])
            _, seq_len, emb_size = lm_target_emb.size()
            # [1,E] -> [1,1,E] -> [B,1,E]
            expand_lm_token = self.special_tokens[1:2].unsqueeze(0).expand(bsz, 1, emb_size)

            # todo: 1.将mem_hidden设置为0, .detach()
            #  [B,mem_size,E];     [B,1,E];      [B,seq_len-1,E]
            lm_emb = torch.cat([expand_lm_token,lm_target_emb],dim=1)

            latter_position_ids = torch.arange(end_idx+1,end_idx+seq_len+2,device=lm_target_emb.device).unsqueeze(0)
            lm_position_ids = torch.cat([compress_token_ids,latter_position_ids-1],dim=1)

            # make three masks
            if self.task_config["use_multi_lora"]:
                mask = make_masks(torch.cat([expand_lm_token,lm_target_emb],dim=1), mem_hidden, compress_prime_token=True)
                if "wo_pe" in self.task_config:
                    outputs = self.model(
                    inputs_embeds=lm_emb,
                    mask=mask,
                )
                else:
                    outputs = self.model(
                    position_ids=lm_position_ids,
                    inputs_embeds=lm_emb,
                    mask=mask,
                )
            else:
                if "wo_pe" in self.task_config:
                    outputs = self.decoder(
                        inputs_embeds=lm_emb,
                        past_key_values=concatenated_past_key_values
                    )
                else:
                    outputs = self.decoder(
                        inputs_embeds=lm_emb,
                        position_ids=latter_position_ids-1,
                        past_key_values=concatenated_past_key_values
                    )

            # [B,mem_size+S,V] -> [B,S,V]
            logits = outputs.logits[:,1:]
            # logits = logits[:, 1:]
            # here, we cat the whole seq's logits
            logits = logits.contiguous().view(-1, self.vocab_size)
            # todo: 这里的instruction_target应该是lm片段，question+answer,然后question-100
            inputs["instruction_target"] = inputs["instruction_target"].contiguous().view(-1).to(logits.device)

            lm_loss = self.loss_fct(logits, inputs["instruction_target"])
            loss_info["lm_loss"] = lm_loss.item()
            tot_loss += lm_loss
            tot_task += 1

        loss = tot_loss/tot_task

        return {"loss":loss, "loss_info":loss_info}

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

    def compute_num_segments(self, total_length):
        assert total_length > 0
        # todo: 后面可以把4改成随意值
        num_segments = math.ceil(total_length / self.segment_len)
        return num_segments
    def lm_inference(self,inputs,segment_size):
        # ->LlamaForCausalLM->LlamaModel->embed_tokens
        bsz, total_length = inputs['input_ids'].size()
        inputs['input_ids'] = inputs['input_ids'][:, :total_length - (total_length % self.head_num)]
        bsz, total_length = inputs['input_ids'].size()
        # num_segments: 片段个数 | segment_mem_size: 片段中mem的大小 | segment_length：片段长度
        num_segments = self.compute_num_segments(total_length)
        segment_length = self.segment_len
        # 收集compress_token
        compress_token = None
        compress_token_ids = None
        all_trimmed_past_key_values = []
        for segment_idx in range(num_segments):
            # 片段token：0-510 -> 510-1020 -> 1020-1530 -> ...
            start_idx = segment_idx * segment_length
            end_idx = min((segment_idx + 1) * segment_length, total_length)
            segment_input_ids = inputs['input_ids'][:, start_idx:end_idx]
            # ->LlamaForCausalLM->LlamaModel->embed_tokens
            # todo:1.
            if self.task_config["use_multi_lora"]:
                mask = {"lm_mask": torch.ones_like(segment_input_ids)}
                inputs_embeds = self.model.model.embed_tokens(segment_input_ids, mask)
            else:
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
            # todo:2.
            if self.task_config["use_multi_lora"]:
                mask = make_masks(inputs_embeds, expand_mem)
                # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
                if "wo_pe" in self.task_config:
                    # print("no pe in here")
                    outputs = self.model(inputs_embeds=encode_inputs_embeds, output_hidden_states=True, mask=mask)
                else:
                    outputs = self.model(position_ids=encode_position_ids, inputs_embeds=encode_inputs_embeds,
                                         output_hidden_states=True, mask=mask)
            else:
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

            past_key_values = outputs.past_key_values
            # print(past_key_values.shape)
            trimmed_past_key_values = tuple(
                (layer_key[:, :, -mem_size:, :], layer_value[:, :, -mem_size:, :])
                for layer_key, layer_value in past_key_values
            )
            all_trimmed_past_key_values.append(trimmed_past_key_values)
            # compress_token = self.compress_project(compress_token)
        # 首先，获取层数（假设所有片段的层数都是一样的）
        num_layers = len(all_trimmed_past_key_values[0])
        merged_past_key_values = []

        # 对于每一层，遍历所有片段，将该层的 key 和 value 分别拼接在一起
        for layer_idx in range(num_layers):
            keys = []
            values = []
            for trimmed in all_trimmed_past_key_values:
                # trimmed 是一个 tuple，其中 trimmed[layer_idx] = (layer_key, layer_value)
                layer_key, layer_value = trimmed[layer_idx]
                keys.append(layer_key)
                values.append(layer_value)
            # 假设 mem_size 的维度在 dim=2 上，将所有片段的 key 和 value 拼接起来
            merged_key = torch.cat(keys, dim=2)
            merged_value = torch.cat(values, dim=2)
            merged_past_key_values.append((merged_key, merged_value))

        # 转换为 tuple 形式
        concatenated_past_key_values = tuple(merged_past_key_values)

        # compress_token = self.compress_project(compress_token)
        if self.task_config["use_multi_lora"]:
            mask = {"lm_mask": torch.ones_like(inputs['lm_targets'])}
            # [B,seq_len-1] -> [B,seq_len-1,E]
            lm_target_emb = self.model.model.embed_tokens(inputs['lm_targets'], mask)
        else:
            lm_target_emb = self.model.model.embed_tokens(inputs['lm_targets'])
        _, seq_len, emb_size = lm_target_emb.size()
        # [1,E] -> [1,1,E] -> [B,1,E]
        expand_lm_token = self.special_tokens[1:2].unsqueeze(0).expand(bsz, 1, emb_size)

        # todo: 1.将mem_hidden设置为0, .detach()
        #  [B,mem_size,E];     [B,1,E];      [B,seq_len-1,E]
        lm_emb = torch.cat([expand_lm_token, lm_target_emb], dim=1)

        latter_position_ids = torch.arange(end_idx, end_idx + seq_len + 1, device=lm_target_emb.device).unsqueeze(0)
        lm_position_ids = torch.cat([compress_token_ids, latter_position_ids], dim=1)


        generate_text = []
        past_key_values = concatenated_past_key_values
        next_inputs_embeds = lm_emb.clone()
        next_position_ids = latter_position_ids.clone()
        if self.task_config["use_multi_lora"]:
            mask = make_masks(torch.cat([expand_lm_token,lm_target_emb],dim=1), compress_token_ids, compress_prime_token=True)
        for i in range(1024):
            # print(f"next_position_ids:{next_position_ids}")
            if self.task_config["use_multi_lora"]:
                if "wo_pe" in self.task_config:
                    out = self.model(inputs_embeds=next_inputs_embeds,
                                     past_key_values=past_key_values,
                                     use_cache=True,
                                     mask=mask)
                else:
                    out = self.model(position_ids=next_position_ids,
                                     inputs_embeds=next_inputs_embeds,
                                     past_key_values=past_key_values,
                                     use_cache=True,
                                     mask=mask)
            else:
                if "wo_pe" in self.task_config:
                    out = self.decoder(inputs_embeds=next_inputs_embeds,
                                     past_key_values=past_key_values,
                                     use_cache=True)
                else:
                    out = self.decoder(
                                     inputs_embeds=next_inputs_embeds,
                                     past_key_values=past_key_values,
                                     position_ids=next_position_ids,
                                     use_cache=True)
            # [B,S,V] -> [B,V]
            logit = out.logits[:, -1]
            past_key_values = out.past_key_values
            # [B,V]->[B]
            next_token_id = torch.argmax(logit, dim=-1)

            # [B]->[B,E]->[B,1,E]
            if self.task_config["use_multi_lora"]:
                mask = {"lm_mask": torch.ones_like(next_token_id)}
                next_inputs_embeds = self.model.model.embed_tokens(next_token_id, mask).unsqueeze(1).to(inputs_embeds.device)
                mask = {"lm_mask": torch.ones_like(next_inputs_embeds)}
            else:
                next_inputs_embeds = self.model.model.embed_tokens(next_token_id).unsqueeze(1).to(inputs_embeds.device)
            # todo: 不是很理解这里每次都是[1,1]和+1的作用
            next_position_ids = next_position_ids[:,-1:]+1 # [1, seq_len]/[1,1] -> [1,1]
            generate_text.append(next_token_id.item())
            if next_token_id.item() == 2:
                return generate_text
        return generate_text


    def cl_inference(self, inputs, segment_size):
        # ->LlamaForCausalLM->LlamaModel->embed_tokens
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
            # todo:1.
            if self.task_config["use_multi_lora"]:
                mask = {"lm_mask": torch.ones_like(segment_input_ids)}
                inputs_embeds = self.model.model.embed_tokens(segment_input_ids, mask)
            else:
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
            # todo:2.
            if self.task_config["use_multi_lora"]:
                mask = make_masks(inputs_embeds, expand_mem)
                # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
                if "wo_pe" in self.task_config:
                    # print("no pe in here")
                    outputs = self.model(inputs_embeds=encode_inputs_embeds, output_hidden_states=True, mask=mask)
                else:
                    outputs = self.model(position_ids=encode_position_ids, inputs_embeds=encode_inputs_embeds,
                                         output_hidden_states=True, mask=mask)
            else:
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
        # [B,mem_size,emb_size] -> [B,mem_size,head_num*vocab_size]
        # logits = self.compress_head(compress_token).float()
        logits = self.compress_head(compress_token).float()
        # [B*mem_size*head_num,vocab_size]
        logits = logits.contiguous().view(-1, self.vocab_size)
        # [b*m*h,v] -> [b*m*h]
        generate_text = torch.argmax(logits, dim=-1).tolist()

        return generate_text

    def vanilla_llama(self, inputs, segment_size):
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
            # todo: 不是很理解这里每次都是[1,1]和+1的作用
            generate_text.append(next_token_id.item())
            if next_token_id.item() == 2:
                return generate_text
        return generate_text


def make_masks(input_token=None, compress_token=None, compress_prime_token=False):
    # make three masks：cl_mask、lm_mask、cl_prime_mask
    mask = {}
    if compress_prime_token:
        lm_zero_mask = torch.zeros_like(compress_token, dtype=torch.bfloat16).to(input_token.device)
        lm_ones_mask = torch.ones_like(input_token, dtype=torch.bfloat16).to(input_token.device)
        lm_mask = torch.cat([lm_zero_mask, lm_ones_mask], dim=1).to(input_token.device)

        cl_prime_ones_mask = torch.ones_like(compress_token, dtype=torch.bfloat16).to(input_token.device)
        cl_prime_zero_mask = torch.zeros_like(input_token, dtype=torch.bfloat16).to(input_token.device)
        cl_prime_mask = torch.cat([cl_prime_ones_mask, cl_prime_zero_mask], dim=1).to(input_token.device)

        mask.update({"cl_prime_mask": cl_prime_mask, "lm_mask": lm_mask,})
    else:
        cl_zero_mask = torch.zeros_like(input_token, dtype=torch.bfloat16).to(input_token.device)
        cl_ones_mask = torch.ones_like(compress_token, dtype=torch.bfloat16).to(input_token.device)
        cl_mask = torch.cat([cl_zero_mask, cl_ones_mask], dim=1).to(input_token.device)

        lm_ones_mask = torch.ones_like(input_token, dtype=torch.bfloat16).to(input_token.device)
        lm_zero_mask = torch.zeros_like(compress_token, dtype=torch.bfloat16).to(input_token.device)
        lm_mask = torch.cat([lm_ones_mask, lm_zero_mask], dim=1).to(input_token.device)

        mask.update({"cl_mask": cl_mask, "lm_mask": lm_mask,})
    return mask



def save_adapter(model,save_path_and_name='adapter.pt', log=False):
    adapter_name = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            if log:
                print("[Save Adapter]",name)
            adapter_name.add(name)
            
    state_dict = model.state_dict()
    adapter_state_dict = {name: param for name, param in state_dict.items() if name in adapter_name}
    torch.save(adapter_state_dict, save_path_and_name)

def load_adapter(model, save_path_and_name='adapter.pt', log=False):
    adapter_state_dict = torch.load(save_path_and_name, map_location='cpu')  # 先加载到CPU
    if log:
        print("Loading adapter parameters:")
        for name, weight in adapter_state_dict.items():
            print(f"[Load Adapter] {name}")
    # 将adapter的权重转移到模型的设备上
    adapter_state_dict = {k: v.to(model.device) for k, v in adapter_state_dict.items()}

    model.load_state_dict(adapter_state_dict, strict=False)
    return model
def freeze(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.requires_grad = False
def load_adapter_to_merge_weight(model, train_adapter='adapter.pt', instruction_adapter="", is_train=False):
    def merge_weight(model):
        for name, module in model.named_children():     # adapter是W'=W+AB -> instruction_adapter是
            if name == "compress_head":
                continue
            if isinstance(module, LinearLoraLayer) or isinstance(module, EmbeddingLoraLayer):
                lora_AB = module.lora_A.data @ module.lora_B.data
                if module.weight.data.shape == lora_AB.shape:
                    module.weight.data += lora_AB * module.scale
                else:
                    module.weight.data += lora_AB.transpose(0,1) * module.scale
            else:
                merge_weight(module)

    def init_lora(model, task_config):
        for name, module in model.named_children():
            if name == "compress_head":
                continue
            if isinstance(module, LinearLoraLayer):
                setattr(model, name,
                        LinearLoraLayer(module.in_features, module.out_features, r=16, weight=module.weight.data.clone()))
            elif isinstance(module, EmbeddingLoraLayer):
                setattr(model, name,
                        EmbeddingLoraLayer(module.num_embeddings, module.embedding_dim, module.padding_idx, r=128, weight=module.weight.data.clone()))
            else:
                # Recursively apply this function to submodules
                init_lora(module, task_config)

    adapter_state_dict = torch.load(train_adapter, map_location='cpu')  # 先加载到CPU
    # 将adapter的权重转移到模型的设备上
    adapter_state_dict = {k: v.to(model.device) for k, v in adapter_state_dict.items()}

    model.load_state_dict(adapter_state_dict, strict=False)
    # W' -> W + AB
    merge_weight(model)
    init_lora(model, task_config="")
    # merge lora weight to origin
    if is_train:
        logging.info("train：merge lora weight to origin")
    else:
        # load A'B'
        adapter_state_dict = torch.load(instruction_adapter, map_location='cpu')  # 先加载到CPU
        # 将adapter的权重转移到模型的设备上
        adapter_state_dict = {k: v.to(model.device) for k, v in adapter_state_dict.items()}
        # finally -> h = W' + A'B' = W + AB + A'B'
        model.load_state_dict(adapter_state_dict, strict=False)
        logging.info("evaluator：no merge lora weight to origin")
    return model

def get_model_for_compress(model_id, task_config, rank):
    def add_compress_lora(model, task_config):
        for name, module in model.named_children():
            if name == "compress_head":
                continue
            if isinstance(module, nn.Linear) and ((name == "q_proj") or (name == "v_proj")):
                setattr(model, name,
                        LinearLoraLayer(module.in_features, module.out_features, r=128, weight=module.weight.data.clone()))
            # elif isinstance(module, nn.Embedding):
            #     setattr(model, name, EmbeddingLoraLayer(module.num_embeddings, module.embedding_dim, module.padding_idx, r=128,
            #                                             weight=module.weight.data.clone()))
            else:
                # Recursively apply this function to submodules
                add_compress_lora(module, task_config)

    def add_multi_lora(model, task_config):
        for name, module in model.named_children():
            if name == "compress_head":
                continue
            if isinstance(module, nn.Linear):
                setattr(model, name,
                        TripleLinearLoraLayer(module.in_features, module.out_features, r_cl=16, r_lm=16, r_cl_prime=16, weight=module.weight.data.clone()))
            elif isinstance(module, nn.Embedding):
                setattr(model, name, TripleEmbeddingLoraLayer(module.num_embeddings, module.embedding_dim, module.padding_idx,
                                                              r_cl=128, r_lm=128, r_cl_prime=128, weight=module.weight.data.clone()))
            else:
                # Recursively apply this function to submodules
                add_multi_lora(module, task_config)
    # config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )
    if task_config["use_multi_lora"]:
        modify_llama()
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
        freeze_model(model)
        add_compress_lora(model, task_config)
    return model


def get_model(model_id, task_config, rank):
    if task_config["task_type"] == "Compress":
        return get_model_for_compress(model_id, task_config, rank)
    raise Exception("Don't exist [{task_type}] task.")

def freeze_model(model):
    for name, param in model.named_parameters():
        if name == "compress_head.weight" or name == "mem_tokens" or name == "special_tokens":
            continue
        param.requires_grad = False
def freeze_decoder(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
def load_model_with_adapter(model_id, task_config, rank, save_path_and_name='adapter.pt', log=False):
    model = get_model(model_id, task_config, rank)
    load_adapter(model, save_path_and_name, log)
    return model
# python /home/liuxinyu/zrs/forget-me-not/models/llama3.py