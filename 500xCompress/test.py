import math
import torch
from click.core import F
from torch import nn


class LinearLoraLayer(nn.Module):
    # No bias in LLama3 LinearLayer
    def __init__(self, in_features, out_features, r=16, weight=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.dropout = nn.Dropout(0.05)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.scale = 2  # The alpha value is usually twice the rank
        # self.lora_A1 = nn.Parameter(torch.zeros((in_features, r), device=self.weight.device, dtype=torch.bfloat16),
        #                            requires_grad=True)
        self.lora_B1 = nn.Parameter(torch.zeros((r, out_features), device=self.weight.device, dtype=torch.bfloat16),
                                   requires_grad=True)

        self.lora_A2 = nn.Parameter(torch.zeros((in_features, r/2), device=self.weight.device, dtype=torch.bfloat16),
                                   requires_grad=True)
        self.lora_B2 = nn.Parameter(torch.zeros((r/2, out_features), device=self.weight.device, dtype=torch.bfloat16),
                                   requires_grad=False)

        self.lora_A3 = nn.Parameter(torch.zeros((in_features, r / 4), device=self.weight.device, dtype=torch.bfloat16),
                                    requires_grad=True)
        self.lora_B3 = nn.Parameter(torch.zeros((r / 4, out_features), device=self.weight.device, dtype=torch.bfloat16),
                                    requires_grad=False)

        self.lora_A4 = nn.Parameter(torch.zeros((in_features, r / 8), device=self.weight.device, dtype=torch.bfloat16),
                                    requires_grad=True)
        self.lora_B4 = nn.Parameter(torch.zeros((r / 8, out_features), device=self.weight.device, dtype=torch.bfloat16),
                                    requires_grad=False)
        self.lora_A5 = nn.Parameter(torch.zeros((in_features, r / 8), device=self.weight.device, dtype=torch.bfloat16),
                                    requires_grad=True)
        nn.init.kaiming_uniform_(self.lora_A2, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_B2, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A3, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_B3, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A4, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_B4, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A5, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.lora_B1, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B1)

    def forward(self, x):
        result = F.linear(x, self.weight)
        result1 = result + self.scale * (x @ self.lora_A2 @ self.lora_B2)
        result2 = result1 + self.scale * (result1 @ self.lora_A3 @ self.lora_B3)
        result3 = result2 + self.scale * (result2 @ self.lora_A4 @ self.lora_B4)
        lora_A1 = torch.cat([self.lora_A2,self.lora_A3,self.lora_A4,self.lora_A5], dim=1)
        result = result3 + self.scale * (result3 @ lora_A1 @ self.lora_B1)
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
        self.lora_A = nn.Parameter(torch.zeros((in_features, r), device=self.weight.device, dtype=torch.bfloat16),
                                   requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros((r, out_features), device=self.weight.device, dtype=torch.bfloat16),
                                   requires_grad=True)
        nn.init.zeros_(self.lora_A)
        nn.init.normal_(self.lora_B)

    def forward(self, x):
        result = F.embedding(x, self.weight, self.padding_idx)
        after_A = F.embedding(x, self.lora_A, self.padding_idx)
        result += self.scale * (after_A @ self.lora_B)
        return result