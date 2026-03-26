import torch
import torch.nn as nn
import torch.nn.functional as F
from model_block.transformer import TransformerExpert_Block, TransformerExpert
from model_block.feature_embedding import FeatureEmbedding2D
from model_block.mamba import MambaExpert
from model_block.common import *
from model_block.resnet import *
from mamba_ssm import Mamba

class MMoE_block(nn.Module):
    def __init__(self, input_dim, seq_len, num_experts, num_tasks, position_tag=None):
        super(MMoE_block, self).__init__()

        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [
                TransformerExpert_Block(embed_dim=input_dim, num_heads=8, mlp_dim=512, seq_len=seq_len,
                                        position_tag=position_tag),
                TransformerExpert_Block(embed_dim=seq_len, num_heads=6, mlp_dim=256, seq_len=input_dim,
                                        position_tag=position_tag),
                Mamba(d_model=input_dim, d_state=16, d_conv=4, expand=1),
                Mamba(d_model=seq_len, d_state=16, d_conv=4, expand=1)]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, input_dim))
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, num_experts + 1)
        )

    def forward(self, x):
        identity = x
        expert_outputs = [identity]
        for i in range(self.num_experts):
            if i % 2 == 0:
                expert_outputs.append(self.experts[i](x))
            else:
                expert_outputs.append(self.experts[i](x.permute(0, 2, 1)).permute(0, 2, 1))

        expert_outputs = torch.stack(expert_outputs, dim=1)

        pooled_x = self.avgpool(x).view(x.size(0), -1)
        gate_values = torch.softmax(self.gate(pooled_x), dim=1).view(expert_outputs.size(0), expert_outputs.size(1), 1,
                                                                     1)
        output = torch.sum(gate_values * expert_outputs, dim=1)

        return output

class MMoE_mlp_block(nn.Module):
    def __init__(self, input_dim, seq_len, num_experts, num_tasks, position_tag=None):
        super(MMoE_mlp_block, self).__init__()

        self.num_tasks = num_tasks
        self.num_experts = num_experts
        if position_tag:
            self.position_embeddings = nn.Parameter(torch.randn(1, seq_len, input_dim))
        experts = [MLP(input_dim, input_dim, seq_len=seq_len, position_tag=position_tag) for _ in
                   range(int(self.num_experts / 2))] + [
                      MLP(seq_len, seq_len, seq_len=input_dim, position_tag=position_tag) for _
                      in
                      range(
                          int(self.num_experts / 2))]
        self.experts = nn.ModuleList(
            experts
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, input_dim))
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, num_experts + 1)
        )

    def forward(self, x):
        identity = x
        expert_outputs = [identity]
        for i in range(self.num_experts):
            if i < self.num_experts / 2:
                expert_outputs.append(self.experts[i](x))
            else:
                expert_outputs.append(self.experts[i](x.permute(0, 2, 1)).permute(0, 2, 1))

        expert_outputs = torch.stack(expert_outputs, dim=1)

        pooled_x = self.avgpool(x).view(x.size(0), -1)
        gate_values = torch.softmax(self.gate(pooled_x), dim=1).view(expert_outputs.size(0), expert_outputs.size(1), 1,
                                                                     1)
        output = torch.sum(gate_values * expert_outputs, dim=1)

        return output

class MMoE_lora_block(nn.Module):
    def __init__(self, input_dim, seq_len, num_experts, num_tasks, position_tag=None):
        super(MMoE_lora_block, self).__init__()

        self.num_tasks = num_tasks
        self.num_experts = num_experts
        experts = [LoraLinear(input_dim, input_dim) for _ in range(int(self.num_experts / 2))] + [
            LoraLinear(seq_len, seq_len) for _
            in
            range(
                int(self.num_experts / 2))]
        self.experts = nn.ModuleList(
            experts
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, input_dim))
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, num_experts + 1)
        )

    def forward(self, x):
        identity = x
        expert_outputs = [identity]
        for i in range(self.num_experts):
            if i < self.num_experts / 2:
                expert_outputs.append(self.experts[i](x))
            else:
                expert_outputs.append(self.experts[i](x.permute(0, 2, 1)).permute(0, 2, 1))

        expert_outputs = torch.stack(expert_outputs, dim=1)

        pooled_x = self.avgpool(x).view(x.size(0), -1)
        gate_values = torch.softmax(self.gate(pooled_x), dim=1).view(expert_outputs.size(0), expert_outputs.size(1), 1,
                                                                     1)
        output = torch.sum(gate_values * expert_outputs, dim=1)

        return output

class MMoE(nn.Module):
    def __init__(self, input_dim, seq_len, num_experts, num_tasks, expert_hidden_dim, tower_hidden_dim, output_dims,
                 block_num=1):
        super(MMoE, self).__init__()

        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.block_num = block_num
        self.seq_len = seq_len

        self.blocks = nn.ModuleList(
            [MMoE_mlp_block(input_dim, seq_len, self.num_experts, self.num_tasks, position_tag=False) for i in
             range(block_num)]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, input_dim))
        self.gates = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, expert_hidden_dim),
            nn.Sigmoid()
        ) for _ in range(num_tasks)])

        self.towers = nn.ModuleList(
            [MLP(expert_hidden_dim, output_dim) for output_dim in output_dims])

    def forward(self, x):
        B = x.size(0)
        out = self.blocks[0](x)
        for i in range(1, self.block_num):
            out = self.blocks[i](out)

        task_outputs = []
        task_rep = []
        x = self.avgpool(x).view(B, -1)
        for i in range(self.num_tasks):
            gate_values = 2 * self.gates[i](x)
            task_output = self.avgpool(out).view(B, -1)
            task_rep.append(gate_values * task_output)
            task_output = self.towers[i](gate_values * task_output)
            task_outputs.append(task_output)

        return task_outputs, task_rep, 0

class MMoE_simple(nn.Module):
    def __init__(self, input_dim, seq_len, num_experts, num_tasks, expert_hidden_dim, tower_hidden_dim, output_dims):
        super(MMoE_simple, self).__init__()

        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.seq_len = seq_len
        self.experts = nn.ModuleList(
            [ResnetExpert(),
             TransformerExpert(embed_dim=input_dim, num_heads=8, seq_length=300, num_blocks=4, mlp_dim=512),
             MambaExpert()]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, input_dim))
        self.gates = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, num_experts)
        ) for _ in range(num_tasks)])

        self.towers = nn.ModuleList(
            [MLP(expert_hidden_dim, tower_hidden_dim, output_dim) for output_dim in output_dims])

    def forward(self, x):
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            if i == 0:
                expert_outputs.append(expert(x))
            else:
                expert_outputs.append(expert(x))

        expert_outputs = torch.stack(expert_outputs, dim=1)

        task_outputs = []
        for i in range(self.num_tasks):
            pooled_x = self.avgpool(x).view(x.size(0), -1)
            gate_values = torch.softmax(self.gates[i](pooled_x), dim=1).unsqueeze(2)
            task_output = torch.sum(expert_outputs * gate_values, dim=1)
            task_output = self.towers[i](task_output)
            task_outputs.append(task_output)

        return task_outputs

class MultiTaskModel(nn.Module):
    def __init__(self, output_dims=[2, 2, 1, 1], input_dim=640, num_experts=4, num_tasks=4,
                 expert_hidden_dim=640,
                 tower_hidden_dim=64, seq_len=300, block_num=4):
        super(MultiTaskModel, self).__init__()
        self.embedding = FeatureEmbedding2D()
        self.mmoe = MMoE(input_dim, seq_len, num_experts, num_tasks, expert_hidden_dim, tower_hidden_dim, output_dims,
                         block_num)

    def count_parameters(self):
        param_num = sum(p.numel() for p in self.parameters() if p.requires_grad == True)
        return print("parameters (M):", param_num / (1024 ** 2))

    def forward(self, x1, x2, x3, x4, x5):
        x = self.embedding(x1, x2, x3, x4, x5)
        return self.mmoe(x)
