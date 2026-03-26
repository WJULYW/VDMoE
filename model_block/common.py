import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class GRL(nn.Module):

    def __init__(self, max_iter):
        super(GRL, self).__init__()
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter

    def forward(self, input):
        self.iter_num += 1
        return input * 1.0

    def backward(self, gradOutput):
        coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter))
                         - (self.high - self.low) + self.low)
        return -coeff * gradOutput

class Discriminator(nn.Module):
    def __init__(self, input_dim, max_iter=4000, sub_num=4):
        super(Discriminator, self).__init__()
        self.ad_net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(input_dim, sub_num)
        )
        self.grl_layer = GRL(max_iter)

    def forward(self, feature):
        adversarial_out = self.ad_net(self.grl_layer(feature))
        return adversarial_out

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super(MultiHeadAttentionBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, embed_dim)
        )

    def forward(self, x):
        x = self.ln1(x)
        attn_output, _ = self.multihead_attn(x, x, x)
        x += attn_output

        x = self.ln2(x)
        mlp_output = self.mlp(x)
        x += mlp_output

        return x

class MLP(nn.Module):
    def __init__(self, input_size, output_size, seq_len=None, position_tag=None):
        super(MLP, self).__init__()
        if position_tag:
            self.position_embeddings = nn.Parameter(torch.randn(1, seq_len, input_size))
        self.position_tag = position_tag
        self.model = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(inplace=True),
            nn.Dropout1d(0.1),
            nn.Linear(input_size, output_size)
        )

    def forward(self, x):
        if self.position_tag:
            x = x + self.position_embeddings
        y = self.model(x)
        return y

class LoraLinear(nn.Linear):

    def __init__(self, in_features, out_features, r=16):
        super().__init__(in_features, out_features)

        self.lora_matrix_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_matrix_A = nn.Parameter(torch.randn(r, in_features))

        self.scaling = 2
        nn.init.kaiming_uniform_(self.lora_matrix_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_matrix_B)

    def forward(self, x):
        return F.linear(
            input=x,
            weight=(self.lora_matrix_B @ self.lora_matrix_A) * self.scaling
        )

class BasicBlock(nn.Module):
    def __init__(self, inplanes, out_planes, stride=2, downsample=1, Res=0, islast=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
        )
        if downsample == 1:
            self.down = nn.Sequential(
                nn.Conv2d(inplanes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_planes)
            )
        self.downsample = downsample
        self.Res = Res
        self.islast = islast

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.Res == 1:
            if self.downsample == 1:
                x = self.down(x)
            out += x
        if self.islast:
            return out
        else:
            return F.relu(out)

class Gate(nn.Module):
    def __init__(self, inplanes, out_planes, stride=2, downsample=1, Res=0, islast=False):
        super(Gate, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
        )
        if downsample == 1:
            self.down = nn.Sequential(
                nn.Conv2d(inplanes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_planes)
            )
        self.downsample = downsample
        self.Res = Res
        self.islast = islast

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.Res == 1:
            if self.downsample == 1:
                x = self.down(x)
            out += x
        if self.islast:
            return out
        else:
            return F.relu(out)
