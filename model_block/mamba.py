import torch
import torch.nn as nn
import torch.nn.functional as F
from model_block.common import MLP, MultiHeadAttentionBlock, BasicBlock
from mamba_ssm import Mamba

class MambaExpert(nn.Module):
    def __init__(self, d_model=640, d_state=16, d_conv=4, expand=2):
        super(MambaExpert, self).__init__()
        self.mamba_blocks = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.avgpool = nn.AdaptiveAvgPool2d((1, d_model))

    def forward(self, x):
        batch_size = x.size(0)

        emb = self.mamba_blocks(x)

        output = self.avgpool(emb)

        return output.view(batch_size, -1)
