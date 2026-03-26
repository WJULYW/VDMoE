import torch
import torch.nn as nn
import torch.nn.functional as F
from model_block.common import MLP, MultiHeadAttentionBlock, BasicBlock
from model_block.feature_embedding import *

class Transformer(nn.Module):
    def __init__(self, embed_dim=640, num_heads=8, seq_length=300, num_blocks=4,
                 mlp_dim=512):
        super(Transformer, self).__init__()
        self.embedding = FeatureEmbedding2D()
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.position_embeddings = nn.Parameter(torch.randn(1, seq_length + 1, embed_dim))

        self.blocks = nn.ModuleList([
            MultiHeadAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim)
            for _ in range(num_blocks)
        ])

        self.head1 = MLP(input_size=640, hidden_size=128, output_size=2)
        self.head2 = MLP(input_size=640, hidden_size=128, output_size=2)
        self.head4 = MLP(input_size=640, hidden_size=128, output_size=300)
        self.head5 = MLP(input_size=640, hidden_size=128, output_size=1)
        self.head6 = MLP(input_size=640, hidden_size=128, output_size=1)

    def forward(self, x1, x2, x3, x4, x5):
        combined = self.embedding(x1, x2, x3, x4, x5)
        batch_size = combined.size(0)

        class_tokens = self.class_token.expand(batch_size, -1, -1)

        combined = torch.cat((class_tokens, combined), dim=1)

        combined = combined + self.position_embeddings

        for block in self.blocks:
            combined = block(combined)

        class_token_output = combined[:, 0, :]
        output1 = self.head1(class_token_output)
        output2 = self.head2(class_token_output)
        RESP = self.head4(class_token_output)
        HR = self.head5(class_token_output)
        RR = self.head6(class_token_output)

        return output1, output2, RESP, HR, RR

class TransformerExpert(nn.Module):
    def __init__(self, embed_dim=640, num_heads=8, seq_length=300, num_blocks=4,
                 mlp_dim=512):
        super(TransformerExpert, self).__init__()
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.position_embeddings = nn.Parameter(torch.randn(1, seq_length + 1, embed_dim))

        self.blocks = nn.ModuleList([
            MultiHeadAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        batch_size = x.size(0)

        class_tokens = self.class_token.expand(batch_size, -1, -1)

        combined = torch.cat((class_tokens, x), dim=1)

        combined = combined + self.position_embeddings

        for block in self.blocks:
            combined = block(combined)

        class_token_output = combined[:, 0, :]

        return class_token_output

class TransformerExpert_Block(nn.Module):
    def __init__(self, embed_dim=640, num_heads=8, mlp_dim=512, seq_len = 300, position_tag=False, class_token=None):
        super(TransformerExpert_Block, self).__init__()
        self.position_tag = position_tag
        if position_tag:
            self.position_embeddings = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        self.att = MultiHeadAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim)

    def forward(self, x):
        if self.position_tag:
            x = x + self.position_embeddings

        combined = self.att(x)

        return combined
