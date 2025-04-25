import torch
import torch.nn as nn
import torch.nn.functional as F
from .multiHeadAttention import MultiHeadAttention


class GraphormerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1, d_sp_enc=8, sp_enc_activation='relu'):
        super(GraphormerBlock, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, d_sp_enc, sp_enc_activation)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, training, mask, min_distance_matrix):
        residual = x
        x_norm = self.layernorm1(x)
        attn_output, _, = self.mha(x_norm, min_distance_matrix, mask)
        attn_output = self.dropout1(attn_output)
        out1 = residual + attn_output

        residual = out1
        out1_norm = self.layernorm2(out1)
        ffn_output = self.ffn(out1_norm)
        ffn_output = self.dropout2(ffn_output)
        out2 = residual + ffn_output

        return out2, _


def point_wise_feed_forward_network(d_model, dff):
    return nn.Sequential(
        nn.Linear(d_model, dff),
        nn.ReLU(),
        nn.Linear(dff, d_model)
    )
