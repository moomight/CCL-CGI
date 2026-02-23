import torch
import torch.nn as nn
import torch.nn.functional as F
from .spatialEncoding import SpatialEncoding


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_sp_enc, sp_enc_activation):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_sp_enc = d_sp_enc
        self.sp_enc_activation = sp_enc_activation
        self.spatial_encoding = SpatialEncoding(self.d_sp_enc, self.sp_enc_activation)

        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, min_distance_matrix, mask=None):
        batch_size = q.size(0)
        spatial_encoding_bias = self.spatial_encoding(min_distance_matrix)  # [batch_size, seq_len_q, 64]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(spatial_encoding_bias)  # (batch_size, seq_len, d_model)
        v = self.wv(spatial_encoding_bias)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, self.num_heads, mask)

        scaled_attention = scaled_attention.permute(0, 2, 1, 3)  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = scaled_attention.contiguous().view(batch_size, -1,
                                                              self.d_model)  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def scaled_dot_product_attention(q, k, v, num_heads, mask=None):
    # 计算 q 和 k 的点积
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    dk = torch.tensor(k.size(-1), dtype=torch.float32)
    # 缩放点积
    scaling = dk ** -0.5
    scaled_attention_logits = matmul_qk * scaling

    # 添加掩码
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # 计算注意力权重
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)

    # 对 v 加权求和
    output = torch.matmul(attention_weights, v)

    return output, attention_weights
