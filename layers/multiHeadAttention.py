import torch
import torch.nn as nn
import torch.nn.functional as F
from .spatialEncoding import SpatialEncoding

class MultiHeadAttention(nn.Module):
    """Multi-head attention with spatial encoding bias (Graphormer-style).
    Keys and values are derived from the spatial encoding of shortest-path distances rather than the input embeddings directly."""
    def __init__(self, d_model, num_heads, d_sp_enc, sp_enc_activation, n_neighbors=8):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_sp_enc = d_sp_enc
        self.sp_enc_activation = sp_enc_activation
        self.n_neighbors = n_neighbors
        self.spatial_encoding = SpatialEncoding(d_sp_enc=self.d_sp_enc,
                                                activation=self.sp_enc_activation,
                                                d_model=d_model,
                                                n_neighbors=n_neighbors)

        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """Reshape (batch, seq_len, d_model) → (batch, num_heads, seq_len, depth)."""
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, min_distance_matrix, mask=None):
        """Args:
            q: input node embeddings, (batch_size, seq_len, d_model).
            min_distance_matrix: shortest-path distances used for spatial encoding.
            mask: optional padding mask.
        Returns:
            output: (batch_size, seq_len, d_model).
            attention_weights: per-head attention weights."""
        batch_size = q.size(0)
        spatial_encoding_bias = self.spatial_encoding(min_distance_matrix)  # [batch_size, seq_len_q, d_model]

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
    """Standard scaled dot-product attention: softmax(QK^T / sqrt(d_k)) V.

    Args:
        q, k, v: query / key / value tensors, (batch, heads, seq, depth).
        num_heads: number of attention heads.
        mask: optional additive mask (masked positions filled with -1e9).
    Returns:
        output: weighted sum of values.
        attention_weights: softmax attention scores."""
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    dk = torch.tensor(k.size(-1), dtype=torch.float32)
    scaling = dk ** -0.5
    scaled_attention_logits = matmul_qk * scaling

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)

    output = torch.matmul(attention_weights, v)

    return output, attention_weights
