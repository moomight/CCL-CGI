import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    def __init__(self, d_model, n_channels):
        super(AttentionFusion, self).__init__()
        self.n_channels = n_channels
        self.d_model = d_model

        self.layernorm = nn.LayerNorm(d_model)

        self.wq = nn.Linear(n_channels * d_model, n_channels * d_model)
        self.wk = nn.Linear(n_channels * d_model, n_channels * d_model)
        self.wv = nn.Linear(n_channels * d_model, n_channels * d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, 1, self.n_channels, self.d_model)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1)

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        dk = k.size(-1)
        scaling = dk ** -0.5
        scaled_attention_logits = matmul_qk * scaling

        v = v.view(batch_size, self.n_channels, self.d_model)

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        attention_weights = attention_weights.view(batch_size, -1, self.n_channels)
        scaled_attention = torch.matmul(attention_weights, v).view(batch_size, self.d_model)

        return self.layernorm(scaled_attention), attention_weights

# attention_layer = AttentionFusion(d_model=128, n_channels=4)
# output, attn_weights = attention_layer(torch.randn(32, 128))
