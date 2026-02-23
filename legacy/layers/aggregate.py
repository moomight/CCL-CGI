import torch.nn as nn


class AttentionAggregate(nn.Module):
    def __init__(self, d_model, num_heads=4):
        self.d_model = d_model
        super(AttentionAggregate, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=0.1)
        self._reset_parameters()

    def _reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        # x 的形状应为 (batch_size, seq_len, embed_dim)
        x = x.transpose(0, 1)  # 调整形状为 (seq_len, batch_size, embed_dim)
        attn_output, attn_weights = self.attention(x, x, x,
                                                   need_weights=True, average_attn_weights=False
                                                   )
        attn_output = attn_output.transpose(0, 1)  # (batch_size, seq_len, embed_dim)
        # 对序列维度进行平均
        output = attn_output.mean(dim=1)  # (batch_size, embed_dim)
        output = self.layernorm(output)
        output = self.dropout(output)
        return output, attn_weights
