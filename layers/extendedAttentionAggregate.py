import torch.nn as nn

class ExtendedAttentionAggregate(nn.Module):
    def __init__(self, input_dim=8, d_model=128, num_heads=4):
        super(ExtendedAttentionAggregate, self).__init__()
        # 先对输入进行投影，将8维映射到d_model维度（比如128维）
        self.input_proj = nn.Linear(input_dim, d_model)
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
        # x 的形状应为 (batch_size, seq_len, input_dim)
        x = self.input_proj(x)  # 投影后的 x 形状为 (batch_size, seq_len, d_model)
        x = x.transpose(0, 1)   # 调整形状为 (seq_len, batch_size, d_model)
        attn_output, attn_weights = self.attention(x, x, x,
                                                   need_weights=True,
                                                   average_attn_weights=False)
        attn_output = attn_output.transpose(0, 1)  # (batch_size, seq_len, d_model)
        # 对序列维度进行平均，得到 (batch_size, d_model)
        output = attn_output.mean(dim=1)
        output = self.layernorm(output)
        output = self.dropout(output)
        return output, attn_weights
