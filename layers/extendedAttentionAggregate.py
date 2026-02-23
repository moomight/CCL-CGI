import torch.nn as nn

class ExtendedAttentionAggregate(nn.Module):
    def __init__(self, input_dim=8, d_model=128, num_heads=4):
        super(ExtendedAttentionAggregate, self).__init__()
        # translated,translated8translatedd_modeltranslated(translated128translated)
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
        # x translated (batch_size, seq_len, input_dim)
        x = self.input_proj(x)  # translated x translated (batch_size, seq_len, d_model)
        x = x.transpose(0, 1)   # translated (seq_len, batch_size, d_model)
        attn_output, attn_weights = self.attention(x, x, x,
                                                   need_weights=True,
                                                   average_attn_weights=False)
        attn_output = attn_output.transpose(0, 1)  # (batch_size, seq_len, d_model)
        # translated,translated (batch_size, d_model)
        output = attn_output.mean(dim=1)
        output = self.layernorm(output)
        output = self.dropout(output)
        return output, attn_weights
