import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialEncoding(nn.Module):
    def __init__(self, d_sp_enc=8, activation='relu'):
        super(SpatialEncoding, self).__init__()
        self.d_sp_enc = d_sp_enc
        self.activation = activation
        self.dense1 = nn.Linear(d_sp_enc, 64)
        self.dense2 = nn.Linear(64, 8)
        self.dropout = nn.Dropout(0.1)

    def forward(self, distances):
        distances = distances.float()
        outputs = F.relu(self.dense1(distances))  # [batch_size, n_nodes, 64]
        outputs = F.relu(self.dense2(outputs))  # [batch_size, n_nodes, 8]
        return outputs
