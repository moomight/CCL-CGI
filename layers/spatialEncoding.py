import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialEncoding(nn.Module):
    def __init__(self, d_sp_enc=64, activation='relu', d_model=8, n_neighbors=8):
        super(SpatialEncoding, self).__init__()
        self.d_sp_enc = d_sp_enc
        self.d_model = d_model
        self.n_neighbors = n_neighbors
        self.activation = activation

        # n_neighbors → d_sp_enc → d_model
        self.dense1 = nn.Linear(n_neighbors, d_sp_enc)
        self.dense2 = nn.Linear(d_sp_enc, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, distances):
        distances = distances.float()
        outputs = F.relu(self.dense1(distances))  # [batch_size, n_nodes, d_sp_enc]
        outputs = F.relu(self.dense2(outputs))  # [batch_size, n_nodes, d_model]
        return outputs
