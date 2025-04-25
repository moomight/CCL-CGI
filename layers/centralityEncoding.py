import torch
import torch.nn as nn
import torch.nn.functional as F


class CentralityEncoding(nn.Module):
    def __init__(self, max_degree, d_model, **kwargs):
        super(CentralityEncoding, self).__init__()
        self.centr_embedding = nn.Embedding(max_degree, d_model)

    def centrality(self, distances):
        centrality = torch.eq(torch.abs(distances), 1).float()
        centrality = torch.sum(centrality, dim=-1)
        return centrality.float()

    def forward(self, distances):
        centrality = self.centrality(distances)
        centrality_encoding = self.centr_embedding(centrality.long())  #[batch_size, n_graphs, n_nodes, 64]

        return centrality_encoding.float()  #[batch_size, n_graphs, n_nodes, 64]
