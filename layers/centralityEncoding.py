import torch
import torch.nn as nn
import torch.nn.functional as F


class CentralityEncoding(nn.Module):
    """Encodes node centrality (degree) from a distance matrix into a learned embedding.
    Computes degree by counting direct neighbours (distance == 1) and maps the integer degree to a d_model-dimensional embedding via nn.Embedding."""
    def __init__(self, max_degree, d_model, **kwargs):
        super(CentralityEncoding, self).__init__()
        self.centr_embedding = nn.Embedding(max_degree, d_model)

    def centrality(self, distances):
        """Count direct neighbours for each node → degree vector."""
        centrality = torch.eq(torch.abs(distances), 1).float()
        centrality = torch.sum(centrality, dim=-1)
        return centrality.float()

    def forward(self, distances):
        """Args:
            distances: distance matrix, shape (batch_size, n_nodes, n_nodes).
        Returns:
            Centrality encoding of shape (batch_size, n_nodes, d_model)."""
        centrality = self.centrality(distances)
        centrality_encoding = self.centr_embedding(centrality.long())  #[batch_size, n_graphs, n_nodes, 64]

        return centrality_encoding.float()  #[batch_size, n_graphs, n_nodes, 64]
