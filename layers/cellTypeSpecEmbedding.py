import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import CentralityEncoding, GraphormerBlock, AttentionFusion
from losses import WeightedBinaryCrossEntropy
from torch.autograd import Variable

class CellTypeSpecEmbedding(nn.Module):
    def __init__(self, config, cell_type_num):
        super(CellTypeSpecEmbedding, self).__init__()
        self.cell_type_num = cell_type_num
        self.config = config
        self.centrEncodingLayer = CentralityEncoding(self.config.max_degree[self.cell_type_num], self.config.d_model).to('cuda')
        self.graphormer_layers_common_channel = nn.ModuleList(
            [GraphormerBlock(self.config.d_model, self.config.num_heads, self.config.dff, self.config.dropout,
                             self.config.d_sp_enc, self.config.sp_enc_activation).to('cuda')
             for _ in range(self.config.n_layers)]
        )
        self.attentionLayer = AttentionFusion(d_model=self.config.d_model, n_channels=self.config.n_graphs).to('cuda')
        # self.Linear = nn.Linear(self.config.d_model, self.config.d_model).double().to('cuda')
        # self.Relu = nn.ReLU().double().to('cuda')
        self.Linear = nn.Linear(self.config.d_model, self.config.d_model).to('cuda')
        self.Relu = nn.ReLU().to('cuda')
        # self.node_neighbor = torch.tensor(self.config.node_neighbor).to('cuda')
        # # self.node_feature = torch.tensor(self.config.node_feature).to('cuda')
        # self.distance_matrix = torch.tensor(self.config.distance_matrix).to('cuda')
        # self.spatial_matrix = torch.tensor(self.config.spatial_matrix).to('cuda')

    def forward(self, input_node_id):
        sub_node_feature, sub_distance, sub_spatial, sub_node_neighbor = self.get_sub_info(input_node_id)
        sub_node_feature = torch.nan_to_num(sub_node_feature, 0.0)
        if self.cell_type_num == 18:
            if torch.any(torch.isnan(sub_node_feature)):
                nan = torch.isnan(sub_node_feature)
                nan_positions = torch.nonzero(nan)
                print(f"\nsub_node_feature:{nan_positions}")
            if torch.any(torch.isnan(sub_distance)):
                nan = torch.isnan(sub_distance)
                nan_positions = torch.nonzero(nan)
                print(f"\nsub_distance:{nan_positions}")
            if torch.any(torch.isnan(sub_spatial)):
                nan = torch.isnan(sub_spatial)
                nan_positions = torch.nonzero(nan)
                print(f"\nsub_spatial:{nan_positions}")

        node_embedding = []
        attention_each_layer = []

        for g in range(self.config.n_graphs):
            centr_encoding = self.centrEncodingLayer(sub_distance[:, g, :, :])
            out = self.get_node_feature(sub_node_feature[:, g, :, :], centr_encoding)
            out = out.float()
            # print(out, out.dtype)
            out = self.Linear(out)
            out = self.Relu(out)

            spatial_matrix_in_subgraphs = sub_spatial[:, g, :, :, :]
            mask = self.create_padding_mask(spatial_matrix_in_subgraphs[:, 0, :, :])
            attention_mask = mask.unsqueeze(1)

            for n in range(self.config.n_layers):
                spatial_matrix_hop = spatial_matrix_in_subgraphs[:, 0, :, :]
                attention_mask_n = attention_mask
                out, attn = self.graphormer_layers_common_channel[n](out, self.config.training, attention_mask_n, spatial_matrix_hop)
                attention_each_layer.append(attn)
                # attn_shape[num_heads, neighbors, neighbors]

            node_embedding.append(out[:, 0, :])

        aggreated_out = torch.cat(node_embedding, dim=-1)
        finalembedding, _ = self.attentionLayer(aggreated_out)

        return finalembedding

    def get_final_embedding_all(self, attention_weights, embeddings):
        embeddings = embeddings.view(-1, self.config.n_graphs, self.config.n_layers, self.config.d_model)
        embeddings = embeddings.permute(0, 2, 1, 3)  # [bz, n_layers, n_graphs, d_model]
        attention_weights = attention_weights.view(-1, 1, self.config.n_graphs).unsqueeze(1)  # [bz, 1, 1, n_graphs]
        attention_weights = attention_weights.repeat(1, self.config.n_layers, 1, 1)  # [bz, n_layers, 1, n_graphs]

        return torch.matmul(attention_weights, embeddings)

    def weight_loss(self, y_true, y_pred):
        l = WeightedBinaryCrossEntropy(self.config.loss_mul)
        return l.get_loss(y_true, y_pred)

    def get_sub_info(self, node_id):
        length = len(node_id)
        node_neighbors = torch.index_select(torch.tensor(self.config.node_neighbor[self.cell_type_num]).to('cuda'), 0, node_id)
        node_neighbors = torch.squeeze(node_neighbors, dim=1)

        # node_feature = torch.index_select(torch.tensor(self.config.node_feature[self.cell_type_num]).to('cuda'), 0, node_neighbors)
        # idx = torch.tensor(self.config.idx[self.cell_type_num]).to('cuda')
        idx = self.config.idx[self.cell_type_num]
        idx_dict = {idx[i]: i for i in range(len(idx))}
        # print(idx_dict)
        flat_node_neighbor = node_neighbors.view(-1)
        flat_node_neighbor_list = [i.to('cpu').tolist() for i in flat_node_neighbor]
        mapped_indices = torch.tensor([idx_dict[i] for i in flat_node_neighbor_list], dtype=torch.long).to('cuda')
        cell_type_node_feature = torch.tensor(self.config.node_feature[self.cell_type_num]).to('cuda')
        node_feature = cell_type_node_feature[mapped_indices]
        # print(cell_type_node_feature[16], node_feature[0])  # equal
        # nn_shape = node_neighbors.shape
        # nn_shape_list = nn_shape.to('cpu').tolist()
        if node_feature.numel() == 0:
            assert f"node_feature == 0"
        node_feature = node_feature.reshape(length, self.config.n_graphs, self.config.n_neighbors, -1)
        # print(node_feature.shape)

        # distance = torch.index_select(torch.tensor(self.config.distance_matrix[self.cell_type_num]).to('cuda'), 0, node_neighbors)
        cell_type_distance_matrix = torch.tensor(self.config.distance_matrix[self.cell_type_num]).to('cuda')
        distance = torch.index_select(cell_type_distance_matrix, 0, flat_node_neighbor)
        distance = distance.reshape(length, self.config.n_graphs, self.config.n_neighbors, -1)
        # print(distance.shape)

        spatial = torch.index_select(torch.tensor(self.config.spatial_matrix[self.cell_type_num]).to('cuda'), 0, node_id)
        spatial = torch.squeeze(spatial, dim=1)
        # print(spatial.shape)

        return node_feature, distance, spatial, node_neighbors

    def create_padding_mask(self, nodes):
        return (nodes == -1).float()

    def get_node_feature(self, node_embedding, centr_encoding):
        node_feature = node_embedding
        node_feature *= torch.sqrt(torch.tensor(self.config.d_model, dtype=torch.float32))
        node_feature += centr_encoding

        return F.dropout(node_feature, p=self.config.dropout, training=self.config.training)
