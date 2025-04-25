from layers import AttentionFusion, CentralityEncoding, GraphormerBlock, AttentionAggregate

from utils import score
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config import LOG_DIR, RESULT_DIR
import pandas as pd
from losses import compute_triplet_loss


class PanCancerGenePredict(pl.LightningModule):
    def __init__(self, config):
        super(PanCancerGenePredict, self).__init__()
        self.config = config
        self.temp = 0.3
        self.threshold = 0.5
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 简化模型，移除LLM部分
        # 定义全连接层进行分类
        # self.fc = nn.Sequential(
        #     nn.Linear(39 * self.config.d_model, 256),
        #     nn.ReLU(),
        #     nn.Dropout(self.config.dropout),
        #     nn.Linear(256, 1)
        # )
        self.fc = nn.Sequential(
            nn.Linear(self.config.d_model, 256),
            # nn.BatchNorm1d(256), # 加入正则化
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(128, 1)
        )
        self.sigmoid = nn.Sigmoid()

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        positive_fraction = torch.tensor(0.25).to('cuda')
        bias_value = -torch.log((1 - positive_fraction) / positive_fraction)
        self.fc[-1].bias.data.fill_(bias_value)

        # 定义Graphormer层
        self.graphormer_layers = nn.ModuleList(
            [GraphormerBlock(self.config.d_model, self.config.num_heads, self.config.dff, self.config.dropout,
                             self.config.d_sp_enc, self.config.sp_enc_activation)
             for _ in range(self.config.n_layers)]
        )
        self.centrEncodingLayer = nn.ModuleList([
            CentralityEncoding(self.config.max_degree[cell_type_num],
                               self.config.d_model)
            for cell_type_num in range(39)]
        )
        self.attentionLayer = AttentionFusion(d_model=self.config.d_model, n_channels=self.config.n_graphs).to('cuda')
        self.aggregateLayer = AttentionAggregate(d_model=self.config.d_model, num_heads=self.config.num_heads).to('cuda')
        self.Linear = nn.Linear(self.config.d_model, self.config.d_model).to('cuda')
        self.Relu = nn.ReLU()

        # 损失函数，仅使用BCELoss，并加入类别权重处理不平衡问题
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0).to('cuda'))

        self.alpha = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.beta = nn.Parameter(torch.tensor(0.1, requires_grad=True))
        self.lambda_reg = self.config.lambda_reg

        self.loss = 0
        self.acc = 0
        self.auc = 0
        self.aupr = 0
        self.f1 = 0
        self.batch_idx = 0
        self.val_acc = 0
        self.val_auc = 0
        self.val_aupr = 0
        self.val_f1 = 0
        self.val_loss = 0
        self.val_idx = 0
        self.test_loss = []
        self.test_acc = []
        self.test_auc = []
        self.test_aupr = []
        self.test_f1 = []
        self.history = {
            "train_loss": [], "train_acc": [], "train_auc": [], "train_aupr": [], "train_f1": [],
            "val_loss": [], "val_acc": [], "val_auc": [], "val_aupr": [], "val_f1": []
        }

    def forward(self, input_node_id):
        finalembedding = self.embedding_layer(input_node_id, self.config)
        finalembedding_ = finalembedding.view(len(input_node_id), -1, self.config.d_model)
        finalembedding_, _ = self.aggregateLayer(finalembedding_) # _: attention weights
        # x = self.fc(finalembedding)
        x = self.fc(finalembedding_)
        # x = self.sigmoid(x)
        # return x, finalembedding
        return x, finalembedding, finalembedding_, _

    def training_step(self, batch, batch_idx):
        self.batch_idx += 1
        x, y = batch
        x = x.to(self.device)
        y = y.float().to(self.device)
        y_predict, celltypeEmbedding, embedding, attention_weight = self(x)
        y_predict = y_predict.squeeze()

        # 处理类别不平衡，计算权重
        pos_weight = (y == 0).sum().float() / (y == 1).sum().float()
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        classification_loss = criterion(y_predict, y)

        triplet_loss = compute_triplet_loss(embedding, y)

        alpha = torch.exp(self.alpha)
        beta = torch.exp(self.beta)

        loss = alpha * classification_loss + beta * triplet_loss + self.lambda_reg * (alpha ** 2 + beta ** 2)

        cpu_loss = loss.item()
        self.loss += cpu_loss
        auc, acc, p, r, f1, aupr, fpr, tpr = score(y, y_predict, self.threshold, True)
        self.acc += acc
        self.auc += auc
        self.aupr += aupr
        self.f1 += f1
        self.log('loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('classification_loss', classification_loss, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('triplet_loss', triplet_loss, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('acc', acc, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('auc', auc, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('aupr', aupr, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('f1', f1, on_step=True, on_epoch=False, prog_bar=True, logger=False)

        return loss

    def on_train_epoch_start(self):
        self.loss = 0
        self.acc = 0
        self.auc = 0
        self.aupr = 0
        self.f1 = 0
        self.val_acc = 0
        self.val_auc = 0
        self.val_aupr = 0
        self.val_f1 = 0
        self.batch_idx = 0
        self.val_idx = 0
        self.val_loss = 0

    def validation_step(self, batch, batch_idx):
        self.val_idx += 1
        x, y = batch
        x = x.to(self.device)
        y = y.float().to(self.device)
        y_predict, celltypeEmbedding, embedding, attention_weight = self(x)
        y_predict = y_predict.squeeze()

        classification_loss = self.criterion(y_predict, y)

        triplet_loss = compute_triplet_loss(embedding, y)

        alpha = torch.exp(self.alpha)
        beta = torch.exp(self.beta)

        loss = alpha * classification_loss + beta * triplet_loss + self.lambda_reg * (alpha ** 2 + beta ** 2)
        val_auc, acc, p, r, f1, aupr, fpr, tpr = score(y, y_predict, self.threshold, False)
        self.val_acc += acc
        self.val_auc += val_auc
        self.val_aupr += aupr
        self.val_f1 += f1
        self.val_loss += loss.item()
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("val_auc", val_auc, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("val_aupr", aupr, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("val_f1", f1, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        return loss

    def on_validation_end(self):
        self.loss /= self.batch_idx
        self.acc /= self.batch_idx
        self.auc /= self.batch_idx
        self.aupr /= self.batch_idx
        self.f1 /= self.batch_idx
        self.history["train_loss"].append(self.loss)
        self.history["train_acc"].append(self.acc)
        self.history["train_auc"].append(self.auc)
        self.history["train_aupr"].append(self.aupr)
        self.history["train_f1"].append(self.f1)
        print(f"\ntrain loss:{self.loss} acc:{self.acc} auc:{self.auc} aupr:{self.aupr} f1:{self.f1}")

        self.val_acc /= self.val_idx
        self.val_auc /= self.val_idx
        self.val_aupr /= self.val_idx
        self.val_f1 /= self.val_idx
        self.val_loss /= self.val_idx
        self.history["val_loss"].append(self.val_loss)
        self.history["val_acc"].append(self.val_acc)
        self.history["val_auc"].append(self.val_auc)
        self.history["val_aupr"].append(self.val_aupr)
        self.history["val_f1"].append(self.val_f1)
        print(f"val loss:{self.val_loss} acc:{self.val_acc} auc:{self.val_auc} aupr:{self.val_aupr} f1:{self.val_f1}")

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.to('cuda')
        y = y.float().to('cuda')
        y_predict, celltypeEmbedding, embedding, attention_weight = self(x)
        y_predict = torch.squeeze(y_predict)

        classification_loss = self.criterion(y_predict, y)

        triplet_loss = compute_triplet_loss(embedding, y)

        alpha = torch.exp(self.alpha)
        beta = torch.exp(self.beta)

        loss = alpha * classification_loss + beta * triplet_loss + self.lambda_reg * (alpha ** 2 + beta ** 2)
        auc, acc, p, r, f1, aupr, fpr, tpr = score(y, y_predict, self.threshold, False)
        self.test_loss.append(loss)
        self.test_acc.append(acc)
        self.test_auc.append(auc)
        self.test_aupr.append(aupr)
        self.test_f1.append(f1)
        self.log('loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('acc', acc, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('auc', auc, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('aupr', aupr, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('f1', f1, on_step=True, on_epoch=False, prog_bar=True, logger=False)

        return loss

    def on_test_end(self):
        ave_loss = sum(self.test_loss) / len(self.test_loss)
        ave_acc = sum(self.test_acc) / len(self.test_acc)
        ave_auc = sum(self.test_auc) / len(self.test_auc)
        ave_aupr = sum(self.test_aupr) / len(self.test_aupr)
        ave_f1 = sum(self.test_f1) / len(self.test_f1)
        path = LOG_DIR + RESULT_DIR
        logs = dict()
        logs["dataset"] = "ALL"
        logs["k_fold"] = self.config.K_Fold
        logs["avg_acc"] = ave_acc
        logs["avg_auc"] = ave_auc
        logs["avg_aupr"] = ave_aupr
        logs_ = pd.DataFrame([logs])
        if self.config.K_Fold == 0:
            logs_.to_csv(path, mode='a', index=False)
        else:
            logs_.to_csv(path, mode='a', index=False, header=False)
        print(f"\n-----test-----\nloss: {ave_loss} acc: {ave_acc} auc: {ave_auc} aupr: {ave_aupr} f1: {ave_f1}")
        self.test_loss.clear()
        self.test_acc.clear()
        self.test_auc.clear()
        self.test_aupr.clear()
        self.test_f1.clear()
        del logs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    def embedding_layer(self, node_id, config):
        finalembedding = []

        if torch.is_tensor(node_id):
            tensor_node_id = node_id.to(self.device)
        else:
            tensor_node_id = torch.tensor(node_id).to(self.device)

        for i in range(39):
            idx = torch.tensor(config.idx[i], dtype=torch.long).to(self.device)
            mask = torch.isin(tensor_node_id, idx)
            exist_node_id = node_id[mask]

            if exist_node_id.numel() > 0:
                embeddings = self.get_cell_type_layer_embeddings(exist_node_id, i)
                cell_type_embedding = torch.zeros(len(node_id), self.config.d_model, requires_grad=True).to(self.device)
                cell_type_embedding[mask] = embeddings
            else:
                cell_type_embedding = torch.zeros(len(node_id), self.config.d_model, requires_grad=True).to(self.device)

            finalembedding.append(cell_type_embedding)

        _finalembedding = torch.stack(finalembedding)
        _finalembedding_transpose = torch.transpose(_finalembedding, dim0=0, dim1=1)
        _finalembedding_reshape = _finalembedding_transpose.reshape(_finalembedding_transpose.size(0), -1)
        return _finalembedding_reshape

    def get_cell_type_layer_embeddings(self, input_node_id, cell_type_num):
        sub_node_feature, sub_distance, sub_spatial, sub_node_neighbor = self.get_sub_info(input_node_id, cell_type_num)
        sub_node_feature = torch.nan_to_num(sub_node_feature, 0.0)

        node_embedding = []

        for g in range(self.config.n_graphs):
            centr_encoding = self.centrEncodingLayer[cell_type_num](sub_distance[:, g, :, :])
            out = self.get_node_feature(sub_node_feature[:, g, :, :], centr_encoding)
            # ablation without centr_encoding
            # out = sub_node_feature[:, g, :, :]
            out = out.float()
            out = self.Linear(out)
            out = self.Relu(out)

            spatial_matrix_in_subgraphs = sub_spatial[:, g, :, :, :]
            mask = self.create_padding_mask(spatial_matrix_in_subgraphs[:, 0, :, :])
            attention_mask = mask.unsqueeze(1)

            for n in range(self.config.n_layers):
                spatial_matrix_hop = spatial_matrix_in_subgraphs[:, 0, :, :]
                attention_mask_n = attention_mask
                out, _ = self.graphormer_layers[n](out, self.config.training, attention_mask_n, spatial_matrix_hop)

            node_embedding.append(out[:, 0, :])

        aggreated_out = torch.cat(node_embedding, dim=-1)
        finalembedding, _ = self.attentionLayer(aggreated_out)

        return finalembedding

    def create_padding_mask(self, nodes):
        return (nodes == -1).float()

    def get_node_feature(self, node_embedding, centr_encoding):
        node_feature = node_embedding
        node_feature *= torch.sqrt(torch.tensor(self.config.d_model, dtype=torch.float32))
        node_feature += centr_encoding

        return F.dropout(node_feature, p=self.config.dropout, training=self.config.training)

    def get_sub_info(self, node_id, cell_type_num):
        length = len(node_id)
        node_neighbors = torch.index_select(torch.tensor(self.config.node_neighbor[cell_type_num]).to(self.device), 0, node_id)
        node_neighbors = torch.squeeze(node_neighbors, dim=1)

        idx = self.config.idx[cell_type_num]
        idx_dict = {idx[i]: i for i in range(len(idx))}
        flat_node_neighbor = node_neighbors.view(-1)
        flat_node_neighbor_list = [i.item() for i in flat_node_neighbor]
        mapped_indices = torch.tensor([idx_dict.get(i, 0) for i in flat_node_neighbor_list], dtype=torch.long).to(self.device)
        cell_type_node_feature = torch.tensor(self.config.node_feature[cell_type_num]).to(self.device)
        node_feature = cell_type_node_feature[mapped_indices]

        if node_feature.numel() == 0:
            raise ValueError("node_feature is empty")

        node_feature = node_feature.reshape(length, self.config.n_graphs, self.config.n_neighbors, -1)

        cell_type_distance_matrix = torch.tensor(self.config.distance_matrix[cell_type_num]).to(self.device)
        distance = torch.index_select(cell_type_distance_matrix, 0, flat_node_neighbor)
        distance = distance.reshape(length, self.config.n_graphs, self.config.n_neighbors, -1)

        spatial = torch.index_select(torch.tensor(self.config.spatial_matrix[cell_type_num]).to(self.device), 0, node_id)
        spatial = torch.squeeze(spatial, dim=1)

        return node_feature, distance, spatial, node_neighbors
