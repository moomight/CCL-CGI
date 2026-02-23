from layers import AttentionFusion, CentralityEncoding, GraphormerBlock, AttentionAggregate

from utils import score
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config import LOG_DIR, RESULT_DIR
import pandas as pd
import os
from losses import compute_triplet_loss_for_hardest_case, compute_triplet_loss_for_all, compute_triplet_loss_old

class PanCancerGenePredict(pl.LightningModule):
    def __init__(self, config):
        super(PanCancerGenePredict, self).__init__()
        self.config = config
        requested_device = str(getattr(self.config, 'device', 'cuda'))
        use_cuda = requested_device.startswith('cuda') and torch.cuda.is_available()
        self.runtime_device = torch.device('cuda' if use_cuda else 'cpu')
        self.temp = 0.3
        self.threshold = 0.5
        self.auto_threshold_by_val_f1 = bool(getattr(self.config, 'auto_threshold_by_val_f1', False))
        seed = 42
        torch.manual_seed(seed)
        if use_cuda:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # translated,translatedLLMtranslated
        # translated
        # self.fc = nn.Sequential(
        #     nn.Linear(39 * self.config.d_model, 256),
        #     nn.ReLU(),
        #     nn.Dropout(self.config.dropout),
        #     nn.Linear(256, 1)
        # )
        self.fc = nn.Sequential(
            nn.Linear(self.config.d_model, 256),
            # nn.BatchNorm1d(256), # translated
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

        positive_fraction = torch.tensor(self.config.positive_fraction, device=self.runtime_device)
        bias_value = -torch.log((1 - positive_fraction) / positive_fraction)
        self.fc[-1].bias.data.fill_(bias_value)

        # translatedGraphormertranslated
        self.graphormer_layers = nn.ModuleList(
            [GraphormerBlock(self.config.d_model, self.config.num_heads, self.config.dff, self.config.dropout,
                             self.config.d_sp_enc, self.config.sp_enc_activation, self.config.n_neighbors)
             for _ in range(self.config.n_layers)]
        )
        self.centrEncodingLayer = nn.ModuleList([
            CentralityEncoding(self.config.max_degree[cell_type_num],
                               self.config.d_model)
            for cell_type_num in range(self.config.n_cell_types)]
        )
        self.attentionLayer = AttentionFusion(d_model=self.config.d_model, n_channels=self.config.n_graphs).to(self.runtime_device)
        self.aggregateLayer = AttentionAggregate(d_model=self.config.d_model, num_heads=self.config.num_heads).to(self.runtime_device)

        # No feature projection needed - d_model is set dynamically based on feature dimensions
        # config.d_model is already set to match feature dimensions (8 or 64) in main.py
        self.Linear = nn.Linear(self.config.d_model, self.config.d_model).to(self.runtime_device)
        self.Relu = nn.ReLU()

        # translated,translatedBCELoss,translated
        # pos_weight = (1 - loss_mul) / loss_mul
        # loss_mul translated,translated 0.2 translated20%translated,80%translated
        # pos_weight = 0.8 / 0.2 = 4.0,translatedlosstranslated4translated
        pos_weight_value = (1.0 - self.config.loss_mul) / self.config.loss_mul
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=self.runtime_device))
        print(f"BCEWithLogitsLoss pos_weight set to {pos_weight_value:.4f} (based on loss_mul={self.config.loss_mul})")

        # ===== GPUtranslated: translatedtensors,translated =====
        # translated idx translated GPU tensors translated
        self.cached_idx_tensors = [
            torch.tensor(config.idx[i], dtype=torch.long, device=self.runtime_device)
            for i in range(self.config.n_cell_types)
        ]

        # translated node_feature, distance_matrix, spatial_matrix, node_neighbor translated GPU tensors
        # translated get_sub_info translated
        self.cached_node_features = [
            torch.tensor(config.node_feature[i], dtype=torch.float32, device=self.runtime_device)
            for i in range(self.config.n_cell_types)
        ]
        self.cached_distance_matrices = [
            torch.tensor(config.distance_matrix[i], dtype=torch.float32, device=self.runtime_device)
            for i in range(self.config.n_cell_types)
        ]
        self.cached_spatial_matrices = [
            torch.tensor(config.spatial_matrix[i], dtype=torch.float32, device=self.runtime_device)
            for i in range(self.config.n_cell_types)
        ]
        self.cached_node_neighbors = [
            torch.tensor(config.node_neighbor[i], dtype=torch.long, device=self.runtime_device)
            for i in range(self.config.n_cell_types)
        ]

        # Precompute per-cell-type membership masks to avoid torch.isin on CUDA.
        # Shape: [n_cell_types, n_global_nodes], bool
        global_node_count = int(self.cached_node_neighbors[0].shape[0]) if self.cached_node_neighbors else 0
        self.cached_membership_masks = []
        for i in range(self.config.n_cell_types):
            membership = torch.zeros(global_node_count, dtype=torch.bool, device=self.runtime_device)
            idx_tensor = self.cached_idx_tensors[i]
            if idx_tensor.numel() > 0:
                safe_idx = torch.clamp(idx_tensor, 0, max(global_node_count - 1, 0))
                membership[safe_idx] = True
            self.cached_membership_masks.append(membership)

        # translated idx_dict (translated) - translated tensor translated
        self.cached_idx_mappings = []
        for i in range(self.config.n_cell_types):
            idx = config.idx[i]
            # translatedtensor (translatedindextranslated)
            max_idx = max(idx) if len(idx) > 0 else 0
            mapping = torch.zeros(max_idx + 1, dtype=torch.long, device=self.runtime_device)
            for local_i, global_idx in enumerate(idx):
                mapping[global_idx] = local_i
            self.cached_idx_mappings.append(mapping)

        # translated sqrt(d_model) translated
        self.sqrt_d_model = torch.sqrt(torch.tensor(self.config.d_model, dtype=torch.float32, device=self.runtime_device))

        self.alpha = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.beta = nn.Parameter(torch.tensor(0.1, requires_grad=True))
        self.lambda_reg = self.config.lambda_reg

        self.loss = 0
        self.acc = 0
        self.auc = 0
        self.aupr = 0
        self.f1 = 0
        self.mcc = 0
        self.batch_idx = 0
        self.val_acc = 0
        self.val_auc = 0
        self.val_aupr = 0
        self.val_f1 = 0
        self.val_mcc = 0
        self.val_loss = 0
        self.val_idx = 0
        self.test_loss = []
        self.test_acc = []
        self.test_auc = []
        self.test_aupr = []
        self.test_f1 = []
        self.test_mcc = []
        self.history = {
            "train_loss": [], "train_acc": [], "train_auc": [], "train_aupr": [], "train_f1": [], "train_mcc": [],
            "val_loss": [], "val_acc": [], "val_auc": [], "val_aupr": [], "val_f1": [], "val_mcc": []
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
        y_predict, y = self._flatten_logits_and_labels(y_predict, y)

        # translated criterion (translated config.loss_mul)
        # translatedbatchtranslated pos_weight
        classification_loss = self.criterion(y_predict, y)

        triplet_loss = compute_triplet_loss_for_all(embedding, y)
        # triplet_loss = compute_triplet_loss_old(embedding, y)

        alpha = torch.exp(self.alpha)
        beta = torch.exp(self.beta)

        loss = alpha * classification_loss + beta * triplet_loss + self.lambda_reg * (alpha ** 2 + beta ** 2)

        cpu_loss = loss.item()
        self.loss += cpu_loss
        auc, acc, p, r, f1, aupr, fpr, tpr, mcc = score(y, y_predict, self.threshold, True)
        self.acc += acc
        self.auc += auc
        self.aupr += aupr
        self.f1 += f1
        self.mcc += mcc
        self.log('loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('classification_loss', classification_loss, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('triplet_loss', triplet_loss, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('acc', acc, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('auc', auc, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('aupr', aupr, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('f1', f1, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('mcc', mcc, on_step=True, on_epoch=False, prog_bar=True, logger=False)

        return loss

    def _flatten_logits_and_labels(self, logits, targets):
        """Ensure logits/targets share shape regardless of dataset label format."""
        logits = logits.view(logits.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        if logits.size(-1) == 1 and targets.size(-1) == 1:
            logits = logits.view(-1)
            targets = targets.view(-1)

        return logits, targets

    def on_train_epoch_start(self):
        self.loss = 0
        self.acc = 0
        self.auc = 0
        self.aupr = 0
        self.f1 = 0
        self.mcc = 0
        self.val_acc = 0
        self.val_auc = 0
        self.val_aupr = 0
        self.val_f1 = 0
        self.val_mcc = 0
        self.batch_idx = 0
        self.val_idx = 0
        self.val_loss = 0

    def validation_step(self, batch, batch_idx):
        self.val_idx += 1
        x, y = batch
        x = x.to(self.device)
        y = y.float().to(self.device)
        y_predict, celltypeEmbedding, embedding, attention_weight = self(x)
        y_predict, y = self._flatten_logits_and_labels(y_predict, y)

        classification_loss = self.criterion(y_predict, y)

        triplet_loss = compute_triplet_loss_for_all(embedding, y)
        # triplet_loss = compute_triplet_loss_old(embedding, y)

        alpha = torch.exp(self.alpha)
        beta = torch.exp(self.beta)

        loss = alpha * classification_loss + beta * triplet_loss + self.lambda_reg * (alpha ** 2 + beta ** 2)
        val_auc, acc, p, r, f1, aupr, fpr, tpr, mcc = score(y, y_predict, self.threshold, False)
        self.val_acc += acc
        self.val_auc += val_auc
        self.val_aupr += aupr
        self.val_f1 += f1
        self.val_mcc += mcc
        self.val_loss += loss.item()
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("val_auc", val_auc, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("val_aupr", aupr, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("val_f1", f1, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("val_mcc", mcc, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        return loss

    def on_validation_end(self):
        self.loss /= self.batch_idx
        self.acc /= self.batch_idx
        self.auc /= self.batch_idx
        self.aupr /= self.batch_idx
        self.f1 /= self.batch_idx
        self.mcc /= self.batch_idx
        self.history["train_loss"].append(self.loss)
        self.history["train_acc"].append(self.acc)
        self.history["train_auc"].append(self.auc)
        self.history["train_aupr"].append(self.aupr)
        self.history["train_f1"].append(self.f1)
        self.history["train_mcc"].append(self.mcc)
        print(f"\ntrain loss:{self.loss} acc:{self.acc} auc:{self.auc} aupr:{self.aupr} f1:{self.f1} mcc:{self.mcc}")

        self.val_acc /= self.val_idx
        self.val_auc /= self.val_idx
        self.val_aupr /= self.val_idx
        self.val_f1 /= self.val_idx
        self.val_mcc /= self.val_idx
        self.val_loss /= self.val_idx
        self.history["val_loss"].append(self.val_loss)
        self.history["val_acc"].append(self.val_acc)
        self.history["val_auc"].append(self.val_auc)
        self.history["val_aupr"].append(self.val_aupr)
        self.history["val_f1"].append(self.val_f1)
        self.history["val_mcc"].append(self.val_mcc)
        print(f"val loss:{self.val_loss} acc:{self.val_acc} auc:{self.val_auc} aupr:{self.val_aupr} f1:{self.val_f1} mcc:{self.val_mcc}")

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.float().to(self.device)
        y_predict, celltypeEmbedding, embedding, attention_weight = self(x)
        y_predict, y = self._flatten_logits_and_labels(y_predict, y)

        classification_loss = self.criterion(y_predict, y)

        triplet_loss = compute_triplet_loss_for_all(embedding, y)
        # triplet_loss = compute_triplet_loss_old(embedding, y)

        alpha = torch.exp(self.alpha)
        beta = torch.exp(self.beta)

        loss = alpha * classification_loss + beta * triplet_loss + self.lambda_reg * (alpha ** 2 + beta ** 2)
        y_score = torch.sigmoid(y_predict) if self.auto_threshold_by_val_f1 else y_predict
        auc, acc, p, r, f1, aupr, fpr, tpr, mcc = score(y, y_score, self.threshold, False)

        # 🚀 GPUtranslated: translated GPU tensor translated,translated CPU-GPU translated
        # translated numpy (translated on_test_end translated)
        if not hasattr(self, 'test_predictions'):
            self.test_predictions = []
            self.test_labels = []

        # Convert logits to probabilities (translated GPU translated)
        y_proba = torch.sigmoid(y_predict)
        self.test_predictions.append(y_proba.detach())  # translated GPU tensor
        self.test_labels.append(y.detach())  # translated GPU tensor

        self.test_loss.append(loss)
        self.test_acc.append(acc)
        self.test_auc.append(auc)
        self.test_aupr.append(aupr)
        self.test_f1.append(f1)
        self.test_mcc.append(mcc)
        self.log('loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('acc', acc, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('auc', auc, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('aupr', aupr, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('f1', f1, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('mcc', mcc, on_step=True, on_epoch=False, prog_bar=True, logger=False)

        return loss

    def on_test_end(self):
        ave_loss = sum(self.test_loss) / len(self.test_loss)
        ave_acc = sum(self.test_acc) / len(self.test_acc)
        ave_auc = sum(self.test_auc) / len(self.test_auc)
        ave_aupr = sum(self.test_aupr) / len(self.test_aupr)
        ave_f1 = sum(self.test_f1) / len(self.test_f1)
        ave_mcc = sum(self.test_mcc) / len(self.test_mcc)

        # Store final metrics for return
        self.test_auc_final = ave_auc
        self.test_aupr_final = ave_aupr
        self.test_acc_final = ave_acc
        self.test_mcc_final = ave_mcc

        # Calculate calibration metrics if predictions are available
        if hasattr(self, 'test_predictions') and len(self.test_predictions) > 0:
            import numpy as np
            from utils.statistical_tests import calculate_calibration_metrics, find_optimal_threshold

            # 🚀 GPUtranslated: translated CPU (translated)
            y_true = torch.cat(self.test_labels, dim=0).cpu().numpy()
            y_pred_proba = torch.cat(self.test_predictions, dim=0).cpu().numpy()

            # 🔬 translated (DeLong test, Bootstrap CI)
            self.test_y_true_raw = y_true
            self.test_y_pred_proba_raw = y_pred_proba

            # Calculate calibration metrics
            calibration = calculate_calibration_metrics(y_true, y_pred_proba, n_bins=10)
            self.test_brier_final = calibration['brier_score']
            self.test_ece_final = calibration['ece']

            # Find optimal threshold (this would normally be done on validation set)
            # For now, we report it on test set for reference
            opt_threshold, opt_f1, opt_precision, opt_recall = find_optimal_threshold(
                y_true, y_pred_proba, metric='f1'
            )
            self.test_optimal_threshold = opt_threshold
            self.test_optimal_precision = opt_precision
            self.test_optimal_recall = opt_recall
            self.test_optimal_f1 = opt_f1

            print(f"\nCalibration Metrics:")
            print(f"  Brier Score: {self.test_brier_final:.4f}")
            print(f"  Expected Calibration Error (ECE): {self.test_ece_final:.4f}")
            print(f"\nOptimal Operating Point (F1-optimized):")
            print(f"  Threshold: {self.test_optimal_threshold:.3f}")
            print(f"  Precision: {self.test_optimal_precision:.4f}")
            print(f"  Recall: {self.test_optimal_recall:.4f}")
            print(f"  F1-Score: {self.test_optimal_f1:.4f}")
        else:
            self.test_brier_final = None
            self.test_ece_final = None
            self.test_optimal_threshold = 0.5
            self.test_optimal_precision = None
            self.test_optimal_recall = None
            self.test_optimal_f1 = None

        path = LOG_DIR + RESULT_DIR
        path = path.replace(".csv", f"_{self.config.model_name}.csv")
        logs = dict()
        logs["dataset"] = "ALL"
        logs["k_fold"] = self.config.K_Fold
        logs["avg_acc"] = ave_acc
        logs["avg_auc"] = ave_auc
        logs["avg_aupr"] = ave_aupr
        logs["avg_mcc"] = ave_mcc
        logs_ = pd.DataFrame([logs])

        write_header = not os.path.exists(path)
        logs_.to_csv(path, mode='a', index=False, header=write_header)
        print(f"\n-----test-----\nloss: {ave_loss} acc: {ave_acc} auc: {ave_auc} aupr: {ave_aupr} f1: {ave_f1} mcc: {ave_mcc}")
        self.test_loss.clear()
        self.test_acc.clear()
        self.test_auc.clear()
        self.test_aupr.clear()
        self.test_f1.clear()
        self.test_mcc.clear()
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
        """
        GPUtranslated: translated,translated for translated tensor translated
        """
        # translated node_id translated
        if not torch.is_tensor(node_id):
            node_id = torch.tensor(node_id, dtype=torch.long, device=self.device)
        elif node_id.device != self.device:
            node_id = node_id.to(self.device)

        batch_size = len(node_id)

        # translated embedding tensor (translated)
        # Shape: (n_cell_types, batch_size, d_model)
        finalembedding = torch.zeros(self.config.n_cell_types, batch_size, self.config.d_model,
                                     dtype=torch.float32, device=self.device)

        # translated cell types
        for i in range(self.config.n_cell_types):
            membership = self.cached_membership_masks[i]
            max_global = int(membership.shape[0] - 1)
            if max_global < 0:
                continue

            node_id_safe = torch.clamp(node_id, 0, max_global)
            mask = torch.index_select(membership, 0, node_id_safe)
            selected_pos = torch.nonzero(mask, as_tuple=False).squeeze(-1)

            if selected_pos.numel() == 0:
                continue

            exist_node_id = torch.index_select(node_id_safe, 0, selected_pos)
            embeddings = self.get_cell_type_layer_embeddings(exist_node_id, i)

            if embeddings.shape[0] != selected_pos.numel():
                keep = min(int(embeddings.shape[0]), int(selected_pos.numel()))
                if keep == 0:
                    continue
                embeddings = embeddings[:keep]
                selected_pos = selected_pos[:keep]

            finalembedding[i].index_copy_(0, selected_pos, embeddings)

        # Reshape: (n_cell_types, batch, d_model) -> (batch, n_cell_types, d_model) -> (batch, n_cell_types*d_model)
        finalembedding = finalembedding.transpose(0, 1).reshape(batch_size, -1)
        return finalembedding

    def get_cell_type_layer_embeddings(self, input_node_id, cell_type_num):
        """
        GPUtranslated: translated
        """
        sub_node_feature, sub_distance, sub_spatial, sub_node_neighbor = self.get_sub_info(input_node_id, cell_type_num)
        sub_node_feature = torch.nan_to_num(sub_node_feature, 0.0)

        batch_size = sub_node_feature.size(0)
        n_graphs = self.config.n_graphs

        # translated node_embedding (translated list.append)
        # Shape: (batch, n_graphs, d_model)
        node_embedding = torch.zeros(batch_size, n_graphs, self.config.d_model,
                                     dtype=torch.float32, device=self.device)

        # translated graph (translated Graphormer layers translated)
        # translated
        for g in range(n_graphs):
            # translated centrality encoding
            centr_encoding = self.centrEncodingLayer[cell_type_num](sub_distance[:, g, :, :])
            out = self.get_node_feature(sub_node_feature[:, g, :, :], centr_encoding)

            out = out.float()
            out = self.Linear(out)
            out = self.Relu(out)

            spatial_matrix_in_subgraphs = sub_spatial[:, g, :, :, :]
            mask = self.create_padding_mask(spatial_matrix_in_subgraphs[:, 0, :, :])
            attention_mask = mask.unsqueeze(1)

            # Graphormer layers - translated
            for n in range(self.config.n_layers):
                spatial_matrix_hop = spatial_matrix_in_subgraphs[:, 0, :, :]
                out, _ = self.graphormer_layers[n](out, self.config.training, attention_mask, spatial_matrix_hop)

            # translated token translated embedding
            node_embedding[:, g, :] = out[:, 0, :]

        # Reshape translated
        # (batch, n_graphs, d_model) -> (batch, n_graphs * d_model)
        aggreated_out = node_embedding.reshape(batch_size, -1)
        finalembedding, _ = self.attentionLayer(aggreated_out)

        return finalembedding

    def create_padding_mask(self, nodes):
        return (nodes == -1).float()

    def get_node_feature(self, node_embedding, centr_encoding):
        """
        GPUtranslated: translated sqrt(d_model),translated tensor
        """
        node_feature = node_embedding
        node_feature = node_feature * self.sqrt_d_model  # translated
        node_feature = node_feature + centr_encoding

        return F.dropout(node_feature, p=self.config.dropout, training=self.config.training)

    def get_sub_info(self, node_id, cell_type_num):
        """
        🚀 GPUtranslated: translated tensors,translated:
        1. translated torch.tensor() translated
        2. CPU -> GPU translated
        3. Python list translated (.item(), list comprehension)
        4. translated
        """
        length = len(node_id)

        num_global_nodes = int(self.cached_node_neighbors[cell_type_num].shape[0])
        node_id_safe = torch.clamp(node_id, 0, max(num_global_nodes - 1, 0))

        # translated node_neighbor tensor (translated GPU translated)
        node_neighbors = torch.index_select(self.cached_node_neighbors[cell_type_num], 0, node_id_safe)
        node_neighbors = node_neighbors.squeeze(dim=1)

        # Node2vec walks may contain -1 for isolated nodes / missing walks.
        # Replace invalid indices with self node_id to keep indexing safe and deterministic.
        if (node_neighbors < 0).any():
            self_id = node_id_safe.view(-1, 1, 1).expand_as(node_neighbors)
            node_neighbors = torch.where(node_neighbors < 0, self_id, node_neighbors)

        # translated idx_mapping tensor translated
        # translated Python translated list comprehension
        flat_node_neighbor = node_neighbors.view(-1)

        # Safety clamp for downstream index_select (distance_matrix is sized by full graph nodes)
        max_valid = int(self.cached_distance_matrices[cell_type_num].shape[0] - 1)
        flat_node_neighbor_safe = torch.clamp(flat_node_neighbor, 0, max_valid)

        # GPU translated (translated Python dict translated)
        mapping = self.cached_idx_mappings[cell_type_num]
        # translated: translated 0
        flat_node_neighbor_clamped = torch.clamp(flat_node_neighbor_safe, 0, len(mapping) - 1)
        mapped_indices = mapping[flat_node_neighbor_clamped]

        feature_n = int(self.cached_node_features[cell_type_num].shape[0])
        mapped_indices = torch.clamp(mapped_indices, 0, max(feature_n - 1, 0))

        # translated node_feature tensor
        node_feature = self.cached_node_features[cell_type_num][mapped_indices]

        if node_feature.numel() == 0:
            raise ValueError("node_feature is empty")

        node_feature = node_feature.reshape(length, self.config.n_graphs, self.config.n_neighbors, -1)

        # translated distance_matrix tensor
        distance = torch.index_select(self.cached_distance_matrices[cell_type_num], 0, flat_node_neighbor_safe)
        distance = distance.reshape(length, self.config.n_graphs, self.config.n_neighbors, -1)

        # translated spatial_matrix tensor
        spatial = torch.index_select(self.cached_spatial_matrices[cell_type_num], 0, node_id_safe)
        spatial = spatial.squeeze(dim=1)

        return node_feature, distance, spatial, node_neighbors
