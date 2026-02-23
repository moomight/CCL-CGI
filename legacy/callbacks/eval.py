from collections import defaultdict

import numpy as np
from lightning.pytorch.callbacks import Callback
import lightning.pytorch as pl
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score, precision_recall_curve, recall_score
import sklearn.metrics as m
from utils import write_log

#添加指标：ACC, AUPR, AUC-ROC, F1 +std

class KGCNMetric(Callback):
    def __init__(self, x_train, y_train, x_val, y_val, dataset):
        self.x_train = x_train
        self.y_val = y_val
        self.y_train = y_train
        self.x_val = x_val
        self.dataset=dataset
        self.threshold=0.6

        super(KGCNMetric, self).__init__()

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        epoch = pl_module.current_epoch + 1
        fold = pl_module.config.K_Fold
        history = pl_module.history
        auc = history["val_auc"][-1]
        aupr = history["val_aupr"][-1]
        acc = history["val_acc"][-1]
        f1 = history["val_f1"][-1]
        logs = dict()
        logs["fold"] = fold
        logs['val_aupr'] = float(aupr)
        logs['val_auc'] = float(auc)
        logs['val_acc'] = float(acc)
        logs['val_f1'] = float(f1)
        logs['epoch_count'] = epoch
        # print(f'Logging Info - epoch: {epoch}, val_auc: {auc}, val_aupr: {aupr}, val_acc: {acc}, val_f1: {f1}')
        write_log('log/train_history_node2vec_tree.txt', logs, mode='a')

    @staticmethod
    def get_user_record(data, is_train):
        user_history_dict = defaultdict(set)
        for interaction in data:
            user = interaction[0]
            item = interaction[1]
            label = interaction[2]
            if is_train or label == 1:
                user_history_dict[user].add(item)
        return user_history_dict

