import sklearn.metrics as m
import os

import torch
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
import lightning.pytorch as pl
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve, precision_score, \
    recall_score
from sklearn.metrics import roc_curve
from callbacks import KGCNMetric
from models.base_model import BaseModel
from utils import _Dataset, BalancedBatchSampler
from models.panCancerGenePredict import PanCancerGenePredict
import numpy as np


class TREE(BaseModel):
    def __init__(self, config):
        super(TREE, self).__init__(config)

    def build(self):
        model = PanCancerGenePredict(self.config).to('cuda')

        return model

    def fit(self, x_train, y_train, x_val, y_val):
        self.model.train()
        pl.seed_everything(42)
        self.callbacks = []
        optimizer = self.model.configure_optimizers()
        self.model.configure_optimizers()
        self.init_callbacks()
        _KGCNMetric = KGCNMetric(x_train, y_train, x_val, y_val, self.config.dataset)
        print('Logging Info - Start training...')
        train_ds = _Dataset(x_train, y_train)
        pos_indices = np.where(y_train == True)[0]
        neg_indices = np.where(y_train == False)[0]
        train_sampler = BalancedBatchSampler(pos_indices, neg_indices, self.config.batch_size, train_ds)
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, sampler=train_sampler, num_workers=8,
                                  pin_memory=True)

        val_ds = _Dataset(x_val, y_val)
        pos_indices = np.where(y_val == True)[0]
        neg_indices = np.where(y_val == False)[0]
        val_sampler = BalancedBatchSampler(pos_indices, neg_indices, self.config.batch_size, val_ds)
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, sampler=val_sampler, num_workers=8,
                                pin_memory=True)
        trainer = Trainer(
            accelerator='gpu',
            devices="auto",
            enable_checkpointing=False,
            num_sanity_val_steps=0,
            log_every_n_steps=49,
            max_epochs=self.config.n_epoch,
            callbacks=self.callbacks
        )

        trainer.fit(self.model, train_loader, val_loader)

        print('Logging Info - training end...')

    def get_variables(self):
        return self.model.trainable_weights

    def predict(self, x):
        self.model.eval()
        self.model.to('cuda')
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.to('cuda')
        output, celltypeEmbedding, _, attention_weight = self.model(x)
        output = torch.sigmoid(output)
        return output

    def test(self, x, y):
        self.model.eval()
        test_ds = _Dataset(x, y)
        pos_indices = np.where(y == True)[0]
        neg_indices = np.where(y == False)[0]
        test_sampler = BalancedBatchSampler(pos_indices, neg_indices, self.config.batch_size, test_ds)
        test_loader = DataLoader(test_ds, batch_size=self.config.batch_size, sampler=test_sampler, num_workers=8,
                                 pin_memory=True,
                                 drop_last=True
                                 )

        trainer = Trainer(
            accelerator='gpu',
            devices="auto",
            max_epochs=self.config.n_epoch
        )
        trainer.test(self.model, test_loader)

    def score(self, x, y, threshold=0.6):  ##要重新实现
        x = torch.tensor(x).to('cuda')
        y_pred = self.predict(x).cpu()
        y_true = y.float().cpu()
        auc = roc_auc_score(y_true=y_true, y_score=y_pred)
        precision, recall, _thresholds = precision_recall_curve(y_true=y_true, probas_pred=y_pred)
        aupr = m.auc(recall, precision)

        fpr, tpr, thr = roc_curve(y_true=y_true, y_score=y_pred)
        y_pred = [1 if prob >= threshold else 0 for prob in y_pred]
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        p = precision_score(y_true=y_true, y_pred=y_pred)
        r = recall_score(y_true=y_true, y_pred=y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred)

        return auc, acc, p, r, f1, aupr, fpr.tolist(), tpr.tolist()

    def load_train_model(self):
        print('Logging Info - Loading model training checkpoint: transformer_Node2vec%s.pkl' % self.config.exp_name)
        name = '{}.pkl'.format(self.config.exp_name)
        model_train_save_path = f"{self.config.checkpoint_dir}/transformer_Node2vec{name}"
        self.model = PanCancerGenePredict.load_state_dict(torch.load(model_train_save_path), strict=False)

        print('Logging Info - Model loaded')

    def load_best_model(self):
        print('Logging Info - Loading model checkpoint: %s.pkl' % self.config.exp_name)
        del self.model
        self.model = PanCancerGenePredict(self.config).to('cuda')
        self.model.load_state_dict(
            state_dict=torch.load(os.path.join(self.config.checkpoint_dir, f'{self.config.exp_name}.pkl')),
            strict=False)
        print('Logging Info - Model loaded')

    def get_attn_weights(self, x):
        self.model.eval()
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x.to('cuda')
        output, celltypeEmbedding, _, attention_weight = self.model(x)
        return attention_weight

    def get_celltype_embeddings(self, x):
        self.model.eval()
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x.to('cuda')
        output, celltypeEmbedding, embeddings, attention_weight = self.model(x)
        celltypeEmbedding = celltypeEmbedding.view(len(x), -1, self.config.d_model)
        return celltypeEmbedding

    def get_embeddings(self, x):
        self.model.eval()
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x.to('cuda')
        output, celltypeEmbedding, embeddings, attention_weight = self.model(x)
        return embeddings
