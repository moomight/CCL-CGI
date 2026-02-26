import sklearn.metrics as m
import os

import torch
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
import lightning.pytorch as pl
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve, precision_score, \
    recall_score, matthews_corrcoef
from sklearn.metrics import roc_curve
from callbacks import KGCNMetric
from models.base_model import BaseModel
from utils import _Dataset, BalancedBatchSampler
from utils.statistical_tests import find_optimal_threshold
from models.panCancerGenePredict import PanCancerGenePredict
import numpy as np

class TREE(BaseModel):
    def __init__(self, config):
        super(TREE, self).__init__(config)

    def _runtime_device(self):
        configured = getattr(self.config, "device", None)
        if configured is not None and str(configured).startswith("cuda") and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def build(self):
        model = PanCancerGenePredict(self.config).to(self._runtime_device())

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

        train_num_workers = max(0, int(os.environ.get('CCL_TRAIN_NUM_WORKERS', 4)))
        val_num_workers = max(0, int(os.environ.get('CCL_VAL_NUM_WORKERS', 16)))
        train_prefetch_factor = max(1, int(os.environ.get('CCL_TRAIN_PREFETCH_FACTOR', 2)))
        val_prefetch_factor = max(1, int(os.environ.get('CCL_VAL_PREFETCH_FACTOR', 4)))
        val_persistent_workers = (val_num_workers > 0) and (os.environ.get('CCL_VAL_PERSISTENT_WORKERS', '1') == '1')

        train_ds = _Dataset(x_train, y_train)
        pos_indices = np.where(y_train == True)[0]
        neg_indices = np.where(y_train == False)[0]
        train_sampler = BalancedBatchSampler(pos_indices, neg_indices, self.config.batch_size, train_ds)
        train_loader_kwargs = dict(
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            num_workers=train_num_workers,
            pin_memory=(self._runtime_device() == 'cuda'),
            persistent_workers=False,
        )
        if train_num_workers > 0:
            train_loader_kwargs['prefetch_factor'] = train_prefetch_factor
        train_loader = DataLoader(train_ds, **train_loader_kwargs)

        val_ds = _Dataset(x_val, y_val)
        pos_indices = np.where(y_val == True)[0]
        neg_indices = np.where(y_val == False)[0]
        val_sampler = BalancedBatchSampler(pos_indices, neg_indices, self.config.batch_size, val_ds)
        val_loader_kwargs = dict(
            batch_size=self.config.batch_size,
            sampler=val_sampler,
            num_workers=val_num_workers,
            pin_memory=(self._runtime_device() == 'cuda'),
            persistent_workers=val_persistent_workers,
        )
        if val_num_workers > 0:
            val_loader_kwargs['prefetch_factor'] = val_prefetch_factor
        val_loader = DataLoader(val_ds, **val_loader_kwargs)
        trainer = Trainer(
            accelerator='gpu' if self._runtime_device() == 'cuda' else 'cpu',
            devices=1,
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
        runtime_device = self._runtime_device()
        self.model.to(runtime_device)
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.to(runtime_device)
        output, celltypeEmbedding, _, attention_weight = self.model(x)
        output = torch.sigmoid(output)
        return output

    def test(self, x, y):
        self.model.eval()
        test_num_workers = max(0, int(os.environ.get('CCL_TEST_NUM_WORKERS', 16)))
        test_prefetch_factor = max(1, int(os.environ.get('CCL_TEST_PREFETCH_FACTOR', 4)))
        test_persistent_workers = (test_num_workers > 0) and (os.environ.get('CCL_TEST_PERSISTENT_WORKERS', '1') == '1')

        test_ds = _Dataset(x, y)
        pos_indices = np.where(y == True)[0]
        neg_indices = np.where(y == False)[0]
        test_sampler = BalancedBatchSampler(pos_indices, neg_indices, self.config.batch_size, test_ds)
        test_loader_kwargs = dict(
            batch_size=self.config.batch_size,
            sampler=test_sampler,
            num_workers=test_num_workers,
            pin_memory=(self._runtime_device() == 'cuda'),
            persistent_workers=test_persistent_workers,
            drop_last=True,
        )
        if test_num_workers > 0:
            test_loader_kwargs['prefetch_factor'] = test_prefetch_factor
        test_loader = DataLoader(test_ds, **test_loader_kwargs)

        trainer = Trainer(
            accelerator='gpu' if self._runtime_device() == 'cuda' else 'cpu',
            devices=1,
            max_epochs=self.config.n_epoch
        )
        trainer.test(self.model, test_loader)

        # Return test metrics from the model with raw predictions
        # The metrics are computed in on_test_end() and stored
        return {
            'metrics': {
                'auc': self.model.test_auc_final if hasattr(self.model, 'test_auc_final') else None,
                'aupr': self.model.test_aupr_final if hasattr(self.model, 'test_aupr_final') else None,
                'acc': self.model.test_acc_final if hasattr(self.model, 'test_acc_final') else None,
                'mcc': self.model.test_mcc_final if hasattr(self.model, 'test_mcc_final') else None,
                'brier': self.model.test_brier_final if hasattr(self.model, 'test_brier_final') else None,
                'ece': self.model.test_ece_final if hasattr(self.model, 'test_ece_final') else None,
                'optimal_threshold': self.model.test_optimal_threshold if hasattr(self.model, 'test_optimal_threshold') else getattr(self.config, 'threshold', 0.5),
                'optimal_precision': self.model.test_optimal_precision if hasattr(self.model, 'test_optimal_precision') else None,
                'optimal_recall': self.model.test_optimal_recall if hasattr(self.model, 'test_optimal_recall') else None,
                'optimal_f1': self.model.test_optimal_f1 if hasattr(self.model, 'test_optimal_f1') else None
            },
            'y_true': self.model.test_y_true_raw if hasattr(self.model, 'test_y_true_raw') else None,
            'y_pred_proba': self.model.test_y_pred_proba_raw if hasattr(self.model, 'test_y_pred_proba_raw') else None
        }

    def score(self, x, y, threshold=0.5):
        x = torch.tensor(x).to(self._runtime_device())
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
        mcc = matthews_corrcoef(y_true=y_true, y_pred=y_pred)

        return auc, acc, p, r, f1, aupr, fpr.tolist(), tpr.tolist(), mcc

    def load_train_model(self):
        print('Logging Info - Loading model training checkpoint: transformer_Node2vec%s.pkl' % self.config.exp_name)
        name = '{}.pkl'.format(self.config.exp_name)
        model_train_save_path = f"{self.config.checkpoint_dir}/transformer_Node2vec{name}"
        self.model = PanCancerGenePredict.load_state_dict(torch.load(model_train_save_path), strict=False)

        print('Logging Info - Model loaded')

    def load_best_model(self, checkpoint_path=None):
        if checkpoint_path is not None:
            resolved_checkpoint_path = checkpoint_path
            print(f'Logging Info - Loading model checkpoint from path: {resolved_checkpoint_path}')
        else:
            resolved_checkpoint_path = os.path.join(self.config.checkpoint_dir, f'{self.config.exp_name}.pkl')
            print('Logging Info - Loading model checkpoint: %s.pkl' % self.config.exp_name)
        del self.model
        runtime_device = self._runtime_device()
        self.model = PanCancerGenePredict(self.config).to(runtime_device)
        self.model.load_state_dict(
            state_dict=torch.load(
                resolved_checkpoint_path,
                map_location=runtime_device,
            ),
            strict=False)
        print('Logging Info - Model loaded')

    def tune_threshold_by_validation(self, x_val, y_val):
        runtime_device = self._runtime_device()
        self.model.eval()
        self.model.to(runtime_device)

        val_dataset = _Dataset(np.array(x_val), np.array(y_val))
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(runtime_device == 'cuda'),
        )

        y_true_batches = []
        y_pred_proba_batches = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(runtime_device)
                y_batch = y_batch.float().to(runtime_device)
                logits, _, _, _ = self.model(x_batch)
                logits = logits.view(logits.size(0), -1)
                y_batch = y_batch.view(y_batch.size(0), -1)
                if logits.size(-1) == 1 and y_batch.size(-1) == 1:
                    logits = logits.view(-1)
                    y_batch = y_batch.view(-1)

                y_true_batches.append(y_batch.detach().cpu())
                y_pred_proba_batches.append(torch.sigmoid(logits).detach().cpu())

        if len(y_true_batches) == 0:
            self.model.threshold = getattr(self.config, 'threshold', 0.5)
            print('Logging Info - Validation threshold tuning skipped: empty validation data, using threshold=0.5')
            return

        y_true = torch.cat(y_true_batches, dim=0).numpy()
        y_pred_proba = torch.cat(y_pred_proba_batches, dim=0).numpy()

        if len(np.unique(y_true)) < 2:
            self.model.threshold = getattr(self.config, 'threshold', 0.5)
            print('Logging Info - Validation threshold tuning skipped: single-class validation labels, using threshold=0.5')
            return

        best_threshold, best_f1, best_precision, best_recall = find_optimal_threshold(
            y_true, y_pred_proba, metric='f1'
        )
        self.model.threshold = float(best_threshold)
        print(
            'Logging Info - Auto threshold selected from validation: '
            f'threshold={self.model.threshold:.3f}, f1={best_f1:.4f}, '
            f'precision={best_precision:.4f}, recall={best_recall:.4f}'
        )

    def get_attn_weights(self, x):
        self.model.eval()
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.to(self._runtime_device())
        output, celltypeEmbedding, _, attention_weight = self.model(x)
        return attention_weight

    def get_celltype_embeddings(self, x):
        self.model.eval()
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.to(self._runtime_device())
        output, celltypeEmbedding, embeddings, attention_weight = self.model(x)
        celltypeEmbedding = celltypeEmbedding.view(len(x), -1, self.config.d_model)
        return celltypeEmbedding

    def get_embeddings(self, x):
        self.model.eval()
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.to(self._runtime_device())
        output, celltypeEmbedding, embeddings, attention_weight = self.model(x)
        return embeddings
