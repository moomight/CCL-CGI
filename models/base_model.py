# -*- coding: utf-8 -*-

import os
from typing import Any

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT

from config import ModelConfig
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

class _EarlyStopping(EarlyStopping, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class MyTQDMProgressBar(TQDMProgressBar):

    def __init__(self):
        super(MyTQDMProgressBar, self).__init__()

class _ModelCheckpoint(pl.Callback):
    def __init__(self, dirpath="checkpoint/", filename="checkpoint", monitor="val_auc", mode="max"):
        super(_ModelCheckpoint, self).__init__()
        os.makedirs(dirpath, exist_ok=True)
        self.name = os.path.join(dirpath, filename)
        # self.train_name = dirpath + "train_" + filename
        self.monitor = monitor
        self.mode = mode
        self.value = 0. if mode == "max" else 1e6

    def on_train_epoch_end(self, trainer, module): # translated
        save_state = {}
        for key, value in module.state_dict().items():
            if 'LLM' not in key:
                save_state[key] = value
        if self.mode == "max" and module.history[self.monitor][-1] > self.value:
            self.value = module.history[self.monitor][-1]
            torch.save(save_state, self.name)
            print(f"model state saved at {self.name}")
        if self.mode == "min" and module.history[self.monitor][-1] < self.value:
            self.value = module.history[self.monitor][-1]
            torch.save(save_state, self.name)
            print(f"model state saved at {self.name}")

class CSVLogger(pl.Callback):
    def __init__(self, dirpath="history/", filename="history"):
        super(CSVLogger, self).__init__()
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        self.name = dirpath + filename
        # self.name += ".csv"

    def on_train_epoch_end(self, trainer, module): # translatedlogtranslated
        history = pd.DataFrame(module.history)
        print(f"---history saved at {self.name}---")
        history.to_csv(self.name, index=False)

    def on_sanity_check_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        print(pl_module.history)

class LearningCurve(pl.Callback):
    def __init__(self, dirpath="../curve/", figsize=(4, 4), names=("val_loss", "val_acc", "val_auc", "val_aupr", "val_f1", "val_mcc")):
        super(LearningCurve, self).__init__()
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        self.dirpath = dirpath
        self.figsize = figsize
        self.names = names

    def on_fit_end(self, trainer, module): # translated.fittranslated
        fold = module.config.K_Fold
        history = module.history
        for i, j in enumerate(self.names):
            plt.figure(figsize=self.figsize)
            plt.title(j)
            plt.plot(history[j], "--o", color='r', label=j)
            plt.legend()
            name = self.dirpath + "fold_" + str(fold) + "_" + str(j) + ".png"
            plt.savefig(name)
        # plt.show()

class BaseModel(object):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.callbacks = []
        self.model = self.build()

    def add_model_checkpoint(self):
        self.callbacks.append(_ModelCheckpoint(
            dirpath=self.config.checkpoint_dir,
            filename='{}.pkl'.format(self.config.exp_name),
            monitor=self.config.checkpoint_monitor,
            # save_best_only=self.config.checkpoint_save_best_only,
            mode=self.config.checkpoint_save_weights_mode,
        ))
        print('Logging Info - Callback Added: ModelCheckPoint...')

    def add_tqdm(self):
        self.callbacks.append(MyTQDMProgressBar())
        print('Logging Info - Callback Added: MyTQDMProgressBar...')

    def add_early_stopping(self):
        self.callbacks.append(_EarlyStopping(
            monitor=self.config.early_stopping_monitor,
            mode=self.config.early_stopping_mode,
            patience=self.config.early_stopping_patience,
            verbose=self.config.early_stopping_verbose
        ))
        print('Logging Info - Callback Added: EarlyStopping...')

    def add_CSVLogger(self):
        self.callbacks.append(CSVLogger(
            filename='{}.csv'.format(self.config.exp_name)
        ))

    def init_callbacks(self):
        # self.add_CSVLogger()
        # self.callbacks.append(LearningCurve())
        self.add_tqdm()
        if 'modelcheckpoint' in self.config.callbacks_to_add:
            self.add_model_checkpoint()
        if 'earlystopping' in self.config.callbacks_to_add:
            self.add_early_stopping()

    def build(self):
        raise NotImplementedError

    def fit(self, x_train, y_train, x_val, y_val):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def score(self, x, y):
        raise NotImplementedError

    def load_weights(self, filename: str):
        self.model.load_weights(filename)

    def load_model(self, filename: str):
        # we only save model's weight instead of the whole model
        self.model.load_weights(filename)

    def load_best_model(self):
        raise NotImplementedError

    def load_swa_model(self):
        raise NotImplementedError

    def layers(self):
        self.model.layers

    def summary(self):
        self.model.summary()
