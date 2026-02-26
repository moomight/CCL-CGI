import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve, precision_score, \
    recall_score, matthews_corrcoef
from sklearn.metrics import roc_curve
import sklearn.metrics as m
import torch
import torch.nn as nn
import h5py
import os

def read_data(path):
    with h5py.File(path, 'r') as f:
        features = f['features'][:]
        network = f['network'][:]
        y_train = f['y_train'][:]
        y_test = f['y_test'][:]
        if 'y_val' in f:
            y_val = f['y_val'][:]
        else:
            y_val = None
        train_mask = f['train_mask'][:]
        test_mask = f['test_mask'][:]
        if 'val_mask' in f:
            val_mask = f['val_mask'][:]
        else:
            val_mask = None
        gene_symbols = f["gene_names"][:]
        has_nan = np.isnan(features).any()
        print(f"features NaN: {has_nan}")
        features = np.nan_to_num(features, nan=0.0)
    return features, network, y_train, y_val, y_test, train_mask, val_mask, test_mask, gene_symbols

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y

def save_batch_to_npy(y_true_batch, y_pred_batch, filename):
    if os.path.exists(filename):
        existing_data = np.load(filename, allow_pickle=True)
        y_true_existing, y_pred_existing = existing_data[0], existing_data[1]

        y_true = np.concatenate((y_true_existing, y_true_batch))
        y_pred = np.concatenate((y_pred_existing, y_pred_batch))
    else:
        y_true, y_pred = y_true_batch, y_pred_batch

    np.save(filename, (y_true, y_pred))

def score(y_true, y_pred, threshold, training):
    # y_pred = torch.sigmoid(y_pred)
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    y_pred = y_pred.detach().numpy()
    y_true = y_true.detach().numpy()
    if len(np.unique(y_true)) > 1:  # Check for both classes in y_true
        auc = roc_auc_score(y_true=y_true, y_score=y_pred)
    else:
        auc = 0
    precision, recall, _thresholds = precision_recall_curve(y_true=y_true, probas_pred=y_pred)
    aupr = m.auc(recall, precision)

    fpr, tpr, thr = roc_curve(y_true=y_true, y_score=y_pred)
    y_pred = [1 if prob >= threshold else 0 for prob in y_pred]
    if not training:
        tpr_fpr(y_true, y_pred)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    p = precision_score(y_true=y_true, y_pred=y_pred)
    r = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true=y_true, y_pred=y_pred)

    return auc, acc, p, r, f1, aupr, fpr.tolist(), tpr.tolist(), mcc

def train(model, criterion, optimizer, epoch_num, dataloader):
    runtime_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with tqdm(total=epoch_num) as t:
        for epoch in range(epoch_num):
            sum_loss = 0
            ave_acc = 0
            ave_auc = 0
            ave_aupr = 0
            ave_f1 = 0
            ave_mcc = 0

            t.set_description('Epoch %i' % epoch)
            count = 0
            for batch in dataloader:
                count += 1
                x_batch, y_batch = batch
                x_batch = x_batch.to(runtime_device)
                y_batch = y_batch.float().to(runtime_device)
                y_predict = model(x_batch)
                y_predict = torch.squeeze(y_predict)
                optimizer.zero_grad()
                loss = criterion(y_predict, y_batch)
                sum_loss += loss
                loss.backward()
                optimizer.step()

                auc, acc, p, r, f1, aupr, fpr, tpr, mcc = score(y_batch, y_predict, 0.5, True)
                ave_acc += acc
                ave_auc += auc
                ave_aupr += aupr
                ave_f1 += f1
                ave_mcc += mcc

            ave_acc = ave_acc / count
            ave_aupr = ave_aupr / count
            ave_auc = ave_auc / count
            ave_f1 = ave_f1 / count
            ave_mcc = ave_mcc / count
            ave_loss = sum_loss / count

            t.set_postfix(loss=ave_loss.item(), acc=ave_acc, auc=ave_auc, aupr=ave_aupr, f1=ave_f1, mcc=ave_mcc)
            t.update(1)

def t_model(model, criterion, dataloader):
    print(f"-------------------test---------------------")
    model.eval()
    runtime_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sum_loss = 0
    ave_acc = 0
    ave_auc = 0
    ave_aupr = 0
    ave_f1 = 0
    ave_mcc = 0

    with tqdm() as t:
        count = 0
        for batch in dataloader:
            count += 1
            x_batch, y_batch = batch
            x_batch = x_batch.to(runtime_device)
            y_batch = y_batch.float().to(runtime_device)
            y_predict = model(x_batch)
            y_predict = torch.squeeze(y_predict)
            loss = criterion(y_predict, y_batch)
            sum_loss += loss

            auc, acc, p, r, f1, aupr, fpr, tpr, mcc = score(y_batch, y_predict, 0.5, False)
            ave_acc += acc
            ave_auc += auc
            ave_aupr += aupr
            ave_f1 += f1
            ave_mcc += mcc
            t.set_postfix(loss=loss.item(), acc=acc, auc=auc, aupr=aupr, f1=f1, mcc=mcc)
            t.update(1)

        ave_acc = ave_acc / count
        ave_aupr = ave_aupr / count
        ave_auc = ave_auc / count
        ave_f1 = ave_f1 / count
        ave_mcc = ave_mcc / count
        ave_loss = sum_loss.item() / count

        print(f"test set: loss:{ave_loss}, ave_acc: {ave_acc}, ave_auc: {ave_auc}, ave_aupr: {ave_aupr}, ave_f1: {ave_f1}, ave_mcc: {ave_mcc}")

def tpr_fpr(y_true, y_pred):
    # y_true = y_true.int()
    true_positive = 0
    true_negative = 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_true[i] == 1:
            true_positive += 1
        elif y_pred[i] == 0 and y_true[i] == 0:
            true_negative += 1

    print(f"True Positive: {true_positive} / {sum(y_true)}, True Negative: {true_negative} / {len(y_true) - sum(y_true)}")
