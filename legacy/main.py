# -*- coding: utf-8 -*-

import os
import gc
import time

import numpy as np
import torch

from utils import format_filename, write_log, pickle_dump, pickle_load
from models import TREE
from config import ModelConfig, LOG_DIR, PERFORMANCE_LOG, PROCESSED_DATA_DIR, ADJ_TEMPLATE, FEATURE_TEMPLATE, SPATIAL_TEMPLATE, SUBGRAPHA_TEMPLATE
import random
import torch.optim as optim
import networkx as nx
from scipy import sparse
import h5py
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd

def read_h5file(path, network_name='network', feature_name='features'):
    with h5py.File(path, 'r') as f:
        # network is the adj of whole ppi network, so we need to extract the adj of specific cell type
        network = f[network_name][:]
        idx = f["idx"][:]
        features = f[feature_name][:]

        gene_symbols = f["gene_names"][:]

    # return network, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, gene_symbols, idx
    return network, features, gene_symbols, idx


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_optimizer(op_type, learning_rate, model_parameters):
    if op_type == 'sgd':
        return optim.SGD(model_parameters, lr=learning_rate)
    elif op_type == 'rmsprop':
        return optim.RMSprop(model_parameters, lr=learning_rate)
    elif op_type == 'adagrad':
        return optim.Adagrad(model_parameters, lr=learning_rate)
    elif op_type == 'adadelta':
        return optim.Adadelta(model_parameters, lr=learning_rate)
    elif op_type == 'adam':
        return optim.Adam(model_parameters, lr=learning_rate)
    else:
        raise ValueError('Optimizer Not Understood: {}'.format(op_type))


def train(Kfold, dataset, train_label, test_label, val_label, train_id, test_id, val_id, n_graphs, n_neighbors, n_layers, spatial_type, idx,
          max_degree, batch_size, embed_dim, num_heads, d_sp_enc, dff, l2_weights, lr, dropout, loss_mul, optimizer, n_epoch, DISTANCE_MATRIX, NODE_FEATURE, NODE_NEIGHBOR, SPATIAL_MATRIX, ADJ,
          callbacks_to_add=None, overwrite=True):


    config = ModelConfig()
    config.d_model = embed_dim
    config.n_layers = n_layers
    config.concat_n_layers = n_layers
    config.l2_weight = l2_weights
    config.dataset=dataset
    config.K_Fold=Kfold
    config.lr = lr
    config.batch_size = batch_size
    config.n_epoch = n_epoch
    config.max_degree = max_degree
    config.num_heads = num_heads
    config.n_graphs = n_graphs
    config.n_neighbors = n_neighbors
    config.dropout = dropout
    config.training = True
    config.d_sp_enc = d_sp_enc
    config.dff = dff
    config.loss_mul = loss_mul
    config.callbacks_to_add = callbacks_to_add
    config.idx = idx
    config.distance_matrix = DISTANCE_MATRIX
    config.node_feature = NODE_FEATURE
    config.node_neighbor = NODE_NEIGHBOR
    config.spatial_matrix = SPATIAL_MATRIX
    config.adj = ADJ

    config.exp_name = f'mlp_triple_spatialType_{spatial_type}_layerNums_{n_layers}_graphsNum_{n_graphs}_neighborsNum_{n_neighbors}_optimizer_{optimizer}_lr_{lr}__epoch_{n_epoch}__fold_{Kfold}'

    print(config.callbacks_to_add)
    callback_str = '_' + '_'.join(config.callbacks_to_add)
    callback_str = callback_str.replace('_modelcheckpoint', '').replace('_earlystopping', '')
    config.exp_name += callback_str

    train_log = {'exp_name': config.exp_name, 'optimizer': optimizer,'epoch': n_epoch, 'learning_rate': lr,
                 'n_graphs':n_graphs, 'n_neighbors':n_neighbors}
    print('Logging Info - Experiment: %s' % config.exp_name)
    model_save_path = os.path.join(config.checkpoint_dir, '{}.pkl'.format(config.exp_name))
    name = 'train_{}.pkl'.format(config.exp_name)
    model_train_save_path = os.path.join(config.checkpoint_dir, name)
    print(f"model save path: {model_save_path}")
    model = TREE(config)

    train_label=np.array(train_label)
    valid_label=np.array(val_label)
    test_label=np.array(test_label)

    train_id = np.array(train_id)
    valid_id = np.array(val_id)
    test_id = np.array(test_id)

    setup_seed(42)

    if not os.path.exists(model_save_path) or overwrite:
        start_time = time.time()
        model.fit(x_train = train_id, y_train=train_label, x_val = valid_id, y_val = valid_label)
        elapsed_time = time.time() - start_time
        print('Logging Info - Training time: %s' % time.strftime("%H:%M:%S",
                                                                 time.gmtime(elapsed_time)))
        train_log['train_time'] = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


    print('Logging Info - Evaluate over test data:')
    model.load_best_model()
    model.test(x=test_id, y=test_label)

    train_log['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    write_log(format_filename(LOG_DIR, PERFORMANCE_LOG), log=train_log, mode='a')
    del model
    gc.collect()
