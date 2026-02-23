# -*- coding: utf-8 -*-
import sys
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
import numpy as np
import h5py
import networkx as nx
from utils import Node2vec
import pandas as pd

sys.path.append(os.getcwd())  # add the env path
from sklearn.model_selection import train_test_split, StratifiedKFold
from main import train


from config import RESULT_LOG, PROCESSED_DATA_DIR, LOG_DIR, MODEL_SAVED_DIR, SUBGRAPHA_TEMPLATE, SPATIAL_TEMPLATE, ModelConfig, ADJ_TEMPLATE, FEATURE_TEMPLATE, SHORT_PATH, cell_type_ppi, K_SETS, GENE_SYMBOLS, GLOBAL_PPI_H5_DIR, RESULT_DIR
from utils import pickle_dump, format_filename, write_log, pickle_load
import warnings
warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision('high')


def cross_validation_sets(y, mask, folds):
    label_idx = np.where(mask == 1)[0] # get indices of labeled genes
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=100)
    splits = kf.split(label_idx, y[label_idx])
    k_sets = []
    for train, test in splits:
        # get the indices in the real y and mask realm
        train_idx = label_idx[train]
        test_idx = label_idx[test]

        k_sets.append((train_idx, y[train_idx], test_idx, y[test_idx]))

        assert len(train_idx) == len(y[train_idx])
        assert len(test_idx) == len(y[test_idx])

    return k_sets


def read_global_ppi(path):
    with h5py.File(path, 'r') as f:
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
    id2symbol = dict()
    symbol2id = dict()
    for i in range(len(gene_symbols)):
        id2symbol[i] = gene_symbols[i]
        symbol2id[gene_symbols[i]] = i
    return network, y_train, y_val, y_test, train_mask, val_mask, test_mask, id2symbol, symbol2id


def read_h5file(path, network_name='network', feature_name='features'):
    with h5py.File(path, 'r') as f:
        # network is the adj of whole ppi network, so we need to extract the adj of specific cell type
        network = f[network_name][:]
        idx = f["idx"][:]
        features = f[feature_name][:]

        gene_symbols = f["gene_names"][:]

    # return network, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, gene_symbols, idx
    return network, features, gene_symbols, idx

def create_subgraphs_randomwalk(dataset, adj, n_graphs, n_neighbors, idx, is_distance = False):

    my_graph = nx.Graph()
    edge_index_begin, edge_index_end = np.where(adj > 0)
    edge_index = np.array([edge_index_begin, edge_index_end]).transpose().tolist()
    n_nodes = adj.shape[0]
    pmat = np.ones(shape=(n_nodes, n_nodes), dtype=int) * np.inf
    # my_graph.add_edges_from(tuple(edge_index))
    print(n_nodes)
    print(len(edge_index_end))
    my_graph = nx.from_numpy_array(adj)
    my_graph = my_graph.to_directed()
    # sub_adj = adj[np.ix_(idx, idx)]
    # sub_graph = nx.from_numpy_array(sub_adj)
    # sub_graph = sub_graph.to_directed()
    # walks = Node2vec(graph=my_graph, path_length=n_neighbors, num_paths=n_graphs, workers=6, dw=True).get_walks() ##[n_nodes, n_graphs, n_neighbors]
    walks = Node2vec(graph=my_graph, path_length=n_neighbors, num_paths=n_graphs, p=1.5, q=1.2, workers=6,
                     dw=False).get_walks()  ##[n_nodes, n_graphs, n_neighbors]

    new_walks = np.zeros(shape=(n_nodes, n_graphs, n_neighbors), dtype=int)
    for i in range(walks.shape[0]):
        new_walks[walks[i][0][0], :, :] = walks[i]

    walks = new_walks

    if is_distance == False:
        path = nx.all_pairs_shortest_path_length(my_graph)
        for node_i, node_ij in path:
            if node_i % 1000 == 0:
                print(node_i)
            for node_j, length_ij in node_ij.items():
                pmat[node_i, node_j] = length_ij
        pmat[pmat == np.inf] = -1
        save_path = "sp/" + dataset + "_sp.h5"
        new_file = h5py.File(save_path, 'w')
        new_file.create_dataset(name="sp", shape=(n_nodes, n_nodes), data=pmat)
    else:
        f = h5py.File("sp/" + dataset + "_sp.h5")
        pmat = f["sp"][:]
        pmat[pmat == np.inf] = -1

    subgraphs_list = []
    for id in range(n_nodes):
        sub_subgraph_list = []
        for g in range(n_graphs):
            node_feature_id = np.array(walks[id, g, :],dtype=int)

            attn_bias = np.concatenate([np.expand_dims(i[node_feature_id, :][:, node_feature_id], 0) for i in [pmat]])

            sub_subgraph_list.append(attn_bias)

        subgraphs_list.append(sub_subgraph_list)

    return walks, np.array(subgraphs_list)


def process_data(DATASET: dict, n_graphs: int, n_neighbors: int, n_layers: int, lr: float, spatial: str, cv_folds: int, dropout: float, loss_mul: float, dff: int, bz: int, distance: bool):

    MAX_DEGREE = []
    IDX = []
    ADJ = []
    degree_path = format_filename(PROCESSED_DATA_DIR, "MAX_DEGREE"), MAX_DEGREE
    print(f"max degree: {degree_path}")

    if not os.path.exists(format_filename(PROCESSED_DATA_DIR, "MAX_DEGREE")):

        for dataset in DATASET.keys():

            print("reading data.....")
            datapath = os.getcwd() + "/h5/" + str(dataset) + ".h5"
            print(datapath)
            adj, features, gene_symbols, idx = read_h5file(datapath)
            max_degree = int(max(np.sum(adj, axis=-1)) + 1)
            degree = np.expand_dims(np.sum(adj, axis=-1), axis=-1)
            MAX_DEGREE.append(max_degree)
            IDX.append(idx)

            if not os.path.exists(format_filename(PROCESSED_DATA_DIR, ADJ_TEMPLATE, dataset=dataset)):
                pickle_dump(format_filename(PROCESSED_DATA_DIR, ADJ_TEMPLATE, dataset=dataset), degree)
                pickle_dump(format_filename(PROCESSED_DATA_DIR, FEATURE_TEMPLATE, dataset=dataset), features)

            neighbor_id = None
            spatial_matrix = None

            subgraph_path = format_filename(PROCESSED_DATA_DIR, SUBGRAPHA_TEMPLATE, dataset=dataset, strategy = spatial, n_channel = n_graphs, n_neighbor = n_neighbors)

            if os.path.exists(subgraph_path) == False:
                # create subgraphs of spec cell type by edges,
                # so there's no need to decrease the whole node number to ppi node number
                neighbor_id, spatial_matrix = create_subgraphs_randomwalk(dataset, adj, n_graphs, n_neighbors, idx, distance)
                pickle_dump(format_filename(PROCESSED_DATA_DIR, SUBGRAPHA_TEMPLATE, dataset=dataset, strategy=spatial,n_channel=n_graphs, n_neighbor=n_neighbors), neighbor_id)
                pickle_dump(format_filename(PROCESSED_DATA_DIR, SPATIAL_TEMPLATE, dataset=dataset, strategy=spatial,n_channel=n_graphs,n_neighbor=n_neighbors), spatial_matrix)

            GENE_SYMBOLS[dataset] = gene_symbols

            pickle_dump(format_filename(PROCESSED_DATA_DIR, "MAX_DEGREE"), MAX_DEGREE)
            pickle_dump(format_filename(PROCESSED_DATA_DIR, "IDX"), IDX)

    else:
        MAX_DEGREE = pickle_load(format_filename(PROCESSED_DATA_DIR, "MAX_DEGREE"))
        IDX = pickle_load(format_filename(PROCESSED_DATA_DIR, "IDX"))

    temp = {'dataset': 'ALL', 'avg_auc': 0.0, 'avg_acc': 0.0, 'avg_aupr': 0.0, 'auc_std': 0.0, 'aupr_std': 0.0}
    results = {'auc': [], 'aupr': [], 'acc': []}
    count = 0

    network, y_train, y_val, y_test, train_mask, val_mask, test_mask, id2symbol, symbol2id = read_global_ppi(GLOBAL_PPI_H5_DIR)

    y_train_val = np.logical_or(y_train, y_val)
    mask_train_val = np.logical_or(train_mask, val_mask)
    k_sets = cross_validation_sets(y=y_train_val, mask=mask_train_val,
                                   folds=cv_folds)  ##split training set and validation set
    K_SETS["global_ppi"] = k_sets
    test_id = np.where(test_mask == 1)[0]  # get indices of labeled genes
    y_test = y_test[test_id]

    DISTANCE_MATRIX = []
    NODE_FEATURE = []
    NODE_NEIGHBOR = []
    SPATIAL_MATRIX = []

    for key in DATASET.keys():
        distance_matrix = np.load(format_filename(PROCESSED_DATA_DIR, ADJ_TEMPLATE, dataset=key), allow_pickle=True)
        node_feature = np.load(format_filename(PROCESSED_DATA_DIR, FEATURE_TEMPLATE, dataset=key), allow_pickle=True)
        node_neighbor = np.load(format_filename(PROCESSED_DATA_DIR, SUBGRAPHA_TEMPLATE, dataset=key, strategy = 'rw', n_channel = n_graphs,n_neighbor = n_neighbors), allow_pickle=True)
        spatial_matrix = np.load(format_filename(PROCESSED_DATA_DIR, SPATIAL_TEMPLATE, dataset=key, strategy = spatial, n_channel = n_graphs,n_neighbor = n_neighbors), allow_pickle=True)

        adj = []

        DISTANCE_MATRIX.append(distance_matrix)
        NODE_FEATURE.append(node_feature)
        NODE_NEIGHBOR.append(node_neighbor)
        SPATIAL_MATRIX.append(spatial_matrix)
        ADJ.append(adj)


    for i in range(cv_folds):

        train_id, y_train, val_id, y_val = k_sets[i]

        print(f"train_id:{len(train_id)}, y_train:{len(y_train)}, val_id:{len(val_id)}, y_val:{len(val_id)}")

        train(
            Kfold=i,
            dataset=DATASET,
            train_label=y_train,
            test_label=y_test,
            val_label=y_val,
            train_id = train_id,
            test_id = test_id,
            val_id = val_id,
            idx=IDX,
            n_graphs = n_graphs,
            n_neighbors = n_neighbors,
            n_layers = n_layers,
            spatial_type = spatial,
            max_degree=MAX_DEGREE,
            batch_size = bz,
            embed_dim = 8,
            num_heads = 4,
            d_sp_enc = dff,
            dff = dff,
            l2_weights=5e-7,
            lr=lr,
            dropout = dropout,
            loss_mul= loss_mul,
            optimizer ='adam',
            n_epoch=200,
            callbacks_to_add=['modelcheckpoint', 'earlystopping'],
            DISTANCE_MATRIX=DISTANCE_MATRIX,
            NODE_NEIGHBOR=NODE_NEIGHBOR,
            NODE_FEATURE=NODE_FEATURE,
            SPATIAL_MATRIX=SPATIAL_MATRIX,
            ADJ=ADJ
        )
        count += 1

    path = LOG_DIR + RESULT_DIR
    logs = pd.read_csv(path)
    temp["avg_auc"] = np.mean(np.array(logs["avg_auc"]))
    temp["avg_acc"] = np.mean(np.array(logs["avg_acc"]))
    temp["avg_aupr"] = np.mean(np.array(logs["avg_aupr"]))
    temp['auc_std'] = np.std(np.array(logs["avg_auc"]))
    temp['aupr_std'] = np.std(np.array(logs["avg_aupr"]))
    write_log(format_filename(LOG_DIR, RESULT_LOG["ALL"]), temp, 'a')
    print(f'Logging Info - {cv_folds} fold result: avg_auc: {temp["avg_auc"]}, avg_acc: {temp["avg_acc"]},'
          f'avg_aupr: {temp["avg_aupr"]}, auc_std: {temp["auc_std"]}, aupr_std: {temp["aupr_std"]}')


if __name__ == '__main__':

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(MODEL_SAVED_DIR):
        os.makedirs(MODEL_SAVED_DIR)
    model_config = ModelConfig()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # if torch.cuda.device_count() != 1:
    #     print(torch.cuda.device_count())
    #     assert "device error"
    # network, y_train, y_val, y_test, train_mask, val_mask, test_mask, id2symbol, symbol2id = read_global_ppi(GLOBAL_PPI_H5_DIR)


    #process_data(dataset: str, n_graphs: int, n_neighbors: int, n_layers: int, lr: float, spatial: str, cv_folds: int)

    process_data(cell_type_ppi, 6, 8, 3, 0.005, "rw", 10, dropout=0.5, loss_mul=0.2, dff=8, bz=256, distance=True)
