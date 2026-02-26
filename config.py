# -*- coding: utf-8 -*-

import os
import torch

PROJECT_ROOT = os.getcwd()

# PROCESSED_DATA_DIR = os.getcwd() + "/pdata"
# LOG_DIR = os.getcwd() + '/log'
# MODEL_SAVED_DIR = os.getcwd() + '/checkpoint'

# GLOBAL_PPI_H5_DIR = os.getcwd() + "/h5/CCL-CGI/global_ppi.h5"
# GLOBAL_CANCER_PPI_DIR = os.getcwd() + "/h5/global_cancer_ppi.h5"

PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "pdata", "CCL-CGI")
LOG_DIR = os.environ.get("CCL_CGI_LOG_DIR", os.path.join(PROJECT_ROOT, 'log'))
MODEL_SAVED_DIR = os.environ.get("CCL_CGI_CHECKPOINT_DIR", os.path.join(PROJECT_ROOT, 'checkpoint'))

GLOBAL_PPI_H5_DIR = os.path.join(PROJECT_ROOT, "h5", "CCL-CGI", "global_ppi.h5")
GLOBAL_CANCER_PPI_DIR = os.path.join(PROJECT_ROOT, "h5", "cancer_global_ppi_edgelist.txt")
GLOBAL_PPI_H5_WITHCANCER_DIR = os.path.join(PROJECT_ROOT, "h5_withcancer", "CCL-CGI", "global_ppi_withcancer.h5")

# Independent validation datasets (e.g., NSCLC)
INDEPENDENT_VALIDATION_DIR = os.path.join(PROJECT_ROOT, "independent_validation")
NSCLC_H5_DIR = INDEPENDENT_VALIDATION_DIR
NSCLC_PROCESSED_DATA_DIR = os.path.join(INDEPENDENT_VALIDATION_DIR, "pdata_NSCLC")
NSCLC_SP_DIR = os.path.join(INDEPENDENT_VALIDATION_DIR, "sp_NSCLC")
NSCLC_GLOBAL_PPI_H5 = os.path.join(INDEPENDENT_VALIDATION_DIR, "global_ppi.h5")

dataset = ['ALL']

#
SPATIAL_TEMPLATE = '{dataset}_method_{strategy}_channel_{n_channel}_neighbor_{n_neighbor}_spatial.npy'
SUBGRAPHA_TEMPLATE = '{dataset}_method_{strategy}_channel_{n_channel}_neighbor_{n_neighbor}_subgraphs.npy'
ADJ_TEMPLATE = '{dataset}_adj.npy'
SHORT_PATH = '{dataset}_sp.npy'
FEATURE_TEMPLATE = '{dataset}_feature.npy'
TRAIN_DATA_TEMPLATE = '{dataset}_train.npy'
DEV_DATA_TEMPLATE = '{dataset}_dev.npy'
TEST_DATA_TEMPLATE = '{dataset}_test.npy'
MAX_DEGREE_TEMPLATE = 'MAX_DEGREE'

RESULT_LOG={
    'ALL': 'cell_type_pancancer_result.txt'
}

PERFORMANCE_LOG = 'CCL_CGI_performance.txt'

RESULT_DIR = '/result_CCL_CGI.csv'

CUDNN_DETERMINISTIC = True   # Set to False for better performance (less reproducible)
CUDNN_BENCHMARK = False      # Set to True for better performance (less reproducible)


cell_type = [
    'T cells3', 'T cells2', 'T cells1',
    'Smooth muscle cells', 'Plasmacytoid dendritic cells', 'Plasma cells', 'Pericytes',
    'Pancreatic stellate cells',
    'NK cells3', 'NK cells2', 'NK cells1',
    'Neutrophils', 'Neurons', 'Myoepithelial cells', 'Monocytes',
    'Macrophages3', 'Macrophages2', 'Macrophages1',
    'Luminal epithelial cells', 'Keratinocytes', 'Hepatocytes', 'Goblet cells',
    'Gamma (PP) cells', 'Fibroblasts', 'Enteroendocrine cells', 'Enterocytes',
    'Endothelial cells2', 'Endothelial cells1',
    'Ductal cells', 'Dendritic cells2', 'Dendritic cells1',
    'Delta cells', 'Cholangiocytes', 'Beta cells',
    'Basal cells', 'B cells', 'Alpha cells', 'Adipocytes', 'Acinar cells'
]

singleR_cell_type = [
 'Acinar_cells_new', 'B_cells_naive_new', 'B_cells_new', 'CMP_new',
 'Cholangiocytes_new', 'Dendritic_cells_new', 'Ductal_cells_new', 'Endothelial_cells_new',
 'Enterocytes_new', 'Enteroendocrine_cells_new', 'Epithelial_cells_new',
 'Erythroid_like_and_erythroid_precursor_cells_new', 'Fibroblasts_new', 'GMP_new',
 'Goblet_cells_new', 'HSC_CD34+_new', 'Keratinocytes_new', 'Luminal_epithelial_cells_new',
 'MEP_new', 'MSC_new', 'Macrophages_new', 'Mammary_epithelial_cells_new',
 'Monocytes_new', 'Myoepithelial_cells_new', 'NK_cells_new', 'Neurons_new',
 'Neutrophils_new', 'Pericytes_new', 'Plasma_cells_new', 'Plasmacytoid_dendritic_cells_new',
 'Pro_Myelocyte_new', 'Pulmonary_alveolar_type_II_cells_new', 'Smooth_muscle_cells_new',
 'T_cells_new']


cell_type_ppi = {
    'CCL-CGI': cell_type,
    'NSCLC': ['B_cell', 'T_cell', 'Myeloid', 'Fibro', 'EC', 'Alveolar'],
    'CPDB_multiomics': ['CPDB_multiomics'],
    "singleR": singleR_cell_type,
    "CCL-CGI_withcancer": cell_type,
    "cell_state": cell_type
}

K_SETS = {}
GENE_SYMBOLS = {}

class ModelConfig(object):
    def __init__(self):

        self.n_layers = 2
        self.d_model = 8
        self.l2_weight = 1e-7  # l2 regularizer weight
        self.lr = 0.005  # learning rate
        self.n_epoch = 200,
        self.dff = 8
        self.max_degree = []
        self.n_neighbors = 3
        self.n_graphs = 3
        self.num_heads = 4
        self.concat_n_layers = self.n_layers
        self.dropout = 0.5
        self.loss_mul = 0.1
        self.d_sp_enc = 8
        self.batch_size = 16
        self.top_dropout = 0.5
        self.d_top = 256
        self.optimizer = 'adam'
        self.model_head = 'average'
        self.sp_enc_activation = "relu"
        self.top_activation = "relu"
        self.embedding_dim = 128

        self.distance_matrix = []
        self.node_feature = []
        self.node_neighbor = []
        self.spatial_matrix = []
        self.idx = []
        self.adj = []
        self.training = None

        self.network_all_ppi = None
        self.y_train_all = None
        self.y_val_all = None
        self.y_test_all = None
        self.train_mask_all = None
        self.val_mask_all = None
        self.test_mask_all = None
        self.id2symbol_all = None
        self.symbol2id_all = None

        self.exp_name = None
        self.model_name = None

        self.checkpoint_dir = MODEL_SAVED_DIR
        self.checkpoint_monitor = 'val_auc'
        # self.checkpoint_monitor = 'val_aupr'
        self.checkpoint_save_best_only = True
        self.checkpoint_save_weights_only = True
        self.checkpoint_save_weights_mode = 'max'
        self.checkpoint_verbose = 1

        # early_stoping configuration
        self.early_stopping_monitor = 'val_auc'
        # self.early_stopping_monitor = 'val_aupr'
        self.early_stopping_mode = 'max'
        self.early_stopping_patience = 30 
        self.early_stopping_verbose = 1
        self.K_Fold = 1
        self.callbacks_to_add = None

        # self.optimizer_monitor = 'val_aupr'
        self.optimizer_monitor = 'val_auc'

        # config for learning rating scheduler and ensembler
        self.swa_start = 3

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.contrastive_weight = 0.1
        self.triplet_loss_weight = 1
        self.lambda_reg = 0.01

        self.n_cell_types = 39  # Number of cell types in the dataset

        self.positive_fraction = None
        self.auto_threshold_by_val_f1 = False
        self.threshold = 0.5  # Default classification threshold; overridden by --auto_threshold_by_val_f1
