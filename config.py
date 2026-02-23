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
# TODO: translatedruntranslateddirectory

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

# cell_type_ppi = {
#     'T cells3': [
#         'cd8-positive,_alpha-beta_cytotoxic_t_cell.txt', 'naive_regulatory_t_cell.txt',
#         'naive_thymus-derived_cd4-positive,_alpha-beta_t_cell.txt', 'cd4-positive_helper_t_cell.txt',
#         'mature_nk_t_cell.txt', 'cd4-positive,_alpha-beta_memory_t_cell.txt', 'dn1_thymic_pro-t_cell.txt',
#         'regulatory_t_cell.txt', 'type_i_nk_t_cell.txt',
#         'cd8-positive,_alpha-beta_cytokine_secreting_effector_t_cell.txt'
#     ],
#     'T cells2': [
#         'cd8-positive,_alpha-beta_cytotoxic_t_cell.txt', 'naive_regulatory_t_cell.txt',
#         'naive_thymus-derived_cd4-positive,_alpha-beta_t_cell.txt', 'cd4-positive_helper_t_cell.txt',
#         'mature_nk_t_cell.txt', 'cd4-positive,_alpha-beta_memory_t_cell.txt', 'dn1_thymic_pro-t_cell.txt',
#         'regulatory_t_cell.txt', 'type_i_nk_t_cell.txt',
#         'cd8-positive,_alpha-beta_cytokine_secreting_effector_t_cell.txt'
#     ],
#     'T cells1': [
#         'cd8-positive,_alpha-beta_cytotoxic_t_cell.txt', 'naive_regulatory_t_cell.txt',
#         'naive_thymus-derived_cd4-positive,_alpha-beta_t_cell.txt', 'cd4-positive_helper_t_cell.txt',
#         'mature_nk_t_cell.txt', 'cd4-positive,_alpha-beta_memory_t_cell.txt', 'dn1_thymic_pro-t_cell.txt',
#         'regulatory_t_cell.txt', 'type_i_nk_t_cell.txt',
#         'cd8-positive,_alpha-beta_cytokine_secreting_effector_t_cell.txt'
#     ],
#     'Smooth muscle cells': [
#         'smooth_muscle_cell.txt', 'vascular_associated_smooth_muscle_cell.txt', 'bronchial_smooth_muscle_cell.txt'
#     ],
#     'Plasmacytoid dendritic cells': [
#         'plasmacytoid_dendritic_cell.txt'
#     ],
#     'Plasma cells': [
#         'plasma_cell.txt'
#     ],
#     'Pericytes': [
#         'pericyte_cell.txt'
#     ],
#     'Pancreatic stellate cells': [
#         'pancreatic_stellate_cell.txt'
#     ],
#     'NK cells3': [
#         'nk_cell.txt'
#     ],
#     'NK cells2': [
#         'nk_cell.txt'
#     ],
#     'NK cells1': [
#         'nk_cell.txt'
#     ],
#     'Neutrophils': [
#         'cd24_neutrophil.txt', 'nampt_neutrophil.txt'
#     ],
#     'Neurons': [
#         'retinal_bipolar_neuron.txt'
#     ],
#     'Myoepithelial cells': [
#         'myoepithelial_cell.txt'
#     ],
#     'Monocytes': [
#         'monocyte.txt', 'classical_monocyte.txt', 'intermediate_monocyte.txt', 'non-classical_monocyte.txt'
#     ],
#     'Macrophages3': [
#         'macrophage.txt'
#     ],
#     'Macrophages2': [
#         'macrophage.txt'
#     ],
#     'Macrophages1': [
#         'macrophage.txt'
#     ],
#     'Luminal epithelial cells': [
#         'luminal_cell_of_prostate_epithelium.txt', 'luminal_epithelial_cell_of_mammary_gland.txt'
#     ],
#     'Keratinocytes': [
#         'keratinocyte.txt'
#     ],
#     'Hepatocytes': [
#         'hepatocyte.txt'
#     ],
#     'Goblet cells': [
#         'goblet_cell.txt', 'respiratory_goblet_cell.txt', 'small_intestine_goblet_cell.txt',
#         'large_intestine_goblet_cell.txt', 'tracheal_goblet_cell.txt'
#     ],
#     'Gamma (PP) cells': [
#         'pancreatic_pp_cell.txt'
#     ],
#     'Fibroblasts': [
#         'fibroblast.txt', 'fibroblast_of_cardiac_tissue.txt', 'alveolar_fibroblast.txt', 'fibroblast_of_breast.txt'
#     ],
#     'Enteroendocrine cells': [
#         'intestinal_enteroendocrine_cell.txt'
#     ],
#     'Enterocytes': [
#         'enterocyte_of_epithelium_of_small_intestine.txt', 'enterocyte_of_epithelium_of_large_intestine.txt',
#         'mature_enterocyte.txt', 'immature_enterocyte.txt'
#     ],
#     'Endothelial cells2': [
#         'bronchial_vessel_endothelial_cell.txt', 'endothelial_cell.txt', 'artery_endothelial_cell.txt',
#         'capillary_endothelial_cell.txt', 'cardiac_endothelial_cell.txt', 'endothelial_cell_of_lymphatic_vessel.txt',
#         'endothelial_cell_of_vascular_tree.txt', 'gut_endothelial_cell.txt', 'retinal_blood_vessel_endothelial_cell.txt',
#         'endothelial_cell_of_artery.txt', 'vein_endothelial_cell.txt', 'endothelial_cell_of_hepatic_sinusoid.txt',
#         'lung_microvascular_endothelial_cell.txt', 'lymphatic_endothelial_cell.txt'
#     ],
#     'Endothelial cells1': [
#         'bronchial_vessel_endothelial_cell.txt', 'endothelial_cell.txt', 'artery_endothelial_cell.txt',
#         'capillary_endothelial_cell.txt', 'cardiac_endothelial_cell.txt', 'endothelial_cell_of_lymphatic_vessel.txt',
#         'endothelial_cell_of_vascular_tree.txt', 'gut_endothelial_cell.txt', 'retinal_blood_vessel_endothelial_cell.txt',
#         'endothelial_cell_of_artery.txt', 'vein_endothelial_cell.txt', 'endothelial_cell_of_hepatic_sinusoid.txt',
#         'lung_microvascular_endothelial_cell.txt', 'lymphatic_endothelial_cell.txt'
#     ],
#     'Ductal cells': [
#         'pancreatic_ductal_cell.txt'
#     ],
#     # because plasmacytoid_dendritic_cell already exists, we doesn't add it into dentritic cells
#     'Dendritic cells2': [
#         'cd1c-positive_myeloid_dendritic_cell.txt', 'liver_dendritic_cell.txt', 'myeloid_dendritic_cell.txt',
#         'cd141-positive_myeloid_dendritic_cell.txt', 'dendritic_cell.txt', 'mature_conventional_dendritic_cell.txt'
#     ],
#     'Dendritic cells1': [
#         'cd1c-positive_myeloid_dendritic_cell.txt', 'liver_dendritic_cell.txt', 'myeloid_dendritic_cell.txt',
#         'cd141-positive_myeloid_dendritic_cell.txt', 'dendritic_cell.txt', 'mature_conventional_dendritic_cell.txt'
#     ],
#     'Delta cells': [
#         'pancreatic_delta_cell.txt'
#     ],
#     'Cholangiocytes': [
#         'intrahepatic_cholangiocyte.txt'
#     ],
#     'Beta cells': [
#         'pancreatic_beta_cell.txt'
#     ],
#     'Basal cells': [
#         'basal_cell.txt', 'basal_cell_of_prostate_epithelium.txt'
#     ],
#     'B cells': [
#         'b_cell.txt'
#     ],
#     'Alpha cells': [
#         'pancreatic_alpha_cell.txt'
#     ],
#     'Adipocytes': [
#         'adipocyte.txt'
#     ],
#     'Acinar cells': [
#         'pancreatic_acinar_cell.txt', 'acinar_cell_of_salivary_gland.txt'
#     ]
# }
cell_type_ppi = {
    'CCL-CGI': cell_type,
    'NSCLC': ['B_cell', 'T_cell', 'Myeloid', 'Fibro', 'EC', 'Alveolar'],
    # Single-file baseline dataset example: ./independent_validation_NSCLC_10033/baseline_NSCLC.h5
    'NSCLC_BASELINE': ['baseline_NSCLC'],
    'CPDB_multiomics': ['CPDB_multiomics'],
    'MTG_multiomics': ['MTG_multiomics'],
    "LTG_multiomics": ["LTG_multiomics"]
}

K_SETS = {}
GENE_SYMBOLS = {}

class ModelConfig(object):
    def __init__(self):

        self.n_layers = 2
        self.d_model = 8
        self.l2_weight = 1e-7  # l2 regularizer weight
        self.lr = 0.001  # learning rate
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
        self.early_stopping_patience = 30 # TODO
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
