---
language:
- en
license: mit
library_name: pytorch
tags:
- biology
- cancer-gene-identification
- single-cell
- graph-neural-network
- pytorch
---

# Model Card for CCL-CGI

## Model Details

### Model Description

CCL-CGI is a graph transformer-based network designed to leverage cellular heterogeneity through contextualized metric-based contrastive learning, integrating scRNA-seq and PPI network data.

The model combines:

- Cell-type-specific graph representations
- Graphormer-style encoding with structural/spatial information
- Attention-based aggregation across cell types
- Joint optimization with classification and metric-based contrastive learning objectives

- **Developed by:** Ying Chang
- **Model type:** Graph-based binary classifier (gene-level cancer gene prediction)
- **Language(s):** Not an NLP model
- **License:** MIT (see `LICENSE`)
- **Repository:** Local project repository


## Uses

### Direct Use

This model is intended to rank/score genes by cancer-gene likelihood under the provided CCL-CGI data setting.

Typical direct uses:

- Reproducing cross-validation performance on CCL-CGI
- Running checkpoint-based inference/evaluation
- Extracting attention weights for biological interpretation

### Downstream Use

Possible downstream uses include candidate cancer gene identification, exploratory biological analysis between cancer gene and cell types, and case-study style interpretation with attention scores.

### Out-of-Scope Use

- Clinical diagnosis or treatment decisions
- Causal interpretation of attention scores as definitive biological mechanisms

### Recommendations

- Although we provided the CPU fallback, GPU is recommended for this repository.


## How to Get Started with the Model

### 1) Environment and dependencies

```bash
python -m venv ccl-cgi
source ccl-cgi/bin/activate
python -m pip install -U pip
python -m pip install uv
uv sync
```

### 2) Train/evaluate on CCL-CGI

```bash
python run.py --dataset_name CCL-CGI --n_cell_types 39 --cv_folds 10 --spatial rw --n_graphs 6 --n_neighbors 8 --n_layers 3 --dff 8 --d_sp_enc 64 --lr 0.005 --dropout 0.5 --loss_mul 0.2 --bz 256 --h5_dir ./h5/CCL-CGI --data_dir ./pdata/CCL-CGI --sp_dir ./sp/CCL-CGI --gpu 0 --seed 42
```

### 3) Run from existing checkpoints

```bash
python run.py --dataset_name CCL-CGI --n_cell_types 39 --fold_idx 8 --model_name CCL_CGI --checkpoint_path ./checkpoint/CCL_CGI_ckpt.pkl --h5_dir ./h5/CCL-CGI --data_dir ./pdata/CCL-CGI --sp_dir ./sp/CCL-CGI --gpu 0 --seed 42
```

Performance logs are written to:

- `log/CCL_CGI_performance_CCL_CGI.csv`


## Training Details

### Training Data

CCL-CGI setup statistics (from `global_ppi.h5`):

- Total genes: `12,956`
- Labeled genes: `3,091`
- Positive genes: `783`
- Negative genes: `2,308`
- Cell types: `39`

Split protocol:

- Original split: `train=2,086`, `val=231`, `test=774`
- Training pool: `train + val = 2,317`
- Test set: `774`
- Training pool : test ≈ `75% : 25%` (on labeled genes)
- 10-fold cross-validation on the training pool

### Training Procedure

#### Preprocessing

- Read H5 graph/feature/label data
- Build/load cached intermediate files in:
  - `pdata/CCL-CGI/`
  - `sp/CCL-CGI/`
- Optional feature sanitation and state-feature normalization

#### Training Hyperparameters (default CCL-CGI setting)

- `n_graphs=6`
- `n_neighbors=8`
- `n_layers=3`
- `dff=8`
- `d_sp_enc=64`
- `lr=0.005`
- `dropout=0.5`
- `loss_mul=0.2`
- `batch_size=256`
- `cv_folds=10`

#### Speeds, Sizes, Times

- Hardware: single A100 GPU
- Training time: ~`1-2` hours per fold
- GPU memory usage: ~`3 GB`


## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The fixed test split from `global_ppi.h5` (`test=774` labeled genes) is used, while CV is conducted on train+val.

#### Factors

- Fold-to-fold variation under 10-fold CV
- Checkpoint choice
- Runtime device and reproducibility settings

#### Metrics

The pipeline reports:

- AUC
- AUPR
- ACC
- MCC


## Technical Specifications

### Model Architecture and Objective

- Graph-based encoder with structural/spatial biases
- Attention fusion and multi-head aggregation across cell types (hierarchical)
- Binary classification objective with additional metric-based contrastive learning components

### Compute Infrastructure

#### Hardware

- GPU preferred (A100 tested), CPU fallback supported in current code

#### Software (see `pyproject.toml`)

- Python `3.10`
- PyTorch `1.13.1`
- lightning / pytorch-lightning `2.1.x`
- numpy `1.26.4`
- pandas `2.2.1`
- scikit-learn `1.4.2`
- networkx `3.3`
- h5py `3.11.0`


## Citation

If you use this model/code, please cite the corresponding project/paper once finalized.


## Model Card Authors

Repository maintainers.


## Model Card Contact

Please use the repository issue tracker or project contact channels.
