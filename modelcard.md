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

CCL-CGI is a graph-based deep learning model for cancer gene identification using multi-cell-type context from single-cell-derived molecular features and a global PPI graph.

The model combines:

- Cell-type-specific graph representations
- Graphormer-style encoding with structural/spatial information
- Attention-based aggregation across cell types
- Joint optimization with classification and metric-learning objectives

- **Developed by:** Project authors/maintainers of this repository
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

Possible downstream uses include candidate gene prioritization pipelines, exploratory biological analysis, and case-study style interpretation with attention scores.

### Out-of-Scope Use

- Clinical diagnosis or treatment decisions
- Use on unseen datasets without proper preprocessing and validation
- Causal interpretation of attention scores as definitive biological mechanisms


## Bias, Risks, and Limitations

- Predictions depend strongly on the provided graph/features and data split.
- Label sparsity and class imbalance can affect calibration and threshold-dependent metrics.
- Attention scores are useful for interpretation but should not be treated as causal proof.
- Performance may vary with checkpoint, fold selection, and hardware/runtime settings.

### Recommendations

- Report full CV protocol and confidence intervals when comparing runs.
- Validate on external/independent datasets before downstream claims.
- Keep preprocessing and hyperparameters consistent with training when using checkpoints.


## How to Get Started with the Model

### 1) Environment and dependencies

```bash
conda create -n ccl-cgi python=3.10 -y
conda activate ccl-cgi
python -m pip install -U pip
python -m pip install uv
uv sync
```

### 2) Train/evaluate on CCL-CGI

```bash
python run.py \
  --dataset_name CCL-CGI \
  --n_cell_types 39 \
  --cv_folds 10 \
  --spatial rw \
  --n_graphs 6 \
  --n_neighbors 8 \
  --n_layers 3 \
  --dff 8 \
  --d_sp_enc 64 \
  --lr 0.005 \
  --dropout 0.5 \
  --loss_mul 0.2 \
  --bz 256 \
  --h5_dir ./h5/CCL-CGI \
  --data_dir ./pdata/CCL-CGI \
  --sp_dir ./sp/CCL-CGI \
  --gpu 0
```

### 3) Run from existing checkpoints

```bash
python run.py \
  --dataset_name CCL-CGI \
  --n_cell_types 39 \
  --cv_folds 10 \
  --reuse_checkpoint \
  --model_name CCL_CGI \
  --h5_dir ./h5/CCL-CGI \
  --data_dir ./pdata/CCL-CGI \
  --sp_dir ./sp/CCL-CGI \
  --gpu 0
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
- Training pool : test ≈ `74.96% : 25.04%` (on labeled genes)
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
- Precision / Recall / F1
- Calibration metrics (Brier score, ECE)
- Optional confidence intervals (DeLong for AUC, Bootstrap for others)

### Results

This repository does not embed one fixed “official” score in the model card because results depend on checkpoint and run configuration.

Use the command above and read:

- `log/CCL_CGI_performance_CCL_CGI.csv`

to obtain the exact metrics for your run/checkpoint.

#### Summary

CCL-CGI is designed for robust CV-based cancer gene scoring with interpretable attention outputs across cell types.


## Model Examination

Attention-weight extraction is supported via:

- `tools/extract_attention_weights.py`

Example:

```bash
python tools/extract_attention_weights.py \
  --checkpoint_path checkpoint/your_model.pkl \
  --data_dir ./pdata/CCL-CGI \
  --global_ppi_h5 ./h5/CCL-CGI/global_ppi.h5 \
  --genes VAV1 TP53 BRCA1 \
  --merge_subtypes \
  --output_csv outputs/attention_scores.csv \
  --heatmap_path outputs/attention_heatmap.png
```


## Environmental Impact

Carbon emissions are not directly measured in this repository.

- **Hardware Type:** A100 GPU (single-card runs)
- **Hours used:** ~1–2 hours per fold (~10–20 hours for full 10-fold)
- **Cloud Provider:** Not specified
- **Compute Region:** Not specified
- **Carbon Emitted:** Not measured


## Technical Specifications

### Model Architecture and Objective

- Graph-based encoder with structural/spatial biases
- Attention fusion and multi-head aggregation across cell types
- Binary classification objective with additional metric-learning components

### Compute Infrastructure

#### Hardware

- GPU preferred (A100 tested), CPU fallback supported in current code

#### Software

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
