# CCL-CGI Configurable Arguments

This document summarizes commonly used runtime arguments in `run.py`.

## Graph Construction

- `--n_graphs`
- `--n_neighbors`
- `--n_layers`
- `--spatial`

## Model Architecture

- `--dff`
- `--d_sp_enc`
- `--num_heads`

## Training

- `--lr`
- `--dropout`
- `--loss_mul`
- `--bz`
- `--cv_folds`

## Data

- `--dataset_name`
- `--n_cell_types`
- `--h5_dir`
- `--data_dir`
- `--sp_dir`

## Runtime Control

- `--gpu`
- `--seed`
- `--fold_idx`
- `--test_run`
- `--reuse_checkpoint`
- `--checkpoint_path`
- `--auto_threshold_by_val_f1`

## Threshold Policy

- Default (no flag): fixed threshold `0.5`
- With `--auto_threshold_by_val_f1`: automatically select the best threshold on the validation split by F1 score
