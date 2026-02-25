# Data preparation
- Download the datasets, `pdata`, and `sp` from [figshare](https://figshare.com/articles/dataset/CCL_CGI_h5_tar_gz/28777580?file=53613494) 
- Unzip into corresponding folders or generate `pdata` and `sp` by code


# SingleR
## Run Command

```bash
python run.py --dataset_name singleR --n_cell_types 34 --cv_folds 10 --spatial rw --n_graphs 12 --n_neighbors 12 --n_layers 4 --dff 64 --num_heads 4 --d_sp_enc 64 --p 1.5 --q 1.2 --lr 0.01 --dropout 0.5 --loss_mul 0.25 --bz 256 --model_name singleR --h5_dir ./h5/SingleR --data_dir ./pdata/SingleR --sp_dir ./sp/SingleR --gpu 0 --seed 42
```

## Run Directly from Checkpoints to View Performance

```bash
python run.py --dataset_name singleR --n_cell_types 34 --fold_idx 1 --cv_folds 10 --spatial rw --n_graphs 12 --n_neighbors 12 --n_layers 4 --dff 64 --num_heads 4 --d_sp_enc 64 --p 1.5 --q 1.2 --lr 0.01 --dropout 0.5 --loss_mul 0.25 --bz 256 --model_name singleR --checkpoint_path ./checkpoint/singleR_ckpt.pkl --h5_dir ./h5/SingleR --data_dir ./pdata/SingleR --sp_dir ./sp/SingleR --gpu 0 --seed 42
```


# NSCLC
## Run Command

```bash
python run.py --model_name NSCLC --dataset_name NSCLC --n_cell_types 6 --h5_dir ./h5/NSCLC --data_dir ./pdata/NSCLC --sp_dir ./sp/NSCLC --cv_folds 10 --fold_idx 1 --n_graphs 6 --n_neighbors 8 --n_layers 6 --dff 8 --d_sp_enc 64 --num_heads 4 --feature_subset all --lr 0.007 --dropout 0.3 --loss_mul 0.25 --bz 256 --spatial rw --p 1.5 --q 1.2 --use_pq_template --seed 42 --gpu 0
```

## Run Directly from Checkpoints to View Performance

```bash
python run.py --model_name NSCLC --dataset_name NSCLC --n_cell_types 6 --h5_dir ./h5/NSCLC --data_dir ./pdata/NSCLC --sp_dir ./sp/NSCLC --cv_folds 10 --fold_idx 1 --n_graphs 6 --n_neighbors 8 --n_layers 6 --dff 8 --d_sp_enc 64 --num_heads 4 --feature_subset all --lr 0.007 --dropout 0.3 --loss_mul 0.25 --bz 256 --spatial rw --p 1.5 --q 1.2 --use_pq_template --checkpoint_path ./checkpoint/NSCLC_ckpt.pkl --seed 42 --gpu 0
```

# CCL-CGI_withcancer
## Run Command

```bash
python run.py --dataset_name CCL-CGI_withcancer --use_cancer_ppi --n_cell_types 39 --cv_folds 10 --spatial rw --n_graphs 6 --n_neighbors 8 --n_layers 3 --dff 8 --d_sp_enc 64 --lr 0.005 --dropout 0.5 --loss_mul 0.2 --bz 256 --h5_dir ./h5/CCL-CGI_withcancer  --data_dir ./pdata/CCL-CGI_withcancer --sp_dir ./sp/CCL-CGI_withcancer --gpu 0 --seed 42
```

## Run Directly from Checkpoints to View Performance

```bash
python run.py --dataset_name CCL-CGI_withcancer --use_cancer_ppi --n_cell_types 39 --fold_idx 5 --model_name CCL_CGI --checkpoint_path ./checkpoint/CCL_CGI_withcancer_ckpt.pkl --h5_dir ./h5/CCL-CGI_withcancer --data_dir ./pdata/CCL-CGI_withcancer --sp_dir ./sp/CCL-CGI_withcancer --gpu 0 --seed 42
```


# cell_state
## Run Command

```bash
python run.py --model_name cell_state --dataset_name cell_state --n_cell_types 39 --h5_dir ./h5/cell_state --data_dir ./pdata/cell_state --sp_dir ./sp/cell_state --cv_folds 10 --n_graphs 6 --n_neighbors 8 --n_layers 4 --dff 256 --d_sp_enc 64 --num_heads 4 --lr 0.001 --dropout 0.3 --loss_mul 0.2 --bz 256 --spatial rw --p 1.0 --q 1.0 --use_64d_features --normalize_state_features --seed 42 --gpu 0
```

## Run Directly from Checkpoints to View Performance

```bash
python run.py --model_name cell_state --dataset_name cell_state --n_cell_types 39 --h5_dir ./h5/cell_state --data_dir ./pdata/cell_state --sp_dir ./sp/cell_state --cv_folds 10 --fold_idx 6 --n_graphs 6 --n_neighbors 8 --n_layers 4 --dff 256 --d_sp_enc 64 --num_heads 4 --lr 0.001 --dropout 0.3 --loss_mul 0.2 --bz 256 --spatial rw --p 1.0 --q 1.0 --use_64d_features --normalize_state_features --checkpoint_path ./checkpoint/cell_state_ckpt.pkl --seed 42 --gpu 0
```
