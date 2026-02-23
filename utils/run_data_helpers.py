import os

import h5py
import numpy as np
from sklearn.model_selection import StratifiedKFold

from config import GLOBAL_PPI_H5_DIR, GLOBAL_PPI_H5_WITHCANCER_DIR
from utils.io import format_filename


def cross_validation_sets(y, mask, folds):
    label_idx = np.where(mask == 1)[0]
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=100)
    splits = kf.split(label_idx, y[label_idx])
    k_sets = []
    for train, test in splits:
        train_idx = label_idx[train]
        test_idx = label_idx[test]

        k_sets.append((train_idx, y[train_idx], test_idx, y[test_idx]))

        assert len(train_idx) == len(y[train_idx])
        assert len(test_idx) == len(y[test_idx])

    return k_sets


def _select_feature_view(features: np.ndarray, use_64d_features: bool) -> np.ndarray:
    if features is None or features.ndim != 2:
        return features

    if use_64d_features:
        if features.shape[1] > 64:
            return features[:, :64]
        return features

    if features.shape[1] > 8:
        return features[:, :8]

    return features


def read_global_ppi(use_cancer_ppi: bool = False, global_ppi_h5: str | None = None):
    print("Reading global PPI ...")
    if global_ppi_h5 is not None:
        path = global_ppi_h5
    else:
        path = GLOBAL_PPI_H5_WITHCANCER_DIR if use_cancer_ppi else GLOBAL_PPI_H5_DIR

    with h5py.File(path, "r") as f:
        network = f["network"][:]

        y_train = f["y_train"][:]
        y_test = f["y_test"][:]
        if "y_val" in f:
            y_val = f["y_val"][:]
        else:
            y_val = None
        train_mask = f["train_mask"][:]
        test_mask = f["test_mask"][:]
        if "val_mask" in f:
            val_mask = f["val_mask"][:]
        else:
            val_mask = None
        gene_symbols = f["gene_names"][:]

    id2symbol = dict()
    symbol2id = dict()
    for i in range(len(gene_symbols)):
        id2symbol[i] = gene_symbols[i]
        symbol2id[gene_symbols[i]] = i

    return network, y_train, y_val, y_test, train_mask, val_mask, test_mask, id2symbol, symbol2id


def resolve_processed_cache_path(
    processed_data_dir: str,
    filename_template: str,
    *,
    dataset: str,
    shared_dataset: str | None = None,
    template_fallbacks: list[str] | None = None,
    **kwargs,
) -> str:
    template_fallbacks = template_fallbacks or []

    candidates: list[str] = []

    candidates.append(format_filename(processed_data_dir, filename_template, dataset=dataset, **kwargs))
    if shared_dataset:
        candidates.append(format_filename(processed_data_dir, filename_template, dataset=shared_dataset, **kwargs))

    for tpl in template_fallbacks:
        candidates.append(format_filename(processed_data_dir, tpl, dataset=dataset, **kwargs))
        if shared_dataset:
            candidates.append(format_filename(processed_data_dir, tpl, dataset=shared_dataset, **kwargs))

    for path in candidates:
        if os.path.exists(path):
            if path != candidates[0]:
                print(f"⚠️  Cache fallback: using `{path}`")
            return path

    return candidates[0]


def read_h5file(path, network_name="network", feature_name="features", use_64d_features: bool = False):
    with h5py.File(path, "r") as f:
        network = f[network_name][:]
        idx = f["idx"][:]
        features = f[feature_name][:]
        features = _select_feature_view(features, use_64d_features=use_64d_features)
        gene_symbols = f["gene_names"][:]

    return network, features, gene_symbols, idx


def read_multiomicsfile(path, use_64d_features: bool = False):
    with h5py.File(path, "r") as f:
        network = f["network"][:]
        features = f["features"][:]
        features = _select_feature_view(features, use_64d_features=use_64d_features)
        gene_symbols = f["gene_names"][:]
        y_train = f["y_train"][:]
        y_test = f["y_test"][:]
        if "y_val" in f:
            y_val = f["y_val"][:]
        else:
            y_val = None

        if "mask_train" in f:
            train_mask = f["mask_train"][:]
        else:
            train_mask = f["train_mask"][:]

        if "mask_test" in f:
            test_mask = f["mask_test"][:]
        else:
            test_mask = f["test_mask"][:]

        if "mask_val" in f:
            val_mask = f["mask_val"][:]
        elif "val_mask" in f:
            val_mask = f["val_mask"][:]
        else:
            val_mask = None

    return network, features, gene_symbols, y_train, y_val, y_test, train_mask, val_mask, test_mask


def read_network_features_idx_for_cache(
    datapath: str,
    *,
    dataset_name: str,
    use_64d_features: bool,
):
    if dataset_name in ["CPDB_multiomics", "MTG_multiomics", "LTG_multiomics"]:
        adj, features, gene_symbols, *_ = read_multiomicsfile(datapath, use_64d_features=use_64d_features)
        idx = np.arange(0, adj.shape[0])
        return adj, features, gene_symbols, idx

    with h5py.File(datapath, "r") as f:
        has_idx = "idx" in f
        has_labels = ("y_train" in f) or ("train_mask" in f)

    if has_idx and (not has_labels):
        adj, features, gene_symbols, idx = read_h5file(
            datapath,
            use_64d_features=use_64d_features,
        )
        return adj, features, gene_symbols, idx

    adj, features, gene_symbols, *_ = read_multiomicsfile(datapath, use_64d_features=use_64d_features)
    idx = np.arange(0, adj.shape[0])
    return adj, features, gene_symbols, idx


def normalize_state_features_inplace(features: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if features is None or features.ndim != 2:
        return features

    features = features.astype(np.float32, copy=False)
    normalized = features

    normalized = np.asarray(normalized, dtype=np.float32)
    normalized[~np.isfinite(normalized)] = 0.0

    col_min = np.min(normalized, axis=0)
    col_max = np.max(normalized, axis=0)
    count_like = (col_min >= 0) & (col_max > 50)
    if np.any(count_like):
        normalized[:, count_like] = np.log1p(normalized[:, count_like])

    mean = np.mean(normalized, axis=0)
    std = np.std(normalized, axis=0)
    const = std < eps
    safe_std = std.copy()
    safe_std[const] = 1.0
    normalized = (normalized - mean) / safe_std
    if np.any(const):
        normalized[:, const] = 0.0

    features[:, :] = np.clip(normalized, -5.0, 5.0)
    return features


def sanitize_features_inplace(features: np.ndarray) -> tuple[np.ndarray, int]:
    if features is None or features.ndim != 2:
        return features, 0

    features = features.astype(np.float32, copy=False)
    invalid_mask = ~np.isfinite(features)
    invalid_count = int(np.sum(invalid_mask))
    if invalid_count > 0:
        features[invalid_mask] = 0.0
    return features, invalid_count
