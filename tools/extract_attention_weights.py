#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import pickle
from collections import OrderedDict

import h5py
import numpy as np
import pandas as pd
import torch

from config import ModelConfig, cell_type_ppi
from models import TREE


def _decode_symbol(symbol):
    if isinstance(symbol, bytes):
        return symbol.decode("utf-8")
    return str(symbol)


def _load_pickle_with_candidates(base_dir, candidates):
    for name in candidates:
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            with open(path, "rb") as file_obj:
                print(f"Loaded: {path}")
                return pickle.load(file_obj)
    raise FileNotFoundError(f"None of these files exists under `{base_dir}`: {candidates}")


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("`--device cuda` requested, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def _load_global_metadata(global_ppi_h5):
    with h5py.File(global_ppi_h5, "r") as h5f:
        gene_names_raw = h5f["gene_names"][:]
        gene_names = [_decode_symbol(item) for item in gene_names_raw]

        y_train = h5f["y_train"][:] if "y_train" in h5f else None
        y_val = h5f["y_val"][:] if "y_val" in h5f else None
        y_test = h5f["y_test"][:] if "y_test" in h5f else None
        train_mask = h5f["train_mask"][:] if "train_mask" in h5f else None
        val_mask = h5f["val_mask"][:] if "val_mask" in h5f else None
        test_mask = h5f["test_mask"][:] if "test_mask" in h5f else None

    symbol2id = {symbol: idx for idx, symbol in enumerate(gene_names)}

    positive_fraction = 0.25
    if y_train is not None and y_val is not None and train_mask is not None and val_mask is not None:
        y_train_val = np.logical_or(y_train, y_val)
        mask_train_val = np.logical_or(train_mask, val_mask)
        train_val_labels = y_train_val[mask_train_val == 1]
        if len(train_val_labels) > 0:
            positive_fraction = float(np.mean(train_val_labels))

    return {
        "gene_names": gene_names,
        "symbol2id": symbol2id,
        "y_test": y_test,
        "test_mask": test_mask,
        "positive_fraction": positive_fraction,
    }


def _resolve_gene_list(args, metadata):
    symbol2id = metadata["symbol2id"]

    if args.genes:
        genes = [item.strip() for item in args.genes if str(item).strip()]
    elif args.gene_file:
        with open(args.gene_file, "r", encoding="utf-8") as file_obj:
            genes = [line.strip() for line in file_obj if line.strip()]
    else:
        y_test = metadata["y_test"]
        test_mask = metadata["test_mask"]
        if y_test is None or test_mask is None:
            raise ValueError(
                "Please provide `--genes` or `--gene_file`, because test labels are not available in global PPI h5."
            )
        positive_test_ids = np.where((test_mask == 1) & (y_test == 1))[0]
        genes = [metadata["gene_names"][index] for index in positive_test_ids]
        print(f"No genes provided. Using all positive test genes: {len(genes)}")

    missing = [gene for gene in genes if gene not in symbol2id]
    if missing:
        raise ValueError(f"These genes are not found in global PPI gene_names: {missing[:10]}")

    return genes


def _load_processed_inputs(args, selected_cell_types):
    max_degree = _load_pickle_with_candidates(
        args.data_dir,
        [
            f"MAX_DEGREE_{args.dataset_name}_{args.n_cell_types}",
            "MAX_DEGREE",
            f"MAX_DEGREE_{args.dataset_name}",
        ],
    )
    idx = _load_pickle_with_candidates(
        args.data_dir,
        [
            f"IDX_{args.dataset_name}_{args.n_cell_types}",
            "IDX",
        ],
    )

    if len(max_degree) < len(selected_cell_types) or len(idx) < len(selected_cell_types):
        raise ValueError(
            f"MAX_DEGREE/IDX length mismatch. "
            f"len(MAX_DEGREE)={len(max_degree)}, len(IDX)={len(idx)}, len(cell_types)={len(selected_cell_types)}"
        )

    distance_matrices = []
    node_features = []
    node_neighbors = []
    spatial_matrices = []

    for cell_type_name in selected_cell_types:
        adj_path = os.path.join(args.data_dir, f"{cell_type_name}_adj.npy")
        feature_path = os.path.join(args.data_dir, f"{cell_type_name}_feature.npy")
        neighbor_path = os.path.join(
            args.data_dir,
            f"{cell_type_name}_method_{args.spatial}_channel_{args.n_graphs}_neighbor_{args.n_neighbors}_subgraphs.npy",
        )
        spatial_path = os.path.join(
            args.data_dir,
            f"{cell_type_name}_method_{args.spatial}_channel_{args.n_graphs}_neighbor_{args.n_neighbors}_spatial.npy",
        )

        if not os.path.exists(adj_path):
            raise FileNotFoundError(f"Missing file: {adj_path}")
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Missing file: {feature_path}")
        if not os.path.exists(neighbor_path):
            raise FileNotFoundError(f"Missing file: {neighbor_path}")
        if not os.path.exists(spatial_path):
            raise FileNotFoundError(f"Missing file: {spatial_path}")

        distance_matrix = np.load(adj_path, allow_pickle=True)
        if isinstance(distance_matrix, np.ndarray) and distance_matrix.dtype == np.uint64:
            distance_matrix = distance_matrix.astype(np.float32, copy=False)

        distance_matrices.append(distance_matrix)
        node_features.append(np.load(feature_path, allow_pickle=True))
        node_neighbors.append(np.load(neighbor_path, allow_pickle=True))
        spatial_matrices.append(np.load(spatial_path, allow_pickle=True))

    return {
        "max_degree": max_degree[: len(selected_cell_types)],
        "idx": idx[: len(selected_cell_types)],
        "distance_matrices": distance_matrices,
        "node_features": node_features,
        "node_neighbors": node_neighbors,
        "spatial_matrices": spatial_matrices,
    }


def _build_model(args, device, selected_cell_types, processed, positive_fraction):
    config = ModelConfig()
    config.device = str(device)
    config.training = False

    config.d_model = args.d_model
    config.n_layers = args.n_layers
    config.concat_n_layers = args.n_layers
    config.n_graphs = args.n_graphs
    config.n_neighbors = args.n_neighbors
    config.d_sp_enc = args.d_sp_enc
    config.dff = args.dff
    config.num_heads = args.num_heads
    config.dropout = args.dropout
    config.loss_mul = args.loss_mul
    config.batch_size = args.batch_size
    config.n_cell_types = len(selected_cell_types)
    config.positive_fraction = positive_fraction

    config.max_degree = processed["max_degree"]
    config.idx = processed["idx"]
    config.distance_matrix = processed["distance_matrices"]
    config.node_feature = processed["node_features"]
    config.node_neighbor = processed["node_neighbors"]
    config.spatial_matrix = processed["spatial_matrices"]

    model = TREE(config)

    state_dict = torch.load(args.checkpoint_path, map_location=device)
    model.model.load_state_dict(state_dict, strict=True)
    model.model.to(device)
    model.model.eval()

    print(f"Loaded checkpoint: {args.checkpoint_path}")
    print(f"Running device: {device}")
    return model


def _extract_attention_scores(model, genes, symbol2id, selected_cell_types, device, save_raw_dir=None):
    rows = []
    index = []

    if save_raw_dir:
        os.makedirs(save_raw_dir, exist_ok=True)

    with torch.no_grad():
        for gene in genes:
            gene_id = symbol2id[gene]
            node = torch.tensor([gene_id], dtype=torch.long, device=device)

            # Shape: [batch, num_heads, n_cell_types, n_cell_types]
            attn_weights = model.get_attn_weights(node)
            attn_array = attn_weights.detach().cpu().numpy()

            # Mean over batch, heads, query-cell-type -> score per key-cell-type
            # Output shape: [n_cell_types]
            scores = attn_array.mean(axis=(0, 1, 2))
            rows.append(scores)
            index.append(gene)

            if save_raw_dir:
                np.save(
                    os.path.join(save_raw_dir, f"{gene}_attn.npy"),
                    attn_array,
                )

    table = pd.DataFrame(rows, index=index, columns=selected_cell_types)
    return table


def _merge_cell_subtypes(table):
    merge_map = OrderedDict(
        {
            "T cells": ["T cells1", "T cells2", "T cells3"],
            "Dendritic cells": ["Dendritic cells1", "Dendritic cells2"],
            "Macrophages": ["Macrophages1", "Macrophages2", "Macrophages3"],
            "NK cells": ["NK cells1", "NK cells2", "NK cells3"],
            "Endothelial cells": ["Endothelial cells1", "Endothelial cells2"],
        }
    )

    columns = list(table.columns)
    for merged_name, members in merge_map.items():
        existing = [member for member in members if member in table.columns]
        if not existing:
            continue

        insert_at = min(columns.index(member) for member in existing)
        table[merged_name] = table[existing].mean(axis=1)
        table = table.drop(columns=existing)

        columns = [col for col in columns if col not in existing]
        columns.insert(insert_at, merged_name)
        table = table[columns]

    return table


def _save_heatmap(table, heatmap_path):
    import matplotlib.pyplot as plt

    fig_width = max(10, int(table.shape[1] * 0.5))
    fig_height = max(4, int(table.shape[0] * 0.4))

    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(table.values, aspect="auto", cmap="viridis")
    plt.colorbar(label="Attention score")
    plt.xticks(range(table.shape[1]), table.columns, rotation=45, ha="right")
    plt.yticks(range(table.shape[0]), table.index)
    plt.title("Attention Weights: Gene vs Cell Type")
    plt.xlabel("Cell Types")
    plt.ylabel("Genes")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Saved heatmap: {heatmap_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract per-gene cell-type attention weights from a trained CCL-CGI checkpoint.")

    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint `.pkl` file.")
    parser.add_argument(
        "--global_ppi_h5",
        type=str,
        default=None,
        help="Path to global PPI h5 file. If None, uses ./h5/<dataset_name>/global_ppi.h5.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing processed npy/pkl files. If None, uses ./pdata/<dataset_name>.",
    )

    parser.add_argument("--dataset_name", type=str, default="CCL-CGI", help="Dataset key in config.cell_type_ppi.")
    parser.add_argument("--n_cell_types", type=int, default=39, help="Number of cell types to use.")

    parser.add_argument("--genes", type=str, nargs="*", default=None, help="Gene symbols to extract.")
    parser.add_argument("--gene_file", type=str, default=None, help="Text file with one gene symbol per line.")

    parser.add_argument("--spatial", type=str, default="rw", choices=["rw", "sp"], help="Spatial strategy used by cache files.")
    parser.add_argument("--n_graphs", type=int, default=6)
    parser.add_argument("--n_neighbors", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--d_model", type=int, default=8)
    parser.add_argument("--dff", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_sp_enc", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--loss_mul", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Runtime device.")
    parser.add_argument("--merge_subtypes", action="store_true", help="Merge subtype columns (T cells1/2/3, etc.) into grouped columns.")

    parser.add_argument("--output_csv", type=str, default="attention_scores.csv", help="Output CSV path.")
    parser.add_argument("--heatmap_path", type=str, default=None, help="Optional heatmap image output path.")
    parser.add_argument("--save_raw_dir", type=str, default=None, help="Optional directory to save raw per-gene attention tensors.")

    return parser.parse_args()


def main():
    args = parse_args()
    device = _resolve_device(args.device)

    if args.dataset_name not in cell_type_ppi:
        raise KeyError(f"Unknown dataset_name `{args.dataset_name}`. Available keys: {list(cell_type_ppi.keys())}")

    if args.data_dir is None:
        args.data_dir = os.path.join(os.getcwd(), "pdata", args.dataset_name)
    if args.global_ppi_h5 is None:
        args.global_ppi_h5 = os.path.join(os.getcwd(), "h5", args.dataset_name, "global_ppi.h5")

    selected_cell_types = cell_type_ppi[args.dataset_name][: args.n_cell_types]
    print(f"Using {len(selected_cell_types)} cell types from `{args.dataset_name}`")

    metadata = _load_global_metadata(args.global_ppi_h5)
    genes = _resolve_gene_list(args, metadata)
    print(f"Extracting attention for {len(genes)} genes")

    processed = _load_processed_inputs(args, selected_cell_types)
    model = _build_model(
        args=args,
        device=device,
        selected_cell_types=selected_cell_types,
        processed=processed,
        positive_fraction=metadata["positive_fraction"],
    )

    table = _extract_attention_scores(
        model=model,
        genes=genes,
        symbol2id=metadata["symbol2id"],
        selected_cell_types=selected_cell_types,
        device=device,
        save_raw_dir=args.save_raw_dir,
    )

    if args.merge_subtypes:
        table = _merge_cell_subtypes(table)

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    table.to_csv(args.output_csv)
    print(f"Saved attention scores: {args.output_csv}")

    if args.heatmap_path:
        heatmap_dir = os.path.dirname(args.heatmap_path)
        if heatmap_dir:
            os.makedirs(heatmap_dir, exist_ok=True)
        _save_heatmap(table, args.heatmap_path)


if __name__ == "__main__":
    main()
