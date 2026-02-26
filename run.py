# -*- coding: utf-8 -*-
import sys
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
import numpy as np

sys.path.append(os.getcwd())  # add the env path
from main import train


from config import RESULT_LOG, LOG_DIR, MODEL_SAVED_DIR, SUBGRAPHA_TEMPLATE, SPATIAL_TEMPLATE, ADJ_TEMPLATE, FEATURE_TEMPLATE, cell_type_ppi, K_SETS, GENE_SYMBOLS, MAX_DEGREE_TEMPLATE
from utils import pickle_dump, format_filename, write_log, pickle_load
from utils.run_data_helpers import (
    cross_validation_sets,
    read_global_ppi,
    resolve_processed_cache_path as _resolve_processed_cache_path,
    read_multiomicsfile,
    read_network_features_idx_for_cache as _read_network_features_idx_for_cache,
    normalize_state_features_inplace as _normalize_state_features_inplace,
    sanitize_features_inplace as _sanitize_features_inplace,
)
from utils.run_graph_helpers import create_subgraphs_randomwalk
from utils.run_metrics_helpers import (
    init_metric_containers,
    collect_fold_metrics,
    summarize_metric_results,
    print_metric_report,
    build_performance_log,
    write_performance_csv,
)
from utils.run_cli_helpers import (
    get_runtime_args,
    setup_runtime_environment,
    print_run_configuration,
)
from utils.run_manifest import generate_run_manifest, print_manifest_summary
import warnings
warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision('high')


def process_data(DATASET: dict, n_graphs: int, n_neighbors: int, n_layers: int, lr: float, spatial: str, cv_folds: int,
                 dropout: float, loss_mul: float, dff: int, bz: int,
                 use_64d_features: bool = False, use_cancer_ppi: bool = False, p: float = 1.5, q: float = 1.2,
                 model_name: str = 'CCL_CGI', test_run: int = None, n_cell_types: int = 39,
                 data_dir: str = None, h5_dir: str = None, sp_dir: str = None, dataset_name: str = 'CCL-CGI',
                 force_preprocess: bool = False, d_sp_enc: int = 64, normalize_state_features: bool = False,
                 num_heads_override: int = None, feature_subset: str = 'all', use_pq_template: bool = False,
                 sanitize_features: bool = False,
                 fold_indices=None,
                 global_ppi_h5: str | None = None,
                 reuse_checkpoint: bool = False,
                 checkpoint_path: str | None = None,
                 threshold: float | None = None,
                 auto_threshold_by_val_f1: bool = False,
                 exclude_cell_types: list[str] | None = None):

    base_n_cell_types = n_cell_types

    excluded_cell_types: list[str] = []
    if exclude_cell_types:
        cleaned: list[str] = []
        for item in exclude_cell_types:
            if item is None:
                continue
            item = str(item).strip()
            if not item:
                continue
            parts = [p.strip() for p in item.split(",") if p.strip()]
            if parts:
                cleaned.extend(parts)
            else:
                cleaned.append(item)
        excluded_cell_types = list(dict.fromkeys(cleaned))
    excluded_cell_types_set = set(excluded_cell_types)

    cell_type_list = DATASET[dataset_name]

    # ========== Path Configuration ==========
    # Use provided paths or fall back to config defaults
    processed_data_dir = data_dir if data_dir is not None else os.path.join(os.getcwd(), "pdata", dataset_name)
    
    # Determine H5 directory based on parameters
    if h5_dir is not None:
        base_h5_dir = h5_dir
    else:
        # Default: use config paths
        if use_cancer_ppi:
            base_h5_dir = os.path.join(os.getcwd(), "h5_withcancer", dataset_name)
        else:
            base_h5_dir = os.path.join(os.getcwd(), "h5", dataset_name)
    
    # Determine SP directory
    sp_path_dir = sp_dir if sp_dir is not None else os.path.join(os.getcwd(), "sp", dataset_name)
    
    print(f"\n{'='*80}")
    print(f"Data Configuration:")
    print(f"  Dataset name: {dataset_name}")
    print(f"  Processed data directory: {processed_data_dir}")
    print(f"  H5 data directory: {base_h5_dir}")
    print(f"  Shortest path directory: {sp_path_dir}")
    print(f"  Number of cell types: {n_cell_types}")
    print(f"  sanitize_features: {sanitize_features}")
    print(f"{'='*80}\n")
    
    # Create local copies of templates to avoid modifying global constants
    if use_pq_template:
        pq_suffix = f"_p{p}_q{q}"
        max_degree_template = MAX_DEGREE_TEMPLATE + pq_suffix
        subgraph_template = SUBGRAPHA_TEMPLATE.replace(".npy", f"{pq_suffix}.npy")
        spatial_template = SPATIAL_TEMPLATE.replace(".npy", f"{pq_suffix}.npy")
    else:
        max_degree_template = MAX_DEGREE_TEMPLATE + f"_{dataset_name}_{n_cell_types}"
        subgraph_template = SUBGRAPHA_TEMPLATE
        spatial_template = SPATIAL_TEMPLATE
    adj_template = ADJ_TEMPLATE
    feature_template = FEATURE_TEMPLATE

    # Modify templates based on configuration
    # use_cancer_ppi affects PPI network-related files (subgraphs, spatial, adj, max_degree)
    if use_cancer_ppi:
        max_degree_template = max_degree_template + "_withCancer"
        subgraph_template = subgraph_template.replace(".npy", "_withCancer.npy")
        spatial_template = spatial_template.replace(".npy", "_withCancer.npy")
        adj_template = adj_template.replace(".npy", "_withCancer.npy")
    
    # Feature file naming (to avoid cache collisions)
    if use_64d_features:
        feature_template = feature_template.replace(".npy", "_stateFeatures.npy")
        max_degree_template = max_degree_template + "_stateFeatures"
    
    MAX_DEGREE = []
    IDX = []
    ADJ = []

    degree_path = format_filename(processed_data_dir, max_degree_template)
    print(f"max degree path: {degree_path}")
    print(f"Configuration: use_cancer_ppi={use_cancer_ppi}, use_64d_features={use_64d_features}")

    # Decide whether to run preprocessing
    needs_preprocessing = force_preprocess or not os.path.exists(degree_path)
    
    if needs_preprocessing:
        if force_preprocess:
            print(f"\n{'='*80}")
            print(f"🔄 FORCE PREPROCESS MODE: Regenerating all preprocessed data")
            print(f"{'='*80}\n")
        else:
            print(f"\n{'='*80}")
            print(f"📦 Preprocessed data not found, generating new data")
            print(f"{'='*80}\n")

        # Only process first n_cell_types from DATASET
        dataset_keys = cell_type_list[:n_cell_types]
        print(f"\n{'='*80}")
        print(f"Processing {len(dataset_keys)} cell types (n_cell_types={n_cell_types})")
        print(f"Cell types: {dataset_keys}")
        print(f"{'='*80}\n")
        
        for dataset in dataset_keys:

            print("reading data.....")
            # Construct H5 file path
            h5_filename = f"{dataset}_withcancer.h5" if use_cancer_ppi else f"{dataset}.h5"
            datapath = os.path.join(base_h5_dir, h5_filename)
            print(datapath)
            adj, features, gene_symbols, idx = _read_network_features_idx_for_cache(
                datapath,
                dataset_name=dataset_name,
                use_64d_features=use_64d_features,
            )
            # Backward-compatible behavior: by default keep only the first 8 dims.
            if (not use_64d_features) and features.ndim == 2 and features.shape[1] > 8:
                features = features[:, :8]
            max_degree = int(max(np.sum(adj, axis=-1)) + 1)
            degree = np.expand_dims(np.sum(adj, axis=-1), axis=-1)
            MAX_DEGREE.append(max_degree)
            IDX.append(idx)

            if not os.path.exists(format_filename(processed_data_dir, adj_template, dataset=dataset)):
                pickle_dump(format_filename(processed_data_dir, adj_template, dataset=dataset), degree)
            if not os.path.exists(format_filename(processed_data_dir, feature_template, dataset=dataset)):
                pickle_dump(format_filename(processed_data_dir, feature_template, dataset=dataset), features)

            neighbor_id = None
            spatial_matrix = None

            subgraph_path = format_filename(processed_data_dir, subgraph_template, dataset=dataset, strategy=spatial, n_channel=n_graphs, n_neighbor=n_neighbors)

            if os.path.exists(subgraph_path) == False:
                # create subgraphs of spec cell type by edges,
                # so there's no need to decrease the whole node number to ppi node number
                neighbor_id, spatial_matrix = create_subgraphs_randomwalk(dataset, adj, n_graphs, n_neighbors, idx, p=p, q=q, use_cancer_ppi=use_cancer_ppi, sp_dir=sp_path_dir, dataset_name=dataset_name)
                pickle_dump(format_filename(processed_data_dir, subgraph_template, dataset=dataset, strategy=spatial, n_channel=n_graphs, n_neighbor=n_neighbors), neighbor_id)
                pickle_dump(format_filename(processed_data_dir, spatial_template, dataset=dataset, strategy=spatial, n_channel=n_graphs, n_neighbor=n_neighbors), spatial_matrix)

            GENE_SYMBOLS[dataset] = gene_symbols

        # Save MAX_DEGREE and IDX for all cell types (accumulated in the loop above)
        # MAX_DEGREE: list of max degrees for each cell type, used to initialize CentralityEncoding layers
        # IDX: list of node indices for each cell type, used for node filtering and index mapping
        pickle_dump(format_filename(processed_data_dir, max_degree_template), MAX_DEGREE)
        pickle_dump(format_filename(processed_data_dir, f"IDX_{dataset_name}_{n_cell_types}"), IDX)

    else:
        MAX_DEGREE = pickle_load(format_filename(processed_data_dir, max_degree_template))
        IDX = pickle_load(format_filename(processed_data_dir, f"IDX_{dataset_name}_{n_cell_types}"))

    temp, results = init_metric_containers()

    if len(cell_type_list) > 1:
        resolved_global_ppi_h5 = global_ppi_h5
        if resolved_global_ppi_h5 is None:
            expected_name = "global_ppi_withcancer.h5" if use_cancer_ppi else "global_ppi.h5"
            candidate = os.path.join(base_h5_dir, expected_name)
            if os.path.exists(candidate):
                resolved_global_ppi_h5 = candidate
            else:
                raise FileNotFoundError(
                    "Global PPI file not found under --h5_dir. "
                    f"Expected: {candidate}. "
                    "Please place the expected file there or pass --global_ppi_h5 explicitly."
                )

        print(f"Using global PPI h5: {resolved_global_ppi_h5}")
        network, y_train, y_val, y_test, train_mask, val_mask, test_mask, id2symbol, symbol2id = read_global_ppi(
            use_cancer_ppi=use_cancer_ppi,
            global_ppi_h5=resolved_global_ppi_h5,
        )
    else:
        h5_filename = f"{cell_type_list[0]}_withcancer.h5" if use_cancer_ppi else f"{cell_type_list[0]}.h5"
        datapath = os.path.join(h5_dir, h5_filename)
        network, features, gene_symbols, y_train, y_val, y_test, train_mask, val_mask, test_mask = read_multiomicsfile(
            datapath,
            use_64d_features=use_64d_features,
        )
        if (not use_64d_features) and features.ndim == 2 and features.shape[1] > 8:
            features = features[:, :8]
        idx = np.arange(0, network.shape[0])

    y_train_val = np.logical_or(y_train, y_val)
    mask_train_val = np.logical_or(train_mask, val_mask)
    k_sets = cross_validation_sets(y=y_train_val, mask=mask_train_val,
                                   folds=cv_folds)  ##split training set and validation set
    K_SETS["global_ppi"] = k_sets
    test_id = np.where(test_mask == 1)[0]  # get indices of labeled genes
    y_test = y_test[test_id]

    train_val_labels = y_train_val[mask_train_val == 1]

    n_positive = np.sum(train_val_labels)
    n_total = len(train_val_labels)
    positive_fraction = n_positive / n_total
    print(f"positive_fraction: {positive_fraction}")
    
    DISTANCE_MATRIX = []
    NODE_FEATURE = []
    NODE_NEIGHBOR = []
    SPATIAL_MATRIX = []

    # Only load first n_cell_types from DATASET
    dataset_keys = cell_type_list[:n_cell_types]
    print(f"\n{'='*80}")
    print(f"Loading data for {len(dataset_keys)} cell types (n_cell_types={n_cell_types})")
    print(f"{'='*80}\n")
    
    if excluded_cell_types:
        base_dataset_keys = list(dataset_keys)
        missing = [ct for ct in excluded_cell_types if ct not in base_dataset_keys]
        if missing:
            print(f"⚠️ exclude_cell_types not found in current list: {missing}")
        if len(MAX_DEGREE) != len(base_dataset_keys) or len(IDX) != len(base_dataset_keys):
            raise ValueError(
                f"Cached MAX_DEGREE/IDX length mismatch: "
                f"len(MAX_DEGREE)={len(MAX_DEGREE)}, len(IDX)={len(IDX)}, len(base_dataset_keys)={len(base_dataset_keys)}. "
                f"Try --force_preprocess with base_n_cell_types={base_n_cell_types}."
            )
        keep_indices = [i for i, ct in enumerate(base_dataset_keys) if ct not in excluded_cell_types_set]
        dataset_keys = [base_dataset_keys[i] for i in keep_indices]
        if len(dataset_keys) == 0:
            raise ValueError("After excluding cell types, no datasets remain to run.")
        MAX_DEGREE = [MAX_DEGREE[i] for i in keep_indices]
        IDX = [IDX[i] for i in keep_indices]
        n_cell_types = len(dataset_keys)
        print(f"After excluding cell types: effective_n_cell_types={n_cell_types}")

    shared_dataset = f"SHARED_{dataset_name}" if dataset_name else None

    # Spatial cache does not depend on (p,q); support older caches without pq suffix.
    spatial_template_fallbacks: list[str] = []
    if use_pq_template:
        spatial_no_pq = SPATIAL_TEMPLATE
        if use_cancer_ppi:
            spatial_no_pq = spatial_no_pq.replace(".npy", "_withCancer.npy")
        spatial_template_fallbacks.append(spatial_no_pq)

    for key in dataset_keys:
        distance_matrix = np.load(format_filename(processed_data_dir, adj_template, dataset=key), allow_pickle=True)
        # Torch does not support uint64 tensors; some cached degree arrays may be uint64 (e.g., from np.sum on uint8 adj).
        if isinstance(distance_matrix, np.ndarray) and distance_matrix.dtype == np.uint64:
            distance_matrix = distance_matrix.astype(np.float32, copy=False)
        node_feature = np.load(format_filename(processed_data_dir, feature_template, dataset=key), allow_pickle=True)
        # Ensure consistent 8-d baseline when state features are disabled (older caches may contain >8 dims)
        if (not use_64d_features) and node_feature.ndim == 2 and node_feature.shape[1] > 8:
            node_feature = node_feature[:, :8]
        # NOTE (2025-12): 4-dim feature ablation was used for a one-off study.
        # It is intentionally disabled now to avoid accidental changes to the default pipeline.
        # If you need it again, re-enable the block below.
        #
        
        if (not use_64d_features) and feature_subset != 'all':
            print(f"⚠️ feature_subset={feature_subset} requested but 4-dim ablation is disabled; proceeding with all 8 dims.")
            feature_subset = 'all'
        if sanitize_features:
            node_feature, invalid_count = _sanitize_features_inplace(node_feature)
            if invalid_count > 0:
                print(f"⚠️ Sanitized {invalid_count} non-finite feature values for `{key}`")
        if use_64d_features and normalize_state_features:
            node_feature = _normalize_state_features_inplace(node_feature)

        # ---------- Ensure subgraph/spatial caches exist ----------
        # New (2026-02): allow changing n_graphs/n_neighbors without forcing full preprocessing.
        # If caches are missing, regenerate them on the fly for this cell type only.
        subgraph_path_rw = format_filename(
            processed_data_dir,
            subgraph_template,
            dataset=key,
            strategy="rw",
            n_channel=n_graphs,
            n_neighbor=n_neighbors,
        )
        subgraph_path_spatial = format_filename(
            processed_data_dir,
            subgraph_template,
            dataset=key,
            strategy=spatial,
            n_channel=n_graphs,
            n_neighbor=n_neighbors,
        )
        spatial_path = format_filename(
            processed_data_dir,
            spatial_template,
            dataset=key,
            strategy=spatial,
            n_channel=n_graphs,
            n_neighbor=n_neighbors,
        )

        if (not os.path.exists(subgraph_path_rw)) or (not os.path.exists(spatial_path)):
            print(f"\n{'='*80}")
            print(f"📦 Missing caches for dataset={key}; generating now...")
            print(f"  subgraph (rw): {subgraph_path_rw}")
            if spatial != "rw":
                print(f"  subgraph ({spatial}): {subgraph_path_spatial}")
            print(f"  spatial ({spatial}): {spatial_path}")
            print(f"{'='*80}\n")

            h5_filename = f"{key}_withcancer.h5" if use_cancer_ppi else f"{key}.h5"
            datapath = os.path.join(base_h5_dir, h5_filename)
            adj_full, _, _, idx_full = _read_network_features_idx_for_cache(
                datapath,
                dataset_name=dataset_name,
                use_64d_features=use_64d_features,
            )

            neighbor_id, spatial_matrix_gen = create_subgraphs_randomwalk(
                dataset=key,
                adj=adj_full,
                n_graphs=n_graphs,
                n_neighbors=n_neighbors,
                idx=idx_full,
                p=p,
                q=q,
                use_cancer_ppi=use_cancer_ppi,
                sp_dir=sp_path_dir,
                dataset_name=dataset_name,
            )
            pickle_dump(subgraph_path_rw, neighbor_id)
            if spatial != "rw":
                pickle_dump(subgraph_path_spatial, neighbor_id)
            pickle_dump(spatial_path, spatial_matrix_gen)

        # Use the standard resolution helpers (supports shared cache + fallbacks)
        node_neighbor_path = _resolve_processed_cache_path(
            processed_data_dir,
            subgraph_template,
            dataset=key,
            shared_dataset=shared_dataset,
            strategy="rw",
            n_channel=n_graphs,
            n_neighbor=n_neighbors,
        )
        node_neighbor = np.load(node_neighbor_path, allow_pickle=True)

        spatial_matrix_path = _resolve_processed_cache_path(
            processed_data_dir,
            spatial_template,
            dataset=key,
            shared_dataset=shared_dataset,
            template_fallbacks=spatial_template_fallbacks,
            strategy=spatial,
            n_channel=n_graphs,
            n_neighbor=n_neighbors,
        )
        spatial_matrix = np.load(spatial_matrix_path, allow_pickle=True)

        adj = []

        DISTANCE_MATRIX.append(distance_matrix)
        NODE_FEATURE.append(node_feature)
        NODE_NEIGHBOR.append(node_neighbor)
        SPATIAL_MATRIX.append(spatial_matrix)
        ADJ.append(adj)

    # Set embed_dim based on feature dimensions
    # 8 dims for basic features, 64 dims when using 64-d features
    embed_dim = NODE_FEATURE[0].shape[-1]
    print(f"Using embed_dim={embed_dim} (use_64d_features={use_64d_features})")
    
    # Calculate derived parameters (for logging)
    # For 64-d models we want more heads (matches earlier intent/comments).
    auto_num_heads = 8 if embed_dim >= 64 else 4
    num_heads = int(num_heads_override) if num_heads_override is not None else auto_num_heads
    if embed_dim % num_heads != 0:
        raise ValueError(f"Invalid num_heads={num_heads} for embed_dim={embed_dim} (must divide evenly).")
    if num_heads_override is not None:
        print(f"Using num_heads override: {num_heads} (auto would be {auto_num_heads})")
    l2_weights = 1e-6 if embed_dim >= 64 else 5e-7
    
    # Determine which folds to run
    subset_mode = fold_indices is not None or test_run is not None
    if fold_indices is not None:
        # Normalize to list[int]
        if isinstance(fold_indices, int):
            fold_indices = [fold_indices]
        selected_folds = list(dict.fromkeys(fold_indices))
        for fi in selected_folds:
            if fi < 0 or fi >= len(k_sets):
                raise ValueError(f"Requested fold_idx {fi} is out of range for cv_folds={cv_folds}")
        actual_folds = len(selected_folds)
        print(f"\n{'='*80}")
        print(f"🎯 Running specified folds only: {selected_folds} (total {cv_folds} folds available)")
        print(f"{'='*80}\n")
    else:
        actual_folds = test_run if test_run is not None else cv_folds
        selected_folds = list(range(actual_folds))
        if test_run is not None:
            print(f"\n{'='*80}")
            print(f"🔧 TEST RUN MODE: Only running first {test_run} fold(s) out of {cv_folds}")
            print(f"{'='*80}\n")

    direct_checkpoint_mode = checkpoint_path is not None
    checkpoint_template_mode = direct_checkpoint_mode and ("{fold}" in checkpoint_path)
    if direct_checkpoint_mode:
        if checkpoint_template_mode:
            print(f"Using checkpoint template mode: {checkpoint_path}")
        else:
            if len(selected_folds) != 1:
                raise ValueError(
                    "When --checkpoint_path points to a single file, please run a single fold (e.g., --fold_idx 8), "
                    "or provide a template path containing {fold}."
                )
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")
            print(f"Using direct checkpoint path: {checkpoint_path}")

    for i in selected_folds:

        train_id, y_train, val_id, y_val = k_sets[i]

        print(f"train_id:{len(train_id)}, y_train:{len(y_train)}, val_id:{len(val_id)}, y_val:{len(val_id)}")

        fold_checkpoint_path = None
        if direct_checkpoint_mode:
            if checkpoint_template_mode:
                fold_checkpoint_path = checkpoint_path.format(fold=i)
                if not os.path.exists(fold_checkpoint_path):
                    raise FileNotFoundError(
                        f"Checkpoint for fold {i} not found from template: {fold_checkpoint_path}"
                    )
            else:
                fold_checkpoint_path = checkpoint_path

        test_metrics = train(
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
            embed_dim = embed_dim,
            num_heads = num_heads,
            d_sp_enc = d_sp_enc,
            dff = dff,
            l2_weights=l2_weights,
            lr=lr,
            dropout = dropout,
            loss_mul= loss_mul,
            optimizer ='adam',
            n_epoch=200,
            callbacks_to_add=['modelcheckpoint', 'earlystopping'],
            model_name=model_name,
            overwrite=(not reuse_checkpoint) and (fold_checkpoint_path is None),
            DISTANCE_MATRIX=DISTANCE_MATRIX,
            NODE_NEIGHBOR=NODE_NEIGHBOR,
            NODE_FEATURE=NODE_FEATURE,
            SPATIAL_MATRIX=SPATIAL_MATRIX,
            ADJ=ADJ,
            n_cell_types=n_cell_types,
            positive_fraction=positive_fraction,
            checkpoint_path=fold_checkpoint_path,
            auto_threshold_by_val_f1=auto_threshold_by_val_f1,
            threshold=threshold,
        )
        
        collect_fold_metrics(results=results, test_metrics=test_metrics)

    macro_avg, macro_std, temp = summarize_metric_results(results=results, temp=temp)

    write_log(format_filename(LOG_DIR, RESULT_LOG["ALL"]), temp, "a")

    print_metric_report(
        temp=temp,
        results=results,
        macro_avg=macro_avg,
        macro_std=macro_std,
        subset_mode=subset_mode,
        fold_indices=fold_indices,
        selected_folds=selected_folds,
        actual_folds=actual_folds,
        cv_folds=cv_folds,
    )

    performance_context = {
        "model_name": model_name,
        "subset_mode": subset_mode,
        "actual_folds": actual_folds,
        "cv_folds": cv_folds,
        "base_n_cell_types": base_n_cell_types,
        "n_cell_types": n_cell_types,
        "excluded_cell_types": excluded_cell_types,
        "n_graphs": n_graphs,
        "n_neighbors": n_neighbors,
        "n_layers": n_layers,
        "embed_dim": embed_dim,
        "dff": dff,
        "num_heads": num_heads,
        "d_sp_enc": d_sp_enc,
        "p": p,
        "q": q,
        "lr": lr,
        "dropout": dropout,
        "loss_mul": loss_mul,
        "bz": bz,
        "l2_weights": l2_weights,
        "spatial": spatial,
        "use_64d_features": use_64d_features,
        "normalize_state_features": normalize_state_features,
        "sanitize_features": sanitize_features,
        "use_cancer_ppi": use_cancer_ppi,
        "feature_subset": feature_subset,
    }
    performance_log = build_performance_log(
        context=performance_context,
        temp=temp,
        results=results,
        macro_avg=macro_avg,
        macro_std=macro_std,
    )
    write_performance_csv(performance_log, model_name=model_name, log_dir=LOG_DIR)


if __name__ == '__main__':
    DEBUG_MODE = False

    args = get_runtime_args(debug_mode=DEBUG_MODE)
    setup_runtime_environment(
        args,
        log_dir=LOG_DIR,
        model_saved_dir=MODEL_SAVED_DIR,
    )
    
    # Generate run manifest for reproducibility
    manifest_path = generate_run_manifest(args, output_dir="history")
    print_manifest_summary(manifest_path)
    
    print_run_configuration(args)

    process_data(
        DATASET=cell_type_ppi,
        n_graphs=args.n_graphs,
        n_neighbors=args.n_neighbors,
        n_layers=args.n_layers,
        lr=args.lr,
        spatial=args.spatial,
        cv_folds=args.cv_folds,
        dropout=args.dropout,
        loss_mul=args.loss_mul,
        dff=args.dff,
        bz=args.bz,
        use_64d_features=args.use_64d_features,
        use_cancer_ppi=args.use_cancer_ppi,
        model_name=args.model_name,
        test_run=args.test_run,
        p=args.p,
        q=args.q,
        n_cell_types=args.n_cell_types,
        data_dir=args.data_dir,
        h5_dir=args.h5_dir,
        sp_dir=args.sp_dir,
        dataset_name=args.dataset_name,
        force_preprocess=args.force_preprocess,
        d_sp_enc=args.d_sp_enc,
        normalize_state_features=args.normalize_state_features,
        num_heads_override=args.num_heads,
        feature_subset=args.feature_subset,
        use_pq_template=args.use_pq_template,
        sanitize_features=getattr(args, 'sanitize_features', False),
        fold_indices=args.fold_idx,
        global_ppi_h5=args.global_ppi_h5,
        reuse_checkpoint=args.reuse_checkpoint,
        checkpoint_path=args.checkpoint_path,
        threshold=args.threshold,
        auto_threshold_by_val_f1=args.auto_threshold_by_val_f1,
        exclude_cell_types=args.exclude_cell_types,
    )
