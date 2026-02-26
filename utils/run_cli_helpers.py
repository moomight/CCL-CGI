import argparse
import os
import random
from types import SimpleNamespace

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="CCL-CGI: Cancer Gene Prediction")

    parser.add_argument("--n_graphs", type=int, default=6, help="Number of subgraphs/channels (default: 6)")
    parser.add_argument("--n_neighbors", type=int, default=8, help="Number of neighbors in random walk (default: 8)")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of layers (default: 3)")
    parser.add_argument("--dff", type=int, default=8, help="Dimension of feed-forward network (default: 8)")
    parser.add_argument("--d_sp_enc", type=int, default=64, help="Dimension of spatial encoding (default: 64)")
    parser.add_argument(
        "--num_heads",
        type=int,
        default=None,
        help="Override number of attention heads (default: auto; 4 for 8d, 8 for 64d)",
    )
    parser.add_argument(
        "--feature_subset",
        type=str,
        default="all",
        choices=["all", "first4", "last4"],
        help="(Deprecated) 4-dim ablation flag kept for compatibility; currently ignored unless re-enabled in code.",
    )

    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate (default: 0.005)")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate (default: 0.5)")
    parser.add_argument("--loss_mul", type=float, default=0.2, help="Loss multiplier (default: 0.2)")
    parser.add_argument("--bz", type=int, default=256, help="Batch size (default: 256)")

    parser.add_argument(
        "--spatial",
        type=str,
        default="rw",
        choices=["rw", "sp"],
        help="Spatial encoding strategy: rw (random walk) or sp (shortest path) (default: rw)",
    )
    parser.add_argument("--cv_folds", type=int, default=10, help="Number of cross-validation folds (default: 10)")

    parser.add_argument(
        "--use_64d_features",
        action="store_true",
        default=False,
        help="Use 64-dimensional feature view (8 + appended 56-d state features when available).",
    )
    parser.add_argument(
        "--use_cancer_ppi",
        action="store_true",
        default=False,
        help="Include cancer-specific PPI network edges (default: False)",
    )
    parser.add_argument(
        "--force_preprocess",
        action="store_true",
        default=False,
        help="Force regeneration of preprocessed data even if cache exists (default: False)",
    )
    parser.add_argument(
        "--normalize_state_features",
        action="store_true",
        default=False,
        help="Normalize appended 56-d state features (log1p for count-like cols + z-score + clip); recommended when --use_64d_features is set.",
    )
    parser.add_argument(
        "--sanitize_features",
        action="store_true",
        default=False,
        help="Replace NaN/Inf values in loaded features with 0 before training (default: False).",
    )
    parser.add_argument(
        "--use_pq_template",
        action="store_true",
        default=False,
        help="Append _p{p}_q{q} suffix when loading/saving subgraph, spatial, and max_degree caches.",
    )

    parser.add_argument("--model_name", type=str, default="CCL_CGI", help="Model name prefix (default: CCL_CGI)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--gpu", type=str, default="3", help="GPU device ID (default: 3)")
    parser.add_argument("--test_run", type=int, default=None, help="For testing: only run first N folds (default: None, run all folds)")
    parser.add_argument(
        "--fold_idx",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of 0-based fold indices to run; overrides test_run count.",
    )
    parser.add_argument(
        "--reuse_checkpoint",
        action="store_true",
        default=False,
        help="Skip training for a fold if its checkpoint already exists, then run test directly.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override default classification threshold (default: use config.py value, 0.5). Ignored when --auto_threshold_by_val_f1 is set.",
    )
    parser.add_argument(
        "--auto_threshold_by_val_f1",
        action="store_true",
        default=False,
        help="Automatically choose classification threshold by maximizing F1 on validation set; default threshold is fixed at 0.5.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional checkpoint path for direct evaluation. Supports `{fold}` placeholder for multi-fold runs.",
    )
    parser.add_argument(
        "--exclude_cell_types",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of cell type names to exclude (e.g., leave-one-cell-type-out).",
    )

    parser.add_argument("--p", type=float, default=1.5, help="Return parameter p for node2vec (default: 1.5)")
    parser.add_argument("--q", type=float, default=1.2, help="Inout parameter q for node2vec (default: 1.2)")
    parser.add_argument("--n_cell_types", type=int, default=39, help="Number of cell types in the dataset (default: 39)")

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory for processed data (pdata). If None, uses ./pdata/<dataset_name> under current project directory",
    )
    parser.add_argument(
        "--h5_dir",
        type=str,
        default=None,
        help="Directory containing H5 files. If None, uses ./h5/<dataset_name> or ./h5_withcancer/<dataset_name> under current project directory",
    )
    parser.add_argument(
        "--sp_dir",
        type=str,
        default=None,
        help="Directory for shortest path data. If None, uses ./sp/<dataset_name> under current project directory",
    )
    parser.add_argument("--dataset_name", type=str, default="CCL-CGI", help="Dataset name prefix (default: CCL-CGI)")
    parser.add_argument(
        "--global_ppi_h5",
        type=str,
        default=None,
        help="Optional override path to global_ppi.h5 (used when dataset has >1 cell types).",
    )

    return parser.parse_args()


def get_debug_args():
    print("=" * 80)
    print("DEBUG MODE: Using predefined configuration")
    print("=" * 80)

    return SimpleNamespace(
        n_graphs=6,
        n_neighbors=8,
        n_layers=3,
        num_heads=4,
        dff=8,
        d_sp_enc=64,
        lr=0.005,
        dropout=0.5,
        loss_mul=0.2,
        bz=256,
        spatial="rw",
        cv_folds=10,
        use_64d_features=False,
        use_cancer_ppi=False,
        force_preprocess=False,
        normalize_state_features=False,
        model_name="CCL_CGI",
        seed=42,
        gpu="0",
        test_run=1,
        p=1.5,
        q=1.2,
        use_pq_template=False,
        fold_idx=None,
        reuse_checkpoint=False,
        checkpoint_path=None,
        threshold=None,
        auto_threshold_by_val_f1=False,
        global_ppi_h5=None,
        n_cell_types=39,
        exclude_cell_types=None,
        data_dir=None,
        h5_dir=None,
        sp_dir=None,
        dataset_name="CCL-CGI",
        feature_subset="all",
        sanitize_features=False,
    )


def get_runtime_args(debug_mode: bool = False):
    return get_debug_args() if debug_mode else parse_args()


def setup_runtime_environment(args, *, log_dir: str, model_saved_dir: str):
    # Import determinism settings from config
    from config import CUDNN_DETERMINISTIC, CUDNN_BENCHMARK
    
    has_cuda = bool(torch.cuda.is_available())
    if has_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        print("⚠️ CUDA is not available. Falling back to CPU execution.")

    seed = args.seed
    
    torch.manual_seed(seed)
    if has_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = CUDNN_DETERMINISTIC
        torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"✓ Random seed: {seed}")
    if args.threshold is not None:
        print(f"✓ Threshold (CLI override): {args.threshold}")
    if has_cuda:
        print(f"✓ cuDNN deterministic: {CUDNN_DETERMINISTIC}")
        print(f"✓ cuDNN benchmark: {CUDNN_BENCHMARK}")

    resolved_data_dir = args.data_dir if args.data_dir is not None else os.path.join(os.getcwd(), "pdata", args.dataset_name)
    resolved_sp_dir = args.sp_dir if args.sp_dir is not None else os.path.join(os.getcwd(), "sp", args.dataset_name)

    os.makedirs(resolved_data_dir, exist_ok=True)
    os.makedirs(resolved_sp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_saved_dir, exist_ok=True)


def print_run_configuration(args):
    from config import CUDNN_DETERMINISTIC, CUDNN_BENCHMARK
    
    print("=" * 80)
    print("CCL-CGI Configuration")
    print("=" * 80)
    print("Model Parameters:")
    print(f"  n_graphs: {args.n_graphs}")
    print(f"  n_neighbors: {args.n_neighbors}")
    print(f"  n_layers: {args.n_layers}")
    print(f"  dff: {args.dff}")
    print("\nTraining Parameters:")
    print(f"  learning_rate: {args.lr}")
    print(f"  dropout: {args.dropout}")
    print(f"  loss_mul: {args.loss_mul}")
    print(f"  batch_size: {args.bz}")
    if args.num_heads is not None:
        print(f"  num_heads (override): {args.num_heads}")

    print("\nExperiment Parameters:")
    print(f"  spatial: {args.spatial}")
    print(f"  cv_folds: {args.cv_folds}")
    if args.test_run is not None:
        print(f"  test_run: {args.test_run} (🔧 Only running first {args.test_run} fold(s))")
    if args.fold_idx is not None:
        print(f"  fold_idx: {args.fold_idx} (🎯 run specified fold(s) only)")
    print(f"  reuse_checkpoint: {args.reuse_checkpoint}")
    if args.checkpoint_path is not None:
        print(f"  checkpoint_path: {args.checkpoint_path}")
    print(f"  auto_threshold_by_val_f1: {args.auto_threshold_by_val_f1}")
    if args.threshold is not None:
        print(f"  threshold (CLI override): {args.threshold}")
    print(f"  seed: {args.seed}")
    print(f"  gpu: {args.gpu}")

    print("\nReproducibility Settings (from config.py):")
    print(f"  random_seed: {args.seed}")
    print(f"  cudnn_deterministic: {CUDNN_DETERMINISTIC}")
    print(f"  cudnn_benchmark: {CUDNN_BENCHMARK}")

    print("\nData Processing Parameters:")
    print(f"  use_64d_features: {args.use_64d_features}")
    print(f"  use_cancer_ppi: {args.use_cancer_ppi}")
    print(f"  p: {args.p}, q: {args.q}, use_pq_template: {args.use_pq_template}")
    if args.use_64d_features:
        print(f"  normalize_state_features: {args.normalize_state_features}")
        print(f"  sanitize_features: {getattr(args, 'sanitize_features', False)}")
    else:
        print(f"  feature_subset: {args.feature_subset}")

    print("\nModel Configuration:")
    print(f"  model_name: {args.model_name}")
    print(f"  n_cell_types: {args.n_cell_types}")
    print("=" * 80)
