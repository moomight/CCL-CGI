#!/usr/bin/env python
"""Build aligned H5 files for the original DecoupleR preprocessing output."""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FEATURE_NAMES = np.asarray(
    [
        "malignant:score",
        "malignant:pval",
        "malignant:pvals_adj",
        "malignant:logfoldchange",
        "non-malignant:score",
        "non-malignant:pval",
        "non-malignant:pvals_adj",
        "non-malignant:logfoldchange",
    ],
    dtype="S",
)


def safe_name(value: str) -> str:
    return re.sub(r"[\\/:]+", "_", str(value)).strip()


def load_cell_type_specs(mapping_json: str | None = None) -> Dict[str, Dict[str, object]]:
    """Return target cell type -> PINNACLE/PPI files used by all H5 build steps."""
    if mapping_json is None:
        raise ValueError("A cell-type-to-PINNACLE-PPI mapping JSON is required.")

    with open(mapping_json, "r", encoding="utf-8") as r:
        raw = json.load(r)

    if not isinstance(raw, dict):
        raise ValueError(f"{mapping_json}: expected a JSON object")
    if "cell_type_to_pinnacle_ppi" in raw:
        raw = raw["cell_type_to_pinnacle_ppi"]
        if not isinstance(raw, dict):
            raise ValueError(f"{mapping_json}: 'cell_type_to_pinnacle_ppi' must be a JSON object")
    specs: Dict[str, Dict[str, object]] = {}
    for cell_type, value in raw.items():
        if isinstance(value, str):
            ppi_files = [value]
        elif isinstance(value, list):
            ppi_files = value
        elif isinstance(value, dict) and "ppi_files" in value:
            ppi_files = value["ppi_files"]
        else:
            raise ValueError(
                f"{mapping_json}: {cell_type!r} must map to a PPI filename string, "
                "a list of PPI files, or an object with key 'ppi_files'"
            )
        specs[str(cell_type)] = {"ppi_files": list(ppi_files)}
    return specs


def decode(values) -> List[str]:
    out = []
    for value in values:
        if isinstance(value, (bytes, np.bytes_)):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def build_graph(ppi_dir: Path, ppi_files: Iterable[str]):
    import networkx as nx

    graph = nx.Graph()
    for name in ppi_files:
        path = ppi_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing PINNACLE/PPI edgelist: {path}")
        graph = nx.compose(graph, nx.read_edgelist(path))
    return graph


def all_ppi_files(specs: Dict[str, Dict[str, object]]) -> List[str]:
    files = []
    for spec in specs.values():
        files.extend(spec["ppi_files"])
    return files


def write_global_ppi(args: argparse.Namespace) -> None:
    import h5py

    h5_dir = Path(args.h5_dir)
    h5_dir.mkdir(parents=True, exist_ok=True)
    specs = load_cell_type_specs(args.cell_type_ppi_json)
    graph = build_graph(Path(args.ppi_dir), all_ppi_files(specs))

    gene_names = sorted(map(str, graph.nodes()))
    gene_to_pos = {g: i for i, g in enumerate(gene_names)}
    n = len(gene_names)
    adj = np.zeros((n, n), dtype=np.uint8)
    for u, v in graph.edges():
        i = gene_to_pos[str(u)]
        j = gene_to_pos[str(v)]
        adj[i, j] = 1
        adj[j, i] = 1

    with open(args.cancer_genes, "rb") as r:
        cancer_genes = set(pickle.load(r))
    with open(args.non_cancer_genes, "rb") as r:
        non_cancer_genes = set(pickle.load(r))

    y_all = np.zeros(n, dtype=bool)
    labeled = []
    for i, gene in enumerate(gene_names):
        if gene in cancer_genes:
            y_all[i] = True
            labeled.append(i)
        elif gene in non_cancer_genes:
            labeled.append(i)

    rng = np.random.default_rng(args.seed)
    labeled = np.asarray(labeled, dtype=np.int64)
    rng.shuffle(labeled)
    train_n = int(len(labeled) * 27 / 40)
    val_n = int(len(labeled) * 3 / 40)
    train_idx = labeled[:train_n]
    val_idx = labeled[train_n : train_n + val_n]
    test_idx = labeled[train_n + val_n :]

    def make_mask(indices: np.ndarray) -> np.ndarray:
        mask = np.zeros(n, dtype=bool)
        mask[indices] = True
        return mask

    train_mask = make_mask(train_idx)
    val_mask = make_mask(val_idx)
    test_mask = make_mask(test_idx)

    out_path = h5_dir / "global_ppi.h5"
    with h5py.File(out_path, "w") as f:
        f.create_dataset("gene_names", data=np.asarray(gene_names, dtype="S"))
        f.create_dataset("network", data=adj, compression="gzip", compression_opts=4, chunks=True)
        f.create_dataset("train_mask", data=train_mask)
        f.create_dataset("val_mask", data=val_mask)
        f.create_dataset("test_mask", data=test_mask)
        f.create_dataset("y_train", data=np.where(train_mask, y_all, False))
        f.create_dataset("y_val", data=np.where(val_mask, y_all, False))
        f.create_dataset("y_test", data=np.where(test_mask, y_all, False))
    print(f"[global] wrote {out_path} genes={n} labeled={len(labeled)}")


def load_global_genes(global_h5: Path) -> List[str]:
    import h5py

    with h5py.File(global_h5, "r") as f:
        return decode(f["gene_names"][:])


def expression_paths(expression_dir: Path, cell_type: str) -> Tuple[Path, Path]:
    base = safe_name(cell_type)
    return (
        expression_dir / f"{base}_malignant_expression_matrix.csv",
        expression_dir / f"{base}_non-malignant_expression_matrix.csv",
    )


def extract_de_features(adata, gene_order: Sequence[str]) -> np.ndarray:
    rg = adata.uns["rank_genes_groups"]
    ranked = rg["names"]
    scores = rg["scores"]
    pvals = rg["pvals"]
    pvals_adj = rg["pvals_adj"]
    logfc = rg.get("logfoldchanges", None)
    available = set(ranked.dtype.names or [])

    features = np.zeros((len(gene_order), 8), dtype=np.float32)
    for row, gene in enumerate(gene_order):
        for offset, group in ((0, "malignant"), (4, "non-malignant")):
            if group not in available:
                continue
            names = ranked[group].astype(str)
            hit = np.where(names == gene)[0]
            if hit.size == 0:
                continue
            i = int(hit[0])
            features[row, offset + 0] = float(scores[group][i])
            features[row, offset + 1] = float(pvals[group][i])
            features[row, offset + 2] = float(pvals_adj[group][i])
            features[row, offset + 3] = float(logfc[group][i]) if logfc is not None else 0.0
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def write_celltype_h5(args: argparse.Namespace) -> None:
    import anndata as ad
    import h5py
    import scanpy as sc

    expression_dir = Path(args.expression_dir)
    h5_dir = Path(args.h5_dir)
    h5_dir.mkdir(parents=True, exist_ok=True)
    global_h5 = Path(args.global_h5) if args.global_h5 else h5_dir / "global_ppi.h5"
    global_genes = load_global_genes(global_h5)
    gene_to_pos = {g: i for i, g in enumerate(global_genes)}
    global_set = set(gene_to_pos)
    n_all = len(global_genes)
    specs = load_cell_type_specs(args.cell_type_ppi_json)

    for cell_type, spec in specs.items():
        malignant_path, non_path = expression_paths(expression_dir, cell_type)
        if not malignant_path.exists() or not non_path.exists():
            msg = f"missing expression CSVs for {cell_type}: {malignant_path}, {non_path}"
            if args.skip_missing:
                print(f"[celltype] skip {msg}")
                continue
            raise FileNotFoundError(msg)

        print(f"[celltype] {cell_type}")
        malignant = pd.read_csv(malignant_path, index_col=0)
        non_malignant = pd.read_csv(non_path, index_col=0)
        malignant.index = malignant.index.astype(str)
        non_malignant.index = non_malignant.index.astype(str)

        adata_m = ad.AnnData(
            malignant.T.values,
            obs=pd.DataFrame(index=malignant.columns.astype(str)),
            var=pd.DataFrame(index=malignant.index),
        )
        adata_n = ad.AnnData(
            non_malignant.T.values,
            obs=pd.DataFrame(index=non_malignant.columns.astype(str)),
            var=pd.DataFrame(index=non_malignant.index),
        )
        adata_m.obs["cell_type"] = pd.Categorical(["malignant"] * adata_m.n_obs)
        adata_n.obs["cell_type"] = pd.Categorical(["non-malignant"] * adata_n.n_obs)
        adata = ad.concat([adata_m, adata_n], join="outer", merge="same")

        graph = build_graph(Path(args.ppi_dir), spec["ppi_files"])
        keep = sorted(
            (set(map(str, adata.var_names)) & set(map(str, graph.nodes())) & global_set),
            key=lambda gene: gene_to_pos[gene],
        )
        if not keep:
            print(f"[celltype] skip {cell_type}: empty expression/PPI/global intersection")
            continue

        adata_sub = adata[:, keep].copy()
        if args.normalize_log1p:
            sc.pp.normalize_total(adata_sub, target_sum=1e4)
            sc.pp.log1p(adata_sub)
        sc.tl.rank_genes_groups(adata_sub, "cell_type", method="wilcoxon")

        gene_order = sorted(adata_sub.var_names.astype(str).tolist(), key=lambda gene: gene_to_pos[gene])
        idx = np.asarray([gene_to_pos[gene] for gene in gene_order], dtype=np.int64)
        features = extract_de_features(adata_sub, gene_order)

        out_path = h5_dir / f"{safe_name(cell_type)}{args.output_suffix}.h5"
        with h5py.File(out_path, "w") as f:
            f.create_dataset("feature_names", data=FEATURE_NAMES)
            f.create_dataset("gene_names", data=np.asarray(gene_order, dtype="S"))
            f.create_dataset("gene_names_all", data=np.asarray(global_genes, dtype="S"))
            f.create_dataset("idx", data=idx)
            f.create_dataset("features", data=features, compression="gzip", compression_opts=4, chunks=True)
            f.create_dataset("features_raw", data=features, compression="gzip", compression_opts=4, chunks=True)
            net = f.create_dataset(
                "network",
                shape=(n_all, n_all),
                dtype=np.uint8,
                compression="gzip",
                compression_opts=4,
                chunks=(min(1024, n_all), min(1024, n_all)),
                fillvalue=0,
            )
            by_row: Dict[int, List[int]] = defaultdict(list)
            for u, v in graph.edges():
                u = str(u)
                v = str(v)
                if u not in gene_to_pos or v not in gene_to_pos:
                    continue
                i = gene_to_pos[u]
                j = gene_to_pos[v]
                by_row[i].append(j)
                by_row[j].append(i)
            for row, cols in by_row.items():
                net[row, np.unique(np.asarray(cols, dtype=np.int64))] = 1
        print(f"[celltype] wrote {out_path} genes={len(gene_order)}")


def celltype_h5_paths(h5_dir: Path, suffix: str, specs: Dict[str, Dict[str, object]]) -> Tuple[List[str], List[Path]]:
    names, paths = [], []
    for cell_type in specs:
        path = h5_dir / f"{safe_name(cell_type)}{suffix}.h5"
        if path.exists():
            names.append(safe_name(cell_type))
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No cell-type H5 files found in {h5_dir}")
    return names, paths


def write_baseline(args: argparse.Namespace) -> None:
    import h5py

    h5_dir = Path(args.h5_dir)
    global_h5 = Path(args.global_h5) if args.global_h5 else h5_dir / "global_ppi.h5"
    specs = load_cell_type_specs(args.cell_type_ppi_json)
    cell_types, paths = celltype_h5_paths(h5_dir, args.input_suffix, specs)

    with h5py.File(global_h5, "r") as g:
        global_gene_raw = g["gene_names"][:]
        global_genes = np.asarray(decode(global_gene_raw), dtype=object)
        network = g["network"][:]
        labels_masks = {key: g[key][:] for key in ("y_train", "y_val", "y_test", "train_mask", "val_mask", "test_mask") if key in g}

    d = args.feature_dim
    features = np.zeros((len(global_genes), len(paths) * d), dtype=np.float32)
    for i, (cell_type, path) in enumerate(zip(cell_types, paths)):
        with h5py.File(path, "r") as f:
            idx = np.asarray(f["idx"][:], dtype=np.int64)
            gene_names = np.asarray(decode(f["gene_names"][:]), dtype=object)
            if not np.array_equal(gene_names, global_genes[idx]):
                raise ValueError(f"{path}: gene_names do not match global_ppi.h5/gene_names[idx]")
            if "gene_names_all" in f:
                gene_names_all = np.asarray(decode(f["gene_names_all"][:]), dtype=object)
                if not np.array_equal(gene_names_all, global_genes):
                    raise ValueError(f"{path}: gene_names_all does not match global_ppi.h5/gene_names")
            features[idx, i * d : (i + 1) * d] = np.nan_to_num(f["features"][:, :d], nan=0.0)
        print(f"[baseline] added {cell_type}")

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as out:
        out.create_dataset("gene_names", data=global_gene_raw)
        out.create_dataset("cell_types", data=np.asarray(cell_types, dtype="S"))
        out.create_dataset("feature_names", data=np.asarray([f"{ct}__f{j}" for ct in cell_types for j in range(d)], dtype="S"))
        out.create_dataset("features", data=features, compression="gzip", compression_opts=4, chunks=True)
        out.create_dataset("network", data=network, compression="gzip", compression_opts=4, chunks=True)
        for key, value in labels_masks.items():
            out.create_dataset(key, data=value)
    print(f"[baseline] wrote {out_path} features={features.shape}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build aligned DecoupleR preprocessing H5 files.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("global", help="Build global_ppi.h5 from PINNACLE edgelists.")
    p.add_argument("--ppi-dir", required=True)
    p.add_argument("--h5-dir", required=True)
    p.add_argument("--cancer-genes", default=str(REPO_ROOT / "preprocessing" / "input" / "cancer_genes.pkl"))
    p.add_argument("--non-cancer-genes", default=str(REPO_ROOT / "preprocessing" / "input" / "non_cancer_genes.pkl"))
    p.add_argument(
        "--cell-type-ppi-json",
        dest="cell_type_ppi_json",
        required=True,
        help="Cell-type-to-PINNACLE-PPI mapping JSON.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=write_global_ppi)

    p = sub.add_parser("celltype", help="Build aligned cell-type H5 files from expression CSVs.")
    p.add_argument("--expression-dir", required=True)
    p.add_argument("--ppi-dir", required=True)
    p.add_argument("--h5-dir", required=True)
    p.add_argument("--global-h5", default=None)
    p.add_argument(
        "--cell-type-ppi-json",
        dest="cell_type_ppi_json",
        required=True,
        help="Cell-type-to-PINNACLE-PPI mapping JSON.",
    )
    p.add_argument("--output-suffix", default="")
    p.add_argument("--skip-missing", action="store_true")
    p.add_argument("--normalize-log1p", action="store_true")
    p.set_defaults(func=write_celltype_h5)

    p = sub.add_parser("baseline", help="Build baseline_data.h5 by scattering cell-type features through idx.")
    p.add_argument("--h5-dir", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--global-h5", default=None)
    p.add_argument(
        "--cell-type-ppi-json",
        dest="cell_type_ppi_json",
        required=True,
        help="Cell-type-to-PINNACLE-PPI mapping JSON.",
    )
    p.add_argument("--input-suffix", default="")
    p.add_argument("--feature-dim", type=int, default=8)
    p.set_defaults(func=write_baseline)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
