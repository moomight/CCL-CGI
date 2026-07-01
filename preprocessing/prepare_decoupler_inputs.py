#!/usr/bin/env python
"""Prepare DecoupleR preprocessing inputs from sample-level expression matrices.

The expected input expression matrix is gene x cell for one sample. Cell types
are mixed at this stage. This script provides three steps:

1. convert: convert raw sample-level gene-by-cell expression matrices into compressed pkl.bz2 matrices.
2. annotate: generate per-sample metaInfo files containing cell IDs, malignancy labels, malignancy scores, and cell-type labels. 
If cell-type and malignancy annotations are already available, this step can be skipped.
The provided annotation files must follow the same metaInfo format produced by this step.
3. extract: use the metaInfo files to split sample-level expression matrices into cell-type-specific malignant and non-malignant expression matrices.
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def context_from_path(path: Path) -> str:
    name = path.name
    for suffix in (".pkl.bz2", ".pkl.gz", ".pickle", ".pkl", ".tsv.gz", ".tsv", ".csv.gz", ".csv"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def list_matrices(matrix_dir: Path, pattern: str) -> List[Path]:
    paths = [path for path in sorted(matrix_dir.glob(pattern)) if path.is_file()]
    if not paths:
        raise FileNotFoundError(f"No matrices found in {matrix_dir} with pattern {pattern}")
    return paths


def read_expression(path: Path) -> pd.DataFrame:
    if path.name.endswith((".pkl", ".pkl.bz2", ".pkl.gz", ".pickle")):
        return pd.read_pickle(path, compression="infer")
    sep = "\t" if ".tsv" in path.name else ","
    return pd.read_csv(path, sep=sep, index_col=0)


def normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(("true", "1", "yes", "y"))


def cmd_convert(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_matrix_dir)
    out_dir = Path(args.out_matrix_dir)
    save_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    cell_to_id: Dict[str, int] = {}
    genes = set()
    for path in list_matrices(raw_dir, args.pattern):
        context = context_from_path(path)
        print(f"[convert] reading {path}")
        df = read_expression(path)
        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)
        genes.update(df.index.tolist())
        for cell_id in df.columns:
            if cell_id not in cell_to_id:
                cell_to_id[cell_id] = len(cell_to_id) + 1
        out_path = out_dir / f"{context}.pkl.bz2"
        df.to_pickle(out_path, compression="bz2")
        print(f"[convert] wrote {out_path} shape={df.shape}")

    gene_to_id = {gene: i + 1 for i, gene in enumerate(sorted(genes))}
    with open(save_dir / "cell_id_dict.pkl", "wb") as w:
        pickle.dump(cell_to_id, w)
    with open(save_dir / "gene_id_dict.pkl", "wb") as w:
        pickle.dump(gene_to_id, w)
    pd.Series(cell_to_id).to_csv(save_dir / "cell_id.txt", sep=" ", header=False)
    pd.Series(gene_to_id).to_csv(save_dir / "gene_id.txt", sep=" ", header=False)
    print(f"[convert] cells={len(cell_to_id)} genes={len(gene_to_id)}")


def cmd_annotate(args: argparse.Namespace) -> None:
    import anndata as ad
    import decoupler as dc
    import scanpy as sc
    from scipy import sparse
    from xgboost import XGBRegressor

    matrix_dir = Path(args.matrix_dir)
    meta_dir = Path(args.meta_dir)
    save_dir = Path(args.save_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    reg = XGBRegressor(objective="binary:logistic", n_estimators=100)
    reg.load_model(args.sc_cancer_model)
    selected_genes = pd.read_csv(args.sc_cancer_genes, sep="\t", index_col=0).iloc[:, 0].astype(str).tolist()

    if args.panglaodb_pickle:
        markers = pd.read_pickle(args.panglaodb_pickle)
    else:
        markers = dc.get_resource("PanglaoDB")
    markers = markers[
        markers["human"]
        & markers["canonical_marker"]
        & (markers["human_sensitivity"] > args.marker_sensitivity)
    ]
    markers = markers[~markers.duplicated(["cell_type", "genesymbol"])]

    cell_type_seen = []
    cell_type_dict = defaultdict(lambda: defaultdict(lambda: {"malignant": [], "non-malignant": []}))

    for path in list_matrices(matrix_dir, args.pattern):
        context = context_from_path(path)
        out_csv = meta_dir / f"{context}_metaInfo.csv"
        if out_csv.exists() and not args.overwrite:
            print(f"[annotate] exists, skip {out_csv}")
            meta = pd.read_csv(out_csv)
        else:
            print(f"[annotate] reading {path}")
            counts = read_expression(path)
            counts.index = counts.index.astype(str)
            counts.columns = counts.columns.astype(str)
            adata = ad.AnnData(counts.T)
            adata.obs_names = adata.obs_names.astype(str)
            obs_names = adata.obs_names.tolist()

            print(f"[annotate] malignancy {context}")
            malignancy_score, malignancy_label = predict_malignancy(
                adata.copy(), reg=reg, selected_genes=selected_genes, threshold=args.malignancy_threshold
            )

            print(f"[annotate] DecoupleR cell type {context}")
            cell_type = annotate_cell_type_decoupler(adata.copy(), sc=sc, dc=dc, markers=markers)

            meta = pd.DataFrame(
                {
                    "cell_id": obs_names,
                    "malignancy": np.asarray(malignancy_label, dtype=bool),
                    "malignancy_score": np.asarray(malignancy_score, dtype=float),
                    "cell_type": np.asarray(cell_type, dtype=object),
                }
            )
            meta.to_csv(out_csv, index=False)
            print(f"[annotate] wrote {out_csv} rows={len(meta)}")

        meta["malignancy"] = normalize_bool(meta["malignancy"])
        for row in meta.itertuples(index=False):
            ct = str(row.cell_type)
            if ct not in cell_type_seen:
                cell_type_seen.append(ct)
            tag = "malignant" if bool(row.malignancy) else "non-malignant"
            cell_type_dict[ct][context][tag].append(str(row.cell_id))

    with open(save_dir / "cell_type_list.pickle", "wb") as w:
        pickle.dump(cell_type_seen, w)
    with open(save_dir / "cell_type_dict.pickle", "wb") as w:
        pickle.dump(dict(cell_type_dict), w)
    with open(save_dir / "malignancy.txt", "w", encoding="utf-8") as w:
        for ct, by_context in cell_type_dict.items():
            m = sum(len(v["malignant"]) for v in by_context.values())
            n = sum(len(v["non-malignant"]) for v in by_context.values())
            w.write(f"{ct}\t{m}\t{n}\n")


def predict_malignancy(adata, reg, selected_genes: Sequence[str], threshold: float):
    import scanpy as sc
    from scipy import sparse

    data = adata
    data.var_names_make_unique()
    sc.pp.filter_genes(data, min_cells=3)
    mito_genes = data.var_names.str.startswith("MT-")
    total = np.ravel(np.sum(data.X, axis=1))
    mito = np.ravel(np.sum(data[:, mito_genes].X, axis=1))
    data.obs["percent_mito"] = np.divide(mito, total, out=np.zeros_like(mito, dtype=float), where=total != 0)
    data.obs["n_counts"] = total
    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    sc.pp.regress_out(data, ["n_counts", "percent_mito"])

    matrix = sparse.csr_matrix(data.X).todense()
    expr = pd.DataFrame(data=matrix, index=np.asarray(data.obs.index), columns=data.var.index)
    pred_mtx = pd.DataFrame(np.zeros((expr.shape[0], len(selected_genes))), index=expr.index)
    gene_to_col = {gene: i for i, gene in enumerate(data.var.index)}
    for i, gene in enumerate(selected_genes):
        j = gene_to_col.get(gene)
        if j is not None:
            pred_mtx.iloc[:, i] = expr.iloc[:, j].values
    score = reg.predict(pred_mtx)
    return score, score > threshold


def annotate_cell_type_decoupler(adata, sc, dc, markers: pd.DataFrame) -> List[str]:
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    adata = adata[adata.obs.pct_counts_mt < 5, :]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["log_norm"] = adata.X.copy()
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"])
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack")
    dc.swap_layer(adata, "log_norm", X_layer_key=None, inplace=True)
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)

    dc.run_ora(mat=adata, net=markers, source="cell_type", target="genesymbol", min_n=3, verbose=False, use_raw=False)
    acts = dc.get_acts(adata, obsm_key="ora_estimate")
    vals = acts.X.ravel()
    finite = vals[np.isfinite(vals)]
    if finite.size:
        acts.X[~np.isfinite(acts.X)] = np.nanmax(finite)
    df = dc.rank_sources_groups(acts, groupby="leiden", reference="rest", method="t-test_overestim_var")
    annotation = df.groupby("group").head(1).set_index("group")["names"].to_dict()
    return [annotation[cluster] for cluster in adata.obs["leiden"]]


def read_meta(meta_path: Path) -> pd.DataFrame:
    meta = pd.read_csv(meta_path)
    required = {"cell_id", "malignancy", "cell_type"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"{meta_path} missing required columns: {sorted(missing)}")
    meta["cell_id"] = meta["cell_id"].astype(str)
    meta["cell_type"] = meta["cell_type"].astype(str)
    meta["malignancy"] = normalize_bool(meta["malignancy"])
    return meta


def cmd_extract(args: argparse.Namespace) -> None:
    matrix_dir = Path(args.matrix_dir)
    meta_dir = Path(args.meta_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_paths = {context_from_path(path): path for path in list_matrices(matrix_dir, args.pattern)}
    meta_by_context = {}
    for context in matrix_paths:
        meta_path = meta_dir / f"{context}_metaInfo.csv"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metaInfo for {context}: {meta_path}")
        meta_by_context[context] = read_meta(meta_path)

    if args.cell_types:
        cell_types = args.cell_types
    else:
        cell_types = sorted({ct for meta in meta_by_context.values() for ct in meta["cell_type"].unique()})

    for cell_type in cell_types:
        print(f"[extract] {cell_type}")
        buffers = {True: [], False: []}
        for context, matrix_path in matrix_paths.items():
            meta = meta_by_context[context]
            meta = meta[meta["cell_type"] == cell_type]
            if meta.empty:
                continue
            counts = read_expression(matrix_path)
            counts.index = counts.index.astype(str)
            counts.columns = counts.columns.astype(str)
            for is_malignant in (True, False):
                cells = meta.loc[meta["malignancy"] == is_malignant, "cell_id"].tolist()
                cells = [cell for cell in cells if cell in counts.columns]
                if not cells:
                    continue
                sub = counts.loc[:, cells].copy()
                sub.columns = [f"{cell}_{context}" for cell in sub.columns]
                buffers[is_malignant].append(sub)
        for is_malignant, frames in buffers.items():
            tag = "malignant" if is_malignant else "non-malignant"
            if not frames:
                print(f"[extract] skip {cell_type} {tag}: no cells")
                continue
            df = pd.concat(frames, axis=1).fillna(0).sort_index()
            safe = re.sub(r"[\\/:]+", "_", cell_type).strip()
            out_path = out_dir / f"{safe}_{tag}_expression_matrix.csv"
            df.to_csv(out_path)
            print(f"[extract] wrote {out_path} shape={df.shape}")


def add_matrix_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pattern", default="*.pkl.bz2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare DecoupleR preprocessing inputs from sample-level matrices.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("convert", help="Convert raw sample-level gene x cell matrices to pkl.bz2.")
    p.add_argument("--raw-matrix-dir", required=True)
    p.add_argument("--out-matrix-dir", required=True)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--pattern", default="*.tsv")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("annotate", help="Generate per-sample metaInfo using scCancer and DecoupleR/PanglaoDB.")
    p.add_argument("--matrix-dir", required=True)
    p.add_argument("--meta-dir", required=True)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--sc-cancer-model", required=True)
    p.add_argument("--sc-cancer-genes", required=True)
    p.add_argument("--panglaodb-pickle", default=None)
    p.add_argument("--marker-sensitivity", type=float, default=0.5)
    p.add_argument("--malignancy-threshold", type=float, default=0.5)
    p.add_argument("--overwrite", action="store_true")
    add_matrix_args(p)
    p.set_defaults(func=cmd_annotate)

    p = sub.add_parser("extract", help="Extract cell-type expression matrices split by malignancy.")
    p.add_argument("--matrix-dir", required=True)
    p.add_argument("--meta-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cell-types", nargs="*", default=None)
    add_matrix_args(p)
    p.set_defaults(func=cmd_extract)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
