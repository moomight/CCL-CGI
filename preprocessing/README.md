# Preprocessing Pipeline

This directory contains the preprocessing code for building a CCL-CGI-format dataset from sample-level single-cell expression matrices and cell-type-specific PPI networks. It can be used to generate H5 files with the same layout as the original DecoupleR-based dataset.

## Input Files

Use these repository-relative locations by default:

- Sample-level expression matrices: `preprocessing/input/expression_matrices/`
- scCancer model and selected-gene list: `preprocessing/input/scCancer/`
- PINNACLE-derived PPI edgelists: `networks/ppi_edgelists/`
- Cell-type-to-PINNACLE-PPI mapping JSON: `preprocessing/cell_type_to_pinnacle_ppi.json`
- Cancer gene labels: `preprocessing/input/cancer_genes.pkl` and `preprocessing/input/non_cancer_genes.pkl`

## Generated Files

The preprocessing scripts write intermediate and final files to these repository-relative locations by default:

- Converted sample matrices: `preprocessing/work/matrices/`
- Per-sample metadata: `preprocessing/work/metaInfo/`
- Cell-type expression matrices: `preprocessing/work/cell_type_expression/`
- Final H5 files: `h5/CCL-CGI/`

Each sample-level expression matrix should be a gene-by-cell matrix containing all cells from one sample. Cell types are mixed at this stage. The PPI directory must contain the edgelist files referenced by `cell_type_to_pinnacle_ppi.json`, for example `B_cells_ppi.txt`, `Fibroblasts_ppi.txt`, and `Smooth_muscle_cells_ppi.txt`.

This pipeline does not automatically download scCancer2. By default, the annotation step expects the malignancy classifier files here:

- `preprocessing/input/scCancer/sc_xgboost.model`
- `preprocessing/input/scCancer/genes-scRNA-tcga-sorted.txt`

Provide a JSON mapping file instead of editing Python config. Start from `preprocessing/cell_type_to_pinnacle_ppi.example.json`: each key is a processed cell type, and each value is one or more PINNACLE subtype PPI edgelists to merge for that cell type. The format is:

```json
{
  "B cells": ["B_cells_ppi.txt"],
  "Fibroblasts": [
    "Fibroblasts_ppi.txt"
  ]
}
```

The JSON keys must match the processed cell-type names used in the extracted expression matrices. The PPI filenames are resolved relative to `--ppi-dir`. Matrix filenames are generated automatically from the cell type, for example `B cells_malignant_expression_matrix.csv` and `B cells_non-malignant_expression_matrix.csv`. `build_decoupler_h5.py` requires this mapping so the PPI networks can be tied back to the expression-derived cell types.

## Expected Input Matrices

The full pipeline starts from sample-level expression matrices. Each input file should contain one sample:

Each CSV should be a gene-by-cell matrix:

- rows: gene symbols
- columns: cell IDs
- values: expression values

After annotation, the pipeline writes cell-type expression matrices split by malignancy. For every cell type listed in `cell_type_to_pinnacle_ppi.json`, the H5 builder consumes the automatically named malignant and non-malignant expression matrices.

## Preprocessing Steps

1. `convert`: convert raw sample-level gene-by-cell expression matrices into compressed `pkl.bz2` matrices.
2. `annotate`: generate per-sample `metaInfo` files containing cell IDs, malignancy labels, malignancy scores, and cell-type labels.
   If cell-type and malignancy annotations are already available, this step can be skipped. The provided annotation files must follow the same `metaInfo` format produced by this step.
3. `extract`: use the `metaInfo` files to split sample-level expression matrices into cell-type-specific malignant and non-malignant expression matrices.

## Run Order

From the repository root:

```bash
python preprocessing/prepare_decoupler_inputs.py convert --raw-matrix-dir preprocessing/input/expression_matrices --out-matrix-dir preprocessing/work/matrices --save-dir preprocessing/work/save --pattern '*.tsv'

python preprocessing/prepare_decoupler_inputs.py annotate --matrix-dir preprocessing/work/matrices --meta-dir preprocessing/work/metaInfo --save-dir preprocessing/work/save --sc-cancer-model preprocessing/input/scCancer/sc_xgboost.model --sc-cancer-genes preprocessing/input/scCancer/genes-scRNA-tcga-sorted.txt

python preprocessing/prepare_decoupler_inputs.py extract --matrix-dir preprocessing/work/matrices --meta-dir preprocessing/work/metaInfo --out-dir preprocessing/work/cell_type_expression

python preprocessing/build_decoupler_h5.py global --ppi-dir networks/ppi_edgelists --h5-dir h5/CCL-CGI --cell-type-ppi-json preprocessing/cell_type_to_pinnacle_ppi.json
python preprocessing/build_decoupler_h5.py celltype --expression-dir preprocessing/work/cell_type_expression --ppi-dir networks/ppi_edgelists --h5-dir h5/CCL-CGI --cell-type-ppi-json preprocessing/cell_type_to_pinnacle_ppi.json --skip-missing
python preprocessing/build_decoupler_h5.py baseline --h5-dir h5/CCL-CGI --output-path h5/CCL-CGI/baseline_data.h5 --cell-type-ppi-json preprocessing/cell_type_to_pinnacle_ppi.json
```

`prepare_decoupler_inputs.py` handles the `convert`, `annotate`, and `extract` steps. `build_decoupler_h5.py` writes aligned H5 files directly.

## Output Contract

All generated cell-type H5 files use the same global gene coordinate system:

- `global_ppi.h5/gene_names` defines the global gene order.
- Each cell-type H5 stores `gene_names_all`, identical to `global_ppi.h5/gene_names`.
- Each feature row maps to global position `idx[row]`.
- `gene_names == gene_names_all[idx]`.
- `network` is global-size and follows `gene_names_all` row/column order.

This is the same alignment convention used by the fixed NSCLC preprocessing output.

## References

```
@article{li2024contextual,
  title={Contextual AI models for single-cell protein biology},
  author={Li, Michelle M and Huang, Yepeng and Sumathipala, Marissa and Liang, Man Qing and Valdeolivas, Alberto and Ananthakrishnan, Ashwin N and Liao, Katherine and Marbach, Daniel and Zitnik, Marinka},
  journal={Nature Methods},
  volume={21},
  number={8},
  pages={1546--1557},
  year={2024},
  doi={10.1038/s41592-024-02341-3},
  publisher={Nature Publishing Group US New York}
}

@article{chen2024sccancer2,
  title={scCancer2: data-driven in-depth annotations of the tumor microenvironment at single-level resolution},
  author={Chen, Zeyu and Miao, Yuxin and Tan, Zhiyuan and Hu, Qifan and Wu, Yanhong and Li, Xinqi and Guo, Wenbo and Gu, Jin},
  journal={Bioinformatics},
  volume={40},
  number={2},
  pages={btae028},
  year={2024},
  doi={10.1093/bioinformatics/btae028},
  publisher={Oxford University Press}
}

@article{badiaimompel2022decoupler,
  title={decoupleR: ensemble of computational methods to infer biological activities from omics data},
  author={Badia-i-Mompel, Pau and V{\'e}lez Santiago, Jes{\'u}s and Braunger, Jana and Geiss, Cornelia and Dimitrov, Daniel and M{\"u}ller-Dott, Sophia and Taus, Petr and Dugourd, Aurelien and Holland, Christian H and Ramirez Flores, Ricardo O and Saez-Rodriguez, Julio},
  journal={Bioinformatics Advances},
  volume={2},
  number={1},
  pages={vbac016},
  year={2022},
  doi={10.1093/bioadv/vbac016},
  publisher={Oxford University Press}
}

@article{franzen2019panglaodb,
  title={PanglaoDB: a web server for exploration of mouse and human single-cell RNA sequencing data},
  author={Franz{\'e}n, Oscar and Gan, Li-Ming and Bj{\"o}rkegren, Johan LM},
  journal={Database},
  volume={2019},
  pages={baz046},
  year={2019},
  doi={10.1093/database/baz046},
  publisher={Oxford University Press}
}
```
