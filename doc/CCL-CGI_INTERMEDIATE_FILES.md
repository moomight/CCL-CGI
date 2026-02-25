# CCL-CGI Intermediate Computation Files

## Directory Structure

- `pdata/CCL-CGI/`
  - `*_adj.npy`
  - `*_feature.npy`
  - `*_subgraphs.npy`
  - `*_spatial.npy`
- `sp/CCL-CGI/`
  - `*_sp.h5` (shortest-path cache files)

## Notes

- You can either download and unzip these intermediate files, or generate them during preprocessing.
- Paths are configurable via runtime arguments `--data_dir` and `--sp_dir`.
