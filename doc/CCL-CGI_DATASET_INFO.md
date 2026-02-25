# CCL-CGI Dataset Information

The following statistics are based on `global_ppi.h5`.

## Basic Statistics

- Total genes: `12,956`
- Labeled genes: `3,091`
- Positive genes: `783`
- Negative genes: `2,308`
- Number of cell types: `31`

- Note: Due to the memory limited during data preprocessing, some expression matrix of certain cell types are divided into several parts, thereby generating totally 39 h5 files.

## Data Split

- Original split:
  - `train = 2,086`
  - `val = 231`
  - `test = 774`
- In training, `train + val` are merged into one training pool (`2,317`), while test stays `774`.
- Training pool : test set ≈ `75% : 25%` (on labeled genes).
- 10-fold cross-validation is performed on the training pool.
