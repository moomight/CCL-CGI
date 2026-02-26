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

In the CCL-CGI setup, we first randomly split the labeled genes into three disjoint subsets and store the resulting labels and masks in the global PPI H5 file (`y_train`, `y_val`, `y_test`, and the corresponding masks), with train+validation accounting for 75% and the test set accounting for 25%. The test set is kept fixed throughout all runs. For model selection, we build a training pool as (train ∪ val) and run stratified 10-fold cross-validation on that pool. In each fold, 9 folds are used for training and 1 fold for validation; evaluation is performed on the same held-out test set, and fold-wise test metrics are averaged for final reporting.


