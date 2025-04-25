# CCL-CGI: Leveraging Cellular Heterogeneity with Contextualized Contrastive Learning for Cancer Gene Identification by Single-cell Sequencing

## Dependencies

* python 3.10
* pytorch_gpu 1.12.1
* scikit-learn 1.4.2
* pytorch-lightning 2.1.0
* pandas 2.2.1
* numpy 1.26.4
* networkx 3.3
* h5py 3.11.0

## Datasets

The dataset can be download from https://doi.org/10.6084/m9.figshare.28777580, and then unzipped to "h5/.".

## Reproducibility

To reproduce the results of CCL-CGI, you are supposed to download the datasets above and put them into the dictionary "h5/.".

The directory structure of CCL-CGI should be:

```
.
│  config.py
│  main.py
│  run.py
│
├─callbacks
│      eval.py
│      __init__.py
│
├─checkpoint             
│
├─h5
│      Acinar cells.h5
│      ...
├─history
├─layers
│      aggregate.py
│      attentionFusion.py
│      cellTypeSpecEmbedding.py
│      centralityEncoding.py
│      extendedAttentionAggregate.py
│      graphormerEncoder.py
│      multiHeadAttention.py
│      spatialEncoding.py
│      tensorToString.py
│      __init__.py
│
├─lightning_logs
├─log
├─losses
│      contrastiveLoss.py
│      triple_loss.py
│      weightedBinaryCrossEntropy.py
│      __init__.py
│
├─models
│      base_model.py
│      panCancerGenePredict.py
│      tree.py
│      __init__.py
│
├─pdata
├─sp
└─utils
        DATASET.py
        io.py
        node2vec.py
        utils.py
        walker.py
        __init__.py
```

You can train CCL-CGI by:

```python
python run.py
```

The model checkpoint is saved in the dictionary "checkpoint/.".
