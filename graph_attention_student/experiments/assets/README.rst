=================
Experiment Assets
=================

This module contains the assets for the various computational experiments structured into 
different folders, which are briefly described below:

- ``splits``: This folder contains JSON files of "test_indices" or "val_indices" for the 
  different datasets. These files are used to split the datasets into training, validation, 
  and test sets. The files are named after the dataset they are used for, e.g.,
  "logp_test_indices.json" contains the test indices for the LogP dataset. The indices 
  JSON files should consist of only a single list of integer indices.