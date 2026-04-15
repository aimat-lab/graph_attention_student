# Changelog

## 1.4.0 - 2026-04-15

Uniformity Regularization

- Added uniformity loss (Wang & Isola, ICML 2020) that encourages graph embeddings to spread
  uniformly on the unit hypersphere per channel, preventing embedding collapse. Uses per-channel
  FIFO queues (shared queue size with MoCo) to measure uniformity over thousands of embeddings.
- New model parameters: `uniformity_factor`, `uniformity_t`.

Training Metrics Dashboard

- Added uniformity loss (`loss_unif`) tracking to `MeganTrainingMetricsCallback`.
- Added embedding uniformity monitoring (mean-vector norm from uniformity queue).
- Replaced "Negative Similarity" plot with "Uniformity" plot in the training dashboard.
- Uniformity loss included in the stacked loss ratio chart.

Documentation

- Fixed the "Python API" example in `README.rst`: added missing `importance_mode='regression'`
  to the `Megan(...)` constructor call, without which `training_step` crashes because
  `training_explanation` never reassigns `loss_expl` from its initial `float`.

Testing

- Added `tests/test_readme_examples.py` covering the README's Python API training and
  load-infer-report snippets end-to-end (scaled-down model and epoch count).
- Added `tests/assets/readme_fixture.csv` (10-row SMILES/target fixture) for those tests.

Housekeeping

- Removed `HISTORY.rst` (superseded by `CHANGELOG.md`).
- Added `release.sh` script automating test → version bump → commit/tag → push →
  GitHub release → build/publish.
- Raised the supported Python floor from 3.8 to 3.9 (`requires-python`, classifiers
  and `noxfile.py` sweep updated to 3.9–3.13); removed the now-unreachable
  `pydyf<0.11.0; python_version<'3.9'` marker and the leftover `poetry-bumpversion`
  dependency from the Poetry era.
- Added `pyarrow>=10.0.0` as a direct dependency so `pl.from_pandas()` handles
  pyarrow-backed pandas string columns (fixes a test failure under newer pandas).
- Introduced a `dev` extra (`nox`, `bump-my-version`, `pytest`) installable via
  `uv pip install -e ".[dev]"`.
- Silenced third-party deprecation noise via `[tool.pytest.ini_options]`
  `filterwarnings` (matplotlib/pyparsing/Pillow, Lightning training UX warnings,
  and Lightning's internal `torch.load` `weights_only` FutureWarning).
- Fixed an invalid `\m` escape in the `negative_log_likelihood` docstring
  (`metrics.py`) by making it a raw string.
- Switched `polars.LazyFrame.collect(streaming=True)` to `collect(engine="streaming")`
  in `torch/data.py` to track the polars 1.25+ API.
- Passed `weights_only=False` explicitly to our own `torch.load` call sites
  (`torch/megan.py`, `torch/model.py`, `tests/test_torch_megan.py`) to silence
  the FutureWarning ahead of PyTorch's default flip.

## 1.3.0 - 2026-03-24

Contrastive Learning

- Replaced SimCLR contrastive loss with MoCo-style queue-based contrastive learning. Uses momentum-updated
  projection heads and per-channel FIFO queues (default 4096) for diverse negative embeddings, decoupling
  the number of negatives from batch size.
- Improved contrastive augmentation strategy: structural edge dropping in non-explained regions, soft
  perturbation of explained regions, and adaptive importance thresholding.
- Reduced from two augmented forward passes to one (single augmented view).
- Added `sim_neg` logging alongside `sim_pos` for monitoring negative similarity.
- Deprecated unused `contrastive_beta` and `contrastive_tau` hard-negative-mining parameters.
- New parameters: `contrastive_queue_size`, `contrastive_momentum`, `contrastive_detach_importance`.

Training Metrics Dashboard

- Added `MeganTrainingMetricsCallback`: generates a 6x5 matplotlib grid each epoch covering loss components,
  loss dynamics, validation quality, explanation quality, gradient health, and hardware utilization.
- Added `batch_metrics` dict to `Megan.training_step` for callback consumption.
- Added ROC-AUC based explanation quality metric (threshold-free separability) with optimal accuracy overlay,
  replacing the fixed-threshold approximation accuracy.

Explanation Loss Improvements

- Fixed median-split bug where samples exactly at the median got incorrect `[False, False]` targets for both
  channels. Now uses a clean `<=` vs `>` partition.
- Restored `regression_margin` support with proper loss masking: samples in the dead zone are excluded from
  the BCE loss entirely rather than receiving wrong targets.
- Switched explanation split from median to mean-based.

Callbacks

- Added `GracefulStopCallback`: first Ctrl+C finishes the current epoch and proceeds to evaluation; second
  Ctrl+C force-quits. Restores original signal handler on teardown.
- Added contrastive warmup scheduling to the base `vgd_torch__megan` experiment via `ContrastiveSchedulerCallback`.

Bugfixes

- Fixed isolated node crash in edge importance pooling: `MaxAggregation` now uses `dim_size=num_nodes` to
  handle graphs with nodes that have no edges.
- Fixed edge_attr/edge_index size mismatch: `forward()` now truncates or pads `edge_attr` to match
  `edge_index` and writes the fix back to `data.edge_attr` for downstream consistency.
- Fixed `training_step` RuntimeError fallback to use `requires_grad=True` so Lightning's backward pass
  doesn't crash on malformed batches.
- Wrapped `training_representation` call in try/except for graceful degradation.
- Fixed projection head double-normalization (removed final BatchNorm before F.normalize).

Experiments & Quality of Life

- Added `vgd_torch__megan__synth2.py` experiment with MoCo contrastive learning enabled.
- Added `e.log_parameters()` at training start to log all experiment parameters.
- Dataset `process.py` is now copied into the experiment archive at training start.
- Increased training example visualizations from 8 to 16.
- Corrected `SPARSITY_FACTOR` comments (removed incorrect "DEPRECATED" label).

## 1.2.0 - 2026-01-22

Data Loading

- Added new random-access data store classes to `torch/data.py` for more efficient data loading:
  - `SmilesStore`: SQLite-backed store for CSV data with random access. Uses thread-local
    connections for multi-worker DataLoader support. Create from CSV via `SmilesStore.from_csv()`.
  - `SmilesGraphStore`: Wraps a `SmilesStore` with a `Processing` instance to convert
    SMILES strings to GraphDict representations on-the-fly.
  - `VisualGraphDatasetStore`: Random-access store for Visual Graph Dataset (VGD) directories.
    Provides direct index-based access (`store[5]` reads `5.json`).
  - `GraphDataLoader`: PyTorch Geometric DataLoader subclass that accepts any `Sequence[GraphDict]`
    and automatically converts to PyG Data objects.
- These new classes provide an alternative to the streaming-based `SmilesDataset` when random
  access is needed (e.g., for validation/test sets or when shuffling is required without reservoir sampling).

## 1.1.0 - 2025-12-02

Command Line Interface

- Fixed a bug where `megan --version` was not working

Experiment Modules

- Added the new experiment module `train_model.py` which is the new base experiment
  that allows for the training of models without a pre-computed visual graph dataset
  and instead uses just a CSV file as the dataset basis.
- Added `train_model__megan.py` which implements the training of a MEGAN model based
  on simple dataset csv files.
- Extended the training scripts to now also track various statistics about the latent
  space of the megan model during the training.

## 1.0.0 - 2025-09-25

Packaging

- Changed the `pyproject.toml` from using poetry to using uv + hatchling now.
- Removed the `torch_scatter` default dependency which should make it possible to install the
  package now with a single pip install operation.
- Added the `weasyprint` dependency for the generation of PDF reports
- Added the `polars` dependency for fast and lazy data frame operations
- bumped the required version for `pycomex` to `0.21.0` as this is the only recent version with backward compatbility to python 3.8
- bumped the required version for `visual_graph_datasets` to `0.17.0` as this is the only recent version with backward compatibility to python 3.8

Functionality

- Changed the default constructor parameters of the `Megan` model.
- Added the `SmilesDataset` class which allows a custom processing based torch Dataset based on
  a CSV / data frame of SMILES strings and target values for easier training. This class implements a
  streaming data loading scheme which makes it possible to handle arbitrarily large datasets with a minimal
  memory footprint.

Documentation

- Added the `00_basic_usage.ipynb` tutorial notebook. Illustrates a basic workflow with model training
  and inference of the trained model.
- Added the `01_full_example.ipynb` tutorial notebook. Illustrates a more involved workflow with train test splitting,
  performance evaluations and manual explanation visualizations.

## 0.18.5 - 2025-05-19

- Added the `torch/_legacy.py` module which contains an older version of the Megan model which needs to
  be supported to enable backwards compatibility for the aggregation prediction model.

## 0.18.4 - 2024-10-16

- Added the `torch/advanced.py` module to contain the advanced functionality that builds on top of the basic
  model functionality.
  - the `explain_value` function directly plots the explanation masks given a domain specific graph representation
    and a model instance.

## 0.18.3 - 2024-10-01

- modified the augementations that are used for the contrastive learning now.
- using the Ruff Linter now
- added the `ruff.toml` configuration file
- removed various unused imports

## 0.18.2 - 2024-08-08

HOTFIX: Removed batchnorm layers in the projection MLPs as this was causing significantly different results when
running the model in eval mode versus in

- modified the GraphAttentionLayerV2 to now use a "look ahead" aggregation of the neighbor nodes as well in the
  message update. significantly improves the explanations for the BA2Motifs dataset.

## 0.18.1 - 2024-08-08

HOTFIX: The `Megan.regression_reference` running mean is now a `nn.Parameter` and therefore also included
when saving/loading the model from persistent files.

## 0.18.0 - 2024-08-08

BACKWARD INCOMPATIBLE CHANGES!

- Completely removed the `kgcnn` and `tensorflow` dependency now as the model is fully ported to torch
  - Remove `graph_attention_student.training` module
  - Remove `graph_attention_student.layers` module
  - Remove `graph_attention_student.data` module
  - Remove `graph_attention_student.models` package
  - Removed all derivations of the `vgd_single.py` experiment modules
- Changes to the model (loading previously exported versions of the model will no longer work!)
  - Using BatchNorm and ELU activation functions in all MLPs now
  - Using BatchNorm and multi layer MLPs for every transformation function in the GraphAttentionLayerV2 now
  - DEPRECATED the `regression_reference` parameter now. On the prediction part of the model this is replaced by
    a running average that calculates the mean of the dataset directly from the batch ground truth labels. For the
    explanation approximation loss, the reference is not locally chosen as the median of the ground truth values in
    each batch.
  - In the calculation of the explanation approximation loss, the model now uses the normalized importances instead
    of the absolute importances. This now prevents the model from cheating the loss by simply decreasing the values
    of the importances further.
  - DEPRECATED the `sparsity_factor` parameter now. Due to the usage of the normalized importances, the sparsity
    can now be more accurately controlled by the `importance_offset` parameter.

Additional changes:

- Updated the examples to be more up-to-date with the current state of the model
- When attempting to load an old model version, there is now an appropriate error message that explains the
  version incompatibility.

## 0.17.0 - 2024-06-28

- Added the `GraphAttentionLayerV2` layer which is an extension of the original `GraphAttentionLayer` layer. The
  new layer now also considers the edge features for the message update and uses MLPs instead of single dense layers.
  These mlps also use batch norm intermediates. This has shown improved convergence speed for almost all datasets.
- Tweaked the value for the importance offset in the "edge" computation of the importance loss so that it produces
  more meaningful results.

## 0.16.3 - 2024-06-07

- Added the new parameter `regression_target` to the default `Megan` class. Possible values are the the string
  literals 'node' and 'edge'. The node-case is the default backwards compatible case where the explanation approximation
  loss is calculated on the basis of the nodes alone. With the new edge-case, the explanation approximation loss is
  based on the edges. Specifically, the edge features as well as the features of the two adjoined nodes. This is a more
  general case as it also considers tasks which are primarily influenced by the edge features and not the node features.

## 0.16.2 - 2024-03-20

- Added the new experiment module `vgd_torch__megan__tadf.py` which trains the MEGAN model on the TADF dataset
  for predicting the singlet-triplet energy gap of molecules.
- The `torch.data.data_from_graph` function now also attaches the `node_coordinates` to the resulting Data object
  as the `data.coords` attribibute - if it exists in the given graph dict.

## 0.16.0 - 2024-03-19

- Added an additional experiment module for training a model on the COMPAS dataset.

MODEL BACKWARDS INCOMPATIBLE

- Made several changes to the torch version of the Megan base model
  - Fixed a crucial bug in the classification implementation of the model, where a softmax operation was applied to the
    classification logits twice which lead to an explosion of the logit values.
  - Implemented the fidelity training loss as a seperate loss term
  - Slightly changed how the explanation approximation loss is computed: Instead of simply summing up the attention values
    themselves. The sum is now computed over learned values based on the initial node features, where the attention values
    are used as weights. This should make it a bit more generic and for example less dependent on the graph / motif size.
  - Added optional labels smoothing for the classification loss to tackle overconfident models
  - Added optional logit normalization for the classification logits to tackle overconfident models

## 0.15.1 - 2024-03-22

. Changed the `vgd_torch.py` base experiment to now define the train test split with a hook because that should be
  more customizable in the future
- Added the `predict_graph` function to the torch model base class which predicts a single graph output to be consistent
  with the tensorflow version

## 0.15.0 - 2024-03-10

- Created a new experiment module `vgd_torch__megan__fia_49k.py` which trains the MEGAN model on the FIA dataset
  for predicting the lewis acidity of molecules.
- Slightly changed the MEGAN model's contrastive learning scheme to now use a projection head before applying the SimCLR
  loss. This is a common practice in the literature and should improve the performance of the model.
- Added the functionality to save the MEGAN model as a PT file to the disk
- Changed the python dependency to allow also newer versions of python

## 0.14.0 - 2024-01-22

The MEGAN pytorch port: The self-explaining Megan graph neural network model has been ported to a pytorch version.
All future developments will likely be done with this pytorch version, due to pytorch's significantly higher
flexibility (it does not need to be compiled into a static graph like tensorflow which enables the use of arbitrary
python during the forward pass and the training step implementation)

- Created a new subpackage `torch` which contains the ported model, custom layer implementations and torch-specific
  data processing utils.
- Created a new set of experiment modules that use the pytorch version of the MEGAN model
  - `vgd_torch.py` the base model that implements the training and evaluation of any `AbstractGraphModel` based model
    without an explenatory aspect
  - `vgd_torch__megan.py` specific implementation for the MEGAN model which includes the explanation specific evaluation
  - `vgd_torch__megan__rb_dual_motifs.py`
  - `vgd_torch__megan__aqsoldb.py`
  - `vgd_torch__megan__mutagenicity.py`

## 0.13.0 - 2023-12-21

- Quality of life improvements to the "vgd_single.py" base experiment. For example it is not possible to inject
  an external list of integer indices to act as the test indices for the experiment. It is also possible to load
  the dataset from the remote file share.
- Changed the base Megan model to now also use dropout layers in the final prediction MLP
- Added the method "predict_graphs_monte_carlo" to the Megan models which can be used to create an uncertainty
  estimation for the model based on the monte-carlo dropout method.

## 0.12.3 - 2023-06-21

- Added the option to return the pooled graph embeddings for a MEGAN model instead of the final prediction and also
  added the method `embedd_graphs` which does this for a list of graph dicts.
- Added the TADF dataset training sub-experiment modules

## 0.12.2 - 2023-05-22

- Added the new method `Megan.leave_one_out_deviations` which is more general

## 0.12.1 - 2023-05-22

- Small fix for the computation of the leave-one-out deviations for the MEGAN model

## 0.12.0 - 2023-05-20

BACKWARDS INCOMPATIBLE - MEGAN update - I changed the basic architecture of the MEGAN model a bit. The
MLP backend is now no longer a concatenation of all the channel-specific graph embeddings. Instead, the
*same* MLP is now used to produce a vector of the final output shape for each of the channels. These
are then added at the end plus a bias weight. This change is motivated by the prior inclusion of the
fidelity training step which turned out to work really well. Conceptionally, it makes more sense to let the
gradients of that fidelity train step affect the MLP as well, but that is only possible with the previously
described changes to the MLP structure so as to not give the model the chance to "cheat" the fidelity.

- Added an experiment which trains the megan model on the "mu" value of the QM9 dataset.

## 0.11.0 - 2023-05-08

MEGAN update - The previously introduced variation `FidelityMegan` turns out not to work great on it's
own, but the developed fidelity train step seems to work very well when integrated into the main megan
model on top of the approximation co-training.

- Added the `train_step_fidelity` method to `Megan` model along with the keyword arguments
  `fidelity_factor` and `fidelity_funcs` to control that behavior.

Fidelity Utils

- Added the module `fidelity` which will contain functions relevant to the computation of fidelity
    - `fidelity.leave_one_out_analysis` can be used to calculate all the leave one out deviations
      for all the pairings of channels and targets.
- Added `visualization.plot_leave_one_out_analysis`
- Added very basic test cases for the above functionality

## 0.10.0 - 2023-05-08

- Added a development version of `FidelityMegan` model which can be trained directly to match a
  fidelity target.
- Added a `keras.load_model` utility function
- Added the `layers.ExplanationGiniRegularization` layer

## 0.9.0 - 2023-05-01

- fixed an important bug with the loading of a previously saved Megan model
- Megan model now implements the `PredictGraphMixin` from `visual_graph_datasets`
- Added the `vgd_counterfactuals` library to the dependencies
- Started to generally move towards the new Functional API of `pycomex`

Examples

- Started working on some actual documented examples
    - `examples/02_saving_models.py`
    - `examples/03_loading_models.py`
    - `examples/04_counterfactuals.py`
- Added `examples/README.rst`

## 0.8.0 - 2023-04-27

- Started moving towards the pycomex functional interface which was introduced in the newest version of
  pycomex
- Changed the pycomex version dependency

## 0.7.2 - 2023-04-27

- Added an alternative explanation co-training parameter to the MEGAN model `regression_weights` which
  will now slowly replace `regression_limits`. The old version will still work but be deprecated in the
  future. This new parameter can be used in junction with `importance_multiplier` to set relative weights
  for negative and positive individually.

## 0.7.1 - 2023-04-01

- Added documentation
- Fixed a bug related to RaggedTensor support in the functions `training.mae` and `training.mse`
- Minor Bug fixes

## 0.7.0 - 2023-03-27

- Moved the contents of `model.py` into individual modules of the `models` package because that module
  was getting way too big.
- Small improvements for the gradient based models and the GNES implementation.
- Changed the version dependency for numpy
- improved the `visualization.plot_regression_map` function

## 0.6.0 - 2023-02-27

- Fixed the classification explanation step in MEGAN
- Added the `keras` model which is important for loading MEGAN models from persistent representation
  on the disk

## 0.5.0 - 2023-01-20

- Added the `templates` folder and some templates to create latex code for the automatic generation
  of tables to represent the experiment results.
- Refactored `examples.vgd_multitask` to `examples.vgd_multitask_megan`: The experiment now utilizes
  the "hook" system of the newest pycomex version. The experiment now does multiple independent experiments
  and the analysis produces a latex table with the mean results for each of the targets.
- Added `examples.vgd_multitask_gnn` as a base experiment which trains a classic GCN model on a multitask
  VGD dataset
- Added `examples.vgd_multitask_gatv2` which trains a GATv2 model on a multitask VGD dataset
- Added `examples.vgd_multitask_gin` which trains a GIN model on a multitask VGD dataset.

## 0.4.0 - 2022-12-19

- Extended `examples.vgd_multitask` to generate more informative artifacts.
- Added function `visualization.plot_regression_fit` which creates a regression plot with the true values
  on the x-axis and the predicted values on the y-axis. But this is not a scatter plot as it was previously
  realized in various experiment scripts. This function will create a heatmap to be more efficient. With
  large amounts of data a scatter plot becomes very memory inefficient when exported as a vector graphic
  PDF!

## 0.3.0 - 2022-12-19

- Added multi-regression explanation support for `Megan`
- Added global graph feature support for `Megan`
- Started to move away from the local implementation of "eye_tracking_dataset" and
  instead replace it's usage with the external package `visual_graph_dataset`.
- Added example `examples.vgd_multitask`
