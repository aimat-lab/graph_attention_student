=========
Changelog
=========

0.3.0 - 19.12.2022
------------------

- Added multi-regression explanation support for `Megan`
- Added global graph feature support for `Megan`
- Started to move away from the local implementation of "eye_tracking_dataset" and
  instead replace it's usage with the external package `visual_graph_dataset`.
- Added example `examples.vgd_multitask`

0.4.0 - 19.12.2022
------------------

- Extended `examples.vgd_multitask` to generate more informative artifacts.
- Added function `visualization.plot_regression_fit` which creates a regression plot with the true values
  on the x-axis and the predicted values on the y-axis. But this is not a scatter plot as it was previously
  realized in various experiment scripts. This function will create a heatmap to be more efficient. With
  large amounts of data a scatter plot becomes very memory inefficient when exported as a vector graphic
  PDF!

0.5.0 - 20.01.2023
------------------

- Added the `templates` folder and some templates to create latex code for the automatic generation
  of tables to represent the experiment results.
- Refactored `examples.vgd_multitask` to `examples.vgd_multitask_megan`: The experiment now utilizes
  the "hook" system of the newest pycomex version. The experiment now does multiple independent experiments
  and the analysis produces a latex table with the mean results for each of the targets.
- Added `examples.vgd_multitask_gnn` as a base experiment which trains a classic GCN model on a multitask
  VGD dataset
- Added `examples.vgd_multitask_gatv2` which trains a GATv2 model on a multitask VGD dataset
- Added `examples.vgd_multitask_gin` which trains a GIN model on a multitask VGD dataset.

0.6.0 - 27.02.2023
------------------

- Fixed the classification explanation step in MEGAN
- Added the `keras` model which is important for loading MEGAN models from persistent representation
  on the disk

0.7.0 - 27.03.2023
------------------

- Moved the contents of `model.py` into individual modules of the `models` package because that module
  was getting way too big.
- Small improvements for the gradient based models and the GNES implementation.
- Changed the version dependency for numpy
- improved the `visualization.plot_regression_map` function

0.7.1 - 01.04.2023
------------------

- Added documentation
- Fixed a bug related to RaggedTensor support in the functions `training.mae` and `training.mse`
- Minor Bug fixes

0.7.2 - 27.04.2023
------------------

- Added an alternative explanation co-training parameter to the MEGAN model `regression_weights` which
  will now slowly replace `regression_limits`. The old version will still work but be deprecated in the
  future. This new parameter can be used in junction with `importance_multiplier` to set relative weights
  for negative and positive individually.

0.8.0 - 27.04.2023
------------------

- Started moving towards the pycomex functional interface which was introduced in the newest version of
  pycomex
- Changed the pycomex version dependency

0.9.0 - 01.05.2023
------------------

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

0.10.0 - 08.05.2023
-------------------

- Added a development version of `FidelityMegan` model which can be trained directly to match a
  fidelity target.
- Added a `keras.load_model` utility function
- Added the `layers.ExplanationGiniRegularization` layer

0.11.0 - 08.05.2023
-------------------

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

0.12.0 - 20.05.2023
-------------------

BACKWARDS INCOMPATIBLE - MEGAN update - I changed the basic architecture of the MEGAN model a bit. The
MLP backend is now no longer a concatenation of all the channel-specific graph embeddings. Instead, the
*same* MLP is now used to produce a vector of the final output shape for each of the channels. These
are then added at the end plus a bias weight. This change is motivated by the prior inclusion of the
fidelity training step which turned out to work really well. Conceptionally, it makes more sense to let the
gradients of that fidelity train step affect the MLP as well, but that is only possible with the previously
described changes to the MLP structure so as to not give the model the chance to "cheat" the fidelity.

- Added an experiment which trains the megan model on the "mu" value of the QM9 dataset.

0.12.1 - 22.05.2023
-------------------

- Small fix for the computation of the leave-one-out deviations for the MEGAN model

0.12.2 - 22.05.2023
-------------------

- Added the new method `Megan.leave_one_out_deviations` which is more general

0.12.3 - 21.06.2023
-------------------

- Added the option to return the pooled graph embeddings for a MEGAN model instead of the final prediction and also 
  added the method `embedd_graphs` which does this for a list of graph dicts.
- Added the TADF dataset training sub-experiment modules

0.13.0 - 21.12.2023
-------------------

- Quality of life improvements to the "vgd_single.py" base experiment. For example it is not possible to inject 
  an external list of integer indices to act as the test indices for the experiment. It is also possible to load 
  the dataset from the remote file share.
- Changed the base Megan model to now also use dropout layers in the final prediction MLP
- Added the method "predict_graphs_monte_carlo" to the Megan models which can be used to create an uncertainty 
  estimation for the model based on the monte-carlo dropout method.
  
0.14.0 - 22.01.2024
-------------------

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

0.15.0 - 10.03.2024
-------------------

- Created a new experiment module `vgd_torch__megan__fia_49k.py` which trains the MEGAN model on the FIA dataset 
  for predicting the lewis acidity of molecules.
- Slightly changed the MEGAN model's contrastive learning scheme to now use a projection head before applying the SimCLR 
  loss. This is a common practice in the literature and should improve the performance of the model.
- Added the functionality to save the MEGAN model as a PT file to the disk
- Changed the python dependency to allow also newer versions of python

0.15.1 - 22.03.2024
-------------------

. Changed the `vgd_torch.py` base experiment to now define the train test split with a hook because that should be 
  more customizable in the future
- Added the `predict_graph` function to the torch model base class which predicts a single graph output to be consistent 
  with the tensorflow version

0.16.0 - 19.03.2024
-------------------

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

0.16.2 - 20.03.2024
-------------------

- Added the new experiment module `vgd_torch__megan__tadf.py` which trains the MEGAN model on the TADF dataset 
  for predicting the singlet-triplet energy gap of molecules.
- The `torch.data.data_from_graph` function now also attaches the `node_coordinates` to the resulting Data object 
  as the `data.coords` attribibute - if it exists in the given graph dict.

0.16.3 - 07.06.2024
-------------------

- Added the new parameter `regression_target` to the default `Megan` class. Possible values are the the string 
  literals 'node' and 'edge'. The node-case is the default backwards compatible case where the explanation approximation 
  loss is calculated on the basis of the nodes alone. With the new edge-case, the explanation approximation loss is 
  based on the edges. Specifically, the edge features as well as the features of the two adjoined nodes. This is a more 
  general case as it also considers tasks which are primarily influenced by the edge features and not the node features.

0.17.0 - 28.06.2024
-------------------

- Added the `GraphAttentionLayerV2` layer which is an extension of the original `GraphAttentionLayer` layer. The 
  new layer now also considers the edge features for the message update and uses MLPs instead of single dense layers.
  These mlps also use batch norm intermediates. This has shown improved convergence speed for almost all datasets.
- Tweaked the value for the importance offset in the "edge" computation of the importance loss so that it produces 
  more meaningful results.

0.18.0 - 08.08.2024
-------------------

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

0.18.1 - 08.08.2024
-------------------

HOTFIX: The `Megan.regression_reference` running mean is now a `nn.Parameter` and therefore also included 
when saving/loading the model from persistent files.

0.18.2 - 08.08.2024
-------------------

HOTFIX: Removed batchnorm layers in the projection MLPs as this was causing significantly different results when 
running the model in eval mode versus in 

- modified the GraphAttentionLayerV2 to now use a "look ahead" aggregation of the neighbor nodes as well in the 
  message update. significantly improves the explanations for the BA2Motifs dataset.

0.18.3 - 01.10.2024
-------------------

- modified the augementations that are used for the contrastive learning now.
- using the Ruff Linter now
- added the `ruff.toml` configuration file
- removed various unused imports

0.18.4 - 16.10.2024
-------------------

- Added the `torch/advanced.py` module to contain the advanced functionality that builds on top of the basic 
  model functionality.
  - the `explain_value` function directly plots the explanation masks given a domain specific graph representation 
    and a model instance.


0.18.5 - 19.05.2025
-------------------

- Added the `torch/_legacy.py` module which contains an older version of the Megan model which needs to 
  be supported to enable backwards compatibility for the aggregation prediction model.


1.0.0 - 25.09.2025
-------------------

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
