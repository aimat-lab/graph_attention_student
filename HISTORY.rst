=========
Changelog
=========

0.3.0 - 19.12.2022
------------------

- Added multi-regression explanation support for ``Megan``
- Added global graph feature support for ``Megan``
- Started to move away from the local implementation of "eye_tracking_dataset" and
  instead replace it's usage with the external package ``visual_graph_dataset``.
- Added example ``examples.vgd_multitask``

0.4.0 - 19.12.2022
------------------

- Extended ``examples.vgd_multitask`` to generate more informative artifacts.
- Added function ``visualization.plot_regression_fit`` which creates a regression plot with the true values
  on the x-axis and the predicted values on the y-axis. But this is not a scatter plot as it was previously
  realized in various experiment scripts. This function will create a heatmap to be more efficient. With
  large amounts of data a scatter plot becomes very memory inefficient when exported as a vector graphic
  PDF!

0.5.0 - 20.01.2023
------------------

- Added the ``templates`` folder and some templates to create latex code for the automatic generation
  of tables to represent the experiment results.
- Refactored ``examples.vgd_multitask`` to ``examples.vgd_multitask_megan``: The experiment now utilizes
  the "hook" system of the newest pycomex version. The experiment now does multiple independent experiments
  and the analysis produces a latex table with the mean results for each of the targets.
- Added ``examples.vgd_multitask_gnn`` as a base experiment which trains a classic GCN model on a multitask
  VGD dataset
- Added ``examples.vgd_multitask_gatv2`` which trains a GATv2 model on a multitask VGD dataset
- Added ``examples.vgd_multitask_gin`` which trains a GIN model on a multitask VGD dataset.

0.6.0 - 27.02.2023
------------------

- Fixed the classification explanation step in MEGAN
- Added the ``keras`` model which is important for loading MEGAN models from persistent representation
  on the disk

0.7.0 - 27.03.2023
------------------

- Moved the contents of ``model.py`` into individual modules of the ``models`` package because that module
  was getting way too big.
- Small improvements for the gradient based models and the GNES implementation.
- Changed the version dependency for numpy
- improved the ``visualization.plot_regression_map`` function

0.7.1 - 01.04.2023
------------------

- Added documentation
- Fixed a bug related to RaggedTensor support in the functions ``training.mae`` and ``training.mse``
- Minor Bug fixes

0.7.2 - 27.04.2023
------------------

- Added an alternative explanation co-training parameter to the MEGAN model ``regression_weights`` which
  will now slowly replace ``regression_limits``. The old version will still work but be deprecated in the
  future. This new parameter can be used in junction with ``importance_multiplier`` to set relative weights
  for negative and positive individually.

0.8.0 - 27.04.2023
------------------

- Started moving towards the pycomex functional interface which was introduced in the newest version of
  pycomex
- Changed the pycomex version dependency

0.9.0 - 01.05.2023
------------------

- fixed an important bug with the loading of a previously saved Megan model
- Megan model now implements the ``PredictGraphMixin`` from ``visual_graph_datasets``
- Added the ``vgd_counterfactuals`` library to the dependencies
- Started to generally move towards the new Functional API of ``pycomex``

Examples

- Started working on some actual documented examples
    - ``examples/02_saving_models.py``
    - ``examples/03_loading_models.py``
    - ``examples/04_counterfactuals.py``
- Added ``examples/README.rst``

0.10.0 - 08.05.2023
-------------------

- Added a development version of ``FidelityMegan`` model which can be trained directly to match a
  fidelity target.
- Added a ``keras.load_model`` utility function
- Added the ``layers.ExplanationGiniRegularization`` layer

0.11.0 - 08.05.2023
-------------------

MEGAN update - The previously introduced variation ``FidelityMegan`` turns out not to work great on it's
own, but the developed fidelity train step seems to work very well when integrated into the main megan
model on top of the approximation co-training.

- Added the ``train_step_fidelity`` method to ``Megan`` model along with the keyword arguments
  ``fidelity_factor`` and ``fidelity_funcs`` to control that behavior.

Fidelity Utils

- Added the module ``fidelity`` which will contain functions relevant to the computation of fidelity
    - ``fidelity.leave_one_out_analysis`` can be used to calculate all the leave one out deviations
      for all the pairings of channels and targets.
- Added ``visualization.plot_leave_one_out_analysis``
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

