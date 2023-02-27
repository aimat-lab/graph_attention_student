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