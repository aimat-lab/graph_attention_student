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