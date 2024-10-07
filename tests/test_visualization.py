import os
import numpy as np
import matplotlib.pyplot as plt


from graph_attention_student.visualization import plot_regression_value_distribution

from .util import ARTIFACTS_PATH


def test_plot_regression_value_distribution():
    """
    regression value distribution should plot the distribution of the regression values.
    """
    values = np.random.random(size=(100, ))
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    plot_regression_value_distribution(
        values=values,
        ax=ax,
    )
    # save the figure into the artifacts path for visual inspection
    fig_path = os.path.join(ARTIFACTS_PATH, 'plot_regression_value_distribution.pdf')
    fig.savefig(fig_path)

# def test_plot_leave_one_out_analysis():
#     """
#     ``plot_leave_one_out_analysis`` is supposed to plot the results of the leave one out analysis into a
#     matrix of num_channels x num_targets and visualize the leave one out effect of every channel on every
#     target in the format of histogram.
#     """
#     num_channels = 2
#     num_targets = 3
#     num_graphs = 20
#     model = MockMegan(
#         importance_channels=num_channels,
#         final_units=[num_targets]
#     )
#     graphs = get_mock_graphs(num_graphs)

#     results = leave_one_out_analysis(
#         model=model,
#         graphs=graphs,
#         num_targets=num_targets,
#         num_channels=num_channels
#     )

#     fig = plot_leave_one_out_analysis(
#         results=results,
#         num_targets=num_targets,
#         num_channels=num_channels,
#     )

#     path = os.path.join(ARTIFACTS_PATH, 'leave_one_out.pdf')
#     fig.savefig(path)


# def test_plot_distribution():
#     values = np.random.normal(loc=3, scale=5, size=10000)
#     fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))

#     result = plot_distribution(ax=ax, values=values, bins=100)
#     ax.legend()

#     assert isinstance(result, dict)
#     save_fig(fig)


# def test_plot_regression_fit():
#     min_value, max_value = -4, 10
#     num_samples = 1000
#     true_values = (max_value - min_value) * np.random.random_sample(num_samples) - min_value
#     pred_values = true_values + np.random.normal(0, 1, num_samples)

#     assert true_values.shape == (num_samples, )
#     assert pred_values.shape == (num_samples, )

#     fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
#     plot_regression_fit(
#         values_true=true_values,
#         values_pred=pred_values,
#         ax=ax,
#     )
#     fig_path = os.path.join(ARTIFACTS_PATH, 'regression_fit.pdf')
#     fig.savefig(fig_path)
