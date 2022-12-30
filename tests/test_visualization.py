import pytest
import os
import numpy as np
import matplotlib.pyplot as plt

from graph_attention_student.visualization import plot_distribution
from graph_attention_student.visualization import plot_regression_fit

from .util import ASSETS_PATH, LOG
from .util import ARTIFACTS_PATH
from .util import save_fig


def test_plot_distribution():
    values = np.random.normal(loc=3, scale=5, size=10000)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))

    result = plot_distribution(ax=ax, values=values, bins=100)
    ax.legend()

    assert isinstance(result, dict)
    save_fig(fig)


def test_plot_regression_fit():
    min_value, max_value = -4, 10
    num_samples = 1000
    true_values = (max_value - min_value) * np.random.random_sample(num_samples) - min_value
    pred_values = true_values + np.random.normal(0, 1, num_samples)

    assert true_values.shape == (num_samples, )
    assert pred_values.shape == (num_samples, )

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    plot_regression_fit(
        values_true=true_values,
        values_pred=pred_values,
        ax=ax,
    )
    fig_path = os.path.join(ARTIFACTS_PATH, 'regression_fit.pdf')
    fig.savefig(fig_path)
