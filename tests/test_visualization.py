import pytest
import numpy as np
import matplotlib.pyplot as plt

from graph_attention_student.visualization import plot_distribution

from .util import ASSETS_PATH, LOG
from .util import save_fig


def test_plot_distribution():
    values = np.random.normal(loc=3, scale=5, size=10000)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))

    result = plot_distribution(ax=ax, values=values, bins=100)
    ax.legend()

    assert isinstance(result, dict)
    save_fig(fig)
