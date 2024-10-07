import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from visual_graph_datasets.data import load_visual_graph_element
from visual_graph_datasets.visualization.base import draw_image
from visual_graph_datasets.visualization.importances import plot_node_importances_border

from graph_attention_student.util import get_version
from graph_attention_student.util import update_nested_dict
from graph_attention_student.util import normalize_importances_fidelity
from graph_attention_student.testing import set_environ

from .util import ASSETS_PATH, ARTIFACTS_PATH

mpl.use('Agg')


def test_get_version():
    """
    31.08.24: The "get_version" util function should return the version string of the package read 
    from the "VERSION" file in the root directory of the package.
    """    
    version = get_version()
    
    # Since the version changes we can only check surface-level if the given string matches 
    # the general pattern of a version string.
    assert version
    assert version != ''
    assert version.count('.') >= 1


def test_overwrite_version():
    """
    31.08.24: The "get_version" util function should return the version string of the package read
    from the "VERSION" file in the root directory of the package. If the environment variable
    "GRAPH_ATTENTION_STUDENT_VERSION_OVERWRITE" is set then the version string should be overwritten
    with the value of that environment variable.
    """
    with set_environ({'GRAPH_ATTENTION_STUDENT_VERSION_OVERWRITE': 'test'}):
        version = get_version()
        assert version == 'test'


def test_normalize_importances_fidelity():
    dataset_path = os.path.join(ASSETS_PATH, 'mock')
    data = load_visual_graph_element(dataset_path, '0')
    graph = data['metadata']['graph']
    node_positions = np.array(graph['image_node_positions'])

    fig, (ax_0, ax_1) = plt.subplots(ncols=2, nrows=1, figsize=(20, 10))
    draw_image(ax_0, data['image_path'])
    draw_image(ax_1, data['image_path'])

    ni = np.ones(shape=(len(graph['node_indices']), 2))
    ni = normalize_importances_fidelity(ni, [0.8, 0.1])

    assert np.mean(ni) != 1
    assert np.mean(ni) != 1

    plot_node_importances_border(ax_0, graph, node_positions, ni[:, 0])
    plot_node_importances_border(ax_1, graph, node_positions, ni[:, 1])

    fig_path = os.path.join(ARTIFACTS_PATH, 'normalize_importances_fidelity.pdf')
    fig.savefig(fig_path)


def test_update_nested_dict():
    # The whole point of this function is that it performs a dict update which respects nested dicts.
    # So whenever a nested dict is found which exists in the original and the new dict then instead of
    # replacing the entire dict of the original with the version of the new one (as is the standard
    # behavior of dict.update()) it performs a dict.update() on those two dicts recursively again.
    original = {
        'nesting1': {
            'value_original': 10
        },
        'nesting2': {
            'value_original': 10
        }
    }
    extension = {
        'value': 20,
        'nesting1': {
            'value_extension': 20
        },
        'nesting2': {
            'value_original': 20
        }
    }

    merged = update_nested_dict(original, extension)
    assert isinstance(merged, dict)
    assert 'value' in merged and merged['value'] == 20
    assert len(merged['nesting1']) == 2
    assert len(merged['nesting2']) == 1
    assert merged['nesting2']['value_original'] == 20
    