import os
import pytest

import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from visual_graph_datasets.data import load_visual_graph_element
from visual_graph_datasets.visualization.base import draw_image
from visual_graph_datasets.visualization.importances import plot_edge_importances_border
from visual_graph_datasets.visualization.importances import plot_node_importances_border

from graph_attention_student.util import update_nested_dict
from graph_attention_student.util import normalize_importances_fidelity

from .util import ASSETS_PATH, ARTIFACTS_PATH

mpl.use('TkAgg')


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
    
    
def test_tensor_matrix_multiplication():
    """
    Understanding how to do matrix multiplication for graph embedding tensors.
    """
    tens = tf.random.normal((32, 16))
    tens_expanded = tf.expand_dims(tens, axis=-2)
    
    product = tf.reduce_sum(tens * tens_expanded, axis=-1)
    product_array = product.numpy()
    print(product_array.shape)
    
    fig, (ax_1, ax_2) = plt.subplots(ncols=2, nrows=1, figsize=(20, 10))
    ax_1.set_title('self product matrix')
    ax_1.imshow(product_array)
    
    # I also need to create a diagonal "eye matrix" with thesame shape to be able to block out all the 
    # auto terms in that result
    eye = 1 - tf.eye(num_rows=product.shape[0], num_columns=product.shape[0])
    eye_array = eye.numpy()
    ax_2.set_title('self product matrix - eye window')
    ax_2.imshow(eye_array)
    
    fig.savefig(os.path.join(ARTIFACTS_PATH, 'test_tensor_matrix_multiplication.pdf'))
    