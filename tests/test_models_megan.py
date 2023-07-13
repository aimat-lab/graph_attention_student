import os

import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
from visual_graph_datasets.data import load_visual_graph_dataset

from graph_attention_student.training import NoLoss
from graph_attention_student.data import tensors_from_graphs
from graph_attention_student.data import mock_importances_from_graphs
from graph_attention_student.models.megan import Megan

from .util import ASSETS_PATH

MOCK_PATH = os.path.join(ASSETS_PATH, 'mock')


def test_megan_basically_works():
    """
    If the default megan model basically works - so can it be constructed without error? Does the 
    predictions and explanations have the correct shape? etc.
    """
    _, index_data_map = load_visual_graph_dataset(MOCK_PATH)
    dataset_length = len(index_data_map)
    
    graphs = [data['metadata']['graph'] for data in index_data_map.values()]
    labels = [data['metadata']['graph']['graph_labels'] for data in index_data_map.values()]

    num_channels = 2
    num_targets = 2
    num_units = 3
    model = Megan(
        units=[num_units, num_units, num_units],
        final_units=[num_targets],
        final_activation='softmax',
        importance_channels=num_channels
    )
    
    # ~ training test
    # here we test if the training can be run without any issues
    x = tensors_from_graphs(graphs)
    y = (np.array(labels), *mock_importances_from_graphs(graphs, num_channels))
    model.compile(
        loss=[ks.losses.MeanSquaredError(), NoLoss(), NoLoss()],
    )
    model.fit(
        x, y,
        batch_size=32,
        epochs=1,
    )
    
    # ~ prediction test
    # In this section we test if the predictions to the extent that all the correct tensor shapes 
    # are created by the output
    predictions = model.predict_graphs(graphs)
    # The least we can expect is that there is exactly one prediction for each element
    assert len(predictions) == dataset_length
    # But then on a more detailed basis we need to check of the shapes of the explanations are correct!
    for prediction, graph in zip(predictions, graphs):
        out, ni, ei = prediction
        assert len(out) == 2
        assert ni.shape == (len(graph['node_indices']), num_channels)
        assert ei.shape == (len(graph['edge_indices']), num_channels)
        
    # ~ embeddings test
    # in this section we test if the generation of the graph embeddings works properly from the 
    # perspective of the tensor shapes
    embeddings = model.embedd_graphs(graphs)
    assert len(embeddings) == len(graphs)
    for embedding, graph in zip(embeddings, graphs):
        assert embedding.shape == (num_units, num_channels)

    
