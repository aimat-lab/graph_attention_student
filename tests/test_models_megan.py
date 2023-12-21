import os
import tempfile
import pytest

import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
from visual_graph_datasets.data import load_visual_graph_dataset

from graph_attention_student.training import NoLoss
from graph_attention_student.data import tensors_from_graphs
from graph_attention_student.data import mock_importances_from_graphs
from graph_attention_student.models.megan import Megan
from graph_attention_student.models.megan import Megan2
from graph_attention_student.models import load_model

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
        importance_channels=num_channels,
        importance_factor=1.0,
    )
    
    # ~ training test
    # here we test if the training can be run without any issues
    x = tensors_from_graphs(graphs)
    y = (np.array(labels), *mock_importances_from_graphs(graphs, num_channels))
    model.compile(
        loss=[ks.losses.MeanSquaredError(), NoLoss(), NoLoss()],
    )
    results = model.fit(
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

    
def test_megan2_basically_works():
    """
    If the MEGAN2 model basically works - so can it be constructed without error? Does the 
    predictions and explanations have the correct shape? etc.
    """
    dev = tf.device('cpu:0')
    dev.__enter__()
    
    _, index_data_map = load_visual_graph_dataset(MOCK_PATH)
    dataset_length = len(index_data_map)
    
    graphs = [data['metadata']['graph'] for data in index_data_map.values()]
    labels = [data['metadata']['graph']['graph_labels'] for data in index_data_map.values()]
    
    num_channels = 2
    num_targets = 2
    num_units = 3
    model = Megan2(
        units=[num_units, num_units, num_units],
        final_units=[num_targets],
        final_activation='softmax',
        importance_channels=num_channels,
        # We want to enable all the additional loss computations to test if they work
        importance_factor=1.0,
        fidelity_factor=0.1,
        fidelity_funcs=[
            lambda a, b: tf.nn.relu(a - b),
            lambda a, b: tf.nn.relu(b - a),
        ],
        contrastive_sampling_factor=0.1,
    )
    
     # ~ training test
    # here we test if the training can be run without any issues. Here we only need to test that the 
    # following code runs without breaking.
    x = tensors_from_graphs(graphs)
    y = (np.array(labels), *mock_importances_from_graphs(graphs, num_channels))
    model.compile(
        optimizer=ks.optimizers.Adam(),
        loss=[ks.losses.CategoricalCrossentropy(), NoLoss(), NoLoss()],
    )
    history = model.fit(
        x, y,
        batch_size=32,
        epochs=3,
    )
    assert not np.isnan(history.history['loss']).all()
    assert not np.isclose(model.var_bias.numpy(), 0).all()
    
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
        assert embedding.shape == (num_channels, num_units)
        
    # ~ saving test
    # In this section we will save the model into a persistent representation on the file system and 
    # then load the model again from that representation to check if that is possible without problems
    with tempfile.TemporaryDirectory() as path:
        
        model.save(path)
        assert len(os.listdir(path)) != 0
    
        model_loaded = load_model(path)
        assert isinstance(model_loaded, Megan2)
        assert not np.isclose(model_loaded.var_bias.numpy(), 0).all()
        
        
@pytest.mark.parametrize('num_targets,num_channels,num_reps', [
    (1, 2, 10),
])
def test_megan_predict_graphs_monte_carlo(num_targets, num_channels, num_reps):
    
    # we need to load the mock dataset for the testing
    _, index_data_map = load_visual_graph_dataset(MOCK_PATH)
    num_elements = len(index_data_map)
    
    # We need to set up the minimal model as well
    model = Megan2(
        units=[32, 32, 32],
        final_units=[32, num_targets],
        final_activation='linear',
        importance_channels=num_channels,
        # We want to enable all the additional loss computations to test if they work
        importance_factor=1.0,
        final_dropout_rate=0.2,
    )
    
    # 
    graphs = [data['metadata']['graph'] for data in index_data_map.values()]
    out_mean, out_std = model.predict_graphs_monte_carlo(
        graphs=graphs,
        num_repetitions=num_reps,
    )
    print('out_mean shape', out_mean.shape)
    print('out_std shape', out_std.shape)
    
    assert isinstance(out_mean, np.ndarray)
    assert out_mean.shape == (num_elements, num_targets)
    
    assert isinstance(out_std, np.ndarray)
    assert out_std.shape == (num_elements, num_targets)
    assert np.mean(out_std) > 1e-6
    print('mean std', np.mean(out_std))