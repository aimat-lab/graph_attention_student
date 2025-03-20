import os
import contextlib

import random
import numpy as np

from visual_graph_datasets.processing.base import ProcessingBase
from graph_attention_student.torch.megan import Megan


@contextlib.contextmanager
def set_environ(data: dict):
    """
    This function can be used as a context manager. It temporarily modifes the environment variables 
    in os.environ with the key value pairs of the given ``data`` dict on __enter__.
    Upon __exit__ it resets the environment variables to the original values.
    """
    original_values = {}
    for key, value in data.items():
        if key in os.environ:
            original_values[key] = os.environ[key]
        
        # we also actually need to modify the environment variables now
        os.environ[key] = value
    
    try:
        yield
        
    # At the end we need to reset the environment to the original values
    finally:
        for key, value in original_values.items():
            os.environ[key] = value


def get_mock_graphs(num: int,
                    num_node_attributes: int = 3,
                    num_edge_attributes: int = 1,
                    num_node_coordinates: int = 3,
                    num_outputs: int = 2,
                    ):
    """
    This function creates a number ``num`` of mock graphs for testing
    
    :param num: how many graphs to generate.
    
    :returns: a list of mock graph dicts
    """
    
    graphs = []
    for i in range(num):
        num_nodes = random.randint(10, 20)
        num_edges = random.randint(10, 20)
        
        node_indices = np.array(list(range(num_nodes)))
        node_attributes = np.random.random(size=(num_nodes, num_node_attributes))
        node_coordinates = np.random.random(size=(num_nodes, num_node_coordinates))
        
        edge_indices = np.column_stack((node_indices, np.roll(node_indices, -1)))
        
        num_edges = len(edge_indices)
        edge_attributes = np.random.random(size=(num_edges, num_edge_attributes)) 
        
        graph = {
            'node_indices': node_indices,
            'node_attributes': node_attributes,
            'node_coordinates': node_coordinates,
            'edge_indices': edge_indices,
            'edge_attributes': edge_attributes,
            'graph_labels': np.random.random(num_outputs)
        }
        graphs.append(graph)

    return graphs


def model_from_processing(processing: ProcessingBase,
                          prediction_mode = 'regression',
                          num_outputs: int = 1,
                          num_channels: int = 2,
                          **kwargs,
                          ) -> Megan:
    """
    Given a Processing instance, this function creates a Megan model that is compatible with the graphs created 
    by that processing instance. This is useful to quickly create a model for testing.
    
    :param processing: an instance of a BaseProcessing subclass.
    :param prediction_mode: the prediction mode of the model (regresssion, classification)
    :param out_dim: the output dimension of the models last layer.

    :returns: a Megan model instance.
    """
    # The number of node and edge features can be derived from the processing instance.
    node_dim = processing.get_num_node_attributes()
    edge_dim = processing.get_num_edge_attributes()
    
    model = Megan(
        node_dim=node_dim,
        edge_dim=edge_dim,
        units=[16, 16, 16],
        hidden_units=32,
        num_channels=num_channels,
        prediction_mode=prediction_mode,
        final_units=[16, num_outputs],
        attention_aggregation='max',
        **kwargs,
    )
    
    return model
