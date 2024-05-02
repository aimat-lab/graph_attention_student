import pytest
import typing as t

import torch
import numpy as np
import visual_graph_datasets.typing as tv
from torch_geometric.data import Data

from graph_attention_student.testing import get_mock_graphs
from graph_attention_student.torch.data import data_from_graph
from graph_attention_student.torch.data import data_list_from_graphs


@pytest.mark.parametrize('num_graphs', [
    10
])
def test_data_list_from_graphs(num_graphs):
    """
    The ``data_list_from_graphs`` function should be able to convert a list of graphs represented 
    as graph dictionaries into a corresponding list of torch geometric Data objects.
    """
    # this function constructs some sample graph dicts that can be used for testing
    graphs: t.List[tv.GraphDict] = get_mock_graphs(num_graphs)
    
    # This function is supposed to convert all those graphs in the list into torch Data elements
    data_list: t.List[Data] = data_list_from_graphs(graphs)
    
    assert isinstance(data_list, list)
    assert len(data_list) == num_graphs
    
    for data in data_list:
        assert isinstance(data, Data)


def test_data_from_graph_node_coordinates_works():
    """
    The ``data_from_graph`` should now be able to process the custom graph dict property 
    "node_coordinates" and transform it into a tensor and attach it to the Data object 
    as the "coords" property.
    """
    # Some random graph
    graph = {
        'node_indices': np.array([0, 1, 2]),
        'node_attributes': np.array([
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
        ]),
        'node_coordinates': np.array([
            [0.1, 0.2],
            [0.4, 0.5],
            [0.7, 0.8],
        ]),
        'edge_indices': np.array([
            [0, 1],
            [1, 2],
            [2, 0]
        ]),
        'edge_attributes': np.array([
            [0, 1],
            [1, 0],
            [1, 0]
        ]),
        'graph_labels': np.array([
            [1]
        ])
    }
    
    data = data_from_graph(graph)
    
    # The "coords" property should now have been added to the Data object and it should be a 
    # float tensor with the same shape as the given node coordinates in the graph dict.
    assert isinstance(data.coords, torch.Tensor)
    assert data.coords.shape == (3, 2)
    assert data.coords.dtype == torch.float32


def test_data_from_graph():
    """
    The ``data_from_graph`` function should be able to convert a graph dict into a torch Data object.
    """
    graph = {
        'node_indices': np.array([0, 1, 2]),
        'node_attributes': np.array([
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
        ]),
        'edge_indices': np.array([
            [0, 1],
            [1, 2],
            [2, 0]
        ]),
        'edge_attributes': np.array([
            [0, 1],
            [1, 0],
            [1, 0]
        ]),
        'graph_labels': np.array([
            [1]
        ])
    }
    
    data = data_from_graph(graph)
    
    assert isinstance(data, Data)
    # node attributes
    assert isinstance(data.x, torch.Tensor)
    assert data.x.shape == (3, 3)
    # edge attributes
    assert isinstance(data.y, torch.Tensor)
    assert data.edge_attr.shape == (3, 2)
    # target value
    assert isinstance(data.y, torch.Tensor)
    assert data.y.shape == (1, 1)