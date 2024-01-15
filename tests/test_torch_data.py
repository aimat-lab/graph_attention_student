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
    
    # this function constructs some sample graph dicts that can be used for testing
    graphs: t.List[tv.GraphDict] = get_mock_graphs(num_graphs)
    
    # This function is supposed to convert all those graphs in the list into torch Data elements
    data_list: t.List[Data] = data_list_from_graphs(graphs)
    
    assert isinstance(data_list, list)
    assert len(data_list) == num_graphs
    
    for data in data_list:
        assert isinstance(data, Data)


def test_data_from_graph():
    
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