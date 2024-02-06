import os
import typing as t

import torch
import numpy as np
import visual_graph_datasets.typing as tv
from torch_geometric.data import Data


def data_from_graph(graph: tv.GraphDict,
                    dtype=torch.float32,
                    ) -> Data:
    """
    Converts a graph dict representation into a ``torch_geometric.data.Data`` instance.
    
    The Data instance will be constructed with the node_attributes, edge_attributes and 
    edge_indices.
    
    :param graph: The graph representation to convert into the Data object
    :param dtype: the torch dtype of the data type to use for the tensor representation of 
        the arrays. Default is float32

    :returns: The Data instance that represents the full graph.
    """
    # In the GraphDict representation, edge_indices is essentially an edge list - a list of 
    # 2-tuples (i, j) which defines an edge from node of index i to node of index j. As 
    # an array this data structure has the shape (E, 2). However, pytorch geometric expects 
    # the edge indices to be defined as a structure of the shape (2, E) so we need to transpose 
    # it here first.
    edge_indices = np.transpose(graph['edge_indices'])
    
    # 24.01.24
    # The graph labels are not always going to be given and we want to support that here as well
    if 'graph_labels' in graph:
        y = torch.tensor(graph['graph_labels'], dtype=dtype)
    else:
        y = torch.tensor([0, ], dtype=dtype)
    
    data = Data(
        x=torch.tensor(graph['node_attributes'], dtype=dtype),
        y=y,
        edge_attr=torch.tensor(graph['edge_attributes'], dtype=dtype),
        edge_index=torch.tensor(edge_indices, dtype=torch.int64),
    )
    
    # After constructing the Data instance, it is possible to attach additional optional 
    # attributes to it. So in the case that there are canonical node and edge explanations 
    # attached to the graph, these are also being added.
    if 'node_importances' in graph:
        data.node_importances = graph['node_importances']
    
    if 'edge_importances' in graph:
        data.edge_importances = graph['edge_importances']
    
    return data


def data_list_from_graphs(graphs: t.List[tv.GraphDict],
                          ) -> t.List[Data]:
    """
    Given a list ``graphs`` of GraphDict graph representations, this function will process those into 
    ``torch_geometric.data.Data`` instances so that they can be used directly for the training of a 
    neural network.
    
    :param graphs: A list of graph dict elements
    
    :returns: A list of Data elements with the same order as the given list of graph dicts
    """
    data_list = []
    for graph in graphs:
        data = data_from_graph(graph)
        data_list.append(data)
        
    return data_list