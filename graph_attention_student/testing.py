import typing as t

import random
import numpy as np
import visual_graph_datasets.typing as tv


def get_mock_graphs(num: int,
                    num_node_attributes: int = 3,
                    num_edge_attributes: int = 1
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
        edge_indices = np.column_stack((node_indices, np.roll(node_indices, -1)))
        num_edges = len(edge_indices)
        edge_attributes = np.random.random(size=(num_edges, num_edge_attributes)) 
        
        graph = {
            'node_indices': node_indices,
            'node_attributes': node_attributes,
            'edge_indices': edge_indices,
            'edge_attributes': edge_attributes,
            'graph_labels': [1]
        }
        graphs.append(graph)

    return graphs