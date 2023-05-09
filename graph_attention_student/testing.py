import random
import numpy as np


def get_mock_graphs(num: int,
                    num_attributes: int = 3):
    graphs = []
    for i in range(num):
        num_nodes = random.randint(10, 20)
        num_edges = random.randint(10, 20)
        graphs.append({
            'node_indices': np.random.random(size=(num_nodes, 1)),
            'node_attributes': np.random.random(size=(num_nodes, num_attributes)),
            'edge_indices': np.random.randint(0, 1, size=(num_edges, 2)),
            'edge_attributes': np.random.random(size=(num_edges, num_attributes))
        })

    return graphs