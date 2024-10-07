import torch

from graph_attention_student.torch.layers import GraphAttentionLayer
from graph_attention_student.testing import get_mock_graphs
from graph_attention_student.torch.data import data_list_from_graphs


def test_graph_attention_layer_forward_pass():
    """
    Checks if the ``GraphAttentionLayer`` class can be instantiated and can perform a forward 
    pass without any errors.
    """
    
    # Setting up some example parameters (arbitrarily chosen) for testing
    in_dim: int = 5
    out_dim: int = 3
    edge_dim: int = 4
    batch_size: int = 10
    
    # creating random mock input data to test if there are no principal errors 
    # during the model forward pass.
    graphs = get_mock_graphs(
        num=batch_size,
        num_node_attributes=in_dim,
        num_edge_attributes=edge_dim,
    )
    data_list = data_list_from_graphs(graphs)
    
    layer = GraphAttentionLayer(
        in_dim=in_dim,
        out_dim=out_dim,
        edge_dim=edge_dim,
    )
    for graph, data in zip(graphs, data_list):
        num_nodes = len(graph['node_indices'])
        num_edges = len(graph['edge_indices'])
        
        node_embeddings, attention = layer(
            x=data.x,
            edge_attr=data.edge_attr,
            edge_index=data.edge_index,
        )
                
        # The node embeddings tensor that it returned by the tensor should have the same number of 
        # nodes as the input structure and additional should transform the second dimension from
        # the original "in_dim" to now "out_dim"
        assert isinstance(node_embeddings, torch.Tensor)
        assert node_embeddings.shape == (num_nodes, out_dim)
        
        # The attention tensor is acually an EDGE tensor, so there should be exactly one (second dimension) 
        # attention value for each edge of the graph.
        assert isinstance(attention, torch.Tensor)
        assert attention.shape == (num_edges, 1)