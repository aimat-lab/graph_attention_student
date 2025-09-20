import os

from visual_graph_datasets.processing.molecules import MoleculeProcessing
from graph_attention_student.testing import model_from_processing
from graph_attention_student.torch.megan import Megan


def test_model_from_processing_basically_works():
    """
    The ``model_from_processing`` function should create a Megan model when given a Processing instance 
    such that the model fits with the the graphs that are created by that processing instance.
    """
    num_outputs = 3
    num_channels = 4
    
    processing = MoleculeProcessing()
    model = model_from_processing(
        processing=processing,
        num_outputs=num_outputs,
        num_channels=num_channels,
        prediction_mode='classification',
    )
    
    assert model is not None
    assert isinstance(model, Megan)
    
    # The model should be able to process a graph that was created from that processing object
    # graph with 8 atoms/nodes.
    # Set model to evaluation mode to avoid BatchNorm issues with single samples
    model.eval()
    graph = processing.process('CCCC=CCCN')
    info = model.forward_graph(graph)
    assert isinstance(info, dict)
    assert 'graph_output' in info
    assert 'node_importance' in info
    assert 'edge_importance' in info
    
    assert info['graph_output'].shape == (num_outputs,)
    assert info['node_importance'].shape[1] == num_channels
    assert info['edge_importance'].shape[1] == num_channels