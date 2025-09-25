import os
import pytest
import random
import tempfile
import typing as t
from rich import print as pprint

import torch
import numpy as np
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from graph_attention_student.visualization import plot_leave_one_out_analysis
from graph_attention_student.testing import get_mock_graphs
from graph_attention_student.torch.data import data_list_from_graphs
from graph_attention_student.torch.model import AbstractGraphModel
from graph_attention_student.torch.megan import Megan, MeganEnsemble

from .util import ARTIFACTS_PATH


def test_saving_loading_megan_works():
    
    model = Megan(
        node_dim=3,
        edge_dim=3,
        units=[32, 32, 32],
        num_channels=2,
        final_units=[32, 1],
    )
    
    with tempfile.TemporaryDirectory() as path:
        
        model_path = os.path.join(path, 'megan.ckpt')
        model.save(model_path)
        assert os.path.exists(model_path)
        
        data = torch.load(model_path)
        assert 'hyper_parameters' in data
        assert 'state_dict' in data
        pprint(data['hyper_parameters'])
        pprint(data['state_dict'])
        
        loaded_model = Megan.load(model_path)
        assert loaded_model is not None
        assert isinstance(loaded_model, Megan)


@pytest.mark.parametrize('num_graphs, node_dim, edge_dim, output_dim', [
    (100, 10, 4, 2),
])
def test_megan_training_sample_weights_work(num_graphs, node_dim, edge_dim, output_dim):
    """
    When adding "graph_weight" property to the graph dicts, this should automatically be converted into 
    the tensor property "train_weight" in the Data instances and this should in turn be recognized by the 
    megan model which will then apply those sample weights during the training_step method.
    """
    # ~ test configuration
    num_channels = 2
    embedding_dim = 32

    graphs = get_mock_graphs(
        num=num_graphs,
        num_node_attributes=node_dim,
        num_edge_attributes=edge_dim,
    )
    # When the graph dict has the "graph_weight" attribute, this should automatically be detected 
    # by the "data_list_from_graphs" function which should then add those weights as a tensor 
    # attribute "train_weight" to the Data instances
    for graph in graphs:
        graph['graph_weight'] = random.random()
    
    data_list = data_list_from_graphs(graphs)
    for data in data_list:
        assert hasattr(data, 'train_weight')
        assert isinstance(data.train_weight, torch.Tensor)
    
    # constructing the megan model
    loader = DataLoader(data_list, batch_size=32, shuffle=False)
    model = Megan(
        node_dim=node_dim,
        edge_dim=edge_dim,
        units=[32, 32, embedding_dim],
        num_channels=num_channels, # for classification co-training must num_channels==num_outputs
        final_units=[32, output_dim],
    )
    assert isinstance(model, Megan)
    assert isinstance(model, AbstractGraphModel)
    
    # performing one train step with the given data and the result from the training step function
    # should be a valid total loss value.
    for data in loader:
        loss = model.training_step(data, 0)
        assert isinstance(loss, torch.Tensor)


@pytest.mark.parametrize('num_graphs, node_dim, edge_dim, output_dim', [
    (100, 10, 4, 3),
])
def test_megan_leave_one_out_deviations(num_graphs, node_dim, edge_dim, output_dim):
    """
    The "leave_one_out_deviations" method for the prediction of B graphs, C outputs and K channels
    should return a numpy array of shape (B, C, K) where the values are the deviations of the
    predictions when leaving out each channel for the prediction of the network.
    """
    # ~ test configuration
    num_channels = 2
    embedding_dim = 32

    graphs = get_mock_graphs(
        num=num_graphs,
        num_node_attributes=node_dim,
        num_edge_attributes=edge_dim,
    )
    
    model = Megan(
        node_dim=node_dim,
        edge_dim=edge_dim,
        units=[32, 32, embedding_dim],
        num_channels=num_channels, # for classification co-training must num_channels==num_outputs
        final_units=[32, output_dim],
    )
    assert isinstance(model, Megan)
    assert isinstance(model, AbstractGraphModel)
    
    # ~ actually testing the leave_one_out_deviations method
    
    deviations = model.leave_one_out_deviations(graphs)
    assert isinstance(deviations, np.ndarray)
    assert deviations.shape == (num_graphs, output_dim, num_channels)

    fig = plot_leave_one_out_analysis(
        deviations, 
        num_targets=output_dim, 
        num_channels=num_channels
    )
    fig.savefig(os.path.join(ARTIFACTS_PATH, f'torch_megan_leave_one_out__{output_dim}_{num_channels}.pdf'))
    

@pytest.mark.parametrize('num_graphs, node_dim, edge_dim', [
    (100, 10, 4),
])
def test_megan_classification_explanation_training_works(num_graphs, node_dim, edge_dim):
    """
    It should be possible to apply the Megan explanation co-training routine for a classification task.
    This test tries to do a simple mock training for a classification task with 3 classes and 
    consequently 3 explanations.
    """
    # ~ test configuration
    embedding_dim = 32

    graphs = get_mock_graphs(
        num=num_graphs,
        num_node_attributes=node_dim,
        num_edge_attributes=edge_dim,
    )
    # Giving each graph an actual target value for the classification
    targets = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    output_dim = len(targets)
    for graph in graphs:
        graph['graph_labels'] = np.array(random.choice(targets), dtype=float)
    
    data_list = data_list_from_graphs(graphs)
    loader = DataLoader(data_list, batch_size=32, shuffle=False)
    model = Megan(
        node_dim=node_dim,
        edge_dim=edge_dim,
        units=[32, 32, embedding_dim],
        num_channels=output_dim, # for classification co-training must num_channels==num_outputs
        importance_units=[32],
        importance_factor=1.0,
        prediction_mode='classification',
        importance_mode='classification',
        regression_reference=0.0,
        final_units=[32, output_dim],
    )
    assert isinstance(model, Megan)
    assert isinstance(model, AbstractGraphModel)

    trainer = pl.Trainer(max_epochs=3, accelerator='cpu')
    trainer.fit(model, train_dataloaders=loader)


@pytest.mark.parametrize('num_graphs, node_dim, edge_dim', [
    (100, 10, 4),
])
def test_megan_regression_explanation_training_works(num_graphs, node_dim, edge_dim):
    """
    It should be possible to apply the Megan explanation co-training routine for a regression task.
    This test tries to do a simple mock training for a regression task and a single target value and
    consequently 2 explanations (negative and positive).
    """
    # ~ test configuration
    embedding_dim = 32
    output_dim = 1

    graphs = get_mock_graphs(
        num=num_graphs,
        num_node_attributes=node_dim,
        num_edge_attributes=edge_dim,
        num_outputs=output_dim,
    )
    # Giving each graph an actual target value for the regression
    for graph in graphs:
        graph['graph_labels'] = np.random.uniform(-1, 1, size=(1, ))
    
    data_list = data_list_from_graphs(graphs)
    loader = DataLoader(data_list, batch_size=32, shuffle=False)
    model = Megan(
        node_dim=node_dim,
        edge_dim=edge_dim,
        units=[32, 32, embedding_dim],
        num_channels=2, # for regression co-training only num_channels==2 is supported!
        importance_units=[32],
        importance_factor=1.0,
        prediction_mode='regression',
        importance_mode='regression',
        regression_reference=0.0,
        sparsity_factor=1.0,
        final_units=[32, 1],
    )
    assert isinstance(model, Megan)
    assert isinstance(model, AbstractGraphModel)

    trainer = pl.Trainer(max_epochs=3, accelerator='cpu')
    trainer.fit(model, train_dataloaders=loader)
    

@pytest.mark.parametrize('num_graphs, node_dim, edge_dim, num_channels', [
    (100, 10, 4, 2),
    (100, 20, 5, 3),
])
def test_megan_training_works(num_graphs, node_dim, edge_dim, num_channels):
    """
    Tests if the ``Megan`` model training process works without errors. This case purely uses the model 
    for the primary prediction without explanation co-training.
    """
    # ~ test configuration
    embedding_dim = 32
    output_dim = 1

    graphs = get_mock_graphs(
        num=num_graphs,
        num_node_attributes=node_dim,
        num_edge_attributes=edge_dim,
        num_outputs=output_dim,
    )
    data_list = data_list_from_graphs(graphs)
    loader = DataLoader(data_list, batch_size=32, shuffle=False)

    model = Megan(
        node_dim=node_dim,
        edge_dim=edge_dim,
        units=[32, 32, embedding_dim],
        num_channels=num_channels,
        importance_factor=0.0,
        final_units=[32, 1],
    )

    trainer = pl.Trainer(max_epochs=3, accelerator='cpu')
    trainer.fit(model, train_dataloaders=loader)
    

@pytest.mark.parametrize('num_graphs, node_dim, edge_dim, num_channels', [
    (100, 10, 4, 2),
])
def test_megan_basically_works(num_graphs, node_dim, edge_dim, num_channels):
    """
    This tests constructs a Megan model with some default parameters, executs the forward pass and 
    tests if that works without errors.
    """
    graphs = get_mock_graphs(
        num=num_graphs,
        num_node_attributes=node_dim,
        num_edge_attributes=edge_dim,
    )    
    output_dim = 1
    embedding_dim = 32
    model = Megan(
        node_dim=node_dim,
        edge_dim=edge_dim,
        units=[32, 32, embedding_dim],
        projection_units=[embedding_dim, ],
        num_channels=num_channels,
        importance_units=[32],
        final_units=[32, 1],
    )
    # In the first instance, the Megan model should be a direct implementation of the AbstractGraphModel
    # base class, which itself should be a LightningModule.
    assert isinstance(model, AbstractGraphModel)
    assert isinstance(model, pl.LightningModule)
    
    # The "forward_model" method is essentially the method which just maps the raw output of the "forward"
    # method itself onto the graph level. The result should be a list of dicts that contains all the 
    # prediction information about the various graphs.
    results: t.List[dict] = model.forward_graphs(graphs)
    for graph, result in zip(graphs, results):
        
        num_nodes = len(graph['node_attributes'])
                
        assert isinstance(result, dict)
        assert len(result) != 0
        
        assert 'graph_output' in result
        assert result['graph_output'].shape == (1, )
        
        assert 'graph_embedding' in result
        assert result['graph_embedding'].shape == (embedding_dim, num_channels)
        
        assert 'node_importance' in result
        assert result['node_importance'].shape == (num_nodes, num_channels)

    # The "predict_graphs" method essentially builds on forward_graphs, but returns a single numpy array 
    # with the shape (B, O) where B is the number of graphs and O the output dimension
    out_pred = model.predict_graphs(graphs)
    assert isinstance(out_pred, np.ndarray)
    assert out_pred.shape == (num_graphs, output_dim)


class TestMeganEnsemble:
    """
    Tests for the MeganEnsemble class, which can be used to bundle multiple individual Megan models into 
    an ensemble to get uncertainty estimates.
    """
    
    def test_saving_loading_megan_ensemble_works(self):
        """
        MeganEnsemble.save should save the ensemble model to the given path as a torch archive file,
        and it should be possible to reconstruct the model from the saved file.
        Additionally, it checks if the forward pass results are the same before saving and after loading.
        """
        # Create mock Megan models
        model1 = Megan(node_dim=3, edge_dim=3, units=[32, 32, 32], num_channels=2, final_units=[32, 1])
        model2 = Megan(node_dim=3, edge_dim=3, units=[32, 32, 32], num_channels=2, final_units=[32, 1])

        # Create MeganEnsemble
        ensemble_model = MeganEnsemble(models=[model1, model2])

        # Create a dummy graph for testing
        graphs = get_mock_graphs(num=1, num_node_attributes=3, num_edge_attributes=3)
        
        # Perform a forward pass before saving
        ensemble_model.eval()
        before_save_result = ensemble_model.forward_graphs(graphs)[0]['graph_output']

        with tempfile.TemporaryDirectory() as path:
            model_path = os.path.join(path, 'megan_ensemble.ckpt')
            ensemble_model.save(model_path)
            assert os.path.exists(model_path)

            # Load the model
            loaded_ensemble_model = MeganEnsemble.load(model_path)
            assert loaded_ensemble_model is not None
            assert isinstance(loaded_ensemble_model, MeganEnsemble)

            # Perform a forward pass after loading
            loaded_ensemble_model.eval()
            after_load_result = loaded_ensemble_model.forward_graphs(graphs)[0]['graph_output']
            

        # Compare the results before saving and after loading
        assert np.allclose(before_save_result, after_load_result), \
            "The forward pass results are different before saving and after loading."

    @pytest.mark.parametrize('num_graphs, node_dim, edge_dim', [(10, 3, 3)])
    def test_megan_ensemble_forward_graphs_works(self, num_graphs, node_dim, edge_dim):
        """
        MeganEnsemble.forward_graphs should execute the forward pass for a list of graphs without errors
        and return a list of dictionaries containing the results.
        """
        # Create mock Megan models
        model1 = Megan(node_dim=node_dim, edge_dim=edge_dim, units=[32, 32, 32], num_channels=2, final_units=[32, 1])
        model2 = Megan(node_dim=node_dim, edge_dim=edge_dim, units=[32, 32, 32], num_channels=2, final_units=[32, 1])

        # Create MeganEnsemble
        ensemble_model = MeganEnsemble(models=[model1, model2])

        # Create mock graphs
        graphs = get_mock_graphs(num=num_graphs, num_node_attributes=node_dim, num_edge_attributes=edge_dim)

        # Execute forward_graphs
        results = ensemble_model.forward_graphs(graphs)

        # Assertions
        assert isinstance(results, list)
        assert len(results) == num_graphs
        for result in results:
            assert isinstance(result, dict)
            assert 'graph_output' in result
            assert 'node_importance' in result
            assert 'edge_importance' in result
            assert 'graph_uncertainty' in result
            assert 'node_importance_std' in result
            assert 'edge_importance_std' in result

    def test_megan_ensemble_forward_captures_mve_status(self):
        """
        Test that MeganEnsemble.forward correctly captures the MVE status of individual member models
        when calculating uncertainty.
        """
        # Create mock Megan models with MVE status
        model1 = Megan(node_dim=3, edge_dim=3, units=[32, 32, 32], num_channels=2, final_units=[32, 1])
        model2 = Megan(node_dim=3, edge_dim=3, units=[32, 32, 32], num_channels=2, final_units=[32, 1])
        model1.mve_active = True
        model2.mve_active = True

        # Create MeganEnsemble
        ensemble_model = MeganEnsemble(models=[model1, model2])

        # Create a dummy graph for testing
        graphs = get_mock_graphs(num=1, num_node_attributes=3, num_edge_attributes=3)

        # Perform a forward pass
        ensemble_model.eval()
        results = ensemble_model.forward_graphs(graphs)

        # basic Assertions
        assert isinstance(results, list)
        assert len(results) == 1
        result = results[0]
        assert 'graph_output' in result
        assert 'graph_uncertainty' in result
        assert result['graph_uncertainty'] is not None
        
        # Now, the results should also contain the "graph_variance_stack" property which should consist of 
        # exactly two tensors that are stacked since there are two members in the ensemble here.
        assert 'graph_variance_stack' in result
        assert isinstance(result['graph_variance_stack'], np.ndarray)
        assert result['graph_variance_stack'].shape[-1] == 2
        
    def test_megan_ensemble_forward_works_without_mve(self):
        # Create mock Megan models with MVE status
        model1 = Megan(node_dim=3, edge_dim=3, units=[32, 32, 32], num_channels=2, final_units=[32, 1])
        model2 = Megan(node_dim=3, edge_dim=3, units=[32, 32, 32], num_channels=2, final_units=[32, 1])
        # NOTE: We explicitly set the MVE status to False here!
        model1.mve_active = False
        model2.mve_active = False

        # Create MeganEnsemble
        ensemble_model = MeganEnsemble(models=[model1, model2])

        # Create a dummy graph for testing
        graphs = get_mock_graphs(num=1, num_node_attributes=3, num_edge_attributes=3)

        # Perform a forward pass
        ensemble_model.eval()
        results = ensemble_model.forward_graphs(graphs)

        # basic Assertions
        assert isinstance(results, list)
        assert len(results) == 1
        result = results[0]
        assert 'graph_output' in result
        assert 'graph_uncertainty' in result
        assert result['graph_uncertainty'] is not None
        
        # Now - without mve capable members - this property should also not be included in the 
        # results here! 
        assert 'graph_variance_stack' not in result


class TestMegan:
    
    @pytest.mark.skip(reason="Method _predict_graphs_approximate does not exist in Megan class")
    @pytest.mark.parametrize('num_graphs, node_dim, edge_dim, num_channels', [(10, 3, 3, 2)])
    def test_predict_graphs_approximate_works(self, num_graphs, node_dim, edge_dim, num_channels):
        """
        Test that the _predict_graphs_approximate method returns the correct shapes and types.
        """
        # Create a Megan model
        model = Megan(node_dim=node_dim, edge_dim=edge_dim, units=[32, 32, 32], num_channels=num_channels, final_units=[32, 1], importance_mode='regression')

        # Create mock graphs
        graphs = get_mock_graphs(num=num_graphs, num_node_attributes=node_dim, num_edge_attributes=edge_dim)

        # Create mock labels
        labels = np.random.uniform(-3, 3, size=(num_graphs, 1))

        # Call the method
        out_true, out_pred = model._predict_graphs_approximate(graphs, labels)
        
        # Compute the accuracy
        accuracy = np.mean((np.round(out_pred) == out_true))
        print(f"Accuracy: {accuracy}")

        # Assertions
        assert isinstance(out_true, np.ndarray)
        assert isinstance(out_pred, np.ndarray)
        assert out_true.shape == (num_graphs, num_channels)
        assert out_pred.shape == (num_graphs, num_channels)
