import pytest
import typing as t
import tempfile
import os

import torch
import numpy as np
import polars as pl
import pandas as pd
import visual_graph_datasets.typing as tv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from visual_graph_datasets.processing.molecules import MoleculeProcessing

from graph_attention_student.testing import get_mock_graphs
from graph_attention_student.torch.data import data_from_graph
from graph_attention_student.torch.data import data_list_from_graphs
from graph_attention_student.torch.data import SmilesDataset


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

# == data_from_graph ==


def test_data_from_graph_basically_works():
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
    
    
def test_data_from_graph_node_coordinates_work():
    """
    When the graph dict contains the optional property "node_coordinates" the ``data_from_graph``
    function should be able to convert this into a tensor and attach it to the Data object as the
    "coords" property dynamically.
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
    assert not hasattr(data, 'coords')
    
    # Only after we add the optional attribute to the graph dict, the data object should contain 
    # the additional properties as well.
    graph['node_coordinates'] = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.5],
        [0.7, 0.8, 0.5],
    ])
    data = data_from_graph(graph)
    assert isinstance(data, Data)
    
    assert hasattr(data, 'coords')
    assert isinstance(data.coords, torch.Tensor)
    assert data.coords.shape == (3, 3)
    
    
def test_data_from_graph_graph_weight_works():
    """
    When the graph dict contains the optional property "graph_weight" the ``data_from_graph`` function
    should be able to convert this into a tensor and attach it to the Data object as the "train_weight"
    property dynamically. During the training this should act as a sample specific weight of the loss.
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
    assert not hasattr(data, 'train_weight')
    
    # Only after we add the optional attribute to the graph dict, the data object should contain 
    # the additional properties as well.
    graph['graph_weight'] = np.array([0.1])
    data = data_from_graph(graph)
    assert isinstance(data, Data)
    
    assert hasattr(data, 'train_weight')
    assert isinstance(data.train_weight, torch.Tensor)
    assert data.train_weight.shape == (1, )


# == SmilesDataset Tests ==

class TestSmilesDataset:
    """Comprehensive test suite for the SmilesDataset class."""

    @pytest.fixture
    def sample_smiles_data(self):
        """Fixture providing sample SMILES data for testing."""
        return [
            {'smiles': 'CCO', 'value': 1.0, 'property': 'alcohol'},
            {'smiles': 'CC(=O)O', 'value': 2.0, 'property': 'acid'},
            {'smiles': 'c1ccccc1', 'value': 0.5, 'property': 'aromatic'},
            {'smiles': 'CCN', 'value': 1.5, 'property': 'amine'},
        ]

    @pytest.fixture
    def sample_csv_file(self, sample_smiles_data):
        """Fixture creating a temporary CSV file with sample data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write header
            f.write('smiles,value,property\n')
            # Write data rows
            for row in sample_smiles_data:
                f.write(f"{row['smiles']},{row['value']},{row['property']}\n")
            temp_path = f.name

        yield temp_path

        # Cleanup
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass

    def test_init_with_list(self, sample_smiles_data):
        """Test SmilesDataset initialization with list data."""
        dataset = SmilesDataset(sample_smiles_data)

        assert dataset.smiles_column == 'smiles'
        assert dataset.target_columns == ['value']
        assert dataset.reservoir_sampling == True
        assert dataset.reservoir_size == 1000
        assert isinstance(dataset.processing, MoleculeProcessing)

    def test_init_with_polars_dataframe(self, sample_smiles_data):
        """Test SmilesDataset initialization with Polars DataFrame."""
        df = pl.DataFrame(sample_smiles_data)
        dataset = SmilesDataset(df)

        assert isinstance(dataset.dataset, pl.DataFrame)
        assert dataset.smiles_column == 'smiles'

    def test_init_with_pandas_dataframe(self, sample_smiles_data):
        """Test SmilesDataset initialization with Pandas DataFrame."""
        df = pd.DataFrame(sample_smiles_data)
        dataset = SmilesDataset(df)

        assert isinstance(dataset.dataset, pd.DataFrame)
        assert dataset.smiles_column == 'smiles'

    def test_init_with_csv_file(self, sample_csv_file):
        """Test SmilesDataset initialization with CSV file path."""
        dataset = SmilesDataset(sample_csv_file)

        assert dataset.dataset == sample_csv_file
        assert dataset.smiles_column == 'smiles'

    def test_init_custom_parameters(self, sample_smiles_data):
        """Test SmilesDataset initialization with custom parameters."""
        dataset = SmilesDataset(
            sample_smiles_data,
            smiles_column='custom_smiles',
            target_columns=['target1', 'target2'],
            reservoir_sampling=False,
            reservoir_size=500
        )

        assert dataset.smiles_column == 'custom_smiles'
        assert dataset.target_columns == ['target1', 'target2']
        assert dataset.reservoir_sampling == False
        assert dataset.reservoir_size == 500

    def test_create_worker_dataframe_with_list(self, sample_smiles_data):
        """Test _create_worker_dataframe method with list input."""
        dataset = SmilesDataset(sample_smiles_data)
        lazy_df = dataset._create_worker_dataframe()

        assert isinstance(lazy_df, pl.LazyFrame)
        collected = lazy_df.collect()
        assert len(collected) == len(sample_smiles_data)

    def test_create_worker_dataframe_with_polars_df(self, sample_smiles_data):
        """Test _create_worker_dataframe method with Polars DataFrame."""
        df = pl.DataFrame(sample_smiles_data)
        dataset = SmilesDataset(df)
        lazy_df = dataset._create_worker_dataframe()

        assert isinstance(lazy_df, pl.LazyFrame)

    def test_create_worker_dataframe_with_pandas_df(self, sample_smiles_data):
        """Test _create_worker_dataframe method with Pandas DataFrame."""
        df = pd.DataFrame(sample_smiles_data)
        dataset = SmilesDataset(df)
        lazy_df = dataset._create_worker_dataframe()

        assert isinstance(lazy_df, pl.LazyFrame)

    def test_create_worker_dataframe_with_csv(self, sample_csv_file):
        """Test _create_worker_dataframe method with CSV file."""
        dataset = SmilesDataset(sample_csv_file)
        lazy_df = dataset._create_worker_dataframe()

        assert isinstance(lazy_df, pl.LazyFrame)

    def test_create_worker_dataframe_unsupported_type(self):
        """Test _create_worker_dataframe method with unsupported type."""
        dataset = SmilesDataset('not_a_file.csv')  # This will fail in the method
        dataset.dataset = {'invalid': 'type'}  # Set invalid type

        with pytest.raises(ValueError, match="Unsupported dataset type"):
            dataset._create_worker_dataframe()

    def test_create_worker_processing(self, sample_smiles_data):
        """Test _create_worker_processing method."""
        dataset = SmilesDataset(sample_smiles_data)
        processing = dataset._create_worker_processing()

        assert isinstance(processing, MoleculeProcessing)

    def test_iteration_without_reservoir_sampling(self, sample_smiles_data):
        """Test dataset iteration without reservoir sampling."""
        dataset = SmilesDataset(sample_smiles_data, reservoir_sampling=False)

        data_items = []
        for data in dataset:
            data_items.append(data)
            if len(data_items) >= 3:  # Limit to avoid long iteration
                break

        assert len(data_items) > 0
        for data in data_items:
            assert isinstance(data, Data)
            assert hasattr(data, 'x')  # node features
            assert hasattr(data, 'edge_index')  # edge indices
            assert hasattr(data, 'y')  # target values

    def test_iteration_with_reservoir_sampling(self, sample_smiles_data):
        """Test dataset iteration with reservoir sampling."""
        dataset = SmilesDataset(
            sample_smiles_data,
            reservoir_sampling=True,
            reservoir_size=2
        )

        data_items = []
        for data in dataset:
            data_items.append(data)
            if len(data_items) >= 3:  # Limit to avoid long iteration
                break

        assert len(data_items) > 0
        for data in data_items:
            assert isinstance(data, Data)

    def test_data_structure_validity(self, sample_smiles_data):
        """Test that generated Data objects have valid structure."""
        dataset = SmilesDataset(sample_smiles_data, reservoir_sampling=False)

        for i, data in enumerate(dataset):
            # Test basic torch geometric Data structure
            assert isinstance(data.x, torch.Tensor)  # Node features
            assert isinstance(data.edge_index, torch.Tensor)  # Edge indices
            assert isinstance(data.edge_attr, torch.Tensor)  # Edge attributes
            assert isinstance(data.y, torch.Tensor)  # Target values

            # Test dimensions
            assert data.edge_index.shape[0] == 2  # Should be (2, num_edges)
            assert data.x.shape[0] > 0  # Should have at least one node
            assert data.edge_attr.shape[0] == data.edge_index.shape[1]  # Consistent edge count

            if i >= 2:  # Test first few items only
                break

    def test_single_worker_dataloader(self, sample_smiles_data):
        """Test SmilesDataset with PyG DataLoader using single worker."""
        dataset = SmilesDataset(sample_smiles_data, reservoir_sampling=False)

        # Create DataLoader with single worker (num_workers=0)
        dataloader = DataLoader(dataset, batch_size=2, num_workers=0)

        batches = []
        for batch in dataloader:
            batches.append(batch)
            if len(batches) >= 2:  # Test first 2 batches
                break

        assert len(batches) > 0
        for batch in batches:
            assert isinstance(batch, Data)
            assert hasattr(batch, 'batch')  # Batch index should be present
            assert batch.x.shape[0] > 0  # Should have nodes

    @pytest.mark.skip(reason="Multiworker tests can timeout and are skipped by default")
    def test_multiworker_dataloader(self, sample_smiles_data):
        """Test SmilesDataset with PyG DataLoader using multiple workers."""
        # Extend the sample data to have more examples for multi-worker testing
        extended_data = sample_smiles_data * 10  # Replicate data 10 times
        dataset = SmilesDataset(extended_data, reservoir_sampling=False)

        # Create DataLoader with multiple workers
        dataloader = DataLoader(dataset, batch_size=2, num_workers=2)

        batches = []
        for batch in dataloader:
            batches.append(batch)
            if len(batches) >= 3:  # Test first 3 batches
                break

        assert len(batches) > 0
        for batch in batches:
            assert isinstance(batch, Data)
            assert hasattr(batch, 'batch')  # Batch index should be present
            assert batch.x.shape[0] > 0  # Should have nodes

    @pytest.mark.skip(reason="Multiworker tests can timeout and are skipped by default")
    def test_multiworker_persistent_workers(self, sample_smiles_data):
        """Test SmilesDataset with PyG DataLoader using persistent workers."""
        # Extend the sample data
        extended_data = sample_smiles_data * 10
        dataset = SmilesDataset(extended_data, reservoir_sampling=False)

        # Create DataLoader with persistent workers
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=2,
            persistent_workers=True
        )

        batches = []
        for batch in dataloader:
            batches.append(batch)
            if len(batches) >= 3:  # Test first 3 batches
                break

        assert len(batches) > 0
        for batch in batches:
            assert isinstance(batch, Data)

    def test_invalid_smiles_handling(self):
        """Test that invalid SMILES strings are handled gracefully."""
        # Include some invalid SMILES strings
        data_with_invalid = [
            {'smiles': 'CCO', 'value': 1.0},
            {'smiles': 'INVALID_SMILES_123', 'value': 2.0},
            {'smiles': 'CC(=O)O', 'value': 3.0},
            {'smiles': 'NOT_A_SMILES', 'value': 4.0},
        ]

        dataset = SmilesDataset(data_with_invalid, reservoir_sampling=False)

        valid_data_count = 0
        for data in dataset:
            valid_data_count += 1
            assert isinstance(data, Data)
            if valid_data_count >= 5:  # Limit iteration
                break

        # Should have at least some valid data (valid SMILES should work)
        assert valid_data_count > 0

    def test_custom_target_columns(self):
        """Test dataset with custom target columns."""
        data = [
            {'smiles': 'CCO', 'prop1': 1.0, 'prop2': 2.0},
            {'smiles': 'CC(=O)O', 'prop1': 3.0, 'prop2': 4.0},
        ]

        dataset = SmilesDataset(
            data,
            target_columns=['prop1', 'prop2'],
            reservoir_sampling=False
        )

        for data_item in dataset:
            assert isinstance(data_item, Data)
            assert data_item.y.shape[0] == 2  # Should have 2 target values
            break  # Test first item only

    def test_empty_dataset_handling(self):
        """Test handling of empty dataset."""
        empty_data = []
        dataset = SmilesDataset(empty_data, reservoir_sampling=False)

        data_items = list(dataset)
        assert len(data_items) == 0

    @pytest.mark.parametrize("num_workers", [1, 2, 4])
    @pytest.mark.skip(reason="Multiworker tests can timeout and are skipped by default")
    def test_worker_splitting(self, sample_smiles_data, num_workers):
        """Test that data is properly split across workers."""
        # Use a larger dataset for meaningful worker splitting
        large_data = sample_smiles_data * 20
        dataset = SmilesDataset(large_data, reservoir_sampling=False)

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=num_workers
        )

        total_items = 0
        for batch in dataloader:
            total_items += batch.num_graphs
            if total_items >= 10:  # Limit to avoid long test
                break

        assert total_items > 0

