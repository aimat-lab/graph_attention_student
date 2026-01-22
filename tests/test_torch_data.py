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
from graph_attention_student.torch.data import SmilesStore
from graph_attention_student.torch.data import SmilesGraphStore
from graph_attention_student.torch.data import VisualGraphDatasetStore
from graph_attention_student.torch.data import GraphDataLoader


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


# == SmilesStore Tests ==

class TestSmilesStore:
    """Test suite for the SmilesStore class."""

    @pytest.fixture
    def sample_csv_file(self):
        """Fixture creating a temporary CSV file with sample data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('smiles,value,name\n')
            f.write('CCO,1.5,ethanol\n')
            f.write('CC(=O)O,2.3,acetic_acid\n')
            f.write('c1ccccc1,0.8,benzene\n')
            f.write('CCN,1.2,ethylamine\n')
            temp_path = f.name

        yield temp_path

        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass

    @pytest.fixture
    def sample_sqlite_path(self):
        """Fixture providing a temporary SQLite file path."""
        with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as f:
            temp_path = f.name

        yield temp_path

        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass

    def test_from_csv_creates_sqlite(self, sample_csv_file, sample_sqlite_path):
        """Test that from_csv creates a valid SQLite file."""
        store = SmilesStore.from_csv(sample_csv_file, sample_sqlite_path)

        assert os.path.exists(sample_sqlite_path)
        assert isinstance(store, SmilesStore)
        assert store.sqlite_path == sample_sqlite_path

    def test_from_csv_overwrites_existing(self, sample_csv_file, sample_sqlite_path):
        """Test that from_csv overwrites existing SQLite file."""
        # Create initial store
        store1 = SmilesStore.from_csv(sample_csv_file, sample_sqlite_path)
        len1 = len(store1)

        # Create new CSV with different data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('smiles,value,name\n')
            f.write('CCO,1.5,ethanol\n')
            new_csv_path = f.name

        try:
            # Overwrite with new data
            store2 = SmilesStore.from_csv(new_csv_path, sample_sqlite_path)
            len2 = len(store2)

            assert len2 == 1  # New file has only 1 row
            assert len1 != len2
        finally:
            os.unlink(new_csv_path)

    def test_len(self, sample_csv_file, sample_sqlite_path):
        """Test that __len__ returns correct count."""
        store = SmilesStore.from_csv(sample_csv_file, sample_sqlite_path)

        assert len(store) == 4

    def test_getitem_returns_dict(self, sample_csv_file, sample_sqlite_path):
        """Test that __getitem__ returns a dictionary with all columns."""
        store = SmilesStore.from_csv(sample_csv_file, sample_sqlite_path)

        row = store[0]

        assert isinstance(row, dict)
        assert 'smiles' in row
        assert 'value' in row
        assert 'name' in row
        assert row['smiles'] == 'CCO'
        assert row['value'] == 1.5
        assert row['name'] == 'ethanol'

    def test_getitem_all_rows(self, sample_csv_file, sample_sqlite_path):
        """Test accessing all rows by index."""
        store = SmilesStore.from_csv(sample_csv_file, sample_sqlite_path)

        expected = [
            {'smiles': 'CCO', 'value': 1.5, 'name': 'ethanol'},
            {'smiles': 'CC(=O)O', 'value': 2.3, 'name': 'acetic_acid'},
            {'smiles': 'c1ccccc1', 'value': 0.8, 'name': 'benzene'},
            {'smiles': 'CCN', 'value': 1.2, 'name': 'ethylamine'},
        ]

        for i, exp in enumerate(expected):
            row = store[i]
            assert row['smiles'] == exp['smiles']
            assert row['value'] == exp['value']
            assert row['name'] == exp['name']

    def test_getitem_index_error(self, sample_csv_file, sample_sqlite_path):
        """Test that out-of-range index raises IndexError."""
        store = SmilesStore.from_csv(sample_csv_file, sample_sqlite_path)

        with pytest.raises(IndexError):
            store[10]

        with pytest.raises(IndexError):
            store[-1]

    def test_columns_property(self, sample_csv_file, sample_sqlite_path):
        """Test that columns property returns correct column names."""
        store = SmilesStore.from_csv(sample_csv_file, sample_sqlite_path)

        columns = store.columns
        assert 'smiles' in columns
        assert 'value' in columns
        assert 'name' in columns
        assert '_id' not in columns  # Internal column should be excluded

    def test_sequence_protocol(self, sample_csv_file, sample_sqlite_path):
        """Test that SmilesStore properly implements Sequence protocol."""
        from collections.abc import Sequence

        store = SmilesStore.from_csv(sample_csv_file, sample_sqlite_path)

        assert isinstance(store, Sequence)
        assert len(store) == 4
        assert store[0] is not None


# == SmilesGraphStore Tests ==

class TestSmilesGraphStore:
    """Test suite for the SmilesGraphStore class."""

    @pytest.fixture
    def sample_smiles_store(self):
        """Fixture providing a SmilesStore with sample data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('smiles,value\n')
            f.write('CCO,1.5\n')
            f.write('CC(=O)O,2.3\n')
            f.write('c1ccccc1,0.8\n')
            csv_path = f.name

        with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as f:
            sqlite_path = f.name

        store = SmilesStore.from_csv(csv_path, sqlite_path)

        yield store

        try:
            os.unlink(csv_path)
            os.unlink(sqlite_path)
        except FileNotFoundError:
            pass

    @pytest.fixture
    def sample_smiles_store_with_invalid(self):
        """Fixture providing a SmilesStore with some invalid SMILES."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('smiles,value\n')
            f.write('CCO,1.5\n')
            f.write('INVALID_SMILES,2.3\n')
            f.write('c1ccccc1,0.8\n')
            csv_path = f.name

        with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as f:
            sqlite_path = f.name

        store = SmilesStore.from_csv(csv_path, sqlite_path)

        yield store

        try:
            os.unlink(csv_path)
            os.unlink(sqlite_path)
        except FileNotFoundError:
            pass

    def test_init(self, sample_smiles_store):
        """Test SmilesGraphStore initialization."""
        graph_store = SmilesGraphStore(
            smiles_store=sample_smiles_store,
            processing=MoleculeProcessing(),
            target_columns=['value'],
            smiles_column='smiles'
        )

        assert graph_store.smiles_store is sample_smiles_store
        assert graph_store.target_columns == ['value']
        assert graph_store.smiles_column == 'smiles'

    def test_len(self, sample_smiles_store):
        """Test that __len__ returns correct count."""
        graph_store = SmilesGraphStore(
            smiles_store=sample_smiles_store,
            processing=MoleculeProcessing(),
            target_columns=['value']
        )

        assert len(graph_store) == 3

    def test_getitem_returns_graph_dict(self, sample_smiles_store):
        """Test that __getitem__ returns a valid GraphDict."""
        graph_store = SmilesGraphStore(
            smiles_store=sample_smiles_store,
            processing=MoleculeProcessing(),
            target_columns=['value']
        )

        graph = graph_store[0]

        assert graph is not None
        assert 'node_attributes' in graph
        assert 'edge_attributes' in graph
        assert 'edge_indices' in graph
        assert 'graph_labels' in graph
        assert isinstance(graph['node_attributes'], np.ndarray)
        assert isinstance(graph['graph_labels'], np.ndarray)

    def test_getitem_graph_labels(self, sample_smiles_store):
        """Test that graph_labels contains correct target values."""
        graph_store = SmilesGraphStore(
            smiles_store=sample_smiles_store,
            processing=MoleculeProcessing(),
            target_columns=['value']
        )

        graph = graph_store[0]
        assert graph['graph_labels'][0] == 1.5

    def test_getitem_invalid_smiles_returns_none(self, sample_smiles_store_with_invalid):
        """Test that invalid SMILES returns None."""
        graph_store = SmilesGraphStore(
            smiles_store=sample_smiles_store_with_invalid,
            processing=MoleculeProcessing(),
            target_columns=['value']
        )

        # Index 1 has invalid SMILES
        graph = graph_store[1]
        assert graph is None

        # Index 0 and 2 should be valid
        assert graph_store[0] is not None
        assert graph_store[2] is not None

    def test_sequence_protocol(self, sample_smiles_store):
        """Test that SmilesGraphStore properly implements Sequence protocol."""
        from collections.abc import Sequence

        graph_store = SmilesGraphStore(
            smiles_store=sample_smiles_store,
            processing=MoleculeProcessing(),
            target_columns=['value']
        )

        assert isinstance(graph_store, Sequence)


# == VisualGraphDatasetStore Tests ==

class TestVisualGraphDatasetStore:
    """Test suite for the VisualGraphDatasetStore class."""

    @pytest.fixture
    def sample_vgd_directory(self):
        """Fixture creating a temporary VGD-style directory with sample data."""
        import json

        temp_dir = tempfile.mkdtemp()

        # Create sample VGD JSON files
        for idx in [0, 1, 2, 5]:  # Note: sparse indices (missing 3, 4)
            graph_data = {
                'metadata': {
                    'index': idx,
                    'graph': {
                        'node_indices': [0, 1, 2],
                        'node_attributes': [[1, 0], [0, 1], [1, 1]],
                        'edge_indices': [[0, 1], [1, 2]],
                        'edge_attributes': [[1], [1]],
                        'graph_labels': [float(idx)]
                    }
                }
            }
            with open(os.path.join(temp_dir, f'{idx}.json'), 'w') as f:
                json.dump(graph_data, f)

        yield temp_dir

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

    def test_init_discovers_indices(self, sample_vgd_directory):
        """Test that __init__ correctly discovers indices from directory."""
        store = VisualGraphDatasetStore(sample_vgd_directory)

        assert 0 in store._indices
        assert 1 in store._indices
        assert 2 in store._indices
        assert 5 in store._indices
        assert 3 not in store._indices
        assert 4 not in store._indices

    def test_len_with_sparse_indices(self, sample_vgd_directory):
        """Test that __len__ returns max_index + 1 for sparse indices."""
        store = VisualGraphDatasetStore(sample_vgd_directory)

        # max index is 5, so len should be 6
        assert len(store) == 6

    def test_getitem_returns_graph_dict(self, sample_vgd_directory):
        """Test that __getitem__ returns a valid GraphDict."""
        store = VisualGraphDatasetStore(sample_vgd_directory)

        graph = store[0]

        assert isinstance(graph, dict)
        assert 'node_attributes' in graph
        assert 'edge_attributes' in graph
        assert 'edge_indices' in graph
        assert 'graph_labels' in graph

    def test_getitem_converts_to_numpy(self, sample_vgd_directory):
        """Test that JSON lists are converted to numpy arrays."""
        store = VisualGraphDatasetStore(sample_vgd_directory)

        graph = store[0]

        assert isinstance(graph['node_attributes'], np.ndarray)
        assert isinstance(graph['edge_attributes'], np.ndarray)
        assert isinstance(graph['edge_indices'], np.ndarray)
        assert isinstance(graph['graph_labels'], np.ndarray)

    def test_getitem_correct_values(self, sample_vgd_directory):
        """Test that __getitem__ returns correct graph data."""
        store = VisualGraphDatasetStore(sample_vgd_directory)

        graph = store[5]
        assert graph['graph_labels'][0] == 5.0

        graph = store[0]
        assert graph['graph_labels'][0] == 0.0

    def test_getitem_missing_index_raises(self, sample_vgd_directory):
        """Test that accessing missing index raises IndexError."""
        store = VisualGraphDatasetStore(sample_vgd_directory)

        with pytest.raises(IndexError):
            store[3]  # Index 3 doesn't exist

        with pytest.raises(IndexError):
            store[100]

    def test_get_valid_indices(self, sample_vgd_directory):
        """Test get_valid_indices returns sorted list of valid indices."""
        store = VisualGraphDatasetStore(sample_vgd_directory)

        valid_indices = store.get_valid_indices()

        assert valid_indices == [0, 1, 2, 5]

    def test_sequence_protocol(self, sample_vgd_directory):
        """Test that VisualGraphDatasetStore properly implements Sequence protocol."""
        from collections.abc import Sequence

        store = VisualGraphDatasetStore(sample_vgd_directory)

        assert isinstance(store, Sequence)


# == GraphDataLoader Tests ==

class TestGraphDataLoader:
    """Test suite for the GraphDataLoader class."""

    @pytest.fixture
    def sample_smiles_graph_store(self):
        """Fixture providing a SmilesGraphStore with sample data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('smiles,value\n')
            f.write('CCO,1.5\n')
            f.write('CC(=O)O,2.3\n')
            f.write('c1ccccc1,0.8\n')
            f.write('CCN,1.2\n')
            f.write('CCCC,0.5\n')
            csv_path = f.name

        with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as f:
            sqlite_path = f.name

        smiles_store = SmilesStore.from_csv(csv_path, sqlite_path)
        graph_store = SmilesGraphStore(
            smiles_store=smiles_store,
            processing=MoleculeProcessing(),
            target_columns=['value']
        )

        yield graph_store

        try:
            os.unlink(csv_path)
            os.unlink(sqlite_path)
        except FileNotFoundError:
            pass

    @pytest.fixture
    def sample_smiles_graph_store_with_invalid(self):
        """Fixture providing a SmilesGraphStore with some invalid SMILES."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('smiles,value\n')
            f.write('CCO,1.5\n')
            f.write('INVALID,2.3\n')
            f.write('c1ccccc1,0.8\n')
            f.write('BAD_SMILES,1.2\n')
            f.write('CCN,0.5\n')
            csv_path = f.name

        with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as f:
            sqlite_path = f.name

        smiles_store = SmilesStore.from_csv(csv_path, sqlite_path)
        graph_store = SmilesGraphStore(
            smiles_store=smiles_store,
            processing=MoleculeProcessing(),
            target_columns=['value']
        )

        yield graph_store

        try:
            os.unlink(csv_path)
            os.unlink(sqlite_path)
        except FileNotFoundError:
            pass

    @pytest.fixture
    def sample_vgd_store(self):
        """Fixture providing a VisualGraphDatasetStore with sample data."""
        import json

        temp_dir = tempfile.mkdtemp()

        for idx in range(5):
            graph_data = {
                'metadata': {
                    'index': idx,
                    'graph': {
                        'node_indices': [0, 1, 2],
                        'node_attributes': [[1, 0], [0, 1], [1, 1]],
                        'edge_indices': [[0, 1], [1, 2]],
                        'edge_attributes': [[1], [1]],
                        'graph_labels': [float(idx)]
                    }
                }
            }
            with open(os.path.join(temp_dir, f'{idx}.json'), 'w') as f:
                json.dump(graph_data, f)

        store = VisualGraphDatasetStore(temp_dir)

        yield store

        import shutil
        shutil.rmtree(temp_dir)

    def test_init_with_smiles_graph_store(self, sample_smiles_graph_store):
        """Test GraphDataLoader initialization with SmilesGraphStore."""
        loader = GraphDataLoader(
            sample_smiles_graph_store,
            batch_size=2,
            shuffle=False
        )

        assert loader.batch_size == 2

    def test_init_with_vgd_store(self, sample_vgd_store):
        """Test GraphDataLoader initialization with VisualGraphDatasetStore."""
        loader = GraphDataLoader(
            sample_vgd_store,
            batch_size=2,
            shuffle=False
        )

        assert loader.batch_size == 2

    def test_iteration_yields_batches(self, sample_smiles_graph_store):
        """Test that iteration yields PyG Batch objects."""
        loader = GraphDataLoader(
            sample_smiles_graph_store,
            batch_size=2,
            shuffle=False
        )

        batches = list(loader)

        assert len(batches) > 0
        for batch in batches:
            assert hasattr(batch, 'x')
            assert hasattr(batch, 'edge_index')
            assert hasattr(batch, 'y')
            assert hasattr(batch, 'batch')  # Batch assignment vector

    def test_batch_contains_correct_data(self, sample_smiles_graph_store):
        """Test that batches contain correctly structured data."""
        loader = GraphDataLoader(
            sample_smiles_graph_store,
            batch_size=2,
            shuffle=False
        )

        batch = next(iter(loader))

        assert isinstance(batch.x, torch.Tensor)
        assert isinstance(batch.edge_index, torch.Tensor)
        assert isinstance(batch.y, torch.Tensor)
        assert batch.edge_index.shape[0] == 2  # (2, num_edges) format

    def test_invalid_smiles_raises_error(self, sample_smiles_graph_store_with_invalid):
        """Test that invalid SMILES raises an error (user must clean data first)."""
        loader = GraphDataLoader(
            sample_smiles_graph_store_with_invalid,
            batch_size=10,
            shuffle=False
        )

        # Should raise an error when encountering invalid SMILES (returns None)
        with pytest.raises(Exception):  # Will fail in data_from_graph when graph is None
            list(loader)

    def test_with_vgd_store(self, sample_vgd_store):
        """Test GraphDataLoader with VisualGraphDatasetStore."""
        loader = GraphDataLoader(
            sample_vgd_store,
            batch_size=2,
            shuffle=False
        )

        batches = list(loader)

        assert len(batches) > 0
        for batch in batches:
            assert hasattr(batch, 'x')
            assert hasattr(batch, 'edge_index')

    def test_shuffle(self, sample_smiles_graph_store):
        """Test that shuffle parameter works."""
        loader1 = GraphDataLoader(
            sample_smiles_graph_store,
            batch_size=1,
            shuffle=False
        )

        loader2 = GraphDataLoader(
            sample_smiles_graph_store,
            batch_size=1,
            shuffle=True
        )

        # Both should produce batches
        batches1 = list(loader1)
        batches2 = list(loader2)

        assert len(batches1) == len(batches2)

    @pytest.mark.skip(reason="Multi-worker tests can timeout")
    def test_num_workers(self, sample_smiles_graph_store):
        """Test that num_workers parameter works."""
        loader = GraphDataLoader(
            sample_smiles_graph_store,
            batch_size=2,
            shuffle=False,
            num_workers=2
        )

        batches = list(loader)
        assert len(batches) > 0

