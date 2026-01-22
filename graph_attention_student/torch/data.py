import typing as t
from typing import List, Optional, Set
import random
import sqlite3
import threading
import json
import os
import logging
from collections.abc import Sequence
from rich.pretty import pprint

import torch
import polars as pl
import pandas as pd
import numpy as np
import visual_graph_datasets.typing as tv
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Data
from torch_geometric.data import Dataset as PyGDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from visual_graph_datasets.processing.molecules import MoleculeProcessing


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
        data.node_importances = torch.tensor(graph['node_importances'], dtype=dtype)
    
    if 'edge_importances' in graph:
        data.edge_importances = torch.tensor(graph['edge_importances'], dtype=dtype)
        
    # 02.05.24
    # If the graph representation contains node coordinates, we can also attach those to the
    # data object. This is going to be especially important for the equivariance extension for the 
    # networks.
    if 'node_coordinates' in graph:
        data.pos = torch.tensor(graph['node_coordinates'], dtype=dtype)
        data.coords = torch.tensor(graph['node_coordinates'], dtype=dtype)
        
    # 28.06.24
    # We optionally also want to support the case where the graph representation contains a
    # weight for the graph. This is going to be used for the weighted loss function during the 
    # model training.
    if 'graph_weight' in graph:
        data.train_weight = torch.tensor(graph['graph_weight'], dtype=dtype)
    
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



class SmilesDataset(IterableDataset):
    """
    Custom torch IterableDataset class which can be used to load a dataset consisting of SMILES string
    representations of molecular graphs. This implementation uses lazy iteration over polars DataFrames
    and optionally employs reservoir sampling.
    """

    def __init__(self,
                 dataset: t.Union[pl.DataFrame, pd.DataFrame, List[dict], str],
                 smiles_column: str = 'smiles',
                 target_columns: List[str] = ['value'],
                 processing=MoleculeProcessing(),
                 reservoir_sampling: bool = True,
                 reservoir_size: int = 1000,
                 ):

        self.dataset = dataset
        self.smiles_column = smiles_column
        self.target_columns = target_columns
        self.reservoir_sampling = reservoir_sampling
        self.reservoir_size = reservoir_size
        self.processing = processing

        # --- reservoir sampling setup ---
        # If reservoir sampling is enabled, we need to set up a reservoir to hold the samples
        # from which we then randomly draw during iteration.
        self.reservoir: List[dict] = []

    def _create_worker_dataframe(self) -> pl.LazyFrame:
        """
        Create a LazyFrame instance for the current worker.
        This method ensures each worker gets its own LazyFrame instance,
        avoiding pickling issues with shared state.
        """
        if isinstance(self.dataset, pl.DataFrame):
            return self.dataset.lazy()

        elif isinstance(self.dataset, pd.DataFrame):
            return pl.from_pandas(self.dataset).lazy()

        elif isinstance(self.dataset, list):
            return pl.DataFrame(self.dataset).lazy()

        elif isinstance(self.dataset, str):
            return pl.scan_csv(self.dataset)

        else:
            raise ValueError(f"Unsupported dataset type: {type(self.dataset)}")

    def _create_worker_processing(self):
        """
        Create a MoleculeProcessing instance for the current worker.
        This ensures each worker gets its own processing instance.
        """
        # For now, just create a default MoleculeProcessing instance
        # This avoids complex pickling issues with the original instance
        return MoleculeProcessing()

    def __iter__(self):
        """
        Iterate over the dataset lazily, optionally applying reservoir sampling.
        Supports multiple workers by splitting data per worker.
        """
        # Get worker info for multi-worker data loading
        worker_info = get_worker_info()

        # Create a fresh LazyFrame for this worker to avoid pickling issues
        lazyframe: pl.LazyFrame = self._create_worker_dataframe()

        # If multiple workers are used, only collect every num_workers-th row,
        # starting at the current worker id. For single-worker loading just collect all.
        if worker_info is None:
            dataframe: pl.DataFrame = lazyframe.collect(streaming=True)
        else:
            n = worker_info.num_workers
            wid = worker_info.id
            dataframe: pl.DataFrame = (
            lazyframe
            .with_row_index("__row_idx")
            .filter((pl.col("__row_idx") % n) == wid)
            .drop("__row_idx")
            .collect(streaming=True)
            )

        # Create a fresh processing instance for this worker
        processing = self._create_worker_processing()

        pprint(worker_info)

        for row in dataframe.iter_rows(named=True):
            
            if self.reservoir_sampling:
                # Handle reservoir sampling
                if len(self.reservoir) < self.reservoir_size:
                    self.reservoir.append(row)
                else:
                    # Replace a random element
                    idx = random.randint(0, self.reservoir_size - 1)
                    self.reservoir[idx] = row

                # Randomly select an element from reservoir
                if len(self.reservoir) > 0:
                    idx = random.choice(range(len(self.reservoir)))
                    row = self.reservoir[idx]
                else:
                    continue

            # The SMILES string is first processed into a graph dict representation using the
            # MoleculeProcessing class from visual_graph_datasets.
            try:
                smiles = row[self.smiles_column]
                targets = [row[col] for col in self.target_columns]

                graph = processing.process(value=smiles)
                graph['graph_labels'] = targets

                # Finally, this function will convert the graph dict into a PyG data object and
                # return it.
                data = data_from_graph(graph)
                yield data
            except Exception:
                # Skip invalid SMILES strings silently during iteration
                continue


class SmilesStore(Sequence):
    """
    SQLite-backed random-access store for CSV data.

    Supports multi-worker DataLoader through lazy connection initialization,
    where each worker opens its own read-only connection on first access.

    Example usage:
        # Create store from CSV
        store = SmilesStore.from_csv('molecules.csv', 'molecules.sqlite')

        # Access individual rows
        row = store[0]  # {'smiles': 'CCO', 'value': 1.23, ...}

        # Get total count
        print(len(store))
    """

    def __init__(self, sqlite_path: str):
        """
        Initialize the SmilesStore with an existing SQLite database.

        :param sqlite_path: Path to the SQLite database file
        """
        self.sqlite_path = sqlite_path
        # Thread-local storage for database connections.
        # When using DataLoader with num_workers > 0, each worker runs in a separate process.
        # SQLite connections cannot be safely shared across processes/threads, so each worker
        # needs its own connection. threading.local() ensures that when _connection is accessed,
        # each worker gets its own isolated connection instance (created lazily on first access).
        self._local = threading.local()
        self._length: Optional[int] = None
        self._columns: Optional[List[str]] = None

    @classmethod
    def from_csv(cls, csv_path: str, sqlite_path: str) -> 'SmilesStore':
        """
        Create a SmilesStore by converting a CSV file to SQLite format.

        If the SQLite file already exists, it will be overwritten.

        :param csv_path: Path to the source CSV file
        :param sqlite_path: Path where the SQLite database will be created
        :returns: A new SmilesStore instance
        """
        # Remove existing file if it exists
        if os.path.exists(sqlite_path):
            os.remove(sqlite_path)

        # Read CSV with polars for efficiency
        df = pl.read_csv(csv_path)

        # Create SQLite database
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.cursor()

        # Build CREATE TABLE statement with dynamic schema
        columns = df.columns
        column_defs = ['_id INTEGER PRIMARY KEY AUTOINCREMENT']

        for col in columns:
            dtype = df[col].dtype
            if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                sql_type = 'INTEGER'
            elif dtype in (pl.Float32, pl.Float64):
                sql_type = 'REAL'
            else:
                sql_type = 'TEXT'
            # Escape column names with double quotes for SQLite
            column_defs.append(f'"{col}" {sql_type}')

        create_sql = f'CREATE TABLE data ({", ".join(column_defs)})'
        cursor.execute(create_sql)

        # Insert all rows
        placeholders = ', '.join(['?' for _ in columns])
        escaped_columns = ', '.join([f'"{col}"' for col in columns])
        insert_sql = f'INSERT INTO data ({escaped_columns}) VALUES ({placeholders})'

        # Convert to list of tuples for insertion
        rows = df.rows()
        cursor.executemany(insert_sql, rows)

        conn.commit()
        conn.close()

        return cls(sqlite_path)

    @property
    def _connection(self) -> sqlite3.Connection:
        """
        Get a thread-local database connection.

        Each thread/worker gets its own connection, created lazily on first access.
        Connections are read-only for safety with multi-worker DataLoader.
        """
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            # Open in read-only mode using URI
            uri = f'file:{self.sqlite_path}?mode=ro'
            self._local.conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    @property
    def columns(self) -> List[str]:
        """Get the list of column names (excluding _id)."""
        if self._columns is None:
            cursor = self._connection.cursor()
            cursor.execute('PRAGMA table_info(data)')
            self._columns = [row[1] for row in cursor.fetchall() if row[1] != '_id']
        return self._columns

    def __getitem__(self, index: int) -> dict:
        """
        Get a row by index.

        :param index: Zero-based row index
        :returns: Dictionary with all columns from the row
        :raises IndexError: If index is out of range
        """
        if index < 0 or index >= len(self):
            raise IndexError(f'Index {index} out of range for SmilesStore of length {len(self)}')

        cursor = self._connection.cursor()
        # SQLite _id is 1-based, so add 1 to the index
        cursor.execute('SELECT * FROM data WHERE _id = ?', (index + 1,))
        row = cursor.fetchone()

        if row is None:
            raise IndexError(f'Index {index} not found in database')

        # Convert Row to dict, excluding _id
        return {key: row[key] for key in row.keys() if key != '_id'}

    def __len__(self) -> int:
        """Get the total number of rows in the store."""
        if self._length is None:
            cursor = self._connection.cursor()
            cursor.execute('SELECT COUNT(*) FROM data')
            self._length = cursor.fetchone()[0]
        return self._length


class SmilesGraphStore(Sequence):
    """
    Wraps a SmilesStore with a Processing instance to convert SMILES strings
    to GraphDict representations on-the-fly.

    Example usage:
        store = SmilesStore.from_csv('molecules.csv', 'molecules.sqlite')
        graph_store = SmilesGraphStore(
            smiles_store=store,
            processing=MoleculeProcessing(),
            target_columns=['solubility'],
            smiles_column='smiles'
        )

        graph = graph_store[0]  # Returns GraphDict
    """

    def __init__(
        self,
        smiles_store: SmilesStore,
        processing,
        target_columns: List[str],
        smiles_column: str = 'smiles'
    ):
        """
        Initialize the SmilesGraphStore.

        :param smiles_store: The underlying SmilesStore for raw data access
        :param processing: Processing instance (e.g., MoleculeProcessing) to convert SMILES to graphs
        :param target_columns: List of column names to use as graph labels
        :param smiles_column: Name of the column containing SMILES strings
        """
        self.smiles_store = smiles_store
        self.processing = processing
        self.target_columns = target_columns
        self.smiles_column = smiles_column
        self._logger = logging.getLogger(__name__)

    def __getitem__(self, index: int) -> Optional[tv.GraphDict]:
        """
        Get a graph by index.

        :param index: Zero-based index
        :returns: GraphDict representation of the molecule, or None if SMILES is invalid
        """
        row = self.smiles_store[index]
        smiles = row[self.smiles_column]

        try:
            graph = self.processing.process(value=smiles)
            graph['graph_labels'] = np.array([row[col] for col in self.target_columns])
            return graph
        except Exception as e:
            self._logger.warning(f'Invalid SMILES at index {index}: {smiles} - {e}')
            return None

    def __len__(self) -> int:
        """Get the total number of items in the store."""
        return len(self.smiles_store)


class VisualGraphDatasetStore(Sequence):
    """
    Random-access store for Visual Graph Dataset (VGD) directories.

    Provides direct index-based access to graph data stored in VGD format,
    where each graph is stored as a JSON file named {index}.json.

    Example usage:
        store = VisualGraphDatasetStore('/path/to/vgd_dataset')
        graph = store[5]  # Loads 5.json
        print(len(store))  # max_index + 1
    """

    def __init__(self, path: str):
        """
        Initialize the store by discovering all graph indices in the directory.

        :param path: Path to the VGD dataset directory
        """
        self.path = path
        self._indices: Set[int] = set()
        self._max_index: int = -1

        # Discover all indices from JSON files in the directory
        for filename in os.listdir(path):
            if filename.endswith('.json'):
                try:
                    # Extract numeric index from filename (e.g., '123.json' -> 123)
                    idx = int(filename[:-5])
                    self._indices.add(idx)
                    self._max_index = max(self._max_index, idx)
                except ValueError:
                    # Skip non-numeric filenames (e.g., '.meta.json')
                    continue

    def __getitem__(self, index: int) -> tv.GraphDict:
        """
        Get a graph by index.

        :param index: The graph index (corresponds to {index}.json file)
        :returns: GraphDict representation of the graph
        :raises IndexError: If the index doesn't exist in the dataset
        """
        if index not in self._indices:
            raise IndexError(f'Index {index} not found in dataset at {self.path}')

        json_path = os.path.join(self.path, f'{index}.json')
        with open(json_path, 'r') as f:
            data = json.load(f)

        # VGD format: graph data is under metadata['graph']
        graph = data['metadata']['graph']

        # Convert lists back to numpy arrays (JSON serialization stores as lists)
        for key in graph:
            if isinstance(graph[key], list):
                graph[key] = np.array(graph[key])

        return graph

    def __len__(self) -> int:
        """
        Get the length of the store.

        Returns max_index + 1 to support sparse indices with direct mapping.
        """
        return self._max_index + 1

    def get_valid_indices(self) -> List[int]:
        """
        Get a sorted list of all valid indices in the dataset.

        Useful when indices are sparse and you need to iterate over actual elements.

        :returns: Sorted list of valid indices
        """
        return sorted(self._indices)


class _GraphSequenceDataset(PyGDataset):
    """
    Internal PyG Dataset wrapper that adapts a Sequence[GraphDict] for use with DataLoader.

    Converts GraphDict to PyG Data objects. The user is responsible for ensuring all
    indices in the graph store are valid - invalid entries will raise errors.
    """

    def __init__(self, graph_store: Sequence):
        """
        Initialize the dataset wrapper.

        :param graph_store: A Sequence that returns GraphDict for all valid indices
        """
        super().__init__()
        self.graph_store = graph_store

    def len(self) -> int:
        """Get the number of items in the dataset."""
        return len(self.graph_store)

    def get(self, idx: int) -> Data:
        """
        Get a PyG Data object by index.

        :param idx: Index into the graph store
        :returns: PyG Data object
        :raises: IndexError if index doesn't exist, or other errors if graph is invalid
        """
        graph = self.graph_store[idx]
        return data_from_graph(graph)


class GraphDataLoader(PyGDataLoader):
    """
    DataLoader that accepts any Sequence[GraphDict] as input.

    Automatically converts GraphDict to PyG Data objects. The user is responsible
    for ensuring the graph store contains only valid entries - invalid entries
    (None values, missing indices) will raise errors during iteration.

    Example usage:
        # With SmilesGraphStore
        store = SmilesStore.from_csv('molecules.csv', 'molecules.sqlite')
        graph_store = SmilesGraphStore(store, MoleculeProcessing(), ['value'])
        loader = GraphDataLoader(graph_store, batch_size=32, num_workers=4)

        for batch in loader:
            predictions = model(batch)

        # With VisualGraphDatasetStore
        vgd_store = VisualGraphDatasetStore('/path/to/dataset')
        loader = GraphDataLoader(vgd_store, batch_size=32)
    """

    def __init__(
        self,
        graph_store: Sequence,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        **kwargs
    ):
        """
        Initialize the GraphDataLoader.

        :param graph_store: A Sequence that returns GraphDict (e.g., SmilesGraphStore, VisualGraphDatasetStore)
        :param batch_size: Number of graphs per batch
        :param shuffle: Whether to shuffle the data
        :param num_workers: Number of worker processes for data loading
        :param kwargs: Additional arguments passed to PyG DataLoader
        """
        dataset = _GraphSequenceDataset(graph_store)
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )
