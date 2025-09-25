import typing as t
from typing import List
import random
from queue import SimpleQueue
from rich.pretty import pprint

import torch
import polars as pl
import pandas as pd
import numpy as np
import visual_graph_datasets.typing as tv
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Data
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

            