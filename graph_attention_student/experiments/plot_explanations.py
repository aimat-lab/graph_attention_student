import os
import json
import pathlib
import random
import typing as t

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from visual_graph_datasets.config import Config
from visual_graph_datasets.web import ensure_dataset
from visual_graph_datasets.data import VisualGraphDatasetReader
from visual_graph_datasets.visualization.importances import create_importances_pdf
from visual_graph_datasets.visualization.importances import create_combined_importances_pdf
from visual_graph_datasets.visualization.importances import plot_node_importances_background
from visual_graph_datasets.visualization.importances import plot_edge_importances_background

PATH = pathlib.Path(__file__).parent.absolute()
np.set_printoptions(precision=2)

# == DATASET PARAMETERS ==
# This section contains the parameters which determine the dataset and how to handle said dataset.

# :param VISUAL_GRAPH_DATASET:
#       This string may be a valid absolute path to a folder on the local system which
#       contains all the elements of a visual graph dataset. Alternatively this string can be
#       a valid unique identifier of a visual graph dataset which can be downloaded from the main
#       remote file share location.
VISUAL_GRAPH_DATASET: str = 'rb_dual_motifs'
# :param INDICES_PATH:
#       This string may be a valid absolute path to a json file on the local system which defines the 
#       indices of the dataset elements to be used for the plotting of the explanations. If this is not 
#       given, the indices will be sampled from the dataset randomly with the given number of indices.
INDICES_PATH: t.Optional[str] = None
# :param ELEMENTS:
#       There is a list of elements here that is the model where there is a 
ELEMENTS: t.Optional[list[str]] = None
# :param NUM_ELEMENTS:
#       This integer number defines how many elements of the dataset are supposed to be sampled for the
#       plotting of the explanations. This parameter will be ignored if a indices file path is given.
NUM_ELEMENTS: int = 100

# == MODEL PARAMETERS ==
# This section contains the parameters which configure the model architecture.

# :param NUM_CHANNELS:
#       This integer number defines how many explanation channels there are in the given model. This 
#       the number of distinct explanations masks that will be created for each input element.
NUM_CHANNELS: int = 2
# :param IMPORTANCE_CHANNEL_LABELS:
#       This dictionary structure can be used to define the human readable names for the various
#       explanation channels that are part of the model. The keys of this dict have to be integer indices
#       of the channels in the order as they appear in the model. The values are string which will be
#       used as the names of these channels within the evaluation visualizations and log messages etc.
IMPORTANCE_CHANNEL_LABELS: dict = {
    0: 'negative',
    1: 'positive',
}


__DEBUG__: bool = True

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

@experiment.hook('load_dataset')
def load_dataset(e: Experiment):
    
    if os.path.exists(e.VISUAL_GRAPH_DATASET):
        dataset_path = e.VISUAL_GRAPH_DATASET
    else:
        config = Config()
        config.load()
        dataset_path = ensure_dataset(e.VISUAL_GRAPH_DATASET, config=config)
        
    e.log(f'loading dataset from: {dataset_path}')
    reader = VisualGraphDatasetReader(
        dataset_path,
        logger=e.logger,
        log_step=1000,
    )
    index_data_map = reader.read()
    return index_data_map


@experiment.hook('create_explanations', default=False, replace=False)
def create_explanations(e: Experiment,
                        index_data_map: dict,
                        indices: list[int],
                        graphs: list[dict],
                        ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    
    e.log('creating random explanations...')

    node_importances_list = []
    edge_importances_list = []
    for graph in graphs:
        
        node_importances = np.random.rand(len(graph['node_indices']), e.NUM_CHANNELS)
        edge_importances = np.random.rand(len(graph['edge_indices']), e.NUM_CHANNELS)

        node_importances_list.append(node_importances)
        edge_importances_list.append(edge_importances)

    return node_importances_list, edge_importances_list


@experiment.hook('create_labels', default=False, replace=False)
def create_labels(e: Experiment,
                  index_data_map: dict,
                  indices: list[int],
                  graphs: list[dict],
                  ) -> list[str]:
    
    e.log('creating labels...')
    labels = ['random' for index in indices]
    return labels


@experiment
def experiment(e: Experiment):
    
    e.log('starting experiment...')
    
    # :hook load_dataset:
    #       This hook is called to load the dataset from the given path or identifier and return the
    #       index_data_map which is a dictionary of the dataset elements with their corresponding indices.
    index_data_map = e.apply_hook(
        'load_dataset',
    )
    
    e.log('choosing the samples to be visualized...')
    if e.INDICES_PATH:
        with open(e.INDICES_PATH, 'r') as f:
            indices = json.load(f)
    else:
        indices = list(index_data_map.keys())
    
    num_elements = min(len(indices), e.NUM_ELEMENTS)
    indices = random.sample(
        indices,
        k=num_elements,
    )   
    e.log(f'chose {len(indices)} samples')
    
    # Now we also want to save the indices into a json file as an artifact of the experiment
    indices_path = os.path.join(e.path, 'indices.json')
    with open(indices_path, 'w') as f:
        json.dump(indices, f)
    
    graphs = [index_data_map[index]['metadata']['graph'] for index in indices]
    image_paths = [index_data_map[index]['image_path'] for index in indices]
    
    # :hook create_explanations:
    #       This hook is called to create the explanations for the given graphs. The explanations are
    #       returned as node_importances_list and edge_importances_list.
    node_importances_list, edge_importances_list = e.apply_hook(
        'create_explanations',
        index_data_map=index_data_map,
        indices=indices,
        graphs=graphs,
    )
    
    # :hook create_labels:
    #       This hook is called to create the labels for the given graphs. The labels are returned as
    #       strings which will be diplayed in the final explanation document alongside each individual 
    #       element as a tittle of the page.
    labels = e.apply_hook(
        'create_labels',
        index_data_map=index_data_map,
        indices=indices,
        graphs=graphs,
    )
    
    pdf_path = os.path.join(e.path, 'explanations.pdf')
    create_importances_pdf(
        graph_list=graphs,
        image_path_list=image_paths,
        node_positions_list=[graph['node_positions'] for graph in graphs],
        importances_map={
            'model': (node_importances_list, edge_importances_list),
        },
        plot_node_importances_cb=plot_node_importances_background,
        plot_edge_importances_cb=plot_edge_importances_background,
        importance_channel_labels=e.IMPORTANCE_CHANNEL_LABELS,
        labels_list=labels,
        output_path=pdf_path,
        logger=e.logger,
    )
    
    # :hook additional_explanations:
    #       This hook can be called to generate additional visualizations.
    e.apply_hook(
        'additional_explanation',
        index_data_map=index_data_map,
        indices=indices,
        graphs=graphs,
    )
    
    
    
experiment.run_if_main()