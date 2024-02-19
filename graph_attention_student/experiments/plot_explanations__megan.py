import os
import pathlib
import typing as t

import numpy as np
import matplotlib.pyplot as plt
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from graph_attention_student.torch.megan import Megan

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
# :param MODEL_PATH:
#       This string may be a valid absolute path to a file on the local system which contains the 
#       trained model. This model file has to be a valid checkpoint representation of a model of the 
#       "Megan" class.
MODEL_PATH: str = 'rb_dual_motifs__2021-09-14_17-01-05__model__best.pth'

experiment = Experiment.extend(
    'plot_explanations.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

@experiment.hook('create_explanations', default=False, replace=True)
def create_explanations(e: Experiment,
                        index_data_map: dict,
                        indices: list[int],
                        graphs: list[dict],
                        ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    
    e.log('loading the Megan model...')
    model = Megan.load_from_checkpoint(e.MODEL_PATH)

    e.log('forward pass of the model')
    infos = model.forward_graphs(graphs)
    
    node_importances_list = []
    edge_importances_list = []
    for graph, info in zip(graphs, infos):
        node_importances_list.append(info['node_importance'])
        edge_importances_list.append(info['edge_importance'])
        
        # We also want to store the information from the prediction to the actual model here in case that we need 
        # that information for any other step of the visualization and we dont want to explicitly pass all this 
        # information around through all the abstraction layers, which is why we simply store it in the graph 
        # dict itself.
        graph['graph_prediction'] = info['graph_output']
        graph['node_importances'] = info['node_importance']
        graph['edge_importances'] = info['edge_importance']

    return node_importances_list, edge_importances_list

experiment.run_if_main()