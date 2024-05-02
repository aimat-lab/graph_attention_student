import os
import json
import pathlib
import textwrap
import typing as t

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

PATH = pathlib.Path(__file__).parent.absolute()

# == DATASET PARAMETERS ==
# This section contains the parameters which determine the dataset and how to handle said dataset.

# :param VISUAL_GRAPH_DATASET:
#       This string may be a valid absolute path to a folder on the local system which
#       contains all the elements of a visual graph dataset. Alternatively this string can be
#       a valid unique identifier of a visual graph dataset which can be downloaded from the main
#       remote file share location.
VISUAL_GRAPH_DATASET: str = 'compas'
# :param INDICES_PATH:
#       This string may be a valid absolute path to a json file on the local system which defines the 
#       indices of the dataset elements to be used for the plotting of the explanations. If this is not 
#       given, the indices will be sampled from the dataset randomly with the given number of indices.
INDICES_PATH: t.Optional[str] = None
# :param NUM_ELEMENTS:
#       This integer number defines how many elements of the dataset are supposed to be sampled for the
#       plotting of the explanations. This parameter will be ignored if a indices file path is given.
NUM_ELEMENTS: int = 20
# :param ELEMENTS:
#       This list defines the original string domain representations of the elements for which the 
#       explanations are supposed to be generated.
#       If this list is given, then the explanations will be generated for these elements only. If 
#       this parameter is None, the elements will be sampled from the dataset instead.
ELEMENTS: list[str] = [
    'C1C=C2C=C3C=C4C=CC5C=C6C7C=C8C=C9C=CC=CC9=CC8=CC=7C=CC6=CC=5C4=CC3=CC2=CC1',
    'C1=CC=C2C=CC3C4C=CC5C6C7C=C8C=C9C=C%10C=CC=CC%10=CC9=CC8=CC=7C=CC=6C=CC=5C=4C=CC=3C2=C1',
    'c1ccc2c(c1)ccc3cc4c(ccc5ccc6ccc7ccc8ccccc8c7c6c45)cc23',
    'C12C=CC=CC1=CC1C=C3C4C=CC=CC=4C4C5C=CC6C7C=CC=CC=7C=CC=6C=5C=CC=4C3=CC=1C=2',
    'C1=CC=C2C(=C1)C=CC3=C2C=CC4=C3C=CC5=C4C=CC6=CC=CC=C65',
    'C12C=CC=CC1=C1C=C3C4C=C5C=CC6C=C7C=CC8C=CC=CC=8C7=CC=6C5=CC=4C4C5C=CC=CC=5C5C=CC=CC=5C=4C3=CC1=CC=2',
    'c1ccc2c(c1)ccc1c2ccc2ccc3ccc4c5ccccc5c5c6ccccc6c6ccccc6c5c4c3c21',
]

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
MODEL_PATH: str = os.path.join(PATH, 'assets', 'models', 'compas.ckpt')

# == VISUALIZATION PARAMETERS ==
# This section contains the parameters which configure the visualization of the explanations.

# :param IMPORTANCE_THRESHOLD:
#       This float value determines the threshold for the binarization of the importance values. This binarization 
#       is only applied for the "combined" visualization type. Every importance value above the threshold will be
#       visualized with the solid color and values below will not be shown at all.
IMPORTANCE_THRESHOLD: float = None


# == EXPERIMENT PARAMETERS ==

experiment = Experiment.extend(
    'plot_explanations__megan.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


@experiment.hook('create_labels', default=False, replace=True)
def create_labels(e: Experiment,
                  index_data_map: dict,
                  indices: list[int],
                  graphs: list[dict],
                  ) -> str:
    """
    This hook is supposed to create a list of labels that can be used as the figure titles in the visualization PDF
    file. The returned label list has to have the same order as the given graphs list.
    
    This implementation will print the index, the SMILES representation of the molecular graph as the title. Additionally 
    the numeric values for the true and predicted target values will be printed as part of the title as well.
    """
    e.log('creating labels with the SMILES and the true and predicted labels...')
    
    labels = []
    for index, graph in zip(indices, graphs):
        metadata = index_data_map[index]['metadata']
        smiles = metadata['smiles']
        smiles = '\n'.join(textwrap.wrap(smiles, width=80))
        label = (
            f'index: {index}\n'
            f'smiles:{smiles}\n'
            f'true: {graph["graph_labels"]} - pred: {graph["graph_prediction"]}\n'
            f'fidelities: {np.round(graph["graph_fidelity"], 2)}'
        )
        labels.append(label)
        
    return labels

experiment.run_if_main()