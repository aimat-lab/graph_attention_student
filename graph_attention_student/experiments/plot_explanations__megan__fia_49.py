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
VISUAL_GRAPH_DATASET: str = 'fia_49k'
# :param INDICES_PATH:
#       This string may be a valid absolute path to a json file on the local system which defines the 
#       indices of the dataset elements to be used for the plotting of the explanations. If this is not 
#       given, the indices will be sampled from the dataset randomly with the given number of indices.
INDICES_PATH: t.Optional[str] = os.path.join(PATH, 'assets', 'splits', 'fia_49k_test_indices.json')
# :param NUM_ELEMENTS:
#       This integer number defines how many elements of the dataset are supposed to be sampled for the
#       plotting of the explanations. This parameter will be ignored if a indices file path is given.
NUM_ELEMENTS: int = 20

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
MODEL_PATH: str = os.path.join(PATH, 'assets', 'models', 'fia_49k.ckpt')

# == VISUALIZATION PARAMETERS ==
# This section contains the parameters which configure the visualization of the explanations.

# :param CHANNEL_COLORS_MAP:
#       This dictionary structure can be used to define the color maps for the various explanation channels.
#       For the "combined" visualization, all the explanation masks are drawn into the same image. These 
#       color maps define how to color each of the different explanation channels to differentiate them.
#       It is color maps and not flat colors because the color will be scaled with the explanation channel's 
#       fidelity value. The keys of this dict have to be integer indices of the channels in the order as they
#       appear in the model. The values are matplotlib color maps which will be used to color the explanation
#       masks in the visualization.
CHANNEL_COLORS_MAP: dict[int, mcolors.Colormap] = {
    0: cm.get_cmap('Blues'),
    1: cm.get_cmap('Reds'),
}
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