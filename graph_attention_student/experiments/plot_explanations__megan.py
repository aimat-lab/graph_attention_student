import os
import pathlib
import typing as t

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from visual_graph_datasets.visualization.importances import create_combined_importances_pdf

from graph_attention_student.torch.megan import Megan
from graph_attention_student.utils import fidelity_from_deviation

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
# :param DATASET_TYPE:
#       This string has to determine the type of the dataset in regards to the target values.
#       This can either be "regression" or "classification". This choice influences how the model
#       is trained (loss function) and ultimately how it is evaluated.
DATASET_TYPE: str = 'regression'

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
                        ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    
    e.log('loading the Megan model...')
    model = Megan.load_from_checkpoint(e.MODEL_PATH)

    e.log('forward pass of the model')
    infos = model.forward_graphs(graphs)
    
    e.log('leave one out deviations...')
    devs = model.leave_one_out_deviations(graphs)
    
    node_importances_list = []
    edge_importances_list = []
    for graph, info, dev in zip(graphs, infos, devs):
        
        # We also want to store the information from the prediction to the actual model here in case that we need 
        # that information for any other step of the visualization and we dont want to explicitly pass all this 
        # information around through all the abstraction layers, which is why we simply store it in the graph 
        # dict itself.
        graph['graph_prediction'] = info['graph_output']
        graph['node_importances'] = info['node_importance']
        graph['edge_importances'] = info['edge_importance']
        
        # graph_deviation: (num_outputs, num_channels)
        graph['graph_deviation'] = dev
        # graph_fidelity: (num_channels, )
        fid = fidelity_from_deviation(dev, dataset_type=e.DATASET_TYPE)
        graph['graph_fidelity'] = fid
        
        node_importances_list.append(info['node_importance'])
        edge_importances_list.append(info['edge_importance'])

    return node_importances_list, edge_importances_list


@experiment.hook('additional_explanation', default=False, replace=True)
def additional_visualization(e: Experiment,
                             index_data_map: dict,
                             indices: list[int],
                             graphs: list[dict],
                             ) -> None:
    
    e.log('creating combined importances pdf...')
    
    image_paths = [index_data_map[index]['image_path'] for index in indices]
    labels = e.apply_hook(
        'create_labels',
        index_data_map=index_data_map,
        indices=indices,
        graphs=graphs,
    )
    
    pdf_path = os.path.join(e.path, 'combined_importances.pdf')
    create_combined_importances_pdf(
        graph_list=graphs,
        image_path_list=image_paths,
        node_positions_list=[graph['node_positions'] for graph in graphs],
        node_importances_list=[graph['node_importances'] for graph in graphs],
        edge_importances_list=[graph['edge_importances'] for graph in graphs],
        graph_fidelity_list=[graph['graph_fidelity'] for graph in graphs],
        label_list=labels,
        channel_colors_map=e.CHANNEL_COLORS_MAP,
        importance_threshold=e.IMPORTANCE_THRESHOLD,
        output_path=pdf_path,
        logger=e.logger,
        log_step=100,
    )


experiment.run_if_main()