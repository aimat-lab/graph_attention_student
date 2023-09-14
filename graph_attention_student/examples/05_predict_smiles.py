"""
This example showcases how an existing model can be loaded from the disk and subsequently be used to 
make a prediction for a given SMILES representation of a molecular graph.

The example also shows how the graph embedding for that given graph can be retrieved from the given 
model and how the explanations can be visualized.
"""
import os
import pathlib
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as ks
from matplotlib.backends.backend_pdf import PdfPages
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from visual_graph_datasets.data import load_visual_graph_element
from visual_graph_datasets.util import dynamic_import
from visual_graph_datasets.processing.base import ProcessingBase
from visual_graph_datasets.visualization.base import draw_image
from visual_graph_datasets.visualization.importances import plot_node_importances_background
from visual_graph_datasets.visualization.importances import plot_edge_importances_background

from graph_attention_student.utils import array_normalize
from graph_attention_student.models import load_model

np.set_printoptions(precision=3)

PATH = pathlib.Path(__file__).parent.absolute()
ASSETS_PATH = os.path.join(PATH, 'assets')

# :param MODEL_PATH:
#       The path of the model folder that contains the MEGAN model to be used to make the prediction
MODEL_PATH = os.path.join(ASSETS_PATH, 'aqsoldb_model')
# :param PROCESSING_PATH:
#       The path to the specific "process.py" module that contains the processing rules to be applied 
#       to convert the smiles string to the graph representation. It is important that this processing 
#       is the exact same as the dataset that was used to train the above model.
PROCESSING_PATH = os.path.join(ASSETS_PATH, 'aqsoldb_process.py')

# :param SMILES:
#       Insert the custom SMILES string to be predicted here.
#       Default is the SMILES representation of the caffeine molecule
SMILES = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'


__DEBUG__ = True

@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):

    # ~ Loading the dataset
    # One thing which the counterfactuals need is the "Processing" instance of the visual graph dataset
    # which implements the conversion of the domain specific representation (SMILES) to the full graph
    # representation.
    module = dynamic_import(e.PROCESSING_PATH)
    processing: ProcessingBase = module.processing

    # ~ Loading the model
    # The central requirement of the counterfactual generation is the model itself. We load a model from
    # persistent memory here so we don't have to go through the lengthy process of training one every time.
    
    e.log(f'loading the model from path "{e.MODEL_PATH}"...')
    model = load_model(e.MODEL_PATH)
    e.log(f'loaded a model with the name: {model.__class__.__name__}')

    # ~ Processing the SMILES
    # Since we are dealing with graph neural networks, the given string SMILES representation of the molecule 
    # first has to be converted into a graph representation. This has to be done with the specific "Processing"
    # instance that is attached to the original dataset to make sure that the resulting graph is compatible 
    # with the model input requirements.
    graph = processing.process(e.SMILES)
    
    # This method will take a list of graph dictionary representations, query the model and return the results
    # For the models of the MEGAN family, those prediction results are a tuple of 3 data structures: the main
    # prediction result, the edge explanation mask and the node explanation mask
    predictions = model.predict_graphs([graph])
    out_pred, ni_pred, ei_pred = predictions[0]
    e.log(f'for the given SMILES "{e.SMILES}" the model predicts: {out_pred}')
    # Better for visualization later 
    ni_pred = array_normalize(ni_pred)
    ei_pred = array_normalize(ei_pred)

    # ~ Creating the graph embedding
    # Another common operation for a given graph representation is to retrieve the model's internal representation 
    # for that graph - the graph embedding vector. This can be achieved with the "embedd_graphs" method in the 
    # same way as the prediction.
    # However it is important to note that strictly speaking MEGAN models do not produce graph embeddings but 
    # rather multiple "explanation embeddings". The direct results of that method will be as many embedding vectors 
    # as the model has explanation channels and each of them represents only the specific subgraph explanation that 
    # is highlighted in that channel.
    
    # embedding shape: (K, D)
    # where D the embedding dimension and K the number of explanation channels
    embeddings = model.embedd_graphs([graph])
    embedding = embeddings[0]
    e.log(f'original embedding shape: {embedding.shape}')
    
    # Now we want to concatenate those for now separate explanation embeddings into a single graph embedding 
    # which has only one dimension
    embedding = embedding.reshape((-1, ))
    e.log(f'concatenated embedding shape: {embedding.shape}')
    e.log(f'embedding: {embedding}')
    
    # ~ Visualizing the explanation
    # Apart from the actual prediction result, the MEGAN models also predict the explanations masks for the 
    # given input element which can be visualized.
    e.log('visualizing the explanations...')
    
    # To visualize the explanations though we first need a visualization of the input graph itself. This can also 
    # be done using the same "Processing" instance.
    # The "create" method will create a new visual graph dataset element representation in the given folder. This 
    # means that it will produce the visualization PNG file and the metadata JSON file.
    element_path = os.path.join(e.path, 'element')
    os.mkdir(element_path)
    processing.create(
        value=e.SMILES,
        index='0',
        width=1000,
        height=1000,
        output_path=element_path
    )
    element_data = load_visual_graph_element(element_path, name='0')
        
    # One specialty of the MEGAN model is that it does not only create a single explanation mask for each prediction 
    # but multiple ones at the same time along multiple "channels". we can infer the number of channels K from the shape 
    # of the mask which is (V, K) in this case where V is the number of nodes in the graph.
    num_channels = ni_pred.shape[1]

    # Now based on this newly created visual graph element a visualization can be created with matplotlib
    fig, rows = plt.subplots(
        ncols=num_channels,
        nrows=1,
        figsize=(10 * num_channels, 10),
        squeeze=False,
    )
    fig.suptitle(f'Prediction: {out_pred}')
    
    for k in range(num_channels):
        ax = rows[0][k]
        
        # First we draw the image that contains the visualization of the graph itself
        draw_image(ax, image_path=element_data['image_path'])
        # and then on top of this we can draw the explanation mask as a heat map
        plot_node_importances_background(
            ax=ax,
            g=element_data['metadata']['graph'],
            node_positions=element_data['metadata']['graph']['node_positions'],
            node_importances=ni_pred[:, k]
        )
        plot_edge_importances_background(
            ax=ax,
            g=element_data['metadata']['graph'],
            node_positions=element_data['metadata']['graph']['node_positions'],
            edge_importances=ei_pred[:, k]
        )
        ax.set_title(f'channel: {k}')
    
    fig_path = os.path.join(e.path, 'explanations.pdf')
    fig.savefig(fig_path)
    


experiment.run_if_main()
