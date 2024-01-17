import os
import typing as t

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from lightning.pytorch.loggers import CSVLogger
from visual_graph_datasets.visualization.importances import create_importances_pdf
from visual_graph_datasets.visualization.importances import plot_node_importances_background
from visual_graph_datasets.visualization.importances import plot_edge_importances_background

from graph_attention_student.visualization import create_embeddings_pdf
from graph_attention_student.visualization import plot_embeddings_3d
from graph_attention_student.torch.data import data_from_graph
from graph_attention_student.torch.data import data_list_from_graphs
from graph_attention_student.torch.model import AbstractGraphModel
from graph_attention_student.torch.megan import Megan

# == MODEL PARAMETERS ==
# The following parameters configure the model architecture.

# :param NUM_CHANNELS:
#       The number of explanation channels for the model.
NUM_CHANNELS: int = 2
# :param CHANNEL_INFOS:
#       This dictionary can be used to add additional information about the explanation channels that 
#       are used in this experiment. The integer keys of the dict are the indices of the channels
#       and the values are dictionaries that contain the information about the channel with that index.
#       This dict has to have as many entries as there are explanation channels defined for the 
#       model. The info dict for each channel may contain a "name" string entry for a human readable name 
#       asssociated with that channel and a "color" entry to define a color of that channel in the 
#       visualizations.
CHANNEL_INFOS: dict = {
    0: {
        'name': 'negative',
        'color': 'skyblue',
    },
    1: {
        'name': 'positive',
        'color': 'coral',
    }
}
# :param UNITS:
#       This list determines the layer structure of the model's graph encoder part. Each element in 
#       this list represents one layer, where the integer value determines the number of hidden units 
#       in that layer of the encoder network.
UNITS: t.List[int] = [32, 32, 32]
# :param IMPORTANCE_UNITS:
#       This list determines the layer structure of the importance MLP which determines the node importance 
#       weights from the node embeddings of the graph. 
#       Each element in this list represents one layer where the integer value determines the number of hidden 
#       units in that layer.
IMPORTANCE_UNITS: t.List[int] = [32, ]
# :param PROJECTION_LAYERS:
#       This list determines the layer structure of the MLP's that act as the channel-specific projections.
#       Each element in this list represents one layer where the integer value determines the number of hidden
#       units in that layer.
PROJECTION_UNITS: t.List[int] = []
# :param FINAL_UNITS:
#       This list determines the layer structure of the model's final prediction MLP. Each element in 
#       this list represents one layer, where the integer value determines the number of hidden units 
#       in that layer of the prediction network.
#       Note that the last value of this list determines the output shape of the entire network and 
#       therefore has to match the number of target values given in the dataset.
FINAL_UNITS: t.List[int] = [32, 1]
# :param IMPORTANCE_FACTOR:
#       This is the coefficient that is used to scale the explanation co-training loss during training.
#       Roughly, the higher this value, the more the model will prioritize the explanations during training.
IMPORTANCE_FACTOR: float = 1.0
# :param SPARSITY_FACTOR:
#       This is the coefficient that is used to scale the explanation sparsity loss during training.
#       The higher this value the more explanation sparsity (less and more discrete explanation masks)
#       is promoted.
SPARSITY_FACTOR: float = 1.0
# :param REGRESSION_REFERENCE:
#       When dealing with regression tasks, an important hyperparameter to set is this reference value in the 
#       range of possible target values, which will determine what part of the dataset is to be considered as 
#       negative / positive in regard to the negative and the positive explanation channel. A good first choice 
#       for this parameter is the average target value of the training dataset. Depending on the results for 
#       that choice it is possible to adjust the value to adjust the explanations.
REGRESSION_REFERENCE: t.Optional[float] = 0.0

__DEBUG__ = True

experiment = Experiment.extend(
    'vgd_torch.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


@experiment.hook('train_model', default=False, replace=True)
def train_model(e: Experiment,
                rep: int,
                index_data_map: dict,
                train_indices: t.List[int],
                test_indices: t.List[int],
                **kwargs,
                ) -> t.Tuple[AbstractGraphModel, pl.Trainer]:
        """
        This hook gets called to actually construct and train a model during the main training loop. This hook 
        receives the full dataset in the format of the index_data_map as well as the train test split in the format 
        of the train_indices and test_indices. The hook is supposed to return a tuple (model, trainer) that were 
        used for the training process.
        
        In this implementation a ``Megan`` model is trained. Besides the main property prediction task, this 
        megan model is also trained to generate a set of explanations about that task at the same time by using the 
        approximative explanation co-training procedure.
        """
        e.log('preparing data for training...')
        graphs_train = [index_data_map[i]['metadata']['graph'] for i in train_indices]
        graphs_test = [index_data_map[i]['metadata']['graph'] for i in test_indices]
        
        train_loader = DataLoader(data_list_from_graphs(graphs_train), batch_size=e.BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(data_list_from_graphs(graphs_test), batch_size=e.BATCH_SIZE, shuffle=False)
        
        e.log(f'Instantiating Megan model - with explanation training...')
        e.log(f'explanation mode: {e.DATASET_TYPE}')
        model = Megan(
            node_dim=e['node_dim'],
            edge_dim=e['edge_dim'],
            units=e.UNITS,
            importance_units=e.IMPORTANCE_UNITS,
            # only if this is a not-None value, the explanation co-training of the model is actually
            # enabled. The explanation co-training works differently for regression and classification tasks
            projection_units=e.PROJECTION_UNITS,
            importance_mode=e.DATASET_TYPE,
            final_units=e.FINAL_UNITS,
            num_channels=e.NUM_CHANNELS,
            importance_factor=e.IMPORTANCE_FACTOR,
            sparsity_factor=e.SPARSITY_FACTOR,
            regression_reference=e.REGRESSION_REFERENCE,
            prediction_mode=e.DATASET_TYPE,
            learning_rate=e.LEARNING_RATE,
        )
        
        e.log(f'starting model training with {e.EPOCHS} epochs...')
        logger = CSVLogger(e[f'path/{rep}'], name='logs')
        trainer = pl.Trainer(
            max_epochs=e.EPOCHS,
            logger=logger,
        )
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=test_loader,
        )
        
        model.to('cpu')
        return model, trainer


@experiment.hook('evaluate_model', default=False, replace=False)
def evaluate_model(e: Experiment,
                   model: AbstractGraphModel,
                   trainer: pl.Trainer,
                   rep: int,
                   index_data_map: dict,
                   train_indices: t.List[int],
                   test_indices: t.List[int],
                   ) -> None:
    """
    This hook is called during the main training loop AFTER the model has been fully trained to evaluate 
    the performance of the trained model on the test set. The hook receives the trained model as a parameter,
    as well as the repetition index, the full index_data_map dataset and the train and test indices.
    
     
    """
    e.log('evaluating Megan explanations...')

    graphs = [data['metadata']['graph'] for data in index_data_map.values()]
    graphs_test = [index_data_map[i]['metadata']['graph'] for i in test_indices]
        
    # out_true np.ndarray: (B, O)
    out_true = np.array([graph['graph_labels'] for graph in graphs_test])
    # out_pred np.ndarray: (B, O)
    out_pred = model.predict_graphs(graphs_test)
    
    
    e.log('visualizing the example graphs...')
    example_indices: t.List[int] = e[f'example_indices/{rep}']
    graphs_example = [index_data_map[i]['metadata']['graph'] for i in example_indices]
    example_infos: t.List[dict] = model.forward_graphs(graphs_example)
    create_importances_pdf(
        graph_list=graphs_example,
        image_path_list=[index_data_map[i]['image_path'] for i in example_indices],
        node_positions_list=[graph['node_positions'] for graph in graphs_example],
        importances_map={
            'megan': (
                [info['node_importance'] for info in example_infos],
                [info['edge_importance'] for info in example_infos],
            )
        },
        output_path=os.path.join(e[f'path/{rep}'], 'example_explanations.pdf'),
        plot_node_importances_cb=plot_node_importances_background,
        plot_edge_importances_cb=plot_edge_importances_background,
    )

    # ~ visualizing the graph embedding space
    # Another thing we would like to do for the MEGAN model is to visualize the graph embedding space to see 
    # how it is structure (if there is any semantic clustering of motifs or not).
     
    e.log(f'visualizating embedding space with {model.embedding_dim} dimensions...')
    if model.embedding_dim <= 3:
        
        e.log('embedding graphs...')
        infos = model.forward_graphs(graphs_test)
        embeddings = np.array([info['graph_embedding'] for info in infos])
        
        # For this task we dont need to differentiate between the different explanations channels, which is 
        # why we simply treat each channel as a different set of embeddings in the batch dimension.
        # embeddings: (N*K, D)
        embeddings_combined = np.concatenate([embeddings[:, :, k] for k in range(e.NUM_CHANNELS)], axis=0)
        # We calculate the local density using the K nearest neighbors of each embedding.
        num_neighbors = 50
        neighbors = NearestNeighbors(
            n_neighbors=num_neighbors+1, 
            algorithm='ball_tree'
        ).fit(embeddings_combined)
        distances, indices = neighbors.kneighbors(embeddings_combined)
        # local_density: (N*K, )
        local_density = np.mean(distances[:, 1:], axis=1)
        local_density = np.max(local_density) - local_density
        min_density = np.min(local_density)
        max_density = np.max(local_density)
        
        e.log('plotting the embeddings...')
        if model.embedding_dim == 3:
            
            # ~ channel embeddings
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title('Graph Embeddings')
            
            x_range = (np.min(embeddings[:, 0, :]), np.max(embeddings[:, 0, :]))
            y_range = (np.min(embeddings[:, 1, :]), np.max(embeddings[:, 1, :]))
            z_range = (np.min(embeddings[:, 2, :]), np.max(embeddings[:, 2, :]))
            
            for k, channel_info in e.CHANNEL_INFOS.items():
                
                plot_embeddings_3d(
                    embeddings=embeddings[:, :, k],
                    ax=ax,
                    color=channel_info['color'],
                    label=channel_info['name'],
                    scatter_kwargs={'alpha': 0.8},
                    x_range=x_range,
                    y_range=y_range,
                    z_range=z_range,
                )
                
            ax.legend()
            fig.savefig(os.path.join(e[f'path/{rep}'], 'embeddings.pdf'))
            fig.savefig(os.path.join(e[f'path/{rep}'], 'embeddings.png'))
            plt.close(fig)

            # density plot
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title('Graph Embeddings with Local Density')
            
            norm = mcolors.Normalize(vmin=min_density, vmax=max_density)
            colormap = plt.cm.RdPu
            alphas = norm(local_density)
            colors = colormap(alphas)

            plot_embeddings_3d(
                embeddings=embeddings_combined,
                ax=ax,
                color=colors,
                label=channel_info['name'],
                scatter_kwargs={'alpha': 0.2},
            )
            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax)
            
            fig.savefig(os.path.join(e[f'path/{rep}'], 'embeddings_density.pdf'))

experiment.run_if_main()