"""
This experiment extends the vgd_torch base experiment for the training of a torch-based model 
on a visual graph dataset. This experiment will train a MEGAN model specifically to not only 
solve a prediction task but also to create explanations about that task.

In this experiment, the evaluation is extended to include the megan generated explanations as 
well. The model evaluation process will create a PDF file visualizing the megan explanations
for all the example elements from the test set.
Optionally if the graph embedding latent space is either 2 or 3 dimensional, it will be 
visualized in a plot as well for each explanation channel independently.

The analysis of this experiment will perform an HDBSCAN clustering on the graph embedding latent 
space of the all the trained models and will create PDF files that will visualize some of the 
elements in the dataset that are the closest to the cluster centroids.
"""
import os
import typing as t

import hdbscan
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
import pytorch_lightning as pl
from matplotlib.backends.backend_pdf import PdfPages
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from lightning.pytorch.loggers import CSVLogger
from visual_graph_datasets.visualization.importances import create_importances_pdf
from visual_graph_datasets.visualization.importances import plot_node_importances_background
from visual_graph_datasets.visualization.importances import plot_edge_importances_background
from visual_graph_datasets.visualization.base import draw_image

from graph_attention_student.utils import export_metadatas_csv
from graph_attention_student.visualization import generate_contrastive_colors
from graph_attention_student.visualization import plot_embeddings_3d
from graph_attention_student.visualization import plot_embeddings_2d
from graph_attention_student.visualization import plot_leave_one_out_analysis
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
# :param IMPORTANCE_OFFSET:
#       This parameter more or less controls how expansive the explanations are - how much of the graph they
#       tend to cover. Higher values tend to lead to more expansive explanations while lower values tend to 
#       lead to sparser explanations. Typical value range 0.5 - 1.5
IMPORTANCE_OFFSET: float = 0.8
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
# :param REGRESSION_MARGIN:
#       When converting the regression problem into the negative/positive classification problem for the 
#       explanation co-training, this determines the margin for the thresholding. Instead of using the regression
#       reference as a hard threshold, values have to be at least this margin value lower/higher than the 
#       regression reference to be considered a class sample.
REGRESSION_MARGIN: t.Optional[float] = 0.0
# :param NORMALIZE_EMBEDDING:
#       This boolean value determines whether the graph embeddings are normalized to a unit length or not.
#       If this is true, the embedding of each individual explanation channel will be L2 normalized such that 
#       it is projected onto the unit sphere.
NORMALIZE_EMBEDDING: bool = True
# :param CONTRASTIVE_FACTOR:
#       This is the factor of the contrastive representation learning loss of the network. If this value is 0 
#       the contrastive repr. learning is completely disabled (increases computational efficiency). The higher 
#       this value the more the contrastive learning will influence the network during training.
CONTRASTIVE_FACTOR: float = 1.0
# :param CONTRASTIVE_NOISE:
#       This float value determines the noise level that is applied when generating the positive augmentations 
#       during the contrastive learning process.
CONTRASTIVE_NOISE: float = 0.0
# :param CONTRASTIVE_TEMP:
#       This float value is a hyperparameter that controls the "temperature" of the contrastive learning loss.
#       The higher this value, the more the contrastive learning will be smoothed out. The lower this value,
#       the more the contrastive learning will be focused on the most similar pairs of embeddings.
CONTRASTIVE_TEMP: float = 1.0
# :param CONTRASTIVE_BETA:
#       This is the float value from the paper about the hard negative mining called the concentration 
#       parameter. It determines how much the contrastive loss is focused on the hardest negative samples.
CONTRASTIVE_BETA: float = 0.1
# :param CONTRASTIVE_TAU:
#       This float value is a hyperparameters of the de-biasing improvement of the contrastive learning loss. 
#       This value should be chosen as roughly the inverse of the number of expected concepts. So as an example 
#       if it is expected that each explanation consists of roughly 10 distinct concepts, this should be chosen 
#       as 1/10 = 0.1
CONTRASTIVE_TAU: float = 0.1
# :param PREDICTION_FACTOR:
#       This is a float value that determines the factor by which the main prediction loss is being scaled 
#       durign the model training. Changing this from 1.0 should usually not be necessary except for regression
#       tasks with a vastly different target value scale.
PREDICTION_FACTOR: float = 1.0


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
        e.log(f' * importance offset: {e.IMPORTANCE_OFFSET}')
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
            importance_offset=e.IMPORTANCE_OFFSET,
            sparsity_factor=e.SPARSITY_FACTOR,
            regression_reference=e.REGRESSION_REFERENCE,
            regression_margin=e.REGRESSION_MARGIN,
            prediction_mode=e.DATASET_TYPE,
            prediction_factor=e.PREDICTION_FACTOR,
            normalize_embedding=e.NORMALIZE_EMBEDDING,
            contrastive_factor=e.CONTRASTIVE_FACTOR,
            contrastive_temp=e.CONTRASTIVE_TEMP,
            contrastive_noise=e.CONTRASTIVE_NOISE,
            contrastive_beta=e.CONTRASTIVE_BETA,
            contrastive_tau=e.CONTRASTIVE_TAU,
            learning_rate=e.LEARNING_RATE,
        )
        
        e.log(f'starting model training with {e.EPOCHS} epochs...')
        logger = CSVLogger(e[f'path/{rep}'], name='logs')
        trainer = pl.Trainer(
            max_epochs=e.EPOCHS,
            logger=logger,
            # accelerator='cpu',
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
    
    Most importantly, this function will evaluate the model twords the explanations that are created by the 
    MEGAN model. It will create a PDF file which visualizes all the explanations that are created by the 
    model on the example graphs from the test set.
    
    Optionally, if the dimensionality of the latent space is either 2 or 3, the latent spaces of the 
    individual channels will be visualized as well. These latent space visualizations will include the 
    density and a potential hdbscan clustering.
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
    
    # ~ explanation fidelity analysis
    # Explanation fidelity is a metric that essentially tells how much a given attributional explanation mask 
    # actually influences the behavior of the model. It is usually measured as the deviation of the model output 
    # if the areas highlighted by the explanation are removed from the input data structure. 
    # The fidelity of the MEGAN model is a special case because it can be calculated for every explanation 
    # channel independently
    
    e.log(f'calculating explanation fidelity...')
    
    # leave_one_out: (B, O, K) numpy
    leave_one_out = model.leave_one_out_deviations(graphs_test)
    fig = plot_leave_one_out_analysis(
        leave_one_out,
        num_channels=e.NUM_CHANNELS,
        num_targets=e.FINAL_UNITS[-1],
    )
    fig.savefig(os.path.join(e[f'path/{rep}'], 'leave_one_out.pdf'))

    # ~ visualizing the graph embedding space
    # Another thing we would like to do for the MEGAN model is to visualize the graph embedding space to see 
    # how it is structure (if there is any semantic clustering of motifs or not).
     
    e.log(f'visualizating embedding space with {model.embedding_dim} dimensions...')
    if model.embedding_dim <= 3:
        
        e.log('embedding graphs...')
        infos = model.forward_graphs(graphs)
        embeddings = np.array([info['graph_embedding'] for info in infos])
        
        # For this task we dont need to differentiate between the different explanations channels, which is 
        # why we simply treat each channel as a different set of embeddings in the batch dimension.
        # embeddings: (N*K, D)
        embeddings_combined = np.concatenate([embeddings[:, :, k] for k in range(e.NUM_CHANNELS)], axis=0)
        # We calculate the local density using the K nearest neighbors of each embedding.
        
        colormap = mpl.cm.RdPu
        num_neighbors = 100
        
        if model.embedding_dim == 3: 
            plot_func = plot_embeddings_3d
            projection = '3d'
        if model.embedding_dim == 2:
            plot_func = plot_embeddings_2d
            projection = None
        
        e.log(f'plotting the embeddings with dimension {model.embedding_dim}...')
        # ~ channel embeddings
        num_cols = 3
        fig = plt.figure(figsize=(8 * num_cols, 8 * e.NUM_CHANNELS))
        fig.suptitle('Graph Embeddings')
        plot_kwargs = {}
        
        x_range = (np.min(embeddings[:, 0, :]), np.max(embeddings[:, 0, :]))
        plot_kwargs['x_range'] = x_range
        y_range = (np.min(embeddings[:, 1, :]), np.max(embeddings[:, 1, :]))
        plot_kwargs['y_range'] = y_range
        if model.embedding_dim == 3:
            z_range = (np.min(embeddings[:, 2, :]), np.max(embeddings[:, 2, :]))
            plot_kwargs['z_range'] = z_range
        
        index = 1
        for k, channel_info in e.CHANNEL_INFOS.items():
            
            embeddings_k = embeddings[:, :, k]
            
            ax_chan = fig.add_subplot(e.NUM_CHANNELS, num_cols, index, projection=projection)
            index += 1
            plot_func(
                embeddings=embeddings[:, :, k],
                ax=ax_chan,
                color=channel_info['color'],
                label=channel_info['name'],
                scatter_kwargs={'alpha': 0.8},
                **plot_kwargs
            )
            ax_chan.legend()
            
            ax_dens = fig.add_subplot(e.NUM_CHANNELS, num_cols, index, projection=projection)
            index += 1
            # We calculate the local density using the K nearest neighbors of each embedding.
            neighbors = NearestNeighbors(
                n_neighbors=num_neighbors+1, 
                algorithm='ball_tree',
            ).fit(embeddings_k)
            distances, indices = neighbors.kneighbors(embeddings_k)
            # local_density: (N, )
            local_density = np.mean(distances[:, 1:], axis=1)
            local_density = np.max(local_density) - local_density
            min_density = np.min(local_density)
            max_density = np.max(local_density)
            norm = mcolors.Normalize(vmin=min_density, vmax=max_density)
            alphas = norm(local_density)
            colors = colormap(alphas)

            plot_func(
                embeddings=embeddings_k,
                ax=ax_dens,
                color=colors,
                label=channel_info['name'],
                scatter_kwargs={'alpha': 0.2},
                **plot_kwargs
            )
            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax_dens, fraction=0.046, pad=0.04)
            
            ax_clus = fig.add_subplot(e.NUM_CHANNELS, num_cols, index, projection=projection)
            index += 1
            clusterer = hdbscan.HDBSCAN(
                min_samples=20,
            )
            cluster_labels = clusterer.fit_predict(embeddings_k)
            cluster_indices = list(set(cluster_labels))
            num_clusters = len(cluster_indices) - 1
            e.log(f' * channel {k} - num elements {len(embeddings_k)} - num clusters: {num_clusters} ')
            if num_clusters < 2: # catching a possible exception
                continue
            
            color_map = generate_contrastive_colors(num_clusters)
            colors = [(1, 1, 1, 1) if label < 0 else color_map[label] for label in cluster_labels]
            plot_func(
                embeddings=embeddings_k,
                ax=ax_clus,
                color=colors,
                label=channel_info['name'],
                scatter_kwargs={'alpha': 0.8},
                **plot_kwargs
            )
            
        fig.savefig(os.path.join(e[f'path/{rep}'], 'embeddings.pdf'))
        fig.savefig(os.path.join(e[f'path/{rep}'], 'embeddings.png'))
        plt.close(fig)


@experiment.analysis
def analysis(e: Experiment):
    """
    This analysis
    """
    e.log('running MEGAN specific analysis...')
    
    # In the analysis routine of the parent experiment, the index data map is loaded from the disk 
    # already and stored in this experiment storage entry, so that it can be reused here.
    index_data_map = e['_index_data_map']
    graphs = [data['metadata']['graph'] for data in index_data_map.values()]
    indices = np.array(list(index_data_map.keys()))
    
    e.log('Clustering the latent space...')
    min_samples = 5
    for rep in range(e.REPETITIONS):
        
        e.log(f'> REP {rep}')
        test_indices = e[f'test_indices/{rep}']
        graphs_test = [index_data_map[index]['metadata']['graph'] for index in test_indices]
        
        model = e[f'_model/{rep}']
        infos = model.forward_graphs(graphs)
        
        for k in range(e.NUM_CHANNELS):
            
            # graph_embeddings_k: (B, D)
            graph_embeddings_k = np.array([info['graph_embedding'][:, k] for info in infos])

            clusterer = hdbscan.HDBSCAN(min_samples=min_samples)
            cluster_labels = clusterer.fit_predict(graph_embeddings_k)
            cluster_indices = list(set(cluster_labels))
            num_clusters = len(cluster_indices) - 1
            e.log(f' * channel {k} - num clusters: {num_clusters}')
            
            with PdfPages(os.path.join(e[f'path/{rep}'], f'cluster__ch{k:02d}.pdf')) as pdf:
                
                for cluster_index in cluster_indices:
                    
                    fig, rows = plt.subplots(
                        ncols=10, 
                        nrows=1, 
                        figsize=(50, 5),
                        squeeze=False,
                    )
                    
                    cluster_centroid = np.mean(graph_embeddings_k[cluster_labels == cluster_index], axis=0, keepdims=True)
                    cosine_distances = pairwise_distances(graph_embeddings_k, cluster_centroid, metric='cosine')
                    cosine_distances = cosine_distances.flatten()
                    closest_indices = np.argsort(cosine_distances)[:10]

                    for i, j in enumerate(closest_indices):
                        
                        info = infos[j]
                        index = indices[j]
                        ax = rows[0][i]
                        graph = index_data_map[index]['metadata']['graph']
                        draw_image(
                            ax=ax,
                            image_path=index_data_map[index]['image_path']
                        )
                        node_importance = info['node_importance'] / np.max(info['node_importance'])
                        edge_importance = info['edge_importance'] / np.max(info['edge_importance'])
                        plot_node_importances_background(
                            ax=ax,
                            g=graph,
                            node_positions=graph['node_positions'],
                            node_importances=node_importance[:, k],
                            radius=50,
                        )
                        plot_edge_importances_background(
                            ax=ax,
                            g=graph,
                            node_positions=graph['node_positions'],
                            edge_importances=edge_importance[:, k],
                            thickness=10,
                        )
                
                    pdf.savefig(fig)
                    plt.close(fig)
            

experiment.run_if_main()