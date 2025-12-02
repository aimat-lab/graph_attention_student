"""
This experiment extends the train_model base experiment for the training of a MEGAN model
from CSV data. Unlike the VGD-based experiments, this experiment loads data directly from CSV
and computes graph representations on-the-fly.

This experiment trains a MEGAN model specifically to not only solve a prediction task but also
to create explanations about that task. The model evaluation process will create a PDF file
visualizing the MEGAN explanations for all the example elements from the test set.

Visualizations are computed on-demand during evaluation and cached to the experiment archive
folder for efficiency.
"""
import os
import tempfile
import typing as t
from typing import List, Optional

import torch
import hdbscan
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pytorch_lightning as pl
from rich.pretty import pprint
from matplotlib.backends.backend_pdf import PdfPages
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from torch_geometric.loader import DataLoader
from lightning.pytorch.loggers import CSVLogger
from visual_graph_datasets.visualization.importances import create_importances_pdf
from visual_graph_datasets.visualization.importances import plot_node_importances_background
from visual_graph_datasets.visualization.importances import plot_edge_importances_background
from visual_graph_datasets.visualization.base import draw_image

from graph_attention_student.visualization import plot_regression_fit
from graph_attention_student.visualization import generate_contrastive_colors
from graph_attention_student.visualization import plot_embeddings_3d
from graph_attention_student.visualization import plot_embeddings_2d
from graph_attention_student.visualization import plot_leave_one_out_analysis
from graph_attention_student.torch.data import data_list_from_graphs
from graph_attention_student.torch.model import AbstractGraphModel
from graph_attention_student.torch.megan import Megan
from graph_attention_student.torch.megan import MveCallback
from graph_attention_student.torch.utils import SwaCallback
from graph_attention_student.torch.utils import ContrastiveSchedulerCallback
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns

mpl.use('Agg')


# == MODEL PARAMETERS ==
# The following parameters configure the model architecture.

# :param NUM_CHANNELS:
#       The number of explanation channels for the model.
NUM_CHANNELS: int = 2
# :param CHANNEL_INFOS:
#       Information about each explanation channel including name and color.
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
#       Layer structure of the model's graph encoder.
UNITS: t.List[int] = [64, 64, 64]
# :param HIDDEN_UNITS:
#       Number of hidden units in the attention layer's transformative networks.
HIDDEN_UNITS: int = 128
# :param IMPORTANCE_UNITS:
#       Layer structure of the importance MLP.
IMPORTANCE_UNITS: t.List[int] = [32, ]
# :param PROJECTION_UNITS:
#       Layer structure of the channel-specific projection MLPs.
PROJECTION_UNITS: t.List[int] = []
# :param FINAL_UNITS:
#       Layer structure of the final prediction MLP.
FINAL_UNITS: t.List[int] = [32, 1]
# :param OUTPUT_NORM:
#       Optional norm constraint for classification logits.
OUTPUT_NORM: t.Optional[float] = None
# :param LABEL_SMOOTHING:
#       Label smoothing parameter for cross entropy loss.
LABEL_SMOOTHING: t.Optional[float] = 0.0
# :param IMPORTANCE_FACTOR:
#       Coefficient for the explanation co-training loss.
IMPORTANCE_FACTOR: float = 1.0
# :param IMPORTANCE_OFFSET:
#       Controls the sparsity of explanation masks.
IMPORTANCE_OFFSET: float = 0.8
# :param SPARSITY_FACTOR:
#       DEPRECATED. Coefficient for explanation sparsity loss.
SPARSITY_FACTOR: float = 1.0
# :param FIDELITY_FACTOR:
#       Coefficient for explanation fidelity loss.
FIDELITY_FACTOR: float = 0.0
# :param REGRESSION_REFERENCE:
#       Reference value for regression explanation training.
REGRESSION_REFERENCE: t.Optional[float] = 0.0
# :param REGRESSION_MARGIN:
#       DEPRECATED. Margin for regression explanation training.
REGRESSION_MARGIN: t.Optional[float] = 0.0
# :param NORMALIZE_EMBEDDING:
#       Whether to L2 normalize graph embeddings.
NORMALIZE_EMBEDDING: bool = False
# :param ATTENTION_AGGREGATION:
#       Strategy for aggregating edge attention across layers ('sum', 'max', 'min').
ATTENTION_AGGREGATION: str = 'max'
# :param CONTRASTIVE_FACTOR:
#       Factor for contrastive representation learning loss.
CONTRASTIVE_FACTOR: float = 0.0
# :param CONTRASTIVE_NOISE:
#       Noise level for positive augmentations in contrastive learning.
CONTRASTIVE_NOISE: float = 0.1
# :param CONTRASTIVE_TEMP:
#       Temperature for contrastive learning loss.
CONTRASTIVE_TEMP: float = 1.0
# :param CONTRASTIVE_BETA:
#       Concentration parameter for hard negative mining.
CONTRASTIVE_BETA: float = 0.1
# :param CONTRASTIVE_TAU:
#       De-biasing parameter for contrastive learning.
CONTRASTIVE_TAU: float = 0.1
# :param USE_CONTRASTIVE_SCHEDULER:
#       Whether to use a scheduler that linearly ramps up the contrastive factor
#       from near zero to the target value over a warmup period.
USE_CONTRASTIVE_SCHEDULER: bool = True
# :param CONTRASTIVE_WARMUP_EPOCHS:
#       Fraction of total epochs over which to linearly ramp up the contrastive factor.
#       E.g., 0.7 means the contrastive factor reaches its target value after 70% of training.
CONTRASTIVE_WARMUP_EPOCHS: float = 0.5
# :param PREDICTION_FACTOR:
#       Factor for scaling the prediction loss.
PREDICTION_FACTOR: float = 1.0
# :param TRAIN_MVE:
#       Whether to train as a mean variance estimator model.
TRAIN_MVE: bool = False
# :param MVE_WARMUP_EPOCHS:
#       Epochs of normal training before switching to MVE loss.
MVE_WARMUP_EPOCHS: int = 50


# == VISUALIZATION PARAMETERS ==

# :param DO_CLUSTERING:
#       Whether to perform clustering analysis.
DO_CLUSTERING: bool = True
# :param EXAMPLE_VALUES:
#       Optional list of domain representations to plot as examples during training.
EXAMPLE_VALUES: List[str] = []

# == EMBEDDING TRACKING PARAMETERS ==

# :param TRACK_EMBEDDING_HEALTH:
#       Whether to track embedding health metrics (norms, cosine similarity, effective rank).
TRACK_EMBEDDING_HEALTH: bool = True
# :param TRACK_EMBEDDING_UNIFORMITY:
#       Whether to track embedding uniformity metrics.
TRACK_EMBEDDING_UNIFORMITY: bool = True
# :param TRACK_EMBEDDING_TARGET_ALIGNMENT:
#       Whether to track embedding-target alignment metrics.
TRACK_EMBEDDING_TARGET_ALIGNMENT: bool = True
# :param TRACK_EMBEDDING_PROJECTIONS:
#       Whether to create PCA and UMAP projection plots.
TRACK_EMBEDDING_PROJECTIONS: bool = True
# :param EMBEDDING_TRACK_FREQUENCY:
#       Frequency (in epochs) at which to track embedding metrics. Set to higher values
#       to reduce computational overhead during training.
EMBEDDING_TRACK_FREQUENCY: int = 1
# :param EMBEDDING_SAMPLE_SIZE:
#       Maximum number of samples to use for expensive pairwise computations.
EMBEDDING_SAMPLE_SIZE: int = 1000


# == TRAINING PARAMETERS ==

# :param BATCH_ACCUMULATE:
#       Number of batches for gradient accumulation.
BATCH_ACCUMULATE: int = 1
# :param LR_SCHEDULER:
#       Learning rate scheduler type ('cyclic' or None).
LR_SCHEDULER: Optional[str] = 'cyclic'
# :param USE_SWA:
#       Whether to use Stochastic Weight Averaging.
USE_SWA: bool = False
# :param SWA_EPOCHS:
#       Number of final epochs for SWA averaging.
SWA_EPOCHS: int = 10


__DEBUG__ = True

experiment = Experiment.extend(
    'train_model.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


@experiment.hook('train_model', default=False, replace=True)
def train_model(e: Experiment,
                index_data_map: dict,
                train_indices: t.List[int],
                test_indices: t.List[int],
                val_indices: t.List[int],
                **kwargs,
                ) -> t.Tuple[AbstractGraphModel, pl.Trainer]:
    """
    Train a MEGAN model with explanation co-training.

    This implementation trains a MEGAN model that generates both predictions and
    explanations for the prediction task.
    """
    e.log('preparing data for training...')
    graphs_train = [index_data_map[i]['metadata']['graph'] for i in train_indices]
    graphs_test = [index_data_map[i]['metadata']['graph'] for i in test_indices]
    graphs_val = [index_data_map[i]['metadata']['graph'] for i in val_indices]

    train_loader = DataLoader(
        data_list_from_graphs(graphs_train),
        batch_size=e.BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        data_list_from_graphs(graphs_test),
        batch_size=e.BATCH_SIZE,
        shuffle=False,
    )

    example_indices = test_indices[:8]
    example_graphs = [index_data_map[i]['metadata']['graph'] for i in example_indices]

    # Generate visualizations for example elements used during training
    for index in example_indices:
        data = index_data_map[index]
        if data['image_path'] is None:
            image_path, node_positions = e.apply_hook(
                'get_visualization',
                index=index,
                metadata=data['metadata']
            )
            data['image_path'] = image_path
            if node_positions is not None:
                data['metadata']['graph']['node_positions'] = node_positions

    class TrainingCallback(pl.Callback):

        def on_keyboard_interrupt(self, trainer: pl.Trainer, pl_module: Megan):
            print('keyboard interrupt...')
            trainer.should_stop = True

        def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: Megan):

            device = pl_module.device
            pl_module.eval()
            pl_module.to('cpu')

            self.track_metrics(trainer, pl_module)
            self.track_examples(trainer, pl_module)
            metrics = self.track_validation(trainer, pl_module)
            e.track_many(metrics)

            e.log(f'epoch {trainer.current_epoch} - model evaluation'
                  f' - {" - ".join([f"{key}: {value:.2f}" for key, value in metrics.items()])}')
            pl_module.train()
            pl_module.to(device)

        def track_validation(self, trainer: pl.Trainer, pl_module: Megan):

            model = pl_module
            results: List[dict] = model.forward_graphs(graphs_val)

            values_true = np.array([graph['graph_labels'] for graph in graphs_val])
            values_pred = np.array([result['graph_output'] for result in results])

            appox_true, approx_pred = model._predict_approximate(
                results=results,
                values_true=values_true
            )
            acc_approx_value = np.mean(appox_true == np.round(approx_pred))

            if e.DATASET_TYPE == 'regression':
                r2_value = r2_score(values_true, values_pred)
                mae_value = mean_absolute_error(values_true, values_pred)

                fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))
                plot_regression_fit(values_true, values_pred, ax=ax)
                ax.set_title(f'Target\nR2: {r2_value:.3f} - MAE: {mae_value:.3f}')
                e.track('val_regression', fig)
                plt.close('all')

                return {'r2': r2_value, 'mae': mae_value, 'approx': acc_approx_value}

            if e.DATASET_TYPE == 'classification':
                values_true = np.argmax(values_true, axis=1)
                values_pred = np.argmax(values_pred, axis=1)

                acc_value = accuracy_score(values_true, values_pred)
                f1_value = f1_score(values_true, values_pred, average='macro')

                cm = confusion_matrix(values_true, values_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted Labels')
                ax.set_ylabel('True Labels')
                ax.set_title('Confusion Matrix')
                e.track('val_confusion_matrix', fig)
                plt.close('all')

                return {'accuracy': acc_value, 'f1': f1_value, 'approx': acc_approx_value}

        def track_examples(self, trainer: pl.Trainer, pl_module: Megan):

            fig, rows = plt.subplots(
                ncols=len(example_indices),
                nrows=e.NUM_CHANNELS,
                figsize=(8 * len(example_indices), 8 * e.NUM_CHANNELS),
                squeeze=False
            )
            fig.suptitle(f'Examples - Epoch {trainer.current_epoch}')

            model = pl_module
            results: List[dict] = model.forward_graphs(example_graphs)

            for c, (index, graph, result) in enumerate(zip(example_indices, example_graphs, results)):

                out_pred = result['graph_output']
                out_true = graph['graph_labels']

                for k in range(e.NUM_CHANNELS):

                    node_importance = result['node_importance'][:, k]
                    edge_importance = result['edge_importance'][:, k]

                    ax = rows[k][c]
                    ax.set_title(f'index {index}\n'
                                 f'channel {k} - {CHANNEL_INFOS[k]["name"]}\n'
                                 f'out_true: {np.round(out_true, 2)} - out_pred: {np.round(out_pred, 2)}\n'
                                 f'importance'
                                 f' - mean: {np.round(np.mean(node_importance), 2):.2f}'
                                 f' - max: {np.round(np.max(node_importance), 2):.2f}'
                                 f' - min: {np.round(np.min(node_importance), 2):.2f}')

                    image_path = index_data_map[index]['image_path']
                    if image_path and os.path.exists(image_path):
                        draw_image(ax=ax, image_path=image_path)
                        if 'node_positions' in graph:
                            plot_node_importances_background(
                                ax=ax,
                                g=graph,
                                color=e.CHANNEL_INFOS[k]['color'],
                                node_importances=result['node_importance'][:, k],
                                node_positions=graph['node_positions'],
                            )
                            plot_edge_importances_background(
                                ax=ax,
                                g=graph,
                                color=e.CHANNEL_INFOS[k]['color'],
                                edge_importances=result['edge_importance'][:, k],
                                node_positions=graph['node_positions'],
                            )

            e.track('examples_training', fig)
            plt.close('all')

        def track_metrics(self, trainer: pl.Trainer, pl_module: Megan):

            for key, value in trainer.callback_metrics.items():
                if value is not None:
                    e.track(key, value.item())

            e.track('lr', trainer.optimizers[0].param_groups[0]['lr'])

    class RecordEmbeddingsCallback(pl.Callback):
        """
        Callback for tracking embedding space metrics and visualizations during training.

        Tracks the following categories:
        - Health: embedding norms, cosine similarities, effective rank, singular values
        - Uniformity: distribution uniformity on the hypersphere
        - Target Alignment: correlation and k-NN smoothness w.r.t. target values
        - Projections: PCA and UMAP 2D visualizations
        """

        def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: Megan):
            e.track('contrastive_factor', pl_module.contrastive_factor)

        def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: Megan):

            # Only track at specified frequency
            if trainer.current_epoch % e.EMBEDDING_TRACK_FREQUENCY != 0:
                return

            # Check if any tracking is enabled
            if not any([
                e.TRACK_EMBEDDING_HEALTH,
                e.TRACK_EMBEDDING_UNIFORMITY,
                e.TRACK_EMBEDDING_TARGET_ALIGNMENT,
                e.TRACK_EMBEDDING_PROJECTIONS,
            ]):
                return

            e.log('recording embedding metrics...')
            device = pl_module.device
            pl_module.eval()
            pl_module.to('cpu')

            # Get embeddings and targets
            infos_test = pl_module.forward_graphs(graphs_test)
            # embeddings: (num_graphs, embedding_dim, num_channels)
            embeddings = np.array([info['graph_embedding'] for info in infos_test])
            targets = np.array([graph['graph_labels'] for graph in graphs_test])
            targets_flat = targets.flatten() if targets.ndim > 1 else targets

            num_channels = embeddings.shape[-1]
            current_epoch = trainer.current_epoch

            # Track each category with error handling to prevent crashes
            if e.TRACK_EMBEDDING_HEALTH:
                try:
                    self.track_health(embeddings, num_channels, current_epoch)
                except Exception as ex:
                    e.log(f'WARNING: track_health failed: {ex}')

            if e.TRACK_EMBEDDING_UNIFORMITY:
                try:
                    self.track_uniformity(embeddings, num_channels, current_epoch)
                except Exception as ex:
                    e.log(f'WARNING: track_uniformity failed: {ex}')

            if e.TRACK_EMBEDDING_TARGET_ALIGNMENT:
                try:
                    self.track_target_alignment(embeddings, targets_flat, num_channels, current_epoch)
                except Exception as ex:
                    e.log(f'WARNING: track_target_alignment failed: {ex}')

            if e.TRACK_EMBEDDING_PROJECTIONS:
                try:
                    self.track_projections(embeddings, targets_flat, num_channels, current_epoch)
                except Exception as ex:
                    e.log(f'WARNING: track_projections failed: {ex}')

            # Original 3D plotting for low-dimensional embeddings
            if pl_module.embedding_dim <= 3:
                try:
                    self.track_3d_embeddings(embeddings, pl_module.embedding_dim, num_channels, current_epoch)
                except Exception as ex:
                    e.log(f'WARNING: track_3d_embeddings failed: {ex}')

            pl_module.train()
            pl_module.to(device)

        def track_health(self, embeddings: np.ndarray, num_channels: int, epoch: int):
            """
            Track embedding health metrics and create visualization figure.

            Figure layout (num_channels rows x 3 columns):
            - Column 1: Embedding norm distribution (histogram)
            - Column 2: Pairwise cosine similarity distribution (histogram)
            - Column 3: Singular value spectrum (line plot)
            """
            fig, axes = plt.subplots(
                nrows=num_channels,
                ncols=3,
                figsize=(15, 4 * num_channels),
                squeeze=False
            )
            fig.suptitle(f'Embedding Health - Epoch {epoch}', fontsize=14)

            for k in range(num_channels):
                emb_k = embeddings[:, :, k]  # (num_graphs, embedding_dim)

                # --- Column 1: Norm distribution ---
                norms = np.linalg.norm(emb_k, axis=1)
                ax_norm = axes[k, 0]
                ax_norm.hist(norms, bins=30, color=e.CHANNEL_INFOS[k]['color'], alpha=0.7, edgecolor='black')
                ax_norm.axvline(np.mean(norms), color='red', linestyle='--', label=f'mean={np.mean(norms):.2f}')
                ax_norm.set_xlabel('L2 Norm')
                ax_norm.set_ylabel('Count')
                ax_norm.set_title(f'Channel {k}: Norm Distribution')
                ax_norm.legend()

                # Track scalar metrics
                e.track(f'emb_norm_mean_ch{k}', np.mean(norms))
                e.track(f'emb_norm_std_ch{k}', np.std(norms))

                # --- Column 2: Cosine similarity distribution ---
                # Sample for efficiency
                n_samples = min(len(emb_k), e.EMBEDDING_SAMPLE_SIZE)
                idx = np.random.choice(len(emb_k), n_samples, replace=False)
                emb_sample = emb_k[idx]
                emb_normed = emb_sample / (np.linalg.norm(emb_sample, axis=1, keepdims=True) + 1e-8)
                cos_sim_matrix = emb_normed @ emb_normed.T
                # Get upper triangle (excluding diagonal)
                triu_idx = np.triu_indices(n_samples, k=1)
                cos_sim_values = cos_sim_matrix[triu_idx]

                ax_cos = axes[k, 1]
                ax_cos.hist(cos_sim_values, bins=50, color=e.CHANNEL_INFOS[k]['color'], alpha=0.7, edgecolor='black')
                ax_cos.axvline(np.mean(cos_sim_values), color='red', linestyle='--',
                              label=f'mean={np.mean(cos_sim_values):.3f}')
                ax_cos.set_xlabel('Cosine Similarity')
                ax_cos.set_ylabel('Count')
                ax_cos.set_title(f'Channel {k}: Pairwise Cosine Similarity')
                ax_cos.set_xlim(-1, 1)
                ax_cos.legend()

                e.track(f'emb_cos_sim_mean_ch{k}', np.mean(cos_sim_values))
                e.track(f'emb_cos_sim_std_ch{k}', np.std(cos_sim_values))

                # --- Column 3: Singular value spectrum ---
                _, s, _ = np.linalg.svd(emb_k, full_matrices=False)
                s_normalized = s / (np.sum(s) + 1e-8)
                cumvar = np.cumsum(s**2) / (np.sum(s**2) + 1e-8)

                # Effective rank
                entropy = -np.sum(s_normalized * np.log(s_normalized + 1e-8))
                effective_rank = np.exp(entropy)

                ax_sv = axes[k, 2]
                ax_sv.bar(range(len(s_normalized[:20])), s_normalized[:20],
                         color=e.CHANNEL_INFOS[k]['color'], alpha=0.7)
                ax_sv.set_xlabel('Singular Value Index')
                ax_sv.set_ylabel('Normalized Singular Value')
                ax_sv.set_title(f'Channel {k}: Singular Values (eff. rank={effective_rank:.1f})')

                # Add cumulative variance on secondary axis
                ax_sv2 = ax_sv.twinx()
                ax_sv2.plot(range(len(cumvar[:20])), cumvar[:20], 'r--', linewidth=2, label='Cum. Var.')
                ax_sv2.set_ylabel('Cumulative Variance', color='red')
                ax_sv2.tick_params(axis='y', labelcolor='red')
                ax_sv2.set_ylim(0, 1.1)

                e.track(f'emb_effective_rank_ch{k}', effective_rank)
                e.track(f'emb_var_top1_ch{k}', cumvar[0])
                e.track(f'emb_var_top5_ch{k}', cumvar[min(4, len(cumvar)-1)])

            fig.tight_layout()
            e.track('embedding_health', fig)
            plt.close(fig)

        def track_uniformity(self, embeddings: np.ndarray, num_channels: int, epoch: int):
            """
            Track embedding uniformity metrics and create visualization figure.

            Figure layout (num_channels rows x 2 columns):
            - Column 1: Pairwise distance distribution on unit sphere
            - Column 2: Angular distribution (angles between embeddings)
            """
            fig, axes = plt.subplots(
                nrows=num_channels,
                ncols=2,
                figsize=(12, 4 * num_channels),
                squeeze=False
            )
            fig.suptitle(f'Embedding Uniformity - Epoch {epoch}', fontsize=14)

            for k in range(num_channels):
                emb_k = embeddings[:, :, k]

                # Normalize to unit sphere
                emb_normed = emb_k / (np.linalg.norm(emb_k, axis=1, keepdims=True) + 1e-8)

                # Sample for efficiency
                n_samples = min(len(emb_k), e.EMBEDDING_SAMPLE_SIZE)
                idx = np.random.choice(len(emb_k), n_samples, replace=False)
                emb_sample = emb_normed[idx]

                # --- Column 1: Pairwise L2 distance distribution on unit sphere ---
                # For normalized vectors: ||a - b||^2 = 2 - 2*cos(a,b)
                cos_sim_matrix = emb_sample @ emb_sample.T
                sq_dists = 2 - 2 * cos_sim_matrix
                triu_idx = np.triu_indices(n_samples, k=1)
                distances = np.sqrt(np.maximum(sq_dists[triu_idx], 0))

                ax_dist = axes[k, 0]
                ax_dist.hist(distances, bins=50, color=e.CHANNEL_INFOS[k]['color'],
                            alpha=0.7, edgecolor='black', density=True)
                ax_dist.axvline(np.mean(distances), color='red', linestyle='--',
                               label=f'mean={np.mean(distances):.3f}')
                ax_dist.set_xlabel('Pairwise L2 Distance')
                ax_dist.set_ylabel('Density')
                ax_dist.set_title(f'Channel {k}: Distance Distribution on Unit Sphere')
                ax_dist.set_xlim(0, 2)
                ax_dist.legend()

                # Compute uniformity metric: log(mean(exp(-t * ||z_i - z_j||^2)))
                t = 2
                uniformity = np.log(np.mean(np.exp(-t * sq_dists[triu_idx])) + 1e-8)
                e.track(f'emb_uniformity_ch{k}', uniformity)

                # --- Column 2: Angular distribution ---
                cos_sim_values = cos_sim_matrix[triu_idx]
                angles = np.arccos(np.clip(cos_sim_values, -1, 1)) * 180 / np.pi  # in degrees

                ax_ang = axes[k, 1]
                ax_ang.hist(angles, bins=50, color=e.CHANNEL_INFOS[k]['color'],
                           alpha=0.7, edgecolor='black', density=True)
                ax_ang.axvline(np.mean(angles), color='red', linestyle='--',
                              label=f'mean={np.mean(angles):.1f}°')
                ax_ang.axvline(90, color='green', linestyle=':', label='90° (orthogonal)')
                ax_ang.set_xlabel('Angle (degrees)')
                ax_ang.set_ylabel('Density')
                ax_ang.set_title(f'Channel {k}: Angular Distribution')
                ax_ang.set_xlim(0, 180)
                ax_ang.legend()

                e.track(f'emb_mean_angle_ch{k}', np.mean(angles))

            fig.tight_layout()
            e.track('embedding_uniformity', fig)
            plt.close(fig)

        def track_target_alignment(self, embeddings: np.ndarray, targets: np.ndarray,
                                   num_channels: int, epoch: int):
            """
            Track embedding-target alignment metrics and create visualization figure.

            Figure layout (num_channels rows x 3 columns):
            - Column 1: Embedding norm vs target scatter plot
            - Column 2: k-NN target smoothness visualization
            - Column 3: Per-dimension correlation with target (top correlations)
            """
            fig, axes = plt.subplots(
                nrows=num_channels,
                ncols=3,
                figsize=(15, 4 * num_channels),
                squeeze=False
            )
            fig.suptitle(f'Target Alignment - Epoch {epoch}', fontsize=14)

            for k in range(num_channels):
                emb_k = embeddings[:, :, k]

                # --- Column 1: Norm vs Target ---
                norms = np.linalg.norm(emb_k, axis=1)
                corr, p_value = spearmanr(norms, targets)

                ax_norm = axes[k, 0]
                scatter = ax_norm.scatter(targets, norms, c=targets, cmap='coolwarm',
                                         alpha=0.5, s=10)
                ax_norm.set_xlabel('Target Value')
                ax_norm.set_ylabel('Embedding Norm')
                ax_norm.set_title(f'Channel {k}: Norm vs Target (ρ={corr:.3f})')
                plt.colorbar(scatter, ax=ax_norm)

                e.track(f'emb_target_norm_corr_ch{k}', corr if not np.isnan(corr) else 0)

                # --- Column 2: k-NN Target Smoothness ---
                k_neighbors = min(10, len(emb_k) - 1)
                nn = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='ball_tree')
                nn.fit(emb_k)
                distances, indices = nn.kneighbors(emb_k)

                # Compute mean absolute target difference to neighbors
                neighbor_indices = indices[:, 1:]  # exclude self
                neighbor_targets = targets[neighbor_indices]
                target_diffs = np.abs(neighbor_targets - targets[:, np.newaxis])
                mean_diff_per_sample = np.mean(target_diffs, axis=1)
                overall_smoothness = np.mean(mean_diff_per_sample)

                ax_knn = axes[k, 1]
                ax_knn.scatter(targets, mean_diff_per_sample, c=targets, cmap='coolwarm',
                              alpha=0.5, s=10)
                ax_knn.axhline(overall_smoothness, color='red', linestyle='--',
                              label=f'mean={overall_smoothness:.3f}')
                ax_knn.set_xlabel('Target Value')
                ax_knn.set_ylabel(f'Mean |Δtarget| to {k_neighbors}-NN')
                ax_knn.set_title(f'Channel {k}: k-NN Target Smoothness')
                ax_knn.legend()

                e.track(f'emb_knn_smoothness_ch{k}', overall_smoothness)

                # --- Column 3: Per-dimension correlations ---
                dim_correlations = []
                for d in range(emb_k.shape[1]):
                    corr_d, _ = spearmanr(emb_k[:, d], targets)
                    dim_correlations.append(corr_d if not np.isnan(corr_d) else 0)

                dim_correlations = np.array(dim_correlations)
                sorted_idx = np.argsort(np.abs(dim_correlations))[::-1]
                top_n = min(20, len(dim_correlations))

                ax_dim = axes[k, 2]
                colors = ['green' if c > 0 else 'red' for c in dim_correlations[sorted_idx[:top_n]]]
                ax_dim.barh(range(top_n), dim_correlations[sorted_idx[:top_n]], color=colors, alpha=0.7)
                ax_dim.set_yticks(range(top_n))
                ax_dim.set_yticklabels([f'dim {i}' for i in sorted_idx[:top_n]])
                ax_dim.set_xlabel('Spearman Correlation')
                ax_dim.set_title(f'Channel {k}: Top {top_n} Dimension-Target Correlations')
                ax_dim.axvline(0, color='black', linewidth=0.5)
                ax_dim.invert_yaxis()

                e.track(f'emb_max_dim_corr_ch{k}', np.max(np.abs(dim_correlations)))

            fig.tight_layout()
            e.track('embedding_target_alignment', fig)
            plt.close(fig)

        def track_projections(self, embeddings: np.ndarray, targets: np.ndarray,
                             num_channels: int, epoch: int):
            """
            Create PCA and UMAP 2D projection plots.

            Figure layout (num_channels rows x 2 columns):
            - Column 1: PCA projection colored by target
            - Column 2: UMAP projection colored by target
            """
            # Check if UMAP is available
            try:
                from umap import UMAP
                umap_available = True
            except ImportError:
                umap_available = False
                e.log('UMAP not available, skipping UMAP projections')

            ncols = 2 if umap_available else 1
            fig, axes = plt.subplots(
                nrows=num_channels,
                ncols=ncols,
                figsize=(7 * ncols, 5 * num_channels),
                squeeze=False
            )
            fig.suptitle(f'Embedding Projections - Epoch {epoch}', fontsize=14)

            for k in range(num_channels):
                emb_k = embeddings[:, :, k]

                # --- Column 1: PCA ---
                if emb_k.shape[1] > 2:
                    pca = PCA(n_components=2)
                    emb_pca = pca.fit_transform(emb_k)
                    var_explained = sum(pca.explained_variance_ratio_)
                else:
                    emb_pca = emb_k[:, :2] if emb_k.shape[1] >= 2 else np.column_stack([emb_k, np.zeros(len(emb_k))])
                    var_explained = 1.0

                ax_pca = axes[k, 0]
                scatter = ax_pca.scatter(emb_pca[:, 0], emb_pca[:, 1], c=targets,
                                        cmap='coolwarm', alpha=0.6, s=15)
                ax_pca.set_xlabel('PC1')
                ax_pca.set_ylabel('PC2')
                ax_pca.set_title(f'Channel {k}: PCA ({var_explained:.1%} var. explained)')
                plt.colorbar(scatter, ax=ax_pca, label='Target')

                e.track(f'emb_pca_var_explained_ch{k}', var_explained)

                # --- Column 2: UMAP ---
                if umap_available:
                    ax_umap = axes[k, 1]
                    try:
                        # Sample if too many points for UMAP
                        n_samples = min(len(emb_k), e.EMBEDDING_SAMPLE_SIZE)
                        if n_samples < len(emb_k):
                            idx = np.random.choice(len(emb_k), n_samples, replace=False)
                            emb_sample = emb_k[idx]
                            targets_sample = targets[idx]
                        else:
                            emb_sample = emb_k
                            targets_sample = targets

                        umap_reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                                           metric='cosine', random_state=42)
                        emb_umap = umap_reducer.fit_transform(emb_sample)

                        scatter = ax_umap.scatter(emb_umap[:, 0], emb_umap[:, 1],
                                                 c=targets_sample, cmap='coolwarm',
                                                 alpha=0.6, s=15)
                        ax_umap.set_xlabel('UMAP1')
                        ax_umap.set_ylabel('UMAP2')
                        ax_umap.set_title(f'Channel {k}: UMAP')
                        plt.colorbar(scatter, ax=ax_umap, label='Target')
                    except Exception as ex:
                        ax_umap.text(0.5, 0.5, f'UMAP failed:\n{str(ex)[:50]}',
                                    ha='center', va='center', transform=ax_umap.transAxes)
                        ax_umap.set_title(f'Channel {k}: UMAP (failed)')

            fig.tight_layout()
            e.track('embedding_projections', fig)
            plt.close(fig)

        def track_3d_embeddings(self, embeddings: np.ndarray, embedding_dim: int,
                               num_channels: int, epoch: int):
            """Original 3D/2D embedding visualization for low-dimensional embeddings."""
            kwargs = {'projection': '3d'} if embedding_dim == 3 else {}

            fig, rows = plt.subplots(
                ncols=num_channels,
                nrows=1,
                figsize=(8 * num_channels, 8),
                squeeze=False,
                subplot_kw=kwargs,
            )
            fig.suptitle(f'Embeddings Epoch {epoch}')

            for k in range(num_channels):
                ax = rows[0][k]
                ax.set_title(f'Channel {k}')
                if embedding_dim == 3:
                    plot_embeddings_3d(
                        embeddings=embeddings[:, :, k],
                        ax=ax,
                        color=e.CHANNEL_INFOS[k]['color'],
                        x_range=[-1.1, 1.1],
                        y_range=[-1.1, 1.1],
                        z_range=[-1.1, 1.1],
                    )
                else:
                    plot_embeddings_2d(
                        embeddings=embeddings[:, :, k],
                        ax=ax,
                        color=e.CHANNEL_INFOS[k]['color'],
                        x_range=[-1.1, 1.1],
                        y_range=[-1.1, 1.1],
                    )

            e.track('embeddings', fig)
            plt.close(fig)

    e.log('Instantiating Megan model - with explanation training...')
    e.log(f'explanation mode: {e.DATASET_TYPE}')
    e.log(f' * importance offset: {e.IMPORTANCE_OFFSET}')
    e.log(f' * hidden units: {e.HIDDEN_UNITS}')
    model = Megan(
        node_dim=e['node_dim'],
        edge_dim=e['edge_dim'],
        units=e.UNITS,
        hidden_units=e.HIDDEN_UNITS,
        importance_units=e.IMPORTANCE_UNITS,
        layer_version='v2',
        projection_units=e.PROJECTION_UNITS,
        importance_mode=e.DATASET_TYPE,
        final_units=e.FINAL_UNITS,
        output_norm=e.OUTPUT_NORM,
        label_smoothing=e.LABEL_SMOOTHING,
        num_channels=e.NUM_CHANNELS,
        importance_factor=e.IMPORTANCE_FACTOR,
        importance_offset=e.IMPORTANCE_OFFSET,
        importance_target='node',
        sparsity_factor=e.SPARSITY_FACTOR,
        fidelity_factor=e.FIDELITY_FACTOR,
        regression_reference=e.REGRESSION_REFERENCE,
        regression_margin=e.REGRESSION_MARGIN,
        prediction_mode=e.DATASET_TYPE,
        prediction_factor=e.PREDICTION_FACTOR,
        normalize_embedding=e.NORMALIZE_EMBEDDING,
        attention_aggregation=e.ATTENTION_AGGREGATION,
        contrastive_factor=e.CONTRASTIVE_FACTOR,
        contrastive_temp=e.CONTRASTIVE_TEMP,
        contrastive_noise=e.CONTRASTIVE_NOISE,
        contrastive_beta=e.CONTRASTIVE_BETA,
        contrastive_tau=e.CONTRASTIVE_TAU,
        learning_rate=e.LEARNING_RATE,
        lr_scheduler=e.LR_SCHEDULER,
    )

    e.log(f'starting model training with {e.EPOCHS} epochs...')
    logger = CSVLogger(e.path, name='logs')

    callbacks = [
        TrainingCallback(),
        RecordEmbeddingsCallback(),
    ]

    if e.USE_SWA:
        callbacks.append(SwaCallback(history_length=e.SWA_EPOCHS, logger=logger))

    if e.TRAIN_MVE:
        callbacks.append(MveCallback(
            warmup_epochs=e.MVE_WARMUP_EPOCHS,
        ))

    if e.USE_CONTRASTIVE_SCHEDULER and e.CONTRASTIVE_FACTOR > 0:
        warmup_epochs = int(e.CONTRASTIVE_WARMUP_EPOCHS * e.EPOCHS)
        callbacks.append(ContrastiveSchedulerCallback(
            target_factor=e.CONTRASTIVE_FACTOR,
            warmup_epochs=warmup_epochs,
        ))

    trainer = pl.Trainer(
        max_epochs=e.EPOCHS,
        logger=logger,
        callbacks=callbacks,
        accumulate_grad_batches=e.BATCH_ACCUMULATE,
        enable_progress_bar=True,
    )

    e.log('starting model training...')
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
    )

    del train_loader
    del test_loader
    torch.cuda.empty_cache()

    model.eval()
    model.to('cpu')

    return model, trainer


@experiment.hook('evaluate_model', default=False, replace=False)
def evaluate_model(e: Experiment,
                   model: AbstractGraphModel,
                   trainer: pl.Trainer,
                   index_data_map: dict,
                   train_indices: t.List[int],
                   test_indices: t.List[int],
                   ) -> None:
    """
    Evaluate MEGAN model including explanation visualization.
    """

    model.eval()

    with tempfile.TemporaryDirectory() as path:
        model_path = os.path.join(path, 'model.ckpt')
        model.save(model_path)
        model = Megan.load(model_path)

    last_layer = model.dense_layers[-1]
    e.log('final layer info:')
    e.log(f' * regression reference: {model.regression_reference}')

    e.log('evaluating Megan explanations...')
    graphs = [data['metadata']['graph'] for data in index_data_map.values()]
    graphs_test = [index_data_map[i]['metadata']['graph'] for i in test_indices]

    out_true = np.array([graph['graph_labels'] for graph in graphs_test])
    out_pred = model.predict_graphs(graphs_test)

    # Generate visualizations for all example elements
    example_indices: t.List[int] = e['indices/example']
    for index in example_indices:
        data = index_data_map[index]
        if data['image_path'] is None:
            image_path, node_positions = e.apply_hook(
                'get_visualization',
                index=index,
                metadata=data['metadata']
            )
            data['image_path'] = image_path
            if node_positions is not None:
                data['metadata']['graph']['node_positions'] = node_positions

    e.log('visualizing the example graphs...')
    graphs_example = [index_data_map[i]['metadata']['graph'] for i in example_indices]
    example_infos: t.List[dict] = model.forward_graphs(graphs_example)

    # Only create explanation PDF if we have visualizations
    image_paths = [index_data_map[i]['image_path'] for i in example_indices]
    if all(p is not None for p in image_paths):
        create_importances_pdf(
            graph_list=graphs_example,
            image_path_list=image_paths,
            node_positions_list=[graph.get('node_positions') for graph in graphs_example],
            importances_map={
                'megan': (
                    [info['node_importance'] for info in example_infos],
                    [info['edge_importance'] for info in example_infos],
                )
            },
            output_path=os.path.join(e.path, 'example_explanations.pdf'),
            plot_node_importances_cb=plot_node_importances_background,
            plot_edge_importances_cb=plot_edge_importances_background,
        )

    # ~ explanation fidelity analysis
    e.log('calculating explanation fidelity...')

    leave_one_out = model.leave_one_out_deviations(graphs_test)
    fig = plot_leave_one_out_analysis(
        leave_one_out,
        num_channels=e.NUM_CHANNELS,
        num_targets=e.FINAL_UNITS[-1],
    )
    fig.savefig(os.path.join(e.path, 'leave_one_out.pdf'))

    # ~ visualizing the graph embedding space
    e.log(f'visualizating embedding space with {model.embedding_dim} dimensions...')
    if model.embedding_dim <= 3:

        e.log('embedding graphs...')
        infos = model.forward_graphs(graphs)
        embeddings = np.array([info['graph_embedding'] for info in infos])

        embeddings_combined = np.concatenate([embeddings[:, :, k] for k in range(e.NUM_CHANNELS)], axis=0)

        colormap = mpl.cm.RdPu
        num_neighbors = 100

        if model.embedding_dim == 3:
            plot_func = plot_embeddings_3d
            projection = '3d'
        if model.embedding_dim == 2:
            plot_func = plot_embeddings_2d
            projection = None

        e.log(f'plotting the embeddings with dimension {model.embedding_dim}...')
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
            neighbors = NearestNeighbors(
                n_neighbors=num_neighbors+1,
                algorithm='ball_tree',
            ).fit(embeddings_k)
            distances, indices = neighbors.kneighbors(embeddings_k)
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
            clusterer = hdbscan.HDBSCAN(min_samples=20)
            cluster_labels = clusterer.fit_predict(embeddings_k)
            cluster_indices = list(set(cluster_labels))
            num_clusters = len(cluster_indices) - 1
            e.log(f' * channel {k} - num elements {len(embeddings_k)} - num clusters: {num_clusters} ')
            if num_clusters < 2:
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

        fig.savefig(os.path.join(e.path, 'embeddings.pdf'))
        fig.savefig(os.path.join(e.path, 'embeddings.png'))
        plt.close(fig)

    e.apply_hook(
        'after_experiment',
        index_data_map=index_data_map,
        model=model,
        trainer=trainer,
    )


@experiment.hook('after_experiment', default=False, replace=False)
def after_experiment(e: Experiment,
                     index_data_map: dict,
                     model: Megan,
                     trainer: pl.Trainer,
                     ) -> None:
    """
    Clean up experiment resources.
    """
    e.log('cleaning up experiment...')

    e.log('deleting the optimizer states...')
    for optimizer in trainer.optimizers:

        optimizer.zero_grad()
        for param in optimizer.state.values():

            if isinstance(param, torch.Tensor):
                param.data = param.data.to('cpu')
                del param

            if isinstance(param, dict):
                for p in param.values():
                    if isinstance(p, torch.Tensor):
                        p.data = p.data.to('cpu')
                        del p

        del optimizer
        torch.cuda.empty_cache()

    e.log('deleting the trainer...')
    del trainer
    torch.cuda.empty_cache()

    e.log('moving the model to the cpu...')
    model.to('cpu')
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        e.log('memory summary:')
        e.log('\n' + torch.cuda.memory_summary(device=None, abbreviated=False))


@experiment.analysis
def analysis(e: Experiment):
    """
    Run MEGAN-specific analysis including clustering.
    """
    e.log('running MEGAN specific analysis...')

    index_data_map = e['_index_data_map']
    graphs = [data['metadata']['graph'] for data in index_data_map.values()]
    indices = np.array(list(index_data_map.keys()))

    if not e.DO_CLUSTERING:
        e.log('skipping clustering analysis...')
        return

    e.log('Clustering the latent space...')
    min_samples = 5

    test_indices = e[f'indices/test']
    graphs_test = [index_data_map[index]['metadata']['graph'] for index in test_indices]

    model = e[f'_model']
    infos = model.forward_graphs(graphs)

    for k in range(e.NUM_CHANNELS):

        graph_embeddings_k = np.array([info['graph_embedding'][:, k] for info in infos])

        clusterer = hdbscan.HDBSCAN(min_samples=min_samples)
        cluster_labels = clusterer.fit_predict(graph_embeddings_k)
        cluster_indices = list(set(cluster_labels))
        num_clusters = len(cluster_indices) - 1
        e.log(f' * channel {k} - num clusters: {num_clusters}')

        # Generate visualizations for cluster examples
        with PdfPages(os.path.join(e.path, f'cluster__ch{k:02d}.pdf')) as pdf:

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

                    # Generate visualization on-demand
                    data = index_data_map[index]
                    if data['image_path'] is None:
                        image_path, node_positions = e.apply_hook(
                            'get_visualization',
                            index=index,
                            metadata=data['metadata']
                        )
                        data['image_path'] = image_path
                        if node_positions is not None:
                            data['metadata']['graph']['node_positions'] = node_positions

                    if data['image_path'] and os.path.exists(data['image_path']):
                        draw_image(ax=ax, image_path=data['image_path'])

                        if 'node_positions' in graph:
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
