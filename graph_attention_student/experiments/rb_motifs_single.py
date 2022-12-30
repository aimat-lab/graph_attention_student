"""
Trains multi-channel MEGAN models and a GNNExplainer baseline on the synthetic graph regression dataset
RbMotifs. This datasets consits of randomly generated colored graphs, where the target value of each graph
is determined by the special subgraph motifs contained in it. The MEGAN models are configured to the special
single-explanation channel case, where only a single attention explanation mask is generated for each
sample. The generated explanations are compared to the known ground truth explanations. In the GNNExplainer
case a simple GCN network is trained on the same dataset and for each sample of the test set a new
GNNExplainer optimization loop is performed to generate the explanations. The experiment is repeated
multiple independent times.
"""
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import json
import time
import random
from typing import List, Callable, Dict, Optional
from pprint import pprint
from collections import Counter

import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_auc_score, r2_score
from scipy.stats import wilcoxon
from imageio.v2 import imread
from kgcnn.literature.GNNExplain import GNNExplainer
from kgcnn.utils.data import ragged_tensor_from_nested_numpy

from pycomex.experiment import Experiment
from pycomex.util import Skippable

from graph_attention_student.util import DATASETS_FOLDER
from graph_attention_student.util import importance_absolute_similarity
from graph_attention_student.util import importance_canberra_similarity
from graph_attention_student.util import array_normalize
from graph_attention_student.util import binary_threshold
from graph_attention_student.util import render_latex
from graph_attention_student.util import latex_table, latex_table_element_median
from graph_attention_student.data import load_eye_tracking_dataset
from graph_attention_student.data import process_graph_dataset
from graph_attention_student.models import MultiAttentionStudent
from graph_attention_student.models import GnnxGCN
from graph_attention_student.training import NoLoss, ExplanationLoss
from graph_attention_student.training import LogProgressCallback
from graph_attention_student.visualization import plot_node_importances, plot_edge_importances

SHORT_DESCRIPTION = (
    'Trains single-channel MEGAN model, as well as GNNExplainer, on the synthetic graph regression dataset '
    'RbMotifs and compares the generated explanations to the known ground truth.'
)

# == META PARAMETERS ==
REPETITIONS = 50

# == DATASET PARAMETERS ==
DATASET_PATH = os.path.join(DATASETS_FOLDER, 'rb_dual_motifs')
TEST_RATIO = 0.1
NUM_EXAMPLES = 100
EXAMPLE_INDICES: Optional[list] = None

# == MODEL PARAMETERS ==
UNITS = [9, 9, 9]
IMPORTANCE_UNITS = []
SPARSITY_FACTOR = 0
IMPORTANCE_FACTOR = 0.0
IMPORTANCE_MULTIPLIER = 1
NUM_CHANNELS = 1
FINAL_UNITS = [3]
DROPOUT_RATE = 0.0
FINAL_DROPOUT_RATE = 0.0
USE_BIAS = True
REGRESSION_LIMITS = [-4, 4]
REGRESSION_BINS = [[-4, 0], [0, 4]]
MEGAN_SWEEP = {
    'megan_0': {
        'IMPORTANCE_FACTOR': 0.0,
        'SPARSITY_FACTOR': 0e-2,
        'IMPORTANCE_SUPERVISION': False,
    },
    'megan_1': {
        'IMPORTANCE_FACTOR': 0.0,
        'SPARSITY_FACTOR': 0e-2,
        'IMPORTANCE_SUPERVISION': True
    }
}

# == TRAINING PARAMETERS ==
LEARNING_RATE = 0.004
BATCH_SIZE = 512
EPOCHS = 250
LOG_STEP = 10
DEVICE = 'gpu:0'
METRIC_KEY = 'mean_squared_error'
IMPORTANCE_SUPERVISION = False

# == GNN EXPLAINER PARAMETERS ==
GNNX_MODEL_UNITS = [32, 32, 32]
GNNX_EPOCHS = 100
GNNX_LEARNING_RATE = 0.2
GNNX_NODE_WEIGHT = 0.01
GNNX_EDGE_WEIGHT = 0.01

# == EVALUATION PARAMETERS ==
FIDELITY_SPARSITY = 0.3
BINARY_THRESHOLD = 0.5
LOG_STEP_EVAL = 10
NUM_RANDOM_MASKS = 3

# == EXPERIMENT PARAMETERS ==
NAMESPACE = 'rb_motifs_single'
BASE_PATH = os.getcwd()
DEBUG = True

# == HELPER METHODS ==


with Skippable(), (e := Experiment(namespace=NAMESPACE, base_path=BASE_PATH, glob=globals())):
    e.info('Loading the dataset...')
    start_time = time.time()
    eye_tracking_dataset = load_eye_tracking_dataset(DATASET_PATH)
    dataset_size = len(eye_tracking_dataset)
    end_time = time.time()
    e.info(f'Loaded dataset with {dataset_size} elements in {end_time - start_time:.2f} seconds')
    e.status()

    e['metric_keys'] = []
    for rep in range(REPETITIONS):
        e.info(f'STARTING REPETITION ({rep+1}/{REPETITIONS})')

        dataset_indices = list(range(dataset_size))
        test_indices = random.sample(dataset_indices, k=int(TEST_RATIO * dataset_size))
        train_indices = [index for index in dataset_indices if index not in test_indices]
        e[f'test_indices/{rep}'] = test_indices
        e[f'train_indices/{rep}'] = train_indices
        e.info(f'Identified {len(train_indices)} elements for training and {len(test_indices)} for testing')

        example_indices = random.sample(test_indices, k=NUM_EXAMPLES)
        e[f'example_indices/{rep}'] = example_indices
        e.info(f'Identified {len(example_indices)} elements as examples')

        # "eye_tracking_dataset" at this point is only a list of dictionaries which contain the metadata info
        # for each respective elements. This now has to be turned into the appropriate tensors to be used by
        # the model.
        dataset = []
        for data in eye_tracking_dataset:
            g = data['metadata']['graph']

            # The target ground truth will be the float value associated with every graph
            value = data['metadata']['value']
            g['graph_labels'] = np.array(value)

            # Since this is the single-channel case we also need to modify the ground truth node and edge
            # importance annotations here: Currently they are split into two channels, but we need to combine
            # them into a single channel
            node_importances = np.array(g['multi_node_importances'])
            node_importances = np.sum(node_importances, axis=-1, keepdims=True)
            g['node_importances'] = node_importances

            edge_importances = np.array(g['multi_edge_importances'])
            edge_importances = np.sum(edge_importances, axis=-1, keepdims=True)
            g['edge_importances'] = edge_importances

            dataset.append(g)

        x_train, y_train, x_test, y_test = process_graph_dataset(dataset, test_indices)
        e.info('Pre-processed the dataset. Now ready for training')

        # ---------------------------------------------------------------------------------------------------
        # -- MULTI ATTENTION STUDENT ------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------
        for model_name, parameters in MEGAN_SWEEP.items():
            e.info(f'Model sweep: {model_name}')
            for parameter_key, parameter_value in parameters.items():
                globals()[parameter_key] = parameter_value
                e.info(f' * {parameter_key}: {eval(parameter_key)}')

            e.info('Training Multi Explanation Graph Attention Network (MEGAN) ...')
            ks.backend.clear_session()
            model: ks.models.Model = MultiAttentionStudent(
                units=UNITS,
                dropout_rate=DROPOUT_RATE,
                importance_factor=IMPORTANCE_FACTOR,
                importance_multiplier=IMPORTANCE_MULTIPLIER,
                importance_units=IMPORTANCE_UNITS,
                importance_channels=NUM_CHANNELS,
                use_bias=USE_BIAS,
                final_units=FINAL_UNITS,
                final_dropout_rate=FINAL_DROPOUT_RATE,
                regression_limits=REGRESSION_LIMITS,
                regression_bins=REGRESSION_BINS,
                sparsity_factor=SPARSITY_FACTOR
            )
            model.compile(
                loss=[
                    ks.losses.MeanSquaredError(),
                    NoLoss() if not IMPORTANCE_SUPERVISION else ExplanationLoss(),
                    NoLoss() if not IMPORTANCE_SUPERVISION else ExplanationLoss(),
                ],
                loss_weights=[
                    1,
                    0 if not IMPORTANCE_SUPERVISION else 1,
                    0 if not IMPORTANCE_SUPERVISION else 1,
                ],
                metrics=[
                    ks.metrics.MeanSquaredError(),
                    ks.metrics.MeanAbsoluteError(),
                ],
                optimizer=ks.optimizers.Adam(learning_rate=LEARNING_RATE),
                run_eagerly=False
            )
            with tf.device(DEVICE):
                history = model.fit(
                    x_train,
                    y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(x_test, y_test),
                    validation_freq=1,
                    callbacks=LogProgressCallback(
                        logger=e.logger,
                        epoch_step=LOG_STEP,
                        identifier=f'val_output_1_{METRIC_KEY}'
                    ),
                    verbose=0
                )
            e.info(f'model parameters: {model.count_params()}')
            e[f'histories/{model_name}/{rep}'] = history.history
            e[f'epochs/{model_name}/{rep}'] = list(range(EPOCHS))

            ni_true = []
            ei_true = []

            ni_pred = []
            ei_pred = []

            # ~ Making all the test set predictions and saving them
            e.info('evaluating test set...')
            for c, index in enumerate(test_indices):
                data = eye_tracking_dataset[index]
                g = dataset[index]

                # First of all we need to query the model to make the actual prediction for the current element
                out_meg, ni_meg, ei_meg = model.predict_single((
                    g['node_attributes'],
                    g['edge_attributes'],
                    g['edge_indices'],
                ))
                e[f'out/true/{rep}/{index}'] = g['graph_labels']
                e[f'out/{model_name}/{rep}/{index}'] = out_meg
                ni_true += np.array(g['node_importances']).flatten().tolist()
                ei_true += np.array(g['edge_importances']).flatten().tolist()

                ni_pred += np.array(ni_meg).flatten().tolist()
                ei_pred += np.array(ei_meg).flatten().tolist()
                # The first transformation we apply to the importances vectors is to apply a normalization on
                # them such that they use the entire value range [0, 1]
                ni_meg = array_normalize(ni_meg)
                ei_meg = array_normalize(ei_meg)
                e[f'ni/{model_name}/{rep}/{index}'] = ni_meg
                e[f'ei/{model_name}/{rep}/{index}'] = ei_meg
                # Then with these relatively raw importances we calculate the cont. similarity value
                # (basically something like a distance metric with a normalized & inverted scale)
                node_sim_meg = importance_absolute_similarity(g['node_importances'], ni_meg)
                edge_sim_meg = importance_absolute_similarity(g['edge_importances'], ei_meg)
                e[f'node_sim/{model_name}/{rep}/{index}'] = node_sim_meg
                e[f'edge_sim/{model_name}/{rep}/{index}'] = edge_sim_meg
                node_sim_can_meg = importance_canberra_similarity(g['node_importances'], ni_meg)
                edge_sim_can_meg = importance_canberra_similarity(g['edge_importances'], ei_meg)
                e[f'node_sim_can/{model_name}/{rep}/{index}'] = node_sim_can_meg
                e[f'edge_sim_can/{model_name}/{rep}/{index}'] = edge_sim_can_meg
                # Then we calculate the natural sparsity percentage of the importances. We do this by first
                # applying a simple threshold to turn the soft importance masks into hard importance masks
                # and then the sparsity is simply the mean of the hard mask.
                sparsity_ni = np.mean(binary_threshold(ni_meg, BINARY_THRESHOLD))
                sparsity_ei = np.mean(binary_threshold(ei_meg, BINARY_THRESHOLD))
                e[f'node_sparsity/{model_name}/{rep}/{index}'] = sparsity_ni
                e[f'edge_sparsity/{model_name}/{rep}/{index}'] = sparsity_ei
                e[f'sparsity_ni/{model_name}/{rep}/{index}'] = sparsity_ni
                e[f'sparsity_ei/{model_name}/{rep}/{index}'] = sparsity_ei
                # Binary threshold
                node_mask = binary_threshold(ni_meg, BINARY_THRESHOLD)
                edge_mask = binary_threshold(ei_meg, BINARY_THRESHOLD)

                out_m_meg, _, _ = model.predict_single((
                    np.array(g['node_attributes']) * (1 - node_mask),
                    np.array(g['edge_attributes']) ,#* edge_mask,
                    g['edge_indices']
                ))
                fidelity_meg = abs(out_m_meg - out_meg)
                e[f'fidelity/{model_name}/{rep}/{index}'] = fidelity_meg

                # In the last step we generate a random explanation and then calculate the auroc and the
                # fidelity for that one which will act as a baseline to understand the other values better

                deltas_rand = []
                for _ in range(NUM_RANDOM_MASKS):
                    node_mask_rand_indices = random.sample(g['node_indices'], k=int(np.sum(node_mask)))
                    node_mask_rand = np.array([0.0 if i in node_mask_rand_indices else 1.0
                                               for i in g['node_indices']])
                    node_mask_rand = np.expand_dims(node_mask_rand, axis=-1)
                    e[f'ni_rand/{model_name}/{rep}/{index}'] = node_mask_rand

                    # edge_mask_rand_indices = random.sample(g['edge_indices'], k=edge_mask_count)
                    # edge_mask_rand = np.array([0.0 if m in edge_mask_rand_indices else 1.0
                    #                            for m in g['edge_indices']])
                    # edge_mask_rand = np.expand_dims(edge_mask_rand, axis=-1)
                    # e[f'ei_rand/{model_name}/{rep}/{index}'] = edge_mask_rand

                    out_rand_meg, _, _ = model.predict_single((
                        np.array(g['node_attributes']) * node_mask_rand,
                        np.array(g['edge_attributes']) ,#* edge_mask_rand,
                        g['edge_indices']
                    ))
                    deltas_rand.append(abs(out_rand_meg - out_meg))
                fidelity_rand_meg = np.mean(deltas_rand)
                e[f'fidelity_rand/{model_name}/{rep}/{index}'] = fidelity_rand_meg

                if c % LOG_STEP_EVAL == 0:
                    e.info(f' * MAS ({c}) '
                           f' - node sim can: {node_sim_can_meg:.2f}'
                           f' - edge sim can: {edge_sim_can_meg:.2f}'
                           f' - fidelity: {fidelity_meg:.2f}'
                           f' - fidelity_rand: {fidelity_rand_meg:.2f}'
                           f' - y_true: {e[f"out/true/{rep}/{index}"]:.2f}')

            fidelity_mean = np.mean(list(e[f'fidelity/{model_name}/{rep}'].values()))
            e[f'fidelity_mean/{model_name}/{rep}'] = fidelity_mean
            fidelity_rand_mean = np.mean(list(e[f'fidelity_rand/{model_name}/{rep}'].values()))
            e[f'fidelity_rand_mean/{model_name}/{rep}'] = fidelity_rand_mean

            wilcoxon_result = wilcoxon(
                list(e[f'fidelity/{model_name}/{rep}'].values()),
                list(e[f'fidelity_rand/{model_name}/{rep}'].values())
            )
            e[f'fidelity_wilcoxon/{model_name}/{rep}'] = wilcoxon_result.pvalue

            auroc_ni = roc_auc_score(ni_true, ni_pred)
            e[f'auroc_ni/{model_name}/{rep}'] = auroc_ni

            auroc_ei = roc_auc_score(ei_true, ei_pred)
            e[f'auroc_ei/{model_name}/{rep}'] = auroc_ei

            mse = history.history['val_output_1_mean_squared_error'][-1]
            rmse = np.sqrt(mse)
            e[f'mse/{model_name}/{rep}'] = mse
            e[f'rmse/{model_name}/{rep}'] = rmse

            r2 = r2_score(
                list(e[f'out/true/{rep}'].values()),
                list(e[f'out/{model_name}/{rep}'].values())
            )
            e[f'r2/{model_name}/{rep}'] = r2
            e.info(f'node auroc: {auroc_ni:.2f}'
                   f' - edge auroc: {auroc_ei:.2f}'
                   f' - r2: {r2:.2f}'
                   f' - fidelity: {fidelity_mean:.2f}'
                   f' - fidelity_rand: {fidelity_rand_mean:.2f}')

        # ---------------------------------------------------------------------------------------------------
        # -- GNN EXPLAINER ----------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------
        e.info('Training GCN for gnn explainer...')
        ks.backend.clear_session()
        gnnx_model = GnnxGCN(
            units=GNNX_MODEL_UNITS,
            activation='kgcnn>leaky_relu',
            final_activation='linear',
            final_units=FINAL_UNITS
        )
        gnnx_model.compile(
            loss=[ks.losses.MeanSquaredError()],
            loss_weights=[1],
            metrics=[ks.metrics.MeanSquaredError()],
            optimizer=ks.optimizers.Adam(learning_rate=LEARNING_RATE),
            run_eagerly=False
        )
        with tf.device(DEVICE):
            history = gnnx_model.fit(
                x_train,
                y_train,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=(x_test, y_test),
                validation_freq=1,
                callbacks=LogProgressCallback(
                    logger=e.logger,
                    epoch_step=LOG_STEP,
                    identifier=f'val_{METRIC_KEY}'
                ),
                verbose=0
            )
        e[f'histories/gcn/{rep}'] = history.history
        e[f'epochs/gcn/{rep}'] = list(range(EPOCHS))
        e.info(f'parameters: {gnnx_model.count_params()}')

        ni_true = []
        ei_true = []

        ni_pred = []
        ei_pred = []

        # ~ Making all the test set predictions and saving them
        e.info('Evaluating test set...')
        for c, index in enumerate(test_indices):
            data = eye_tracking_dataset[index]
            g = dataset[index]

            gnnx = GNNExplainer(
                gnnx_model,
                compile_options={
                    'loss': ks.losses.mean_squared_error,
                    'optimizer': ks.optimizers.Adam(learning_rate=GNNX_LEARNING_RATE),
                },
                gnnexplaineroptimizer_options={
                    'edge_mask_loss_weight': GNNX_EDGE_WEIGHT,
                    'node_mask_loss_weight': GNNX_NODE_WEIGHT,
                    'feature_mask_loss_weight': 0
                },
                fit_options={
                    'epochs': GNNX_EPOCHS,
                    'batch_size': 1,
                    'verbose': 0,
                }
            )
            x = (
                ragged_tensor_from_nested_numpy([g['node_attributes']]),
                ragged_tensor_from_nested_numpy([g['edge_attributes']]),
                ragged_tensor_from_nested_numpy([g['edge_indices']])
            )
            insp = gnnx.explain(x, inspection=True)
            ni_gnnx, ei_gnnx = gnnx.get_explanation()
            out_gcn = gnnx_model.predict(x).numpy().tolist()[0]
            e[f'out/gnnx/{rep}/{index}'] = out_gcn
            # The first transformation we apply to the importances vectors is to apply a normalization on
            # them such that they use the entire value range [0, 1]
            ni_gnnx = array_normalize(ni_gnnx)
            ei_gnnx = array_normalize(ei_gnnx)
            e[f'ni/gnnx/{rep}/{index}'] = ni_gnnx
            e[f'ei/gnnx/{rep}/{index}'] = ei_gnnx
            ni_true += np.array(g['node_importances']).flatten().tolist()
            ei_true += np.array(g['edge_importances']).flatten().tolist()

            ni_pred += np.array(ni_gnnx).flatten().tolist()
            ei_pred += np.array(ei_gnnx).flatten().tolist()
            # Then with these relatively raw importances we calculate the cont. similarity value (basically
            # something like a distance metric with a normalized & inverted scale)
            node_sim_gnnx = importance_absolute_similarity(g['node_importances'], ni_gnnx)
            edge_sim_gnnx = importance_absolute_similarity(g['edge_importances'], ei_gnnx)
            e[f'node_sim/gnnx/{rep}/{index}'] = node_sim_gnnx
            e[f'edge_sim/gnnx/{rep}/{index}'] = edge_sim_gnnx
            node_sim_can_gnnx = importance_canberra_similarity(g['node_importances'], ni_gnnx)
            edge_sim_can_gnnx = importance_canberra_similarity(g['edge_importances'], ei_gnnx)
            e[f'node_sim_can/gnnx/{rep}/{index}'] = node_sim_can_gnnx
            e[f'edge_sim_can/gnnx/{rep}/{index}'] = edge_sim_can_gnnx
            # Then we calculate the natural sparsity percentage of the importances. We do this by first
            # applying a simple threshold to turn the soft importance masks into hard importance masks
            # and then the sparsity is simply the mean of the hard mask.
            sparsity_ni = np.mean(binary_threshold(ni_gnnx, BINARY_THRESHOLD))
            sparsity_ei = np.mean(binary_threshold(ei_gnnx, BINARY_THRESHOLD))
            e[f'node_sparsity/gnnx/{rep}/{index}'] = sparsity_ni
            e[f'edge_sparsity/gnnx/{rep}/{index}'] = sparsity_ei
            e[f'sparsity_ni/gnnx/{rep}/{index}'] = sparsity_ni
            e[f'sparsity_ei/gnnx/{rep}/{index}'] = sparsity_ei
            # to binary mask
            node_mask = binary_threshold(ni_gnnx, BINARY_THRESHOLD)
            edge_mask = binary_threshold(ei_gnnx, BINARY_THRESHOLD)

            out_m_gcn = gnnx_model.predict([
                x[0] * ragged_tensor_from_nested_numpy([node_mask]),
                x[1] * ragged_tensor_from_nested_numpy([edge_mask]),
                x[2]
            ]).numpy().tolist()[0]
            fidelity_gnnx = abs(out_m_gcn - out_gcn)
            e[f'fidelity/gnnx/{rep}/{index}'] = fidelity_gnnx

            # In the last step we generate a random explanation and then calculate the auroc and the
            # fidelity for that one which will act as a baseline to understand the other values better

            deltas_rand = []
            for _ in range(NUM_RANDOM_MASKS):
                node_mask_rand_indices = random.sample(g['node_indices'], k=int(np.sum(node_mask)))
                node_mask_rand = np.array([0.0 if i in node_mask_rand_indices else 1.0
                                           for i in g['node_indices']])
                node_mask_rand = np.expand_dims(node_mask_rand, axis=-1)
                e[f'ni_rand/{model_name}/{rep}/{index}'] = node_mask_rand

                # edge_mask_rand_indices = random.sample(g['edge_indices'], k=edge_mask_count)
                # edge_mask_rand = np.array([0.0 if m in edge_mask_rand_indices else 1.0
                #                            for m in g['edge_indices']])
                # edge_mask_rand = np.expand_dims(edge_mask_rand, axis=-1)
                # e[f'ei_rand/{model_name}/{rep}/{index}'] = edge_mask_rand

                out_rand_gcn = gnnx_model.predict([
                    x[0] * ragged_tensor_from_nested_numpy([node_mask_rand]),
                    x[1] ,#* ragged_tensor_from_nested_numpy([edge_mask_rand]),
                    x[2]
                ]).numpy().tolist()[0]
                deltas_rand.append(abs(out_rand_gcn - out_gcn))
            fidelity_rand_gnnx = np.mean(deltas_rand)
            e[f'fidelity_rand/gnnx/{rep}/{index}'] = fidelity_rand_gnnx

            if c % LOG_STEP_EVAL == 0:
                e.info(f' § GNNX ({c})'
                       f' - node sim can: {node_sim_can_gnnx:.2f}'
                       f' - edge sim can: {edge_sim_can_gnnx:.2f}'
                       f' - fidelity: {fidelity_gnnx:.2f}({out_gcn:.2f}>{out_m_gcn:.2f})'
                       f' - fidelity_rand: {fidelity_rand_gnnx:.2f}'
                       f' - gnnx_epochs: {len(insp["total_loss"])}'
                       f' - gnnx_loss: {insp["total_loss"][0]:.2f}>{insp["total_loss"][-1]:.2f}')

        fidelity_mean = np.mean(list(e[f'fidelity/gnnx/{rep}'].values()))
        e[f'fidelity_mean/gnnx/{rep}'] = fidelity_mean
        fidelity_rand_mean = np.mean(list(e[f'fidelity_rand/gnnx/{rep}'].values()))
        e[f'fidelity_rand_mean/gnnx/{rep}'] = fidelity_rand_mean

        wilcoxon_result = wilcoxon(
            list(e[f'fidelity/gnnx/{rep}'].values()),
            list(e[f'fidelity_rand/gnnx/{rep}'].values())
        )
        e[f'fidelity_wilcoxon/gnnx/{rep}'] = wilcoxon_result.pvalue

        auroc_ni = roc_auc_score(ni_true, ni_pred)
        e[f'auroc_ni/gnnx/{rep}'] = auroc_ni

        auroc_ei = roc_auc_score(ei_true, ei_pred)
        e[f'auroc_ei/gnnx/{rep}'] = auroc_ei

        mse = history.history['val_mean_squared_error'][-1]
        rmse = np.sqrt(mse)
        e[f'mse/gnnx/{rep}'] = mse
        e[f'rmse/gnnx/{rep}'] = rmse

        r2 = r2_score(
            list(e[f'out/true/{rep}'].values()),
            list(e[f'out/gnnx/{rep}'].values())
        )
        e[f'r2/gnnx/{rep}'] = r2
        e.info(f'node auroc: {auroc_ni:.2f}'
               f' - edge auroc: {auroc_ei:.2f}'
               f' - r2: {r2:.2f}'
               f' - fidelity: {fidelity_mean:.2f}'
               f' - fidelity_rand: {fidelity_rand_mean:.2f}')

        # ~ Drawing the examples
        e.info('Drawing examples...')
        examples_path = os.path.join(e.path, f'{rep}_examples.pdf')
        ncols = 2 + len(MEGAN_SWEEP)
        with PdfPages(examples_path) as pdf:
            for index in example_indices:
                g = dataset[index]
                data = eye_tracking_dataset[index]
                g['node_coordinates'] = np.array(g['node_coordinates'])
                image = np.asarray(imread(data['image_path']))

                fig, rows = plt.subplots(ncols=ncols, nrows=1, figsize=(ncols * 8, 8), squeeze=False)
                fig.suptitle(f'Rep {rep} - Element {index}\n'
                             f'Ground Truth: {e[f"out/true/{rep}/{index}"]:.3f}')

                column_index = 0
                # -- GROUND TRUTH --
                ax_gt = rows[0][column_index]
                ax_gt.set_title('Ground Truth\n\n')
                ax_gt.imshow(image, extent=(0, image.shape[0], 0, image.shape[1]))
                plot_node_importances(
                    g=g,
                    ax=ax_gt,
                    vmax=np.max(g['node_importances']),
                    node_importances=np.squeeze(g['node_importances'], axis=-1),
                    node_coordinates=g['node_coordinates']
                )
                plot_edge_importances(
                    g=g,
                    ax=ax_gt,
                    vmax=np.max(g['edge_importances']),
                    edge_importances=np.squeeze(g['edge_importances'], axis=-1),
                    node_coordinates=g['node_coordinates']
                )
                column_index += 1

                # -- MULTI ATTENTION STUDENT --
                for model_name in MEGAN_SWEEP.keys():
                    ax_mas = rows[0][column_index]
                    ax_mas.set_title(f'Model "{model_name}"\n'
                                     f'Prediction: {e[f"out/{model_name}/{rep}/{index}"]:.2f}\n'
                                     f'ni sim can: {e[f"node_sim_can/{model_name}/{rep}/{index}"]:.2f} - '
                                     f'ei sim can: {e[f"edge_sim_can/{model_name}/{rep}/{index}"]:.2f}\n '
                                     f'fid: {e[f"fidelity/{model_name}/{rep}/{index}"]:.2f} - '
                                     f'fid rand: {e[f"fidelity_rand/{model_name}/{rep}/{index}"]:.2f}')
                    ax_mas.imshow(image, extent=(0, image.shape[0], 0, image.shape[1]))
                    ni_mas = np.array(e[f'ni/{model_name}/{rep}/{index}'])
                    plot_node_importances(
                        g=g,
                        ax=ax_mas,
                        vmax=np.max(ni_mas),
                        node_importances=np.squeeze(ni_mas, axis=-1),
                        node_coordinates=g['node_coordinates']
                    )
                    ei_mas = np.array(e[f'ei/{model_name}/{rep}/{index}'])
                    plot_edge_importances(
                        g=g,
                        ax=ax_mas,
                        vmax=np.max(ei_mas),
                        edge_importances=np.squeeze(ei_mas, axis=-1),
                        node_coordinates=g['node_coordinates']
                    )
                    column_index += 1

                # -- GNN EXPLAINER --
                ax_gnnx = rows[0][column_index]
                model_name = 'gnnx'
                ax_gnnx.set_title(f'Model "{model_name}"\n'
                                  f'Prediction: {e[f"out/{model_name}/{rep}/{index}"]:.2f}\n'
                                  f'ni sim can: {e[f"node_sim_can/{model_name}/{rep}/{index}"]:.2f} - '
                                  f'ei sim can: {e[f"edge_sim_can/{model_name}/{rep}/{index}"]:.2f}\n '
                                  f'fid: {e[f"fidelity/{model_name}/{rep}/{index}"]:.2f} - '
                                  f'fid rand: {e[f"fidelity_rand/{model_name}/{rep}/{index}"]:.2f}')
                ax_gnnx.imshow(image, extent=(0, image.shape[0], 0, image.shape[1]))
                ni_gnnx = np.array(e[f'ni/gnnx/{rep}/{index}'])
                plot_node_importances(
                    g=g,
                    ax=ax_gnnx,
                    vmax=np.max(ni_gnnx),
                    node_importances=np.squeeze(ni_gnnx, axis=-1),
                    node_coordinates=g['node_coordinates']
                )
                ei_gnnx = np.array(e[f'ei/gnnx/{rep}/{index}'])
                plot_edge_importances(
                    g=g,
                    ax=ax_gnnx,
                    vmax=np.max(ei_gnnx),
                    edge_importances=np.squeeze(ei_gnnx, axis=-1),
                    node_coordinates=g['node_coordinates']
                )
                column_index += 1

                pdf.savefig(fig)
                plt.close(fig)

        e.info(f'finished repetition {rep+1}')
        e.status()


with Skippable(), e.analysis:
    model_names = ['gnnx'] + list(MEGAN_SWEEP.keys())
    # This flag determines whether runs where one of the models did not converge should be excluded
    CLEAN_RESULTS = True

    CLEAN_REPETITIONS = list(range(REPETITIONS))
    if CLEAN_RESULTS:
        e.info('excluding non-convergent repetitions...')
        for rep in range(REPETITIONS):
            for model_name in model_names:
                if e[f'mse/{model_name}/{rep}'] > 2.5:
                    CLEAN_REPETITIONS.remove(rep)
                    break

        e.info(f'{len(CLEAN_REPETITIONS)} repetitions remaining')

    # -- POST CALCULATIONS --
    e.info('post hoc calculations...')
    for rep in range(REPETITIONS):
        for model_name in model_names:
            r2 = r2_score(
                list(e[f'out/true/{rep}'].values()),
                list(e[f'out/{model_name}/{rep}'].values())
            )
            e[f'r2/{model_name}/{rep}'] = r2

            e[f'ni_auroc/{model_name}/{rep}'] = e[f'auroc_ni/{model_name}/{rep}']
            e[f'ei_auroc/{model_name}/{rep}'] = e[f'auroc_ei/{model_name}/{rep}']

    # -- SIMPLE STATISTICS --
    e.info('Printing simple statistics')
    REPETITIONS = e.parameters['REPETITIONS']
    metrics = ['node_sim', 'edge_sim', 'node_sim_can', 'edge_sim_can',
               'node_sparsity', 'edge_sparsity',
               'fidelity', 'fidelity_rand']

    for metric in metrics:
        e.info('')
        e.info(metric.upper())
        for model_name in model_names:
            metric_values = []
            for rep in CLEAN_REPETITIONS:
                for index in e[f'{metric}/{model_name}/{rep}'].keys():
                    value = e[f'{metric}/{model_name}/{rep}/{index}']
                    if value is not None:
                        metric_values.append(value)

            e.info(f'')
            e.info(f'* {model_name} / {metric}')
            e.info(f'  - min: {np.min(metric_values):.2f} - max: {np.max(metric_values):.2f}')
            e.info(f'  - mean: {np.mean(metric_values):.2f}: (std: {np.std(metric_values):.2f})')
            e.info(f'  - (0.25 - median - 0.75): ({np.quantile(metric_values, 0.25):.2f} - '
                   f'{np.median(metric_values):.2f} - {np.quantile(metric_values, 0.75):.2f})')

    # -- RENDERING LATEX TABLE --
    e.info('rendering latex table...')
    column_names = [
        'Model',
        'MSE',
        '$R^2$',
        'Node AUROC',
        'Edge AUROC',
        'Sparsity',
        'Fidelity',
        'Fidelity Rand.'
    ]
    rows = []
    for model_name in model_names:
        row = []
        if 'megan' in model_name:
            parameters = MEGAN_SWEEP[model_name]
            model_identifier = r'$\text{MEGAN}^1_{' + f'{parameters["IMPORTANCE_FACTOR"]:.2f}' + r'}$'
        else:
            model_identifier = r'$\text{GNNX}_{\text{GCN}}$'
        row.append(model_identifier)
        row.append([e[f'mse/{model_name}/{rep}']
                    for rep in CLEAN_REPETITIONS])
        row.append([e[f'r2/{model_name}/{rep}']
                    for rep in CLEAN_REPETITIONS])
        row.append([e[f'auroc_ni/{model_name}/{rep}']
                    for rep in CLEAN_REPETITIONS])
        row.append([e[f'auroc_ei/{model_name}/{rep}']
                    for rep in CLEAN_REPETITIONS])
        row.append([e[f'node_sparsity/{model_name}/{rep}/{index}']
                    for rep in CLEAN_REPETITIONS for index in e[f'test_indices/{rep}']])
        row.append([e[f'fidelity/{model_name}/{rep}/{index}']
                    for rep in CLEAN_REPETITIONS for index in e[f'test_indices/{rep}']])
        row.append([e[f'fidelity_rand/{model_name}/{rep}/{index}']
                    for rep in CLEAN_REPETITIONS for index in e[f'test_indices/{rep}']])
        rows.append(row)

    content, table = latex_table(
        column_names=column_names,
        rows=rows,
        list_element_cb=latex_table_element_median
    )
    e.commit_raw('table.tex', table)
    pdf_path = os.path.join(e.path, 'table.pdf')
    render_latex({'content': table}, output_path=pdf_path)
    e.info('rendered latex table')