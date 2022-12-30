"""
Trains multi-channel MEGAN models on the synthetic graph regression dataset RbMotifs. This datasets consits
of randomly generated colored graphs, where the target value of each graph is determined by the special
subgraph motifs contained in it. The MEGAN models are trained with two explanation channels: One to
represent the positive evidence w.r.t. to a reference value and the other for the negative evidence.
The experiment compares multiple hyperparameter configurations, mainly the impact of the "importance factor"
which is the weight assigned to the self-supervised explanation-only training step. The experiment is
repeated multiple independent times.
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
from imageio.v2 import imread
from pycomex.experiment import Experiment
from pycomex.util import Skippable

from graph_attention_student.util import DATASETS_FOLDER
from graph_attention_student.util import importance_absolute_similarity
from graph_attention_student.util import importance_canberra_similarity
from graph_attention_student.util import array_normalize
from graph_attention_student.util import binary_threshold
from graph_attention_student.util import render_latex, latex_table, latex_table_element_median
from graph_attention_student.data import load_eye_tracking_dataset
from graph_attention_student.data import process_graph_dataset
from graph_attention_student.models import MultiAttentionStudent
from graph_attention_student.training import NoLoss, ExplanationLoss
from graph_attention_student.training import LogProgressCallback
from graph_attention_student.visualization import plot_node_importances, plot_edge_importances

SHORT_DESCRIPTION = (
    'Trains multi-channel MEGAN model on the synthetic graph regression dataset RbMotifs. Compares the '
    'generated explanations to the known ground truth.'
)

# == META PARAMETERS ==
REPETITIONS = 50

# == DATASET PARAMETERS ==
DATASET_PATH = os.path.join(DATASETS_FOLDER, 'rb_dual_motifs')
TEST_RATIO = 0.1
NUM_EXAMPLES = 100
EXAMPLE_INDICES: Optional[list] = None

# == MODEL PARAMETERS ==
UNITS = [3, 3, 3]
IMPORTANCE_UNITS = []
SPARSITY_FACTOR = 0
IMPORTANCE_FACTOR = 0.0
IMPORTANCE_MULTIPLIER = 1
IMPORTANCE_SPARSITY = 0
NUM_CHANNELS = 2
FINAL_UNITS = [3]
DROPOUT_RATE = 0.0
FINAL_DROPOUT_RATE = 0.0
REGRESSION_LIMITS = [-4, 4]
REGRESSION_BINS = [
    [-5, -1.5],
    [1.5, 5]
]
USE_BIAS = True

MEGAN_SWEEP = {
    'megan_0': {
        'IMPORTANCE_FACTOR': 0.0,
        'SPARSITY_FACTOR': 1e-1,
        'IMPORTANCE_SUPERVISION': False,
    },
    'megan_1': {
        'IMPORTANCE_FACTOR': 0.1,
        'IMPORTANCE_MULTIPLIER': 10,
        'SPARSITY_FACTOR': 1e-1,
        'IMPORTANCE_SPARSITY': 0.,
        'IMPORTANCE_SUPERVISION': False,
    },
    'megan_2': {
        'IMPORTANCE_FACTOR': 1.0,
        'IMPORTANCE_MULTIPLIER': 10,
        'SPARSITY_FACTOR': 1e-1,
        'IMPORTANCE_SPARSITY': 0.,
        'IMPORTANCE_SUPERVISION': False,
    },
    'megan_3': {
        'IMPORTANCE_FACTOR': 0.0,
        'SPARSITY_FACTOR': 1e-1,
        'IMPORTANCE_SPARSITY': 0.,
        'IMPORTANCE_SUPERVISION': True,
    }
}

# == TRAINING PARAMETERS ==
LEARNING_RATE = 0.002
BATCH_SIZE = 512
EPOCHS = 250
LOG_STEP = 10
DEVICE = 'gpu:0'
METRIC_KEY = 'mean_squared_error'
IMPORTANCE_SUPERVISION = False

# == EVALUATION PARAMETERS ==
FIDELITY_SPARSITY = 0.15
BINARY_THRESHOLD = 0.5
LOG_STEP_EVAL = 20
CHANNEL_NAMES = {
    0: 'NEGATIVE',
    1: 'POSITIVE'
}
CHANNEL_DIRECTIONS = {
    0: -1,
    1: +1,
}

# == EXPERIMENT PARAMETERS ==
NAMESPACE = 'rb_motifs_multi'
BASE_PATH = os.getcwd()
DEBUG = True

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

            # The dataset already comes with ground truth importance annotations which are split into two
            # channels corresponding to "positive" and "negative" influences
            node_importances = np.array(g['multi_node_importances'])
            g['node_importances'] = node_importances

            edge_importances = np.array(g['multi_edge_importances'])
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
                importance_sparsity=IMPORTANCE_SPARSITY,
                use_bias=USE_BIAS,
                final_units=FINAL_UNITS,
                final_dropout_rate=FINAL_DROPOUT_RATE,
                regression_bins=REGRESSION_BINS,
                regression_limits=REGRESSION_LIMITS,
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

            # ~ Making all the test set predictions and saving them
            ni_true_all = []
            ei_true_all = []

            ni_pred_all = []
            ei_pred_all = []

            e.info('Evaluating test set...')
            for c, index in enumerate(test_indices):
                data = eye_tracking_dataset[index]
                g = dataset[index]

                # First of all we need to query the model to make the actual prediction for the current element
                out_meg, ni_meg, ei_meg = model.predict_single((
                    g['node_attributes'],
                    g['edge_attributes'],
                    g['edge_indices']
                ))
                out_true = g['graph_labels'].tolist()
                e[f'out/true/{rep}/{index}'] = out_true
                e[f'out/{model_name}/{rep}/{index}'] = out_meg
                e[f'absolute_error/{model_name}/{rep}/{index}'] = np.abs(out_meg - out_true)
                e[f'squared_error/{model_name}/{rep}/{index}'] = np.square(out_meg - out_true)
                ni_true_all += np.array(g['node_importances']).flatten().tolist()
                ei_true_all += np.array(g['edge_importances']).flatten().tolist()
                # The first transformation we apply to the importances vectors is to apply a normalization on
                # them such that they use the entire value range [0, 1]
                ni_meg = array_normalize(ni_meg)
                ei_meg = array_normalize(ei_meg)
                # ni_mas = np.array(ni_mas)
                # ei_mas = np.array(ei_mas)
                e[f'ni/{model_name}/{rep}/{index}'] = ni_meg.tolist()
                e[f'ei/{model_name}/{rep}/{index}'] = ei_meg.tolist()
                ni_pred_all += ni_meg.flatten().tolist()
                ei_pred_all += ei_meg.flatten().tolist()
                # Then with these relatively raw importances we calculate the cont. similarity value (basically
                # something like a distance metric with a normalized & inverted scale)
                node_sim_abs_meg = importance_absolute_similarity(g['node_importances'], ni_meg)
                edge_sim_abs_meg = importance_absolute_similarity(g['edge_importances'], ei_meg)
                e[f'node_sim_abs/{model_name}/{rep}/{index}'] = node_sim_abs_meg
                e[f'edge_sim_abs/{model_name}/{rep}/{index}'] = edge_sim_abs_meg
                node_sim_can_meg = importance_canberra_similarity(g['node_importances'], ni_meg)
                edge_sim_can_meg = importance_canberra_similarity(g['edge_importances'], ei_meg)
                e[f'node_sim_can/{model_name}/{rep}/{index}'] = node_sim_can_meg
                e[f'edge_sim_can/{model_name}/{rep}/{index}'] = edge_sim_can_meg
                # Then we calculate the natural sparsity percentage of the importances. We do this by first
                # applying a simple threshold to turn the soft importance masks into hard importance masks
                # and then the sparsity is simply the mean of the hard mask.
                e[f'node_sparsity/{model_name}/{rep}/{index}'] = np.mean(binary_threshold(ni_meg, BINARY_THRESHOLD))
                e[f'edge_sparsity/{model_name}/{rep}/{index}'] = np.mean(binary_threshold(ei_meg, BINARY_THRESHOLD))
                # This part here also turns the soft masks into hard masks, but by turning the top K values
                # into 1s and the other values to 0s (basically dynamic threshold). We do this because that
                # is the fairer starting point for the auroc & fidelity calculation
                node_mask_count = int(FIDELITY_SPARSITY * len(g['node_indices']))
                edge_mask_count = int(FIDELITY_SPARSITY * len(g['edge_indices']))
                # ni_mas[:, 0] = binary_threshold_k(ni_mas[:, 0], k=node_mask_count).tolist()
                # ei_mas[:, 0] = binary_threshold_k(ei_mas[:, 0], k=edge_mask_count).tolist()

                # ~ IMPORTANCE MASKING FIDELITY
                fidelity_im = 0
                for k in range(NUM_CHANNELS):
                    _ni_mod = np.copy(ni_meg)
                    _ni_mod[:, k] = 0
                    _out_mod, _, _ = model.predict_single((
                        g['node_attributes'],
                        g['edge_attributes'],
                        g['edge_indices']
                    ), external_node_importances=_ni_mod)
                    fidelity_im += - CHANNEL_DIRECTIONS[k] * (_out_mod - out_meg)
                    e[f'out_mod_im/{model_name}/{rep}/{index}/{k}'] = _out_mod

                e[f'fidelity_im/{model_name}/{rep}/{index}'] = fidelity_im

                if c % LOG_STEP_EVAL == 0:
                    e.info(f' * {model_name.upper()} ({c}) '
                           f' - node sim abs: {node_sim_abs_meg:.2f}'
                           f' - edge sim abs: {edge_sim_abs_meg:.2f}'
                           f' - y_pred: {out_meg:.2f}'
                           f' - y_true: {e[f"out/true/{rep}/{index}"]:.2f}'
                           f' - fidelity_imp: {fidelity_im:.2f}')

            # ~ Calculating overarching metrics
            ni_auroc = roc_auc_score(ni_true_all, ni_pred_all)
            ei_auroc = roc_auc_score(ei_true_all, ei_pred_all)
            e[f'ni_auroc/{model_name}/{rep}'] = ni_auroc
            e[f'ei_auroc/{model_name}/{rep}'] = ei_auroc

            r2 = r2_score(
                list(e[f'out/true/{rep}'].values()),
                list(e[f'out/{model_name}/{rep}'].values())
            )
            e[f'r2/{model_name}/{rep}'] = r2

            e.info(f' * {model_name.upper()} TOTAL '
                   f' - r2: {r2:.2f}'
                   f' - node auroc: {ni_auroc:.2f}'
                   f' - edge auroc: {ei_auroc:.2f}')

        # ~ Drawing the examples
        e.info('Drawing examples...')
        examples_path = os.path.join(e.path, f'{rep}_examples.pdf')
        ncols = 1 + len(MEGAN_SWEEP)
        with PdfPages(examples_path) as pdf:
            for index in example_indices:
                g = dataset[index]
                data = eye_tracking_dataset[index]
                g['node_coordinates'] = np.array(g['node_coordinates'])
                image = np.asarray(imread(data['image_path']))

                fig, rows = plt.subplots(ncols=ncols, nrows=NUM_CHANNELS, figsize=(ncols * 8, 2 * 8),
                                         squeeze=False)
                fig.suptitle(f'Rep {rep} - Element {index}\n'
                             f'Ground Truth: {e[f"out/true/{rep}/{index}"]:.3f}')

                for row_index in range(NUM_CHANNELS):
                    column_index = 0

                    # -- GROUND TRUTH --
                    ax_gt = rows[row_index][column_index]
                    ax_gt.set_title('Ground Truth\n\n')
                    ax_gt.set_ylabel(f'Channel {row_index}: {CHANNEL_NAMES[row_index]}')
                    ax_gt.imshow(image, extent=(0, image.shape[0], 0, image.shape[1]))
                    node_importances = np.array(g['node_importances'])
                    plot_node_importances(
                        g=g,
                        ax=ax_gt,
                        vmax=np.max(node_importances[:, row_index]),
                        node_importances=node_importances[:, row_index],
                        node_coordinates=g['node_coordinates']
                    )
                    edge_importances = np.array(g['edge_importances'])
                    plot_edge_importances(
                        g=g,
                        ax=ax_gt,
                        vmax=np.max(edge_importances[:, row_index]),
                        edge_importances=edge_importances[:, row_index],
                        node_coordinates=g['node_coordinates']
                    )
                    column_index += 1

                    # -- MULTI ATTENTION STUDENT --
                    for model_name in MEGAN_SWEEP.keys():
                        ax_mas = rows[row_index][column_index]
                        if row_index == 0:
                            ax_mas.set_title(
                                f'Model "{model_name}"\n'
                                f'Prediction: {e[f"out/{model_name}/{rep}/{index}"]:.2f}\n'
                                f'ni sim abs: {e[f"node_sim_abs/{model_name}/{rep}/{index}"]:.2f} - '
                                f'ei sim abs: {e[f"edge_sim_abs/{model_name}/{rep}/{index}"]:.2f}\n '
                                f'ni sim can: {e[f"node_sim_can/{model_name}/{rep}/{index}"]:.2f} - '
                                f'ei sim can: {e[f"edge_sim_can/{model_name}/{rep}/{index}"]:.2f}'
                            )

                        ax_mas.imshow(image, extent=(0, image.shape[0], 0, image.shape[1]))
                        ni_mas = np.array(e[f'ni/{model_name}/{rep}/{index}'])
                        plot_node_importances(
                            g=g,
                            ax=ax_mas,
                            vmax=1,
                            #vmax=np.max(ni_mas),
                            #vmax=np.max(ni_mas[:, row_index]),
                            node_importances=ni_mas[:, row_index],
                            node_coordinates=g['node_coordinates']
                        )
                        ei_mas = np.array(e[f'ei/{model_name}/{rep}/{index}'])
                        plot_edge_importances(
                            g=g,
                            ax=ax_mas,
                            vmax=1,
                            #vmax=np.max(ei_mas),
                            #vmax=np.max(ei_mas[:, row_index]),
                            edge_importances=ei_mas[:, row_index],
                            node_coordinates=g['node_coordinates']
                        )
                        column_index += 1

                pdf.savefig(fig)
                plt.close(fig)

        e.info(f'finished repetition {rep + 1}')
        e.status()


with Skippable(), e.analysis:

    NUM_WORST_BEST = 50

    # -- LOADING DATASET ------------------------------------------------------------------------------------
    start_time = time.time()
    eye_tracking_dataset = load_eye_tracking_dataset(DATASET_PATH)
    dataset_size = len(eye_tracking_dataset)
    dataset_indices = list(range(dataset_size))

    # "eye_tracking_dataset" at this point is only a list of dictionaries which contain the metadata info
    # for each respective elements. This now has to be turned into the appropriate tensors to be used by
    # the model.
    dataset = []
    for data in eye_tracking_dataset:
        g = data['metadata']['graph']

        # The target ground truth will be the float value associated with every graph
        value = data['metadata']['value']
        g['graph_labels'] = np.array(value)

        # The dataset already comes with ground truth importance annotations which are split into two
        # channels corresponding to "positive" and "negative" influences
        node_importances = np.array(g['multi_node_importances'])
        g['node_importances'] = node_importances

        edge_importances = np.array(g['multi_edge_importances'])
        g['edge_importances'] = edge_importances

        dataset.append(g)

    end_time = time.time()
    e.info(f'Loaded dataset with {dataset_size} elements in {end_time - start_time:.2f} seconds')
    y_trues = [float(g['graph_labels']) for g in dataset]

    # -- ADDITIONAL CALCULATIONS ----------------------------------------------------------------------------
    e.info('Post-hoc calculations...')
    for rep in range(REPETITIONS):
        e.info(f'REP {rep + 1}/{REPETITIONS}')
        test_indices = e[f'test_indices/{rep}']
        for model_name in MEGAN_SWEEP.keys():
            ni_true_all = []
            ei_true_all = []

            ni_pred_all = []
            ei_pred_all = []
            for index in test_indices:
                g = dataset[index]
                ni_true = np.array(g['node_importances'])
                ei_true = np.array(g['edge_importances'])
                ni_true_all += ni_true.flatten().tolist()
                ei_true_all += ei_true.flatten().tolist()

                ni_pred = np.array(e[f'ni/{model_name}/{rep}/{index}'])
                ei_pred = np.array(e[f'ei/{model_name}/{rep}/{index}'])
                ni_pred_all += ni_pred.flatten().tolist()
                ei_pred_all += ei_pred.flatten().tolist()

            ni_auc = roc_auc_score(ni_true_all, ni_pred_all)
            ei_auc = roc_auc_score(ei_true_all, ei_pred_all)

            e[f'ni_auroc/{model_name}/{rep}'] = ni_auc
            e[f'ei_auroc/{model_name}/{rep}'] = ei_auc

            r2 = r2_score(
                list(e[f'out/true/{rep}'].values()),
                list(e[f'out/{model_name}/{rep}'].values())
            )
            e[f'r2/{model_name}/{rep}'] = r2

            e.info(f' * {model_name:<8} '
                   f' - ni auroc: {ni_auc:.3f}'
                   f' - ei auroc: {ei_auc:.3f}'
                   f' - r2: {r2:.3f}')

    e.info('TOTAL:')
    for model_name in MEGAN_SWEEP.keys():
        ni_aucs = []
        ei_aucs = []
        r2s = []
        for rep in range(REPETITIONS):
            ni_aucs.append(e[f'ni_auroc/{model_name}/{rep}'])
            ei_aucs.append(e[f'ei_auroc/{model_name}/{rep}'])
            r2s.append(e[f'r2/{model_name}/{rep}'])

        e.info(f' * {model_name:<8} '
               f' - ni auroc: {np.mean(ni_aucs):.2f} ({np.std(ni_aucs):.2f})'
               f' - ei auroc: {np.mean(ei_aucs):.2f} ({np.std(ei_aucs):.2f})'
               f' - r2: {np.mean(r2s):.2f} ({np.std(r2s):.2f})')

    # -- SIMPLE STATISTICS ----------------------------------------------------------------------------------
    e.info('Printing simple statistics')
    metrics = ['absolute_error', 'squared_error',
               'node_sim_abs', 'node_sim_can', 'edge_sim_abs', 'edge_sim_can',
               'node_sparsity', 'edge_sparsity',
               'fidelity_im']
    model_names = list(e.parameters['MEGAN_SWEEP'].keys())

    for metric in metrics:
        e.info('')
        e.info(metric.upper())
        for model_name in model_names:
            metric_values = []
            for rep in range(REPETITIONS):
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

    # -- RENDERING LATEX TABLE ------------------------------------------------------------------------------
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
    for model_name, parameters in MEGAN_SWEEP.items():
        row = []
        model_identifier = r'$\text{MEGAN}^2_{' + f'{parameters["IMPORTANCE_FACTOR"]:.2f}' + r'}$'
        row.append(model_identifier)
        row.append([e[f'squared_error/{model_name}/{rep}/{index}']
                    for rep in range(REPETITIONS) for index in e[f'test_indices/{rep}']])
        row.append([e[f'r2/{model_name}/{rep}']
                    for rep in range(REPETITIONS)])
        row.append([e[f'ni_auroc/{model_name}/{rep}']
                    for rep in range(REPETITIONS)])
        row.append([e[f'ei_auroc/{model_name}/{rep}']
                    for rep in range(REPETITIONS)])
        row.append([e[f'node_sparsity/{model_name}/{rep}/{index}']
                    for rep in range(REPETITIONS) for index in e[f'test_indices/{rep}']])
        row.append([e[f'fidelity_im/{model_name}/{rep}/{index}']
                    for rep in range(REPETITIONS) for index in e[f'test_indices/{rep}']])
        row.append('-')
        rows.append(row)

    content, table = latex_table(
        column_names=column_names,
        rows=rows,
        list_element_cb=latex_table_element_median
    )
    print(table)
    e.commit_raw('table.tex', table)
    pdf_path = os.path.join(e.path, 'table.pdf')
    render_latex({'content': table}, output_path=pdf_path)
    e.info('rendered latex table')

    # -- DRAWING EXAMPLES -----------------------------------------------------------------------------------
    e.info('')
    e.info('Drawing best and worst examples according to different criteria')


    def combined_absolute_similarity(rep: int, index: int, model_name: Optional[str] = 'megan_2'):
        model_values = {}
        for name in e['node_sim_abs'].keys():
            node_sim = e[f'node_sim_abs/{name}/{rep}/{index}']
            edge_sim = e[f'edge_sim_abs/{name}/{rep}/{index}']
            model_values[name] = node_sim + edge_sim

        if model_name is None:
            return np.max(model_values.values())

        else:
            return model_values[model_name]


    def combined_canberra_similarity(rep: int, index: int, model_name: Optional[str] = 'megan_2'):
        model_values = {}
        for name in e['node_sim_can'].keys():
            node_sim = e[f'node_sim_can/{name}/{rep}/{index}']
            edge_sim = e[f'edge_sim_can/{name}/{rep}/{index}']
            model_values[name] = node_sim + edge_sim

        if model_name is None:
            return np.max(model_values.values())

        else:
            return model_values[model_name]


    criteria = [
        (combined_absolute_similarity, ['node_sim_abs', 'edge_sim_abs']),
        (combined_canberra_similarity, ['node_sim_can', 'edge_sim_can'])
    ]
    for criterion, metrics in criteria:
        criterion_name = criterion.__name__
        criterion_description = criterion.__doc__
        e.info(f' * {criterion_name}')

        value_tuples = []
        for rep in range(REPETITIONS):
            for index in e[f'test_indices/{rep}']:
                value = criterion(rep, index)
                value_tuples.append(((rep, index), value))

        # Now we need to sort this list according to the values and based on that sorting we can simply
        # get the N first and last elements of that list which will be the best and worst elements
        value_tuples = sorted(value_tuples, key=lambda t: t[1])

        best_tuples = [value_tuples[i] for i in range(NUM_WORST_BEST)]
        worst_tuples = list(reversed([value_tuples[-(i + 1)] for i in range(NUM_WORST_BEST)]))

        # And then we can generate the pdf with the examples
        path = os.path.join(e.path, f'{criterion_name}__best_worst.pdf')
        ncols = 1 + len(MEGAN_SWEEP)
        with PdfPages(path) as pdf:

            for (rep, index), value in reversed(best_tuples + worst_tuples):
                g = dataset[index]
                data = eye_tracking_dataset[index]
                g['node_coordinates'] = np.array(g['node_coordinates'])
                image = np.asarray(imread(data['image_path']))

                fig, rows = plt.subplots(ncols=ncols, nrows=NUM_CHANNELS, figsize=(ncols * 8, 2 * 8),
                                         squeeze=False)
                fig.suptitle(f'Rep {rep} - Element {index}\n'
                             f'{criterion_name}: {value:.2f}')

                for row_index in range(NUM_CHANNELS):
                    column_index = 0

                    # -- GROUND TRUTH --
                    ax_gt = rows[row_index][column_index]
                    ax_gt.set_title('Ground Truth\n\n')
                    ax_gt.set_ylabel(f'Channel {row_index}: {CHANNEL_NAMES[row_index]}')
                    ax_gt.imshow(image, extent=(0, image.shape[0], 0, image.shape[1]))
                    node_importances = np.array(g['node_importances'])
                    plot_node_importances(
                        g=g,
                        ax=ax_gt,
                        vmax=np.max(node_importances[:, row_index]),
                        node_importances=node_importances[:, row_index],
                        node_coordinates=g['node_coordinates']
                    )
                    edge_importances = np.array(g['edge_importances'])
                    plot_edge_importances(
                        g=g,
                        ax=ax_gt,
                        vmax=np.max(edge_importances[:, row_index]),
                        edge_importances=edge_importances[:, row_index],
                        node_coordinates=g['node_coordinates']
                    )
                    column_index += 1

                    # -- MULTI ATTENTION STUDENT --
                    for model_name in MEGAN_SWEEP.keys():
                        ax_mas = rows[row_index][column_index]
                        if row_index == 0:
                            title_lines = [f'Model {model_name}']
                            for metric in metrics:
                                value = e[f'{metric}/{model_name}/{rep}/{index}']
                                title_lines.append(f'{metric}: {value:.2f}')
                            ax_mas.set_title('\n'.join(title_lines))

                        ax_mas.imshow(image, extent=(0, image.shape[0], 0, image.shape[1]))
                        ni_mas = np.array(e[f'ni/{model_name}/{rep}/{index}'])
                        plot_node_importances(
                            g=g,
                            ax=ax_mas,
                            vmax=np.max(ni_mas),
                            # vmax=np.max(ni_mas[:, row_index]),
                            node_importances=ni_mas[:, row_index],
                            node_coordinates=g['node_coordinates']
                        )
                        ei_mas = np.array(e[f'ei/{model_name}/{rep}/{index}'])
                        plot_edge_importances(
                            g=g,
                            ax=ax_mas,
                            vmax=np.max(ei_mas),
                            # vmax=np.max(ei_mas[:, row_index]),
                            edge_importances=ei_mas[:, row_index],
                            node_coordinates=g['node_coordinates']
                        )
                        column_index += 1

                pdf.savefig(fig)
                plt.close(fig)