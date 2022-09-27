"""
Trains multi-channel MEGAN model on the solubility dataset. The solubility datasets consists of molecular
graphs with the task to predict the water solubility of the corresponding molecule. The MEGAN models are
trained with two explanation channels: One to represent the positive evidence w.r.t. to a reference value
and the other for the negative evidence. Evaluates the prediction performance as well as the generated
explanations. The experiment is repeated multiple independent times.
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

import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from nltk.tokenize import word_tokenize
from imageio.v2 import imread
from sklearn.metrics import r2_score

from pycomex.experiment import Experiment
from pycomex.util import Skippable

from graph_attention_student.util import array_normalize
from graph_attention_student.util import binary_threshold
from graph_attention_student.util import render_latex, latex_table, latex_table_element_median
from graph_attention_student.util import DatasetError
from graph_attention_student.data import load_eye_tracking_dataset
from graph_attention_student.data import process_graph_dataset
from graph_attention_student.models import MultiAttentionStudent
from graph_attention_student.training import NoLoss
from graph_attention_student.training import LogProgressCallback
from graph_attention_student.visualization import plot_node_importances, plot_edge_importances

SHORT_EXPLANATION = (
    'Trains a multi-channel MEGAN model on the solubility dataset. Evaluates prediction performance and the '
    'generated explanations.'
)

# == META PARAMETERS ==
REPETITIONS = 50

# == DATASET PARAMETERS ==
# [!] Insert the path to the dataset folder here.
DATASET_PATH = '/home/jonas/Data/Programming/PyCharm/graph_attention_student/graph_attention_student' \
               '/datasets/aqsoldb'
NUM_EXAMPLES = 100
TEST_RATIO = 0.1
EXAMPLE_INDICES: Optional[list] = None

# == MODEL PARAMETERS ==
UNITS = [45, 40, 35, 30, 25]
IMPORTANCE_UNITS = [20]
IMPORTANCE_FACTOR = 0.0
IMPORTANCE_SPARSITY = 0.0
IMPORTANCE_EXCLUSIVITY = 0.0
IMPORTANCE_MULTIPLIER = 1
SPARSITY_FACTOR = 0.0
NUM_CHANNELS = 2
FINAL_UNITS = [30, 20, 10]
DROPOUT_RATE = 0.2
FINAL_DROPOUT_RATE = 0.00
IMPORTANCE_DROPOUT_RATE = 0.0
USE_BIAS = True
REGRESSION_LIMITS = [-16, 3]
REGRESSION_REFERENCE = -2
REGRESSION_BINS = [
    [-10, -6],
    [-1, 4]
]
MODEL_SWEEP = {
    'megan_0': {
        'IMPORTANCE_FACTOR': 0.5,
        'IMPORTANCE_MULTIPLIER': 10,
        'IMPORTANCE_SPARSITY': 0e-0,
        'SPARSITY_FACTOR': 0.1
    },
}

# == TRAINING PARAMETERS ==
LEARNING_RATE = 0.001
BATCH_SIZE = 512
EPOCHS = 250
LOG_STEP = 10
METRIC_KEY = 'mean_squared_error'

# == EVALUATION PARAMETERS
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
NAMESPACE = 'solubility_multi'
BASE_PATH = os.getcwd()
DEBUG = True

with Skippable(), (e := Experiment(namespace=NAMESPACE, base_path=BASE_PATH, glob=globals())):

    e.info('Loading the dataset...')
    if not os.path.exists(DATASET_PATH) or not os.path.isdir(DATASET_PATH):
        raise DatasetError(f'The given dataset path "{DATASET_PATH}" either does not exists or is not a '
                           f'valid dataset folder. Please make sure to specify a valid movie reviews '
                           f'dataset before running the experiment again')

    start_time = time.time()
    dataset_dict: Dict[str, dict] = dict(enumerate(load_eye_tracking_dataset(DATASET_PATH)))
    dataset_size = len(dataset_dict)
    end_time = time.time()
    e.info(f'loaded dataset with {dataset_size} elements '
           f'in {end_time - start_time:.2f} seconds')
    first_element = list(dataset_dict.values())[0]
    e.info(f'dataset element keys: {str(list(first_element.keys()))}')

    e.info('Preparing dataset for training...')
    dataset_indices = list(dataset_dict.keys())
    # test_indices = [int(k) for k, v in dataset_dict.items() if v['metadata']['type'] == 'val']
    test_indices = [int(k) for k, v in dataset_dict.items()
                    if v['metadata']['type'] == 'test']
    train_indices = [int(k) for k, v in dataset_dict.items()
                     if v['metadata']['type'] == 'train']
    e['test_indices'] = test_indices
    e['train_indices'] = train_indices
    e.info(f'Identified {len(train_indices)} train elements and {len(test_indices)} test_elements')

    example_indices = random.sample(test_indices, k=NUM_EXAMPLES)
    e['example_indices'] = example_indices
    e.info(f'Identified {len(example_indices)} example elements from the test set')

    # We need to initialize the dataset list here with actual values, because we need to do indexed
    # assignments to get the elements to their correct position within this list...
    dataset = [None for _ in range(dataset_size)]
    for key, data in dataset_dict.items():
        index = int(key)

        g = data['metadata']['graph']
        # We need to create the target labels as a one hot encoded classification vector
        g['graph_labels'] = np.array(data['metadata']['solubility'])

        data['metadata']['graph'] = g
        dataset[index] = g

    x_train, y_train, x_test, y_test = process_graph_dataset(dataset, test_indices)
    e.info('Processed dataset into tensors')

    for rep in range(REPETITIONS):
        e.info(f'REPETITION {rep + 1}')

        for model_name, parameters in MODEL_SWEEP.items():
            e.info(f'Model sweep: {model_name}')
            for parameter_key, parameter_value in parameters.items():
                globals()[parameter_key] = parameter_value
                e.info(f' * {parameter_key}: {eval(parameter_key)}')

            e.info('Starting model training...')
            model: ks.models.Model = MultiAttentionStudent(
                units=UNITS,
                dropout_rate=DROPOUT_RATE,
                importance_factor=IMPORTANCE_FACTOR,
                importance_multiplier=IMPORTANCE_MULTIPLIER,
                importance_sparsity=IMPORTANCE_SPARSITY,
                importance_exclusivity=0,
                importance_units=IMPORTANCE_UNITS,
                importance_channels=NUM_CHANNELS,
                importance_dropout_rate=IMPORTANCE_DROPOUT_RATE,
                use_bias=USE_BIAS,
                final_units=FINAL_UNITS,
                final_dropout_rate=FINAL_DROPOUT_RATE,
                final_activation='linear',
                regression_bins=REGRESSION_BINS,
                regression_limits=REGRESSION_LIMITS,
                regression_reference=REGRESSION_REFERENCE,
                sparsity_factor=SPARSITY_FACTOR,
            )
            model.compile(
                loss=[
                    ks.losses.MeanSquaredError(),
                    NoLoss(),
                    NoLoss()
                ],
                loss_weights=[
                    1,
                    0,
                    0
                ],
                metrics=[
                    ks.metrics.MeanSquaredError(),
                    ks.metrics.MeanAbsoluteError(),
                ],
                optimizer=ks.optimizers.Adam(learning_rate=LEARNING_RATE),
                run_eagerly=False
            )
            with tf.device('gpu:0'):
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
            e[f'history/{rep}'] = history.history
            e[f'epochs/{rep}'] = list(range(EPOCHS))
            e.info(f'number of parameters: {model.count_params()}')

            e.info('Evaluating on test set...')
            for c, index in enumerate(test_indices):
                data = dataset_dict[index]
                g = data['graph']

                out_true = data['metadata']['solubility']
                e[f"out/true/{rep}/{index}"] = out_true
                out, ni, ei = model.predict_single([
                    g['node_attributes'],
                    g['edge_attributes'],
                    g['edge_indices']
                ])
                e[f'out/{model_name}/{rep}/{index}'] = out
                ni = array_normalize(ni)
                ei = array_normalize(ei)
                e[f'ni/{model_name}/{rep}/{index}'] = ni
                e[f'ei/{model_name}/{rep}/{index}'] = ei

                # ~ Sparsity
                sparsity_ni = np.mean(binary_threshold(ni, 0.5))
                sparsity_ei = np.mean(binary_threshold(ei, 0.5))
                e[f'sparsity_ni/{model_name}/{rep}/{index}'] = sparsity_ni
                e[f'sparsity_ei/{model_name}/{rep}/{index}'] = sparsity_ei

                # ~ Fidelity
                fidelity_im = 0
                for k in range(NUM_CHANNELS):
                    ni_mod = np.copy(ni)
                    ni_mod[:, k] = 0
                    out_mod, _, _ = model.predict_single((
                        g['node_attributes'],
                        g['edge_attributes'],
                        g['edge_indices']
                    ), external_node_importances=ni_mod)
                    fidelity_im += CHANNEL_DIRECTIONS[k] * (out - out_mod)
                    e[f'out_mod_fidelity/{model_name}/{rep}/{index}/{k}'] = out_mod

                e[f'fidelity_im/{model_name}/{rep}/{index}'] = fidelity_im

                if c % LOG_STEP_EVAL == 0:
                    e.info(f' ({c})'
                           f' - gt: {out_true:.2f}'
                           f' - pred: {out:.2f}'
                           f' - fidelity: {fidelity_im:.2f}')

            # ~ Calculation metrics for total test set
            r2 = r2_score(
                list(e[f'out/true/{rep}'].values()),
                list(e[f'out/{model_name}/{rep}'].values())
            )
            e[f'r2/{model_name}/{rep}'] = r2

            mse = history.history['val_output_1_mean_squared_error'][-1]
            e[f'mse/{model_name}/{rep}'] = mse

            rmse = np.sqrt(mse)
            e[f'rmse/{model_name}/{rep}'] = rmse

            e.info(f'r2: {r2:.2f} - mse: {mse:.2f} - rmse: {rmse:.2f}')

            # ~ Drawing the examples
            e.info('Drawing examples...')
            examples_path = os.path.join(e.path, f'{rep}_examples.pdf')
            ncols = len(MODEL_SWEEP)
            with PdfPages(examples_path) as pdf:
                for index in example_indices:
                    data = dataset_dict[index]
                    g = data['graph']
                    g['node_coordinates'] = np.array(g['node_coordinates'])
                    image = np.asarray(imread(data['image_path']))

                    fig, rows = plt.subplots(ncols=ncols, nrows=NUM_CHANNELS, figsize=(ncols * 8, 2 * 8),
                                             squeeze=False)
                    fig.suptitle(f'Element {index}\n'
                                 f'{data["metadata"]["smiles"]}\n'
                                 f'Ground Truth: {e[f"out/true/{rep}/{index}"]:.3f}')

                    for row_index in range(NUM_CHANNELS):
                        column_index = 0

                        # -- MULTI ATTENTION STUDENT --
                        for i, model_name in enumerate(MODEL_SWEEP.keys()):
                            ax_mas = rows[row_index][column_index]
                            if i == 0:
                                ax_mas.set_ylabel(f'Channel {row_index}: {CHANNEL_NAMES[row_index]}')

                            if row_index == 0:
                                ax_mas.set_title(
                                    f'Model "{model_name}"\n'
                                    f'Prediction: {e[f"out/{model_name}/{rep}/{index}"]:.2f}\n'
                                    f'Fidelity: {e[f"fidelity_im/{model_name}/{rep}/{index}"]:.2}'
                                )

                            ax_mas.imshow(image, extent=(0, image.shape[0], 0, image.shape[1]))
                            ni_mas = np.array(e[f'ni/{model_name}/{rep}/{index}'])
                            plot_node_importances(
                                g=g,
                                ax=ax_mas,
                                # vmax=1,
                                vmax=np.max(ni_mas),
                                # vmax=np.max(ni_mas[:, row_index]),
                                node_importances=ni_mas[:, row_index],
                                node_coordinates=g['node_coordinates']
                            )
                            ei_mas = np.array(e[f'ei/{model_name}/{rep}/{index}'])
                            plot_edge_importances(
                                g=g,
                                ax=ax_mas,
                                # vmax=1,
                                vmax=np.max(ei_mas),
                                # vmax=np.max(ei_mas[:, row_index]),
                                edge_importances=ei_mas[:, row_index],
                                node_coordinates=g['node_coordinates']
                            )
                            column_index += 1

                    pdf.savefig(fig)
                    plt.close(fig)

            e.info(f'finished repetition {rep + 1}')
            e.status()


with Skippable(), e.analysis:

    # -- RENDERING LATEX TABLE --
    e.info('Rendering latex table...')
    column_names = [
        'Model',
        'RMSE',
        '$R^2$',
        'Node Sparsity',
        'Edge Sparsity',
        r'$\text{Fidelity}^{*}$'
    ]
    rows = []
    for model_name, parameters in MODEL_SWEEP.items():
        row = []
        model_identifier = r'$\text{MEGAN}^{2}_{' + f'{parameters["IMPORTANCE_FACTOR"]:.1f}' + r'}$'
        row.append(model_identifier)
        row.append([e[f'rmse/{model_name}/{rep}']
                    for rep in range(REPETITIONS)])
        row.append([e[f'r2/{model_name}/{rep}']
                    for rep in range(REPETITIONS)])
        row.append([e[f'sparsity_ni/{model_name}/{rep}/{index}']
                    for rep in range(REPETITIONS) for index in e[f'test_indices']])
        row.append([e[f'sparsity_ei/{model_name}/{rep}/{index}']
                    for rep in range(REPETITIONS) for index in e[f'test_indices']])
        row.append([e[f'fidelity_im/{model_name}/{rep}/{index}']
                    for rep in range(REPETITIONS) for index in e[f'test_indices']])
        rows.append(row)

    content, table = latex_table(
        column_names=column_names,
        rows=rows,
        list_element_cb=latex_table_element_median,
    )
    e.commit_raw('table.tex', table)
    pdf_path = os.path.join(e.path, 'table.pdf')
    render_latex({'content': table}, output_path=pdf_path)
    e.info(f'rendered latex table: {pdf_path}')
