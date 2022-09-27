"""
Trains a multi-channel MEGAN model on the MovieReviews dataset. The MovieReviews dataset consists of movie
reviews from IMDB whose sentiment has to be classified into either of the two classes "positive" or
"negative". The textual dataset has been pre-converted into a graph dataset. The MEGAN model is trained with
two explanation channels - one for each class. The experiment is repeated multiple independent times.
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
from sklearn.metrics import f1_score

from pycomex.experiment import Experiment
from pycomex.util import Skippable

from graph_attention_student.util import TEMPLATE_ENV
from graph_attention_student.util import render_latex
from graph_attention_student.util import latex_table, latex_table_element_median
from graph_attention_student.util import DatasetError
from graph_attention_student.util import array_normalize
from graph_attention_student.util import binary_threshold
from graph_attention_student.data import load_text_graph_dataset
from graph_attention_student.data import load_eye_tracking_dataset
from graph_attention_student.data import process_graph_dataset
from graph_attention_student.models import MultiAttentionStudent
from graph_attention_student.training import NoLoss
from graph_attention_student.training import LogProgressCallback
from graph_attention_student.visualization import fixed_row_layout
from graph_attention_student.visualization import plot_text_graph

SHORT_DESCRIPTION = (
    'Trains multi-channel MEGAN model on the MovieReviews dataset.'
)

# == META PARAMETERS ==
REPETITIONS = 50

# == DATASET PARAMETERS ==
# [!] Insert the path to the dataset folder here.
DATASET_PATH = '/home/jonas/Data/Programming/PyCharm/graph_attention_student/graph_attention_student' \
               '/experiments/process_movie_reviews/debug/dataset'
NUM_EXAMPLES = 20
EXAMPLE_INDICES: Optional[list] = [506, 779, 1999, 1856, 1080, 435, 1554, 35, 1855]

# == MODEL PARAMETERS ==
UNITS = [45, 40, 35, 30, 25]
IMPORTANCE_UNITS = [20]
IMPORTANCE_FACTOR = 0.5
IMPORTANCE_MULTIPLIER = 10
IMPORTANCE_SPARSITY = 0
SPARSITY_FACTOR = 0.1
NUM_CHANNELS = 2
FINAL_UNITS = [30, 20, 10]
DROPOUT_RATE = 0.2
FINAL_DROPOUT_RATE = 0.0
USE_BIAS = True

# == TRAINING PARAMETERS ==
LEARNING_RATE = 0.001
BATCH_SIZE = 50
EPOCHS = 100
LOG_STEP = 5
METRIC_KEY = 'categorical_accuracy'
DEVICE = 'gpu:0'

# == EVALUATION PARAMETERS ==
LOG_STEP_EVAL = 20

# == EXPERIMENT PARAMETERS ==
NAMESPACE = 'movie_reviews_multi'
BASE_PATH = os.getcwd()
DEBUG = True

with Skippable(), (e := Experiment(namespace=NAMESPACE, base_path=BASE_PATH, glob=globals())):

    e.info('Loading the dataset...')
    if not os.path.exists(DATASET_PATH) or not os.path.isdir(DATASET_PATH):
        raise DatasetError(f'The given dataset path "{DATASET_PATH}" either does not exists or is not a '
                           f'valid dataset folder. Please make sure to specify a valid movie reviews '
                           f'dataset before running the experiment again')

    start_time = time.time()
    dataset_dict: Dict[str, dict] = load_text_graph_dataset(DATASET_PATH, logger=e.logger, log_interval=250)
    dataset_size = len(dataset_dict)
    end_time = time.time()
    e.info(f'loaded dataset with {dataset_size} elements '
           f'in {end_time - start_time:.2f} seconds')
    first_element = list(dataset_dict.values())[0]
    e.info(f'dataset element keys: {str(list(first_element.keys()))}')

    e.info('Preparing dataset for training...')
    dataset_indices = list(dataset_dict.keys())
    test_indices = [int(k) for k, v in dataset_dict.items() if v['metadata']['type'] == 'val']
    train_indices = [int(k) for k, v in dataset_dict.items() if int(k) not in test_indices]
    e['test_indices'] = test_indices
    e['train_indices'] = train_indices
    e.info(f'Identified {len(train_indices)} train elements and {len(test_indices)} test_elements')

    # We need to initialize the dataset list here with actual values, because we need to do indexed
    # assignments to get the elements to their correct position within this list...
    dataset = [None for _ in range(dataset_size)]
    for key, data in dataset_dict.items():
        index = int(key)

        g = data['metadata']['graph']
        # We need to create the target labels as a one hot encoded classification vector
        g['graph_labels'] = np.array([1 if int(c) == int(data['metadata']['class']) else 0
                                      for c in data['metadata']['classes'].keys()])

        dataset[index] = g

    x_train, y_train, x_test, y_test = process_graph_dataset(dataset, test_indices)
    e.info('Processed dataset into tensors')

    for rep in range(REPETITIONS):
        e.info(f'REPETITION ({rep + 1}/{REPETITIONS})')

        example_indices = EXAMPLE_INDICES + random.sample(test_indices,
                                                          k=NUM_EXAMPLES - len(EXAMPLE_INDICES))
        e[f'example_indices/{rep}'] = example_indices
        e.info(f'Identified {len(example_indices)} example elements from the test set')

        e.info('Starting model training...')
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
            final_activation='softmax'
        )
        model.compile(
            loss=[
                ks.losses.CategoricalCrossentropy(),
                NoLoss(),
                NoLoss()
            ],
            loss_weights=[
                1,
                0,
                0
            ],
            metrics=[
                ks.metrics.CategoricalAccuracy(),
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

        e[f'history/{rep}'] = history.history
        e[f'epochs/{rep}'] = list(range(EPOCHS))
        e.info(f'history keys: {list(history.history.keys())}')

        # -- Evaluating test set --
        e.info('Evaluating test set...')
        for c, index in enumerate(test_indices):
            g = dataset[index]

            # First we are going to save all of the model outputs itself
            out, node_importances, edge_importances = model.predict_single([
                g['node_attributes'],
                g['edge_attributes'],
                g['edge_indices'],
            ])
            out_true = g['graph_labels']
            e[f'out/true/{index}'] = out_true
            e[f'out/{rep}/{index}'] = out
            ni = array_normalize(np.array(node_importances))
            ei = array_normalize(np.array(edge_importances))
            e[f'ni/{rep}/{index}'] = ni
            e[f'ei/{rep}/{index}'] = ei

            # Sparsity is super easy
            sparsity_ni = np.mean(binary_threshold(ni, 0.5))
            sparsity_ei = np.mean(binary_threshold(ei, 0.5))
            e[f'sparsity_ni/{rep}/{index}'] = sparsity_ni
            e[f'sparsity_ei/{rep}/{index}'] = sparsity_ei

            # Fidelity
            fidelity_im = 0
            for k in range(NUM_CHANNELS):
                ni_mod = np.copy(ni)
                ni_mod[:, k] = 0
                out_mod, _, _ = model.predict_single((
                    g['node_attributes'],
                    g['edge_attributes'],
                    g['edge_indices']
                ), external_node_importances=ni_mod)
                fidelity_im += (np.array(out) - np.array(out_mod))[k]
                e[f'out_mod_fidelity/{rep}/{index}/{k}'] = out_mod

            e[f'fidelity_im/{rep}/{index}'] = fidelity_im

            if c % LOG_STEP_EVAL == 0:
                e.info(f' * ({c}/{len(test_indices)}) {index}'
                       f' - true: {[f"{v:.2f}" for v in out_true]}'
                       f' - pred: {[f"{v:.2f}" for v in out]}'
                       f' - sparsity_ni: {sparsity_ni:.2f}'
                       f' - sparsity_ei: {sparsity_ei:.2f}'
                       f' - fidelity: {fidelity_im:.2f}')

        acc = history.history['val_output_1_categorical_accuracy'][-1]
        e[f'acc/{rep}'] = acc

        y_true = [np.argmax(v) for v in e['out/true'].values()]
        y_pred = [np.argmax(v) for v in e[f'out/{rep}'].values()]
        f1 = f1_score(y_true, y_pred)
        e[f'f1/{rep}'] = f1

        e.info(f'acc: {acc:.2f} - f1: {f1:.2f}')

        # -- Rendering examples as plots and latex --
        e.info('rendering examples...')
        examples_name = f'{rep:02d}_examples'
        examples_path = os.path.join(e.path, examples_name)
        os.mkdir(examples_path)
        explanations_path = os.path.join(examples_path, f'explanations.pdf')
        with PdfPages(explanations_path) as pdf:

            for index in e[f'example_indices/{rep}']:
                g = dataset[int(index)]
                g['node_positions'] = fixed_row_layout(g, ncols=20)
                out, node_importances, edge_importances = model.predict_single([
                    g['node_attributes'],
                    g['edge_attributes'],
                    g['edge_indices'],
                ])
                label = 'neg' if np.argmax(out) == 0 else 'pos'

                # -- MATPLOTLIB VISUALIZATION --
                fig, rows = plt.subplots(nrows=2, ncols=2, figsize=(60, 60), squeeze=False)
                ax_true_neg, ax_true_pos = rows[0]
                max_true = np.max(g['node_importances'])
                plot_text_graph(
                    g=g,
                    ax=ax_true_neg,
                    node_importances=[v[0] for v in g['node_importances']],
                    edge_importances=[v[0] for v in g['edge_importances']],
                    vmax=max_true,
                    do_edges=False
                )
                plot_text_graph(
                    g=g,
                    ax=ax_true_pos,
                    node_importances=[v[1] for v in g['node_importances']],
                    edge_importances=[v[1] for v in g['edge_importances']],
                    vmax=max_true,
                    do_edges=False
                )

                ax_pred_neg, ax_pred_pos = rows[1]
                max_pred = np.max(node_importances)
                plot_text_graph(
                    g=g,
                    ax=ax_pred_neg,
                    node_importances=[v[0] for v in node_importances],
                    edge_importances=[v[0] for v in edge_importances],
                    vmax=max_pred,
                    do_edges=False
                )
                plot_text_graph(
                    g=g,
                    ax=ax_pred_pos,
                    node_importances=[v[1] for v in node_importances],
                    edge_importances=[v[1] for v in edge_importances],
                    vmax=max_pred,
                    do_edges=False
                )

                # Defining what the columns and the rows show
                ax_true_neg.set_title('NEGATIVE', fontsize=50)
                ax_true_pos.set_title('POSITIVE', fontsize=50)

                ax_true_neg.set_ylabel('Ground Truth', fontsize=50)
                ax_pred_neg.set_ylabel('Prediction', fontsize=50)

                pdf.savefig(fig)
                plt.close(fig)

                # -- LATEX VISUALIZATION --
                norm = mcolors.Normalize(vmin=0, vmax=max_pred)
                importance_template = TEMPLATE_ENV.get_template('importances.tex.j2')
                latex_negative = importance_template.render(
                    token_list=g['node_strings'],
                    importance_list=[norm(v[0]) * 0.8 for v in node_importances]
                )
                e.commit_raw(os.path.join(examples_name, f'{index}_neg.tex'), latex_negative)
                latex_positive = importance_template.render(
                    token_list=g['node_strings'],
                    importance_list=[norm(v[1]) * 0.8 for v in node_importances]
                )
                e.commit_raw(os.path.join(examples_name, f'{index}_pos.tex'), latex_positive)

                try:
                    output_path = os.path.join(examples_path, f'explanation_{index}_{label}.pdf')
                    render_latex(
                        kwargs={'negative': latex_negative, 'positive': latex_positive},
                        output_path=output_path,
                        template_name='movie_review.tex.j2'
                    )
                except:
                    pass


with Skippable(), e.analysis:

    # -- POST CALCULATIONS --
    e.info('post-hoc calculations...')
    if 'f1' not in e.data:
        for rep in range(REPETITIONS):
            y_true = [np.argmax(v) for v in e['out/true'].values()]
            y_pred = [np.argmax(v) for v in e[f'out/{rep}'].values()]
            f1 = f1_score(y_true, y_pred)
            e[f'f1/{rep}'] = f1

    # -- RENDERING LATEX TABLE --
    e.info('Rendering latex table...')
    column_names = [
        'Model',
        'F1',
        'Node Sparsity',
        'Edge Sparsity',
        r'$\text{Fidelity}^{*}$'
    ]
    rows = []
    row = []
    model_identifier = r'$\text{MEGAN}^{2}_{' + f'{IMPORTANCE_FACTOR:.1f}' + r'}$'
    row.append(model_identifier)
    row.append([e[f'f1/{rep}']
                for rep in range(REPETITIONS)])
    row.append([e[f'sparsity_ni/{rep}/{index}']
                for rep in range(REPETITIONS) for index in e[f'test_indices']])
    row.append([e[f'sparsity_ei/{rep}/{index}']
                for rep in range(REPETITIONS) for index in e[f'test_indices']])
    row.append([e[f'fidelity_im/{rep}/{index}']
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