"""
This example shows how the internal implementation of GNES can be trained in an explanation supervised
manner using the synthetic rb motifs dataset

**IMPORTANT IMPLEMENTATION CAVEAT**
One important thing to note: Currently we are not able to provide an efficient implementation of GradCAM
(which GNES is based on) using Keras RaggedTensors. In the current implementation this leads to the
restriction that the model needs to know the exact size of each training batch prior to training. This
means that the training dataset has to have a number of elements which is exactly divisible by the
batch size, such that every batch is guaranteed to have the same size!
"""
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import typing as t

import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pycomex.experiment import Experiment
from pycomex.util import Skippable
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from imageio.v2 import imread

import graph_attention_student.typing as tc
from graph_attention_student.util import DATASETS_FOLDER
from graph_attention_student.util import array_normalize
from graph_attention_student.data import process_graph_dataset
from graph_attention_student.data import load_eye_tracking_dataset_dict
from graph_attention_student.training import LogProgressCallback
from graph_attention_student.training import NoLoss, ExplanationLoss
from graph_attention_student.training import mae
from graph_attention_student.models import GnesGradientModel
from graph_attention_student.models import grad_importances, grad_cam_importances
from graph_attention_student.visualization import plot_node_importances, plot_edge_importances


# == DATASET PARAMETERS ==
DATASET_PATH = os.path.join(DATASETS_FOLDER, 'rb_dual_motifs')
METADATA_CONTAINS_INDICES = True
TRAIN_RATIO = 0.9
NUM_EXAMPLES = 50

# == MODEL PARAMETERS ==
MODEL_CLASS = GnesGradientModel
IMPORTANCES_FUNC = lambda *args: grad_importances(*args, use_absolute=True, use_relu=False)
UNITS = [5, 5, 5]

# == TRAINING PARAMETERS ==
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.01
SPARSITY_FACTOR = 1.0
EXPLANATION_SUPERVISION = True

# == EVALUATION PARAMETERS ==
EVAL_BATCH_SIZE = 300

# == EXPERIMENT PARAMETERS ==
PATH = os.getcwd()
NAMESPACE = 'results/gnes_example'
DEBUG = True
with Skippable(), (e := Experiment(base_path=PATH, namespace=NAMESPACE, glob=globals())):
    e.info('starting gnes example...')

    e.info('loading rb motifs dataset...')
    dataset_dict = load_eye_tracking_dataset_dict(DATASET_PATH, logger=e.logger)
    dataset_size = len(dataset_dict)
    dataset_index_dict = {}
    indices = []
    dataset: t.List[tc.GraphDict] = []
    for c, (name, data) in enumerate(dataset_dict.items()):
        metadata = data['metadata']
        g = metadata['graph']

        if METADATA_CONTAINS_INDICES:
            index = metadata['index']
        else:
            index = c

        g['node_importances'] = np.sum(np.array(g['multi_node_importances']), axis=-1, keepdims=True)
        g['edge_importances'] = np.sum(np.array(g['multi_edge_importances']), axis=-1, keepdims=True)

        g['edge_attributes'] = np.array(g['edge_attributes'], dtype=np.float)

        # With this here we want to support the possibility that the "value" annotation might already
        # be a multi target annotation (as is the case with classification datasets for example) but usually
        # this will be a single regression value and we will have to wrap that into a list first
        if isinstance(metadata['value'], list):
            g['graph_labels'] = np.array(metadata['value'])
        else:
            g['graph_labels'] = np.array([metadata['value']])

        indices.append(index)
        dataset_index_dict[index] = data

    dataset = [t[1]['metadata']['graph'] for t in sorted(dataset_index_dict.items(), key=lambda t: t[0])]

    e.info('determining train test split...')
    # At this point we need to determine the number of training samples exactly such that it is divisible
    # by the batch size! Because the implementation of GradCAM currently requires that
    num_train_samples = int(TRAIN_RATIO * dataset_size)
    if num_train_samples % BATCH_SIZE != 0:
        remaining = num_train_samples % BATCH_SIZE
        num_train_samples -= remaining

    e.info(f'training dataset of size {num_train_samples}')

    train_indices = random.sample(indices, k=num_train_samples)
    test_indices = [index for index in indices if index not in train_indices]
    example_indices = random.sample(test_indices, k=NUM_EXAMPLES)
    x_train, y_train, x_test, y_test = process_graph_dataset(dataset, test_indices=test_indices)

    e.info('creating the model...')
    model: ks.models.Model = MODEL_CLASS(
        units=UNITS,
        num_outputs=1,
        batch_size=BATCH_SIZE,
        importance_func=IMPORTANCES_FUNC,
        sparsity_factor=SPARSITY_FACTOR
    )
    model.compile(
        loss=[
            ks.losses.MeanSquaredError(),
            ExplanationLoss(loss_function=mae) if EXPLANATION_SUPERVISION else NoLoss(),
            ExplanationLoss(loss_function=mae) if EXPLANATION_SUPERVISION else NoLoss(),
        ],
        loss_weights=[
            1,
            1,
            1
        ],
        metrics=[
            ks.metrics.MeanSquaredError(),
            ks.metrics.MeanAbsoluteError(),
        ],
        optimizer=ks.optimizers.Adam(learning_rate=LEARNING_RATE),
        run_eagerly=False
    )

    with tf.device('gpu:0'):
        # NOTE: validation during the training is explicitly disabled because the validation dataset would
        # have to be divisible by the batch size as well. As that would be too much of a hassle we rather
        # disable it and perform the assessment manually afterwards.
        hist = model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            # validation_data=(x_test, y_test),
            # validation_freq=1,
            callbacks=LogProgressCallback(
                logger=e.logger,
                epoch_step=1,
                identifier=f'output_1_loss'
            ),
            verbose=0
        )
        e.info('finished model training')
        e.info(f'model with {model.count_params()} parameters')
        history = hist.history
        e['history'] = history

        e.info('evaluating on the test set...')
        eval_indices = test_indices.copy()
        current = 0
        while current < len(test_indices):
            num_samples = min(len(test_indices) - current, EVAL_BATCH_SIZE)

            eval_indices = test_indices[current:current+num_samples]
            node_input = x_test[0][current:current+num_samples]
            edge_input = x_test[1][current:current+num_samples]
            edge_index_input = x_test[2][current:current+num_samples]

            x_eval = (node_input, edge_input, edge_index_input)
            out_pred, ni_pred, ei_pred = [v.numpy() for v in model(x_eval, batch_size=num_samples)]
            out_true = y_test[0][current:current+num_samples]
            for c, index in enumerate(eval_indices):
                e[f'out/pred/{index}'] = float(out_pred[c][0])
                e[f'out/true/{index}'] = float(out_true[c][0])
                e[f'ni/{index}'] = array_normalize(ni_pred[c])
                e[f'ei/{index}'] = array_normalize(ei_pred[c])

            current += num_samples
            e.info(f' * evaluated ({current}/{len(test_indices)})')

        e.info('calculating evaluation metrics...')
        errors = []
        ni_true = []
        ni_pred = []
        ei_true = []
        ei_pred = []
        for index in test_indices:
            data = dataset_index_dict[index]
            g = data['metadata']['graph']

            error = e[f'out/true/{index}'] - e[f'out/pred/{index}']
            errors.append(error)

            ni_pred += np.array(e[f'ni/{index}']).flatten().tolist()
            ni_true += np.array(g['node_importances']).flatten().tolist()

            ei_pred += np.array(e[f'ei/{index}']).flatten().tolist()
            ei_true += np.array(g['edge_importances']).flatten().tolist()

        mse = np.mean(np.square(errors))
        mae = np.mean(np.abs(errors))
        e['mse'] = mse
        e['mae'] = mae
        e.info(f' * mse: {mse}')

        r2 = r2_score(list(e['out/true'].values()), list(e['out/pred'].values()))
        e['r2'] = r2
        e.info(f' * r2: {r2}')

        ni_auc = roc_auc_score(ni_true, ni_pred)
        ei_auc = roc_auc_score(ei_true, ei_pred)
        e['ni_auc'] = ni_auc
        e['ei_auc'] = ei_auc
        e.info(f' * ni_auc: {ni_auc}')
        e.info(f' * ei_auc: {ei_auc}')

        e.info('drawing examples...')
        path = os.path.join(e.path, 'examples.pdf')
        num_channels = 1
        with PdfPages(path) as pdf:
            for c, index in enumerate(example_indices):
                ni = np.array(e[f'ni/{index}'])
                ei = np.array(e[f'ei/{index}'])

                data = dataset_index_dict[index]
                metadata = data['metadata']
                g = metadata['graph']
                g['node_coordinates'] = np.array(g['node_coordinates'])
                image = np.asarray(imread(data['image_path']))

                fig, rows = plt.subplots(ncols=2, nrows=num_channels, figsize=(16, 16),
                                         squeeze=False)

                for row_index in range(num_channels):
                    column_index = 0

                    # -- GROUND TRUTH --
                    ax_gt = rows[row_index][column_index]
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

                    # -- EXPLANATIONS --
                    ax_mas = rows[row_index][column_index]
                    ax_mas.imshow(image, extent=(0, image.shape[0], 0, image.shape[1]))
                    plot_node_importances(
                        g=g,
                        ax=ax_mas,
                        vmax=np.max(ni),
                        node_importances=ni[:, row_index],
                        node_coordinates=g['node_coordinates']
                    )
                    plot_edge_importances(
                        g=g,
                        ax=ax_mas,
                        vmax=np.max(ei),
                        edge_importances=ei[:, row_index],
                        node_coordinates=g['node_coordinates']
                    )
                    column_index += 1

                pdf.savefig(fig)
                plt.close(fig)

