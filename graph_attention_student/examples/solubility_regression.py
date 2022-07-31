"""
This module contains an example of how the ``MultiAttentionStudent`` model can be trained to predict the
aq. solubility of molecules and also produce multi-channel explanations in the process.

This module is structured as a ``pycomex`` computational experiment module. Check out the ``pycomex``
project to see the benefits this provides:

https://github.com/the16thpythonist/pycomex
"""
import os
import pathlib
import random
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import matplotlib.pyplot as plt
from imageio.v2 import imread
from matplotlib.backends.backend_pdf import PdfPages
from pycomex.experiment import Experiment

from graph_attention_student.util import DATASETS_FOLDER, PATH
from graph_attention_student.data import load_eye_tracking_dataset
from graph_attention_student.data import process_graph_dataset
from graph_attention_student.layers import StaticMultiplicationEmbedding
from graph_attention_student.models import MultiAttentionStudent
from graph_attention_student.training import LogProgressCallback
from graph_attention_student.training import NoLoss
from graph_attention_student.visualization import plot_node_importances, plot_edge_importances

# Dataset related
DATASET_PATH = os.path.join(DATASETS_FOLDER, 'solubility_aqsoldb')
TEST_RATIO = 0.1
NUM_EXAMPLES = 100

# Model related
IMPORTANCE_CHANNELS = [-10, 10]
NUM_CHANNELS = len(IMPORTANCE_CHANNELS)
DROPOUT_RATE = 0.2
SPARSITY_FACTOR = 0.1
EXCLUSIVITY_FACTOR = 0

# Training related
LEARNING_RATE = 0.01
BATCH_SIZE = 256
EPOCHS = 500

NAMESPACE = 'solubility_regression'
BASE_PATH = os.getcwd()
DEBUG = True
with Experiment(namespace=NAMESPACE, base_path=BASE_PATH, glob=globals()) as e:
    e.prepare()

    # ~ Loading dataset

    # This function loads a "eye tracking dataset": This simply means that the dataset is structured in a
    # special way, namely like this: The dataset consists of an entire folder, where every element has two
    # files. One is a JSON file, which actually contains the graph structure that represents the element as
    # well as the prediction target etc. And then there is also an image file for each element which is a
    # visual representation of the graph. We can later use this to visualize the explanations generated by
    # the model.
    eye_tracking_dataset: List[dict] = load_eye_tracking_dataset(DATASET_PATH)
    dataset_size = len(eye_tracking_dataset)
    e.info(f'loaded dataset with {dataset_size} elements')

    dataset_indices = list(range(dataset_size))
    test_indices = random.sample(dataset_indices, k=int(TEST_RATIO * dataset_size))
    example_indices = random.sample(test_indices, k=NUM_EXAMPLES)
    train_indices = [i for i in dataset_indices if i not in test_indices]
    e['test_indices'] = test_indices
    e['example_indices'] = example_indices
    e.info(f'randomly chose {len(test_indices)} test indices')

    # "eye_tracking_dataset" at this point is only a list of dictionaries which contain the metadata info
    # for each respective elements. This now has to be turned into the appropriate tensors to be used by
    # the model.
    dataset = []
    for data in eye_tracking_dataset:
        g = data['metadata']['graph']
        g['graph_labels'] = np.array(data['metadata']['solubility'])
        dataset.append(g)

    x_train, y_train, x_test, y_test = process_graph_dataset(dataset, test_indices)

    model: ks.models.Model = MultiAttentionStudent(
        units=[10, 7, 5],
        dropout_rate=DROPOUT_RATE,
        sparsity_factor=SPARSITY_FACTOR,
        exclusivity_factor=EXCLUSIVITY_FACTOR,
        importance_channels=NUM_CHANNELS,
        final_units=[],
        final_activation='softmax',
        lay_additional_cb=lambda: StaticMultiplicationEmbedding(values=IMPORTANCE_CHANNELS)
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
            ks.metrics.MeanAbsoluteError()
        ],
        optimizer=ks.optimizers.Adam(learning_rate=LEARNING_RATE),
        run_eagerly=False
    )

    e.info('starting to train the model...')
    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        validation_freq=1,
        callbacks=LogProgressCallback(
            logger=e.logger,
            epoch_step=25,
            identifier='val_output_1_mean_squared_error'
        ),
        verbose=0
    )
    e['history'] = history.history
    e['epochs'] = list(range(EPOCHS))

    # Running the whole test set through the model and saving the predictions
    for index in test_indices:
        g = dataset[index]
        prediction, node_importances, edge_importances = model.predict_single([
            g['node_attributes'],
            g['edge_attributes'],
            g['edge_indices']
        ])
        e[f'predictions/{index}/prediction'] = prediction
        e[f'predictions/{index}/node_importances'] = node_importances
        e[f'predictions/{index}/edge_importances'] = edge_importances

    # All of this code will automatically also be copied into the "analysis.py" file in the results folder
    # which can then also be executed again independently.
    with e.analysis:

        # visualizing the prediction metric over training
        fig, (ax_pred) = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))
        ax_pred.set_title('Main Metric over Training Epochs')
        ax_pred.set_xlabel('Epochs')
        ax_pred.set_ylabel('MSE')
        ax_pred.set_ylim([0, 4])
        ax_pred.plot(e['epochs'], e['history/val_output_1_mean_squared_error'], c='blue', label='test')
        ax_pred.plot(e['epochs'], e['history/output_1_mean_squared_error'], c='blue', alpha=0.5,
                     label='train')
        ax_pred.legend()
        e.commit_fig('prediction_over_epochs.pdf', fig)

        # visualizing the explanations
        explanations_path = os.path.join(e.path, 'explanations.pdf')
        with PdfPages(explanations_path) as pdf:

            for index in e['example_indices']:
                data = eye_tracking_dataset[index]
                g = data['metadata']['graph']

                fig, rows = plt.subplots(nrows=1, ncols=NUM_CHANNELS, figsize=(8*NUM_CHANNELS, 8),
                                         squeeze=False)
                fig.suptitle(f'Ground Truth: {data["metadata"]["solubility"]:.3f} - '
                             f'Prediction: {e["predictions"][str(index)]["prediction"]:.3f}')
                image = np.asarray(imread(data['image_path']))

                for ax_index, ax in enumerate(rows[0]):
                    ax.set_title(f'Channel {ax_index}: {IMPORTANCE_CHANNELS[ax_index]}')
                    ax.imshow(image, extent=(0, image.shape[0], 0, image.shape[1]))
                    plot_node_importances(
                        g=g,
                        ax=ax,
                        vmax=max(np.array(e[f'predictions/{index}/node_importances'])[:, ax_index]),
                        node_importances=np.array(e[f'predictions/{index}/node_importances'])[:, ax_index],
                        node_coordinates=data['graph']['node_coordinates']
                    )
                    plot_edge_importances(
                        g=g,
                        ax=ax,
                        vmax=max(np.array(e[f'predictions/{index}/edge_importances'])[:, ax_index]),
                        edge_importances=np.array(e[f'predictions/{index}/edge_importances'])[:, ax_index],
                        node_coordinates=data['graph']['node_coordinates']
                    )

                pdf.savefig(fig)
                plt.close(fig)
